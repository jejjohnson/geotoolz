"""Tests for `geopatcher.cog` — the async face of the obstore COG engine.

Every async read is checked against the matching sync ``georeader.read``
function on a `RasterioReader` over the same file: the async path only
changes *how* the source pixels are fetched, never the result. Skipped
unless the ``[obstore-cog]`` extra is installed; no network (``LocalStore``).
"""

from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.windows
from georeader import read
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform as transform_points, transform_bounds
from rasterio.windows import Window


pytest.importorskip("obstore")
pytest.importorskip("async_geotiff")

from obstore.store import LocalStore

from geopatcher import (
    AsyncSpatialPatcher,
    ObstoreCogField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
    cog,
)
from geopatcher._src.fields import obstore_cog as oc_mod


# 3 bands, 90 x 110 with 32-px tiles: partial edge tiles on both axes.
_H, _W = 90, 110
_TRANSFORM = Affine(30.0, 0.0, 500_000.0, 0.0, -30.0, 4_000_000.0)
_NODATA = 9


def _write_cog(path: Path, *, overviews: tuple[int, ...] = (2, 4)) -> Path:
    rng = np.random.default_rng(0)
    data = rng.integers(100, 5000, size=(3, _H, _W)).astype("uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_H,
        width=_W,
        count=3,
        dtype="uint16",
        crs="EPSG:32630",
        transform=_TRANSFORM,
        nodata=_NODATA,
        tiled=True,
        blockxsize=32,
        blockysize=32,
        compress="deflate",
    ) as dst:
        dst.write(data)
        if overviews:
            dst.build_overviews(list(overviews), rasterio.enums.Resampling.nearest)
    return path


@pytest.fixture
def cog_path(tmp_path: Path) -> Path:
    return _write_cog(tmp_path / "scene.tif")


def _open(path: Path, **kwargs) -> cog.AsyncCogReader:
    return asyncio.run(
        cog.AsyncCogReader.open(
            f"file://{path}",
            store=LocalStore(prefix=str(path.parent)),
            path=path.name,
            **kwargs,
        )
    )


def _assert_same(got: GeoTensor, expected: GeoTensor) -> None:
    assert isinstance(got, GeoTensor)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
    assert got.transform.almost_equals(expected.transform, precision=1e-6)
    assert rasterio.crs.CRS.from_user_input(
        got.crs
    ) == rasterio.crs.CRS.from_user_input(expected.crs)
    assert got.fill_value_default == expected.fill_value_default


# ---------------------------------------------------------------- metadata
def test_metadata_matches_rasterio_reader(cog_path: Path):
    reader = _open(cog_path)
    sync = RasterioReader(str(cog_path))
    assert reader.shape == sync.shape == (3, _H, _W)
    assert reader.transform == sync.transform
    assert reader.bounds == pytest.approx(sync.bounds)
    assert reader.res == pytest.approx(sync.res)
    assert reader.dtype == np.dtype(sync.dtype)
    assert reader.fill_value_default == sync.fill_value_default == _NODATA
    assert rasterio.crs.CRS.from_user_input(reader.crs) == sync.crs
    assert reader.footprint().equals(sync.footprint())
    assert reader.footprint("EPSG:4326").equals_exact(sync.footprint("EPSG:4326"), 1e-9)


def test_views_do_no_io_and_compose(cog_path: Path, monkeypatch):
    reader = _open(cog_path)
    calls: list[int] = []

    async def _spy(level, coords):
        calls.append(len(coords))
        return []

    monkeypatch.setattr(oc_mod, "_fetch_and_decode_tiles", _spy)
    view = reader.read_from_window(Window(10, 20, 40, 30))
    inner = view.read_from_window(Window(5, 5, 8, 8))
    assert calls == []  # views are lazy
    assert inner.window_focus == Window(15, 25, 8, 8)
    assert inner.transform == _TRANSFORM * Affine.translation(15, 25)
    assert inner.shape == (3, 8, 8)


# ---------------------------------------------------------------- windows
_WINDOWS = [
    Window(0, 0, _W, _H),
    Window(17, 9, 50, 40),  # straddles tile seams
    Window(90, 70, 40, 40),  # past the right / bottom edge
    Window(-7, -5, 20, 20),  # negative offsets
    Window(10.4, 20.6, 30.2, 10.7),  # fractional: snapped outward
]


@pytest.mark.parametrize("window", _WINDOWS, ids=str)
def test_read_from_window_matches_sync(cog_path: Path, window: Window):
    reader = _open(cog_path)
    got = asyncio.run(cog.read_from_window(reader, window))
    expected = read.read_from_window(
        RasterioReader(str(cog_path)),
        window_utils_round_outer(window),
        trigger_load=True,
        boundless=True,
    )
    _assert_same(got, expected)


def window_utils_round_outer(window: Window) -> Window:
    from georeader import window_utils

    return window_utils.round_outer_window(window)


def test_boundless_false_clips_and_rejects_disjoint(cog_path: Path):
    reader = _open(cog_path)
    clipped = asyncio.run(
        reader.read_from_window(Window(90, 70, 40, 40), boundless=False).load()
    )
    assert clipped.shape == (3, _H - 70, _W - 90)
    with pytest.raises(rasterio.windows.WindowError):
        reader.read_from_window(Window(_W + 5, 0, 4, 4), boundless=False)


def test_load_many_fetches_shared_tiles_once(cog_path: Path, monkeypatch):
    reader = _open(cog_path)
    observed: list[list[tuple[int, int]]] = []
    original = oc_mod._fetch_and_decode_tiles

    async def _spy(level, coords):
        observed.append(list(coords))
        return await original(level, coords)

    monkeypatch.setattr(oc_mod, "_fetch_and_decode_tiles", _spy)
    windows = [Window(0, 0, 20, 20), Window(5, 5, 20, 20), Window(40, 40, 30, 30)]
    chips = asyncio.run(reader.load_many(windows))
    assert len(observed) == 1  # one batch for all windows
    assert len(observed[0]) == len(set(observed[0]))  # no tile twice
    sync = RasterioReader(str(cog_path))
    for chip, window in zip(chips, windows, strict=True):
        _assert_same(chip, read.read_from_window(sync, window, trigger_load=True))


# ---------------------------------------------------------------- read_from_*
def test_read_from_bounds_matches_sync(cog_path: Path):
    reader = _open(cog_path)
    sync = RasterioReader(str(cog_path))
    bounds = (500_500.0, 3_998_000.0, 502_100.0, 3_999_500.0)
    got = asyncio.run(cog.read_from_bounds(reader, bounds, pad_add=(2, 2)))
    _assert_same(
        got, read.read_from_bounds(sync, bounds, pad_add=(2, 2), trigger_load=True)
    )
    bounds_wgs = transform_bounds("EPSG:32630", "EPSG:4326", *bounds)
    got = asyncio.run(cog.read_from_bounds(reader, bounds_wgs, crs_bounds="EPSG:4326"))
    expected = read.read_from_bounds(
        sync, bounds_wgs, crs_bounds="EPSG:4326", trigger_load=True
    )
    _assert_same(got, expected)


def test_read_from_polygon_matches_sync(cog_path: Path):
    from shapely.geometry import Polygon

    reader = _open(cog_path)
    sync = RasterioReader(str(cog_path))
    poly = Polygon([(500_300, 3_999_700), (502_000, 3_998_500), (501_000, 3_997_800)])
    for surrounding in (False, True):
        got = asyncio.run(
            cog.read_from_polygon(reader, poly, window_surrounding=surrounding)
        )
        expected = read.read_from_polygon(
            sync, poly, window_surrounding=surrounding, trigger_load=True
        )
        _assert_same(got, expected)


def test_read_from_center_coords_matches_sync(cog_path: Path):
    reader = _open(cog_path)
    sync = RasterioReader(str(cog_path))
    for center in (
        (501_000.0, 3_998_500.0),
        (500_050.0, 3_999_950.0),
    ):  # inside, at corner
        got = asyncio.run(cog.read_from_center_coords(reader, center, (16, 16)))
        expected = read.read_from_center_coords(
            sync, center, (16, 16), trigger_load=True
        )
        _assert_same(got, expected)


# ---------------------------------------------------------------- reprojection
@pytest.mark.parametrize(
    "bounds",
    [
        (500_600.0, 3_998_000.0, 502_400.0, 3_999_400.0),  # inside
        (502_500.0, 3_996_500.0, 504_000.0, 3_998_000.0),  # over the bottom-right edge
        (600_000.0, 3_000_000.0, 600_300.0, 3_000_300.0),  # off the image
    ],
    ids=["inside", "edge", "outside"],
)
@pytest.mark.parametrize("resampling", [Resampling.cubic_spline, Resampling.nearest])
def test_read_reproject_matches_sync(cog_path: Path, bounds, resampling):
    reader = _open(cog_path)
    sync = RasterioReader(str(cog_path))
    kwargs = {
        "dst_crs": "EPSG:32631",
        "resampling": resampling,
        "resolution_dst_crs": 25.0,
    }
    # Bounds are given in the source CRS; express them in the destination CRS.
    dst_bounds = transform_bounds("EPSG:32630", "EPSG:32631", *bounds)
    got = asyncio.run(cog.read_reproject(reader, bounds=dst_bounds, **kwargs))
    expected = read.read_reproject(sync, bounds=dst_bounds, **kwargs)
    _assert_same(got, expected)


def test_read_reproject_like_matches_sync(cog_path: Path):
    reader = _open(cog_path)
    sync = RasterioReader(str(cog_path))
    # A WGS84 grid over the scene's south-west corner: part inside, part off.
    (lon,), (lat,) = transform_points(
        "EPSG:32630", "EPSG:4326", [500_600.0], [3_997_900.0]
    )
    like = GeoTensor(
        np.zeros((1, 40, 50), dtype="float32"),
        transform=Affine(0.0003, 0.0, lon, 0.0, -0.0003, lat),
        crs="EPSG:4326",
        fill_value_default=0,
    )
    got = asyncio.run(cog.read_reproject_like(reader, like))
    _assert_same(got, read.read_reproject_like(sync, like))


def test_read_to_crs_matches_sync(cog_path: Path):
    reader = _open(cog_path)
    got = asyncio.run(cog.read_to_crs(reader, "EPSG:4326"))
    _assert_same(got, read.read_to_crs(RasterioReader(str(cog_path)), "EPSG:4326"))


def test_read_from_tile_is_a_web_mercator_reprojection(cog_path: Path):
    import mercantile

    reader = _open(cog_path)
    sync = RasterioReader(str(cog_path))
    (lon,), (lat,) = transform_points(  # the scene centre
        "EPSG:32630", "EPSG:4326", [500_000.0 + 15 * _W], [4_000_000.0 - 15 * _H]
    )
    for z in (12, 15):
        tile = mercantile.tile(lon, lat, z)
        got = asyncio.run(cog.read_from_tile(reader, tile.x, tile.y, z))
        b = mercantile.xy_bounds(tile)
        expected = read.read_reproject(
            sync,
            dst_crs="EPSG:3857",
            dst_transform=rasterio.transform.from_bounds(*b, 256, 256),
            window_out=Window(0, 0, 256, 256),
        )
        _assert_same(got, expected)
        assert got.shape == (3, 256, 256)
    off = asyncio.run(cog.read_from_tile(reader, 0, 0, 3))  # far from the scene
    assert (np.asarray(off) == _NODATA).all()


# ---------------------------------------------------------------- overviews
def test_overviews_and_reader_overview(cog_path: Path):
    reader = _open(cog_path)
    assert reader.overviews() == RasterioReader(str(cog_path)).overviews() == [2, 4]
    for level in (0, 1):
        ovr = reader.reader_overview(level)
        sync = RasterioReader(str(cog_path), overview_level=level)
        assert ovr.shape == sync.shape
        assert ovr.transform.almost_equals(sync.transform, precision=1e-9)
        assert ovr.overviews() == []
        _assert_same(
            asyncio.run(ovr.read_from_window(Window(3, 2, 20, 15)).load()),
            read.read_from_window(sync, Window(3, 2, 20, 15), trigger_load=True),
        )
        # Same field as opening the overview IFD directly.
        direct = _open(cog_path, ifd_index=ovr.field.ifd_index)
        assert direct.shape == ovr.shape
    with pytest.raises(ValueError, match="out of range"):
        reader.reader_overview(5)


# ---------------------------------------------------------------- integration
def test_pickle_roundtrip(cog_path: Path):
    view = _open(cog_path).read_from_window(Window(10, 10, 20, 20))
    clone = pickle.loads(pickle.dumps(view))
    _assert_same(asyncio.run(clone.load()), asyncio.run(view.load()))


def test_open_inside_a_running_loop(cog_path: Path):
    async def main() -> tuple[int, ...]:
        reader = await cog.AsyncCogReader.open(
            f"file://{cog_path}",
            store=LocalStore(prefix=str(cog_path.parent)),
            path=cog_path.name,
        )
        tiles = await asyncio.gather(
            cog.read_from_tile(reader, 2017, 1622, 12),
            reader.read_from_window(Window(0, 0, 8, 8)).load(),
        )
        return tiles[1].shape

    assert asyncio.run(main()) == (3, 8, 8)


def test_async_patcher_reads_obstore_cog_field_via_aselect(cog_path: Path):
    """`ObstoreCogField.aselect` plugs the field straight into `AsyncSpatialPatcher`."""
    field = ObstoreCogField.from_url(
        f"file://{cog_path}",
        store=LocalStore(prefix=str(cog_path.parent)),
        path=cog_path.name,
    )
    kwargs = {
        "geometry": SpatialRectangular(size=(32, 32)),
        "sampler": SpatialRegularStride(step=32),
        "window": SpatialBoxcar(),
        "aggregation": SpatialOverlapAdd(),
    }

    async def collect() -> list:
        return [p async for p in AsyncSpatialPatcher(**kwargs).asplit(field)]

    async_patches = asyncio.run(collect())
    sync_patches = list(SpatialPatcher(**kwargs).split(field))
    assert len(async_patches) == len(sync_patches) > 0
    for a, s in zip(async_patches, sync_patches, strict=True):
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(s.data))
