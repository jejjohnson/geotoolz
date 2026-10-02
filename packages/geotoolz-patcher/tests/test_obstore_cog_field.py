"""Tests for ``ObstoreCogField`` — COG reads via obstore + async-tiff.

Skipped unless both the ``obstore`` and ``async-tiff`` extras are
installed (the ``[obstore-cog]`` extra). The tests write a small
tiled GeoTIFF to a tempdir, point an obstore ``LocalStore`` at it,
and exercise the full read path — including the batched
``select_many`` route — without any network. One test serves the
fixture over a local range-capable HTTP server to cover ``HTTPStore``.

Every fixture uses a lossless codec, so reads must match
``rasterio.read`` bit for bit.
"""

from __future__ import annotations

import functools
import http.server
import pickle
import threading
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.crs
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.windows import Window


pytest.importorskip("obstore")
pytest.importorskip("async_tiff")

from georeader.geotensor import GeoTensor
from obstore.store import LocalStore

from geopatcher import (
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher._src.fields.obstore_cog import (
    ObstoreCogField,
    _tile_range_for_window,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cog_path(tmp_path: Path) -> Path:
    """Write a tiny tiled GeoTIFF (the COG-shape minimum) to a tempdir."""
    path = tmp_path / "test.tif"
    height = 64
    width = 64
    data = np.arange(height * width, dtype=np.float32).reshape(height, width)
    transform = from_bounds(500_000, 4_000_000, 500_640, 4_000_640, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32629",
        transform=transform,
        tiled=True,
        blockxsize=16,
        blockysize=16,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def cog_field(tmp_path: Path, cog_path: Path) -> ObstoreCogField:
    """Open the fixture COG through a LocalStore (no network)."""
    store = LocalStore(prefix=str(tmp_path))
    return ObstoreCogField.from_url(
        url=f"file://{cog_path}",
        store=store,
        path="test.tif",
    )


# ---------------------------------------------------------------------------
# Tile-range math (pure function — no obstore needed)
# ---------------------------------------------------------------------------


def test_tile_range_single_tile():
    # 4x4 window starting at (0,0) on a 16x16 tile grid → one tile.
    r = _tile_range_for_window(
        Window(col_off=0, row_off=0, width=4, height=4),
        tile_w=16,
        tile_h=16,
        image_w=64,
        image_h=64,
    )
    assert r == (0, 0, 0, 0)


def test_tile_range_spans_multiple_tiles():
    # 32x32 window starting at (8,8) on a 16x16 tile grid → tiles
    # (0,0), (0,1), (1,0), (1,1), (2,1), (1,2), (2,2).
    r = _tile_range_for_window(
        Window(col_off=8, row_off=8, width=32, height=32),
        tile_w=16,
        tile_h=16,
        image_w=64,
        image_h=64,
    )
    assert r == (0, 0, 2, 2)


def test_tile_range_clamps_to_image_bounds():
    # Window extends past the image — tile range clamps.
    r = _tile_range_for_window(
        Window(col_off=48, row_off=48, width=32, height=32),
        tile_w=16,
        tile_h=16,
        image_w=64,
        image_h=64,
    )
    assert r == (3, 3, 3, 3)


def test_tile_range_entirely_outside_image_returns_empty():
    r = _tile_range_for_window(
        Window(col_off=100, row_off=100, width=8, height=8),
        tile_w=16,
        tile_h=16,
        image_w=64,
        image_h=64,
    )
    # Sentinel for "empty range" — assembly fills with nodata.
    assert r[2] < r[0] or r[3] < r[1]


# ---------------------------------------------------------------------------
# Domain + open
# ---------------------------------------------------------------------------


def test_open_parses_domain(cog_field: ObstoreCogField):
    domain = cog_field.domain
    assert domain.shape == (1, 64, 64)
    assert domain.res == (10.0, 10.0)
    # CRS: EPSG:32629 may come back as a pyproj.CRS object.
    assert "32629" in str(domain.crs)


def test_open_rejects_striped_tiff(tmp_path: Path):
    """Striped TIFFs (the default ``tiled=False``) must raise."""
    path = tmp_path / "striped.tif"
    height = 32
    width = 32
    data = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32629",
        transform=from_bounds(0, 0, 320, 320, width, height),
        tiled=False,
    ) as dst:
        dst.write(data, 1)
    store = LocalStore(prefix=str(tmp_path))
    with pytest.raises(ValueError, match="must be tiled"):
        ObstoreCogField.from_url(
            url=f"file://{path}",
            store=store,
            path="striped.tif",
        )


# ---------------------------------------------------------------------------
# select / select_many — correctness against rasterio
# ---------------------------------------------------------------------------


def test_select_matches_rasterio(cog_path: Path, cog_field: ObstoreCogField):
    """One window read via obstore-cog matches the rasterio reference."""
    window = Window(col_off=8, row_off=12, width=24, height=20)
    got = cog_field.select(window)
    with rasterio.open(cog_path) as src:
        expected = src.read(window=window)
        expected_transform = rasterio.windows.transform(window, src.transform)
    assert isinstance(got, GeoTensor)
    np.testing.assert_array_equal(got.values, expected)
    assert got.transform == expected_transform


def test_select_many_matches_per_window_selects(cog_field: ObstoreCogField):
    """Batched read == sequential reads, value-by-value."""
    windows = [
        Window(col_off=0, row_off=0, width=8, height=8),
        Window(col_off=16, row_off=8, width=16, height=12),
        Window(col_off=24, row_off=24, width=20, height=20),
        Window(col_off=40, row_off=40, width=16, height=16),
    ]
    batched = cog_field.select_many(windows)
    individual = [cog_field.select(w) for w in windows]
    assert len(batched) == len(individual) == 4
    for got, want in zip(batched, individual, strict=True):
        np.testing.assert_array_equal(got.values, want.values)
        assert got.transform == want.transform


def test_select_many_empty_returns_empty(cog_field: ObstoreCogField):
    assert cog_field.select_many([]) == []


def test_select_many_all_windows_outside_image_keeps_band_axis(
    cog_field: ObstoreCogField,
):
    """Regression: out-of-image-only batches must still emit (bands, h, w).

    Previously the empty-tile-range fallback dropped the band axis
    when no tile was decoded — making output shape depend on batch
    composition. Now we derive bands+dtype from the IFD so the shape
    contract holds regardless of what's in the chunk.
    """
    out = cog_field.select_many([Window(col_off=100, row_off=100, width=8, height=8)])
    assert len(out) == 1
    # Single-band fixture COG → (1, 8, 8); important: 3D, not 2D.
    assert out[0].shape == (1, 8, 8)
    assert out[0].dtype == np.float32  # matches the fixture's dtype


def test_select_many_dedups_tile_fetches(cog_field: ObstoreCogField, monkeypatch):
    """Two windows sharing tiles should issue one batched fetch.

    ``ifd.fetch_tiles`` is implemented in Rust and is read-only on
    the IFD instance, so monkeypatching it directly raises
    ``AttributeError``. Instead we patch the module-level
    ``_fetch_and_decode_tiles`` helper that ``select_many`` calls; it
    receives the deduped coord list as its second argument, so
    asserting on its inputs proves the dedup semantics.
    """
    from geopatcher._src.fields import obstore_cog as oc_mod

    observed: list[list[tuple[int, int]]] = []
    original = oc_mod._fetch_and_decode_tiles

    async def _spy(ifd, coords):
        observed.append(list(coords))
        return await original(ifd, coords)

    monkeypatch.setattr(oc_mod, "_fetch_and_decode_tiles", _spy)
    # Two windows both fully inside the (0,0) tile.
    cog_field.select_many(
        [
            Window(col_off=0, row_off=0, width=8, height=8),
            Window(col_off=4, row_off=4, width=8, height=8),
        ]
    )
    assert len(observed) == 1  # one batched call
    # De-dup: should be exactly one tile (0,0), not two.
    assert observed[0] == [(0, 0)]


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def test_with_data_returns_geotensor(cog_field: ObstoreCogField):
    arr = np.zeros((1, 4, 4), dtype=np.float32)
    geo = cog_field.with_data(arr)
    assert isinstance(geo, GeoTensor)
    assert geo.crs is cog_field.domain.crs


# ---------------------------------------------------------------------------
# Real-world COG layouts — every fixture is checked against rasterio
# ---------------------------------------------------------------------------

# 40 x 50 with 16-px tiles: partial edge tiles on both axes, so reads near
# the right / bottom edge exercise the tile-padding clamp.
_H, _W = 40, 50
_ORIGIN_TRANSFORM = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_400.0)


def _ramp(count: int, dtype: str = "uint16") -> np.ndarray:
    return (np.arange(count * _H * _W) % 6000).reshape(count, _H, _W).astype(dtype)


def _write_cog(
    path: Path,
    data: np.ndarray,
    *,
    transform: Affine = _ORIGIN_TRANSFORM,
    crs: str | None = "EPSG:32629",
    overviews: tuple[int, ...] = (),
    tags: dict[str, str] | None = None,
    **profile,
) -> Path:
    """Write a tiled GeoTIFF with rasterio; ``profile`` overrides creation options."""
    options = {"compress": "deflate", "blockxsize": 16, "blockysize": 16}
    options.update(profile)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[-2],
        width=data.shape[-1],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        tiled=True,
        **options,
    ) as dst:
        dst.write(data)
        if tags:
            dst.update_tags(**tags)
        if overviews:
            dst.build_overviews(list(overviews), Resampling.nearest)
    return path


def _open(path: Path, **kwargs) -> ObstoreCogField:
    return ObstoreCogField.from_url(
        url=f"file://{path}",
        store=LocalStore(prefix=str(path.parent)),
        path=path.name,
        **kwargs,
    )


# Windows: whole image, straddling tile seams, past the right/bottom edge
# (partial edge tiles), negative offsets, and entirely outside.
_WINDOWS = [
    Window(0, 0, _W, _H),
    Window(3, 5, 20, 17),
    Window(40, 30, 16, 16),
    Window(-4, -6, 16, 16),
    Window(60, 50, 8, 8),
]


def _assert_matches_rasterio(
    field: ObstoreCogField, path: Path, window: Window, **open_kwargs
) -> None:
    chip = field.select(window)
    with rasterio.open(path, **open_kwargs) as src:
        fill = src.nodata if src.nodata is not None else 0
        inside = (
            window.col_off >= 0
            and window.row_off >= 0
            and window.col_off + window.width <= src.width
            and window.row_off + window.height <= src.height
        )
        # In-image windows read directly: rasterio's boundless path goes
        # through a warped VRT, which resamples rotated grids.
        expected = (
            src.read(window=window)
            if inside
            else src.read(window=window, boundless=True, fill_value=fill)
        )
        transform = rasterio.windows.transform(window, src.transform)
        crs = src.crs
    assert isinstance(chip, GeoTensor)
    np.testing.assert_array_equal(chip.values, expected)
    assert chip.values.dtype == expected.dtype
    assert chip.transform.almost_equals(transform, precision=1e-9)
    assert chip.fill_value_default == fill or (
        np.isnan(chip.fill_value_default) and np.isnan(fill)
    )
    assert (chip.crs is None) == (crs is None)
    if crs is not None:
        assert rasterio.crs.CRS.from_user_input(chip.crs) == crs


@pytest.mark.parametrize("interleave", ["pixel", "band"])
@pytest.mark.parametrize("window", _WINDOWS, ids=str)
def test_band_interleaved(tmp_path: Path, interleave: str, window: Window):
    """3-band COGs match rasterio for both PlanarConfiguration values.

    Regression: band-interleaved (PlanarConfiguration=2) tiles arrive
    band-first from async-tiff and were transposed a second time.
    """
    path = _write_cog(tmp_path / f"{interleave}.tif", _ramp(3), interleave=interleave)
    field = _open(path)
    assert int(field.ifd.planar_configuration) == (1 if interleave == "pixel" else 2)
    assert field.domain.shape == (3, _H, _W)
    _assert_matches_rasterio(field, path, window)


@pytest.mark.parametrize("compress", ["deflate", "lzw", "zstd", "none"])
def test_lossless_codecs_match_rasterio(tmp_path: Path, compress: str):
    path = _write_cog(
        tmp_path / f"{compress}.tif", _ramp(2, "int16"), compress=compress
    )
    _assert_matches_rasterio(_open(path), path, Window(0, 0, _W, _H))


@pytest.mark.parametrize(
    ("dtype", "nodata"), [("uint16", 65535), ("int16", -9999), ("float32", -1.5)]
)
def test_nodata_propagates(tmp_path: Path, dtype: str, nodata: float):
    """GDAL_NODATA → domain, out-of-image fill, pad fill, with_data."""
    path = _write_cog(tmp_path / "nd.tif", _ramp(1, dtype), nodata=nodata)
    field = _open(path)
    assert field.domain.nodata == nodata
    assert field.fill_value_default == nodata
    for window in _WINDOWS:
        _assert_matches_rasterio(field, path, window)
    # Out-of-image region is nodata, not 0.
    chip = field.select(Window(_W - 2, 0, 4, 4))
    assert (chip.values[..., 2:] == nodata).all()
    restored = field.with_data(np.zeros((1, _H, _W), dtype=dtype))
    assert restored.fill_value_default == nodata
    # boundary="pad" pads with nodata.
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16), boundary="pad"),
        sampler=SpatialRegularStride(step=16),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    edge = [p for p in patcher.split(field) if p.anchor == (32, 48)]
    assert len(edge) == 1
    values = np.asarray(edge[0].data)
    assert values.shape == (1, 16, 16)
    assert (values[..., 8:, :] == nodata).all()
    assert (values[..., :, 2:] == nodata).all()


def test_nan_nodata(tmp_path: Path):
    data = _ramp(1, "float32")
    data[0, :4, :4] = np.nan
    path = _write_cog(tmp_path / "nan.tif", data, nodata=np.nan)
    field = _open(path)
    assert np.isnan(field.domain.nodata)
    _assert_matches_rasterio(field, path, Window(-2, -2, 8, 8))
    assert np.isnan(field.select(Window(-2, -2, 8, 8)).values[0, 0, 0])


def test_no_nodata_fills_zero(tmp_path: Path):
    path = _write_cog(tmp_path / "plain.tif", _ramp(1) + 1)
    field = _open(path)
    assert field.domain.nodata is None
    assert field.fill_value_default == 0
    np.testing.assert_array_equal(field.select(Window(_W, 0, 4, 4)).values, 0)


def test_pixel_is_point_origin(tmp_path: Path):
    """GTRasterTypeGeoKey=2 → half-pixel shift, exactly like GDAL/rasterio."""
    corner = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 2560.0)
    path = _write_cog(
        tmp_path / "pip.tif",
        _ramp(1, "float32"),
        transform=corner,
        tags={"AREA_OR_POINT": "Point"},
    )
    field = _open(path)
    assert int(field.ifd.geo_key_directory.raster_type) == 2
    # The tiepoint stores the pixel *centre* (5, 2555); GDAL reports the corner.
    assert list(field.ifd.model_tiepoint)[3:5] == [5.0, 2555.0]
    with rasterio.open(path) as src:
        assert src.transform == corner
        assert field.domain.transform == src.transform
        assert field.domain.bounds == tuple(src.bounds)
    _assert_matches_rasterio(field, path, Window(3, 5, 20, 17))


def test_raster_tiepoint_offset(tmp_path: Path):
    """A tiepoint anchored at raster (I, J) != (0, 0) is honoured."""
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "tiepoint.tif"
    # Raster pixel (I=2, J=3) sits at model (500020, 4000370): the origin
    # is therefore (500000, 4000400).
    geokeys = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32629)
    tifffile.imwrite(
        path,
        _ramp(1)[0],
        tile=(16, 16),
        compression="zlib",
        extratags=[
            (33550, "d", 3, (10.0, 10.0, 0.0)),
            (33922, "d", 6, (2.0, 3.0, 0.0, 500_020.0, 4_000_370.0, 0.0)),
            (34735, "H", len(geokeys), geokeys),
        ],
    )
    field = _open(path)
    with rasterio.open(path) as src:
        assert src.transform == _ORIGIN_TRANSFORM
        assert field.domain.transform == src.transform
    _assert_matches_rasterio(field, path, Window(3, 5, 20, 17))


def test_model_transformation(tmp_path: Path):
    """A rotated grid is written as ModelTransformationTag — honour it."""
    rotated = _ORIGIN_TRANSFORM * Affine.rotation(20.0)
    path = _write_cog(tmp_path / "rot.tif", _ramp(1), transform=rotated)
    field = _open(path)
    assert field.ifd.model_transformation is not None
    assert field.ifd.model_tiepoint is None
    with rasterio.open(path) as src:
        assert field.domain.transform.almost_equals(src.transform, precision=1e-6)
        assert field.domain.res == pytest.approx(src.res)
        assert field.domain.bounds == pytest.approx(tuple(src.bounds))
    _assert_matches_rasterio(field, path, Window(3, 5, 20, 17))


@pytest.mark.parametrize("interleave", ["pixel", "band"])
def test_overview_ifd(tmp_path: Path, interleave: str):
    """Overviews inherit CRS / origin / nodata from IFD 0, scaled pixel size."""
    path = _write_cog(
        tmp_path / "ovr.tif",
        _ramp(3),
        overviews=(2, 4),
        nodata=7,
        interleave=interleave,
    )
    for level in (0, 1):
        field = _open(path, ifd_index=level + 1)
        with rasterio.open(path, overview_level=level) as src:
            assert field.domain.shape == (3, src.height, src.width)
            assert field.domain.transform.almost_equals(src.transform, precision=1e-9)
            assert field.domain.res == pytest.approx(src.res)
            assert field.domain.bounds == pytest.approx(tuple(src.bounds))
            assert field.domain.nodata == src.nodata == 7
        assert "32629" in str(field.domain.crs)
        for window in (Window(0, 0, 13, 10), Window(-3, 2, 16, 16)):
            _assert_matches_rasterio(field, path, window, overview_level=level)


def test_pickle_roundtrip(tmp_path: Path):
    path = _write_cog(tmp_path / "pk.tif", _ramp(3), overviews=(2,), nodata=7)
    field = _open(path, ifd_index=1, timeout=30.0)
    clone = pickle.loads(pickle.dumps(field))
    assert clone is not field
    assert clone.ifd_index == 1
    assert clone.timeout == 30.0
    assert clone.domain == field.domain
    window = Window(-2, 1, 9, 9)
    np.testing.assert_array_equal(
        clone.select(window).values, field.select(window).values
    )


def test_fractional_and_negative_windows_snap_outward(tmp_path: Path):
    """Fractional windows are floored / ceiled, never truncated toward 0."""
    path = _write_cog(tmp_path / "frac.tif", _ramp(1), nodata=1)
    field = _open(path)
    chip = field.select(Window(-2.5, 3.2, 10.1, 7.6))
    # floor(-2.5) = -3, floor(3.2) = 3, ceil(7.6) = 8, ceil(10.8) = 11.
    snapped = Window(-3, 3, 11, 8)
    assert chip.shape == (1, 8, 11)
    _assert_matches_rasterio(field, path, snapped)
    np.testing.assert_array_equal(chip.values, field.select(snapped).values)
    assert chip.transform == field.select(snapped).transform
    # Float noise does not grow the window.
    assert field.select(Window(4.0000001, 2.9999999, 8, 8)).shape == (1, 8, 8)


# ---------------------------------------------------------------------------
# Patcher round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interleave", ["pixel", "band"])
def test_split_merge_through_spatial_patcher(tmp_path: Path, interleave: str):
    """split → merge on a file-backed COG reproduces ``rasterio.read``.

    ``boundary="shrink"`` so the edge chips (40 x 50 is not a multiple
    of 16) are read clipped and still cover the whole domain.
    """
    path = _write_cog(
        tmp_path / "sm.tif", _ramp(3, "float32"), nodata=-1, interleave=interleave
    )
    field = _open(path)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16), boundary="shrink"),
        sampler=SpatialRegularStride(step=16),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    patches = list(patcher.split(field))
    assert all(isinstance(p.data, GeoTensor) for p in patches)
    for p in patches:
        row, col = p.anchor
        assert p.data.transform == field.domain.transform * Affine.translation(col, row)
    merged = patcher.merge(patches, field.domain)
    with rasterio.open(path) as src:
        expected = src.read()
    np.testing.assert_array_equal(np.asarray(merged), expected)
    restored = field.with_data(merged)
    assert restored.transform == field.domain.transform
    assert restored.fill_value_default == -1


def test_parallel_map_batched_path_matches_rasterio(tmp_path: Path):
    """``parallel_map`` drives ``select_many``; the result still merges exactly."""
    from geopatcher.runners import parallel_map

    path = _write_cog(tmp_path / "pm.tif", _ramp(3, "float32"), interleave="band")
    field = _open(path)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 10)),
        sampler=SpatialRegularStride(step=(8, 10)),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    out = parallel_map(patcher, field, lambda d: np.asarray(d) * 2, n_workers=2)
    merged = patcher.merge(out, field.domain)
    with rasterio.open(path) as src:
        np.testing.assert_array_equal(np.asarray(merged), src.read() * 2)


# ---------------------------------------------------------------------------
# HTTPStore against a local range-capable server
# ---------------------------------------------------------------------------


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` + single ``Range: bytes=a-b`` support."""

    def log_message(self, format: str, *args: object) -> None:
        pass  # keep pytest output quiet

    def do_GET(self) -> None:
        header = self.headers.get("Range")
        if not header:
            super().do_GET()
            return
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        start_s, _, end_s = header.removeprefix("bytes=").partition("-")
        if start_s:
            start = int(start_s)
            end = min(int(end_s), len(body) - 1) if end_s else len(body) - 1
        else:  # suffix range: the last N bytes
            start, end = max(0, len(body) - int(end_s)), len(body) - 1
        chunk = body[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Type", "image/tiff")
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def http_root(tmp_path: Path):
    handler = functools.partial(_RangeHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_http_store_via_pool(tmp_path: Path, http_root: str):
    """Pooled ``HTTPStore`` (no explicit store) reads match rasterio; pickles."""
    path = _write_cog(tmp_path / "remote.tif", _ramp(3), nodata=7, interleave="band")
    field = ObstoreCogField.from_url(
        f"{http_root}/remote.tif",
        storage_options={"client_options": {"allow_http": True}},
    )
    assert field.store is None
    for window in (Window(0, 0, _W, _H), Window(40, 30, 16, 16)):
        _assert_matches_rasterio(field, path, window)
    clone = pickle.loads(pickle.dumps(field))
    window = Window(3, 5, 20, 17)
    np.testing.assert_array_equal(
        clone.select(window).values, field.select(window).values
    )


@pytest.mark.parametrize(("photometric", "atol"), [("YCbCr", 4), ("RGB", 2)])
def test_jpeg_within_documented_tolerance(tmp_path: Path, photometric: str, atol: int):
    """JPEG tiles decode via async-tiff, not libjpeg: a few DN apart (documented)."""
    yy, xx = np.mgrid[0:64, 0:64]
    data = np.stack([xx * 4, yy * 4, (xx + yy) * 2]).astype("uint8")
    path = _write_cog(
        tmp_path / "jpeg.tif",
        data,
        compress="jpeg",
        photometric=photometric,
        blockxsize=32,
        blockysize=32,
    )
    got = _open(path).select(Window(0, 0, 64, 64)).values.astype(int)
    with rasterio.open(path) as src:
        expected = src.read().astype(int)
    assert np.abs(got - expected).max() <= atol
