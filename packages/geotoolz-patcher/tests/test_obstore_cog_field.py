"""Tests for ``ObstoreCogField`` — COG reads via obstore + async-tiff.

Skipped unless both the ``obstore`` and ``async-tiff`` extras are
installed (the ``[obstore-cog]`` extra). The tests write a small
tiled GeoTIFF to a tempdir, point an obstore ``LocalStore`` at it,
and exercise the full read path — including the batched
``select_many`` route — without any network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window


pytest.importorskip("obstore")
pytest.importorskip("async_tiff")

from obstore.store import LocalStore

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
    # Sentinel for "empty range" — assembly fills with zeros.
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
        expected = src.read(1, window=window)
    # Drop singleton band axis from obstore-cog (band-first layout).
    np.testing.assert_array_equal(got.squeeze(), expected)


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
        np.testing.assert_array_equal(got, want)


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
    from georeader.geotensor import GeoTensor

    arr = np.zeros((1, 4, 4), dtype=np.float32)
    geo = cog_field.with_data(arr)
    assert isinstance(geo, GeoTensor)
    assert geo.crs is cog_field.domain.crs
