"""Tests for ``build_raster_catalog(concurrency="async")``.

The async path is opt-in; existing callers see no behaviour change.
These tests assert:

1. ``concurrency="async"`` produces a byte-identical catalog to
   ``concurrency="sequential"`` over the same input.
2. ``max_concurrent`` rejects invalid values.
3. ``concurrency`` rejects invalid values.

The headline goal — measurable wall-clock improvement on WAN reads —
is covered by an opt-in benchmark script under
``tests/bench/bench_async_build.py``; CI doesn't measure remote I/O.
"""

from __future__ import annotations

import pytest

from geocatalog import build_raster_catalog


REGEX = r"S2_T29SND_(?P<date>\d{8})"


@pytest.fixture
def four_tiles(utm29_tile_factory):
    """Four UTM29 tiles spread across two acquisition dates."""
    return [
        utm29_tile_factory((500_000, 4_000_000, 500_320, 4_000_320), "20240101"),
        utm29_tile_factory((500_320, 4_000_000, 500_640, 4_000_320), "20240101"),
        utm29_tile_factory((500_000, 4_000_320, 500_320, 4_000_640), "20240115"),
        utm29_tile_factory((500_320, 4_000_320, 500_640, 4_000_640), "20240115"),
    ]


# --- equivalence --------------------------------------------------------


def test_async_equivalent_to_sequential(four_tiles):
    seq = build_raster_catalog(
        four_tiles, filename_regex=REGEX, concurrency="sequential"
    )
    asn = build_raster_catalog(four_tiles, filename_regex=REGEX, concurrency="async")
    assert len(seq) == len(asn) == 4
    # Sort both by filepath to compare row-by-row (async may return
    # rows in completion order, which can differ from input order).
    seq_df = seq.gdf.sort_values("filepath").reset_index(drop=True)
    asn_df = asn.gdf.sort_values("filepath").reset_index(drop=True)
    # Bounds, times, CRS should all match byte-for-byte.
    for col in ("filepath", "start_time", "end_time", "crs"):
        assert (seq_df[col] == asn_df[col]).all(), f"{col} mismatch"
    # Geometry equality via the shapely interface (element-wise).
    assert (seq_df.geometry.geom_equals(asn_df.geometry)).all()


def test_async_with_target_crs(four_tiles):
    seq = build_raster_catalog(
        four_tiles,
        filename_regex=REGEX,
        concurrency="sequential",
        target_crs="EPSG:4326",
    )
    asn = build_raster_catalog(
        four_tiles, filename_regex=REGEX, concurrency="async", target_crs="EPSG:4326"
    )
    assert len(seq) == len(asn) == 4


# --- validation ---------------------------------------------------------


def test_async_rejects_invalid_concurrency(four_tiles):
    with pytest.raises(ValueError, match="concurrency must be"):
        build_raster_catalog(four_tiles, filename_regex=REGEX, concurrency="parallel")  # type: ignore[arg-type]


def test_async_rejects_zero_max_concurrent(four_tiles):
    with pytest.raises(ValueError, match="max_concurrent must be"):
        build_raster_catalog(
            four_tiles, filename_regex=REGEX, concurrency="async", max_concurrent=0
        )


def test_async_works_inside_running_event_loop(four_tiles):
    """Calling the sync builder from inside a running loop must work.

    Regression for the PR #62 review: ``asyncio.run`` raises
    ``RuntimeError`` when invoked from a running loop. Jupyter,
    FastAPI handlers, and pytest-asyncio all hit this path.
    """
    import asyncio

    async def _from_loop():
        # Calling the *sync* builder from an async context is the
        # interesting case — equivalent to Jupyter's "auto-await" cell.
        return build_raster_catalog(
            four_tiles, filename_regex=REGEX, concurrency="async"
        )

    catalog = asyncio.run(_from_loop())
    assert len(catalog) == 4


def test_async_skips_unmatched_files(four_tiles, tmp_path):
    bogus = tmp_path / "no_date_here.tif"
    # Write minimal valid GeoTIFF so the open succeeds; it just won't
    # match the regex and should be skipped (consistent with
    # sequential behaviour).
    import rasterio
    from rasterio.transform import from_bounds

    with rasterio.open(
        bogus,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:32629",
        transform=from_bounds(500_000, 4_000_000, 500_080, 4_000_080, 8, 8),
    ) as dst:
        dst.write(__import__("numpy").zeros((1, 8, 8), dtype="uint8"))

    catalog = build_raster_catalog(
        [*four_tiles, bogus], filename_regex=REGEX, concurrency="async"
    )
    assert len(catalog) == 4
