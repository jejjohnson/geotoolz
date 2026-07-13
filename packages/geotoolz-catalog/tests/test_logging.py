"""Smoke tests for the loguru migration: library-quiet by default, opt-in via enable."""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from loguru import logger


@pytest.fixture
def loguru_sink():
    """Attach a fresh in-memory loguru sink for the test, then remove it."""
    buf = io.StringIO()
    handler_id = logger.add(buf, level="TRACE", format="{name} | {message}")
    yield buf
    logger.remove(handler_id)


def _trigger_regex_miss_warning(
    tile_factory: Callable[..., Path], tmp_path: Path
) -> None:
    """Hit a real WARNING path inside `geocatalog._src.raster`.

    `_filepath_to_row` logs ``Skipping {filepath}: filename does not match
    regex`` when a filename doesn't match the supplied pattern. The call
    site is inside the geocatalog package, so loguru's
    `disable("geocatalog")` filter applies to it.
    """
    from geocatalog._src.raster import _filepath_to_row

    # Build a valid raster, then rename it so the regex misses.
    src = tile_factory((500000, 4000000, 510000, 4010000), "20240601")
    target = tmp_path / "no_date_here.tif"
    src.rename(target)
    pattern = re.compile(r"^S2_(?P<date>\d{8})_.*$")
    _filepath_to_row(
        target,
        filename_regex=pattern,
        date_format="%Y%m%d",
        target_crs=None,
    )


def test_library_silent_by_default(
    loguru_sink: io.StringIO,
    utm29_tile_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """`import geocatalog` should not emit log records to attached sinks."""
    import geocatalog  # noqa: F401  -- side effect: disable("geocatalog")

    _trigger_regex_miss_warning(utm29_tile_factory, tmp_path)

    assert "does not match regex" not in loguru_sink.getvalue()


def test_enable_routes_records_to_sink(
    loguru_sink: io.StringIO,
    utm29_tile_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """`logger.enable("geocatalog")` is the documented opt-in."""
    import geocatalog  # noqa: F401

    logger.enable("geocatalog")
    try:
        _trigger_regex_miss_warning(utm29_tile_factory, tmp_path)
    finally:
        # Restore the library-quiet default so other tests aren't affected.
        logger.disable("geocatalog")

    output = loguru_sink.getvalue()
    assert "does not match regex" in output
    assert "geocatalog._src.raster" in output
