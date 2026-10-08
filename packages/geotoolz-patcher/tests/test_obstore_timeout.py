"""Timeout + strict-metadata behaviour of the obstore COG reader.

These tests exercise `ObstoreCogField.select_many`'s network deadline,
its concurrent tile grouping and internal-mask fill, and the thin
metadata guards over async-geotiff (`_dtype`, `_crs_or_none`,
`_tiepoint_offset`, `_parse_nodata`) against fake async-geotiff
objects, so they need neither the ``obstore`` nor the
``async-geotiff`` extra, and touch no network.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import Affine

from geopatcher._src.fields.obstore_cog import (
    _TILES_PER_FETCH,
    ObstoreCogDomain,
    ObstoreCogField,
    _crs_or_none,
    _dtype,
    _fetch_and_decode_tiles,
    _parse_nodata,
    _tiepoint_offset,
    _with_timeout,
)


URL = "s3://bucket/scene.tif"


@dataclass
class _Window:
    """Duck-typed stand-in for ``rasterio.windows.Window``."""

    col_off: int
    row_off: int
    width: int
    height: int


class _FakeIfd:
    """Tile grid of the fake level: 16x16 tiles over a 32x32 image."""

    tile_width = 16
    tile_height = 16
    image_width = 32
    image_height = 32


def _tile(x: int, y: int, data: np.ndarray, mask: np.ndarray | None = None):
    """An ``async_geotiff.Tile`` look-alike (``mask`` is True where valid)."""
    return SimpleNamespace(x=x, y=y, array=SimpleNamespace(data=data, mask=mask))


class _FakeLevel:
    """Minimal async-geotiff level: one float32 band of 7.0 per tile.

    Records every ``fetch_tiles`` call (its coords and start time) so
    tests can check grouping and concurrency.
    """

    def __init__(self, delay: float = 0.0, mask: np.ndarray | None = None) -> None:
        self.ifd = _FakeIfd()
        self._delay = delay
        self._mask = mask
        self.calls: list[list[tuple[int, int]]] = []
        self.starts: list[float] = []

    async def fetch_tiles(self, coords: list[tuple[int, int]]) -> list[SimpleNamespace]:
        self.calls.append(list(coords))
        self.starts.append(time.perf_counter())
        if self._delay:
            await asyncio.sleep(self._delay)
        data = np.full((1, 16, 16), 7.0, dtype=np.float32)
        return [_tile(x, y, data, self._mask) for x, y in coords]


def _field(
    level: _FakeLevel, timeout: float | None, nodata: float | None = None
) -> ObstoreCogField:
    domain = ObstoreCogDomain(
        crs=None,
        transform=Affine.identity(),
        shape=(1, 32, 32),
        bounds=(0.0, 0.0, 32.0, 32.0),
        res=(1.0, 1.0),
        nodata=nodata,
    )
    return ObstoreCogField(
        url=URL,
        level=level,
        ifd=level.ifd,
        domain=domain,
        timeout=timeout,
        dtype=np.dtype("float32"),
    )


class TestSelectManyTimeout:
    def test_stalled_fetch_raises_timeouterror_naming_url_and_batch(self) -> None:
        field = _field(_FakeLevel(delay=30.0), timeout=0.05)
        window = _Window(col_off=0, row_off=0, width=8, height=8)
        with pytest.raises(TimeoutError, match=r"1 tiles.*s3://bucket/scene\.tif"):
            field.select_many([window])  # type: ignore[list-item]

    def test_fast_fetch_completes_within_deadline(self) -> None:
        field = _field(_FakeLevel(), timeout=30.0)
        window = _Window(col_off=0, row_off=0, width=8, height=8)
        out = field.select_many([window])  # type: ignore[list-item]
        assert out[0].shape == (1, 8, 8)
        np.testing.assert_array_equal(out[0], 7.0)

    def test_timeout_none_disables_the_deadline(self) -> None:
        field = _field(_FakeLevel(delay=0.1), timeout=None)
        window = _Window(col_off=0, row_off=0, width=4, height=4)
        out = field.select_many([window])  # type: ignore[list-item]
        assert out[0].shape == (1, 4, 4)

    def test_default_timeout_is_two_minutes(self) -> None:
        field = _field(_FakeLevel(), timeout=120.0)
        assert field.timeout == 120.0
        assert ObstoreCogField.__dataclass_fields__["timeout"].default == 120.0


class TestWithTimeoutHelper:
    def test_expiry_message_names_the_operation(self) -> None:
        async def _stall() -> None:
            await asyncio.sleep(30.0)

        with pytest.raises(TimeoutError, match=r"opening COG 'x' timed out"):
            asyncio.run(
                _with_timeout(_stall(), timeout=0.01, message="opening COG 'x'")
            )

    def test_none_timeout_passes_result_through(self) -> None:
        async def _value() -> int:
            return 42

        assert asyncio.run(_with_timeout(_value(), timeout=None, message="x")) == 42


class TestConcurrentTileGroups:
    def test_tiles_fetched_once_in_row_major_groups(self) -> None:
        level = _FakeLevel()
        coords = [(x, y) for x in range(5) for y in range(4)]  # column-major
        out = asyncio.run(_fetch_and_decode_tiles(level, coords))
        assert len(out) == len(coords)
        fetched = [xy for call in level.calls for xy in call]
        assert sorted(fetched) == sorted(coords)  # each tile exactly once
        assert all(len(call) <= _TILES_PER_FETCH for call in level.calls)
        assert fetched == sorted(coords, key=lambda xy: (xy[1], xy[0]))

    def test_groups_run_concurrently(self) -> None:
        level = _FakeLevel(delay=0.2)
        coords = [(x, 0) for x in range(3 * _TILES_PER_FETCH)]
        start = time.perf_counter()
        asyncio.run(_fetch_and_decode_tiles(level, coords))
        assert len(level.calls) == 3
        # Sequential groups would take >= 0.6 s.
        assert time.perf_counter() - start < 0.5

    def test_internal_mask_fills_invalid_pixels(self) -> None:
        valid = np.ones((16, 16), dtype=bool)
        valid[:4, :4] = False
        field = _field(_FakeLevel(mask=valid), timeout=None, nodata=-1.0)
        out = np.asarray(field.select(_Window(0, 0, 8, 8)))  # type: ignore[arg-type]
        np.testing.assert_array_equal(out[0, :4, :4], -1.0)
        np.testing.assert_array_equal(out[0, 4:, 4:], 7.0)


class TestDtypeFailsLoud:
    def test_upstream_dtype_passes_through(self) -> None:
        geotiff = SimpleNamespace(dtype=np.dtype("uint16"), ifd=_FakeIfd())
        assert _dtype(geotiff, url=URL) == np.dtype("uint16")

    def test_unsupported_tags_raise_valueerror_naming_file(self) -> None:
        geotiff = SimpleNamespace(dtype=None, ifd=_FakeIfd())
        with pytest.raises(ValueError, match=r"cannot derive a dtype.*scene\.tif"):
            _dtype(geotiff, url=URL)


class TestCrsOrNone:
    def test_upstream_crs_passes_through(self) -> None:
        level = SimpleNamespace(crs="EPSG:32629")
        assert _crs_or_none(level, url=URL) == "EPSG:32629"

    def test_unbuildable_crs_warns_and_returns_none(self) -> None:
        class _Level:
            @property
            def crs(self) -> None:
                raise ValueError("Missing ellipsoid")

        with pytest.warns(RuntimeWarning, match=r"could not build the CRS.*scene"):
            assert _crs_or_none(_Level(), url=URL) is None


class TestTiepointOffset:
    def _geotiff(self, tiepoint, scale=(10.0, 10.0, 0.0)):
        return SimpleNamespace(
            ifd=SimpleNamespace(model_tiepoint=tiepoint, model_pixel_scale=scale)
        )

    def test_origin_tiepoint_is_zero(self) -> None:
        tie = (0.0, 0.0, 0.0, 5.0, 10.0, 0.0)
        assert _tiepoint_offset(self._geotiff(tie)) == (0.0, 0.0)

    def test_raster_tiepoint_is_returned(self) -> None:
        tie = (2.0, 3.0, 0.0, 5.0, 10.0, 0.0)
        assert _tiepoint_offset(self._geotiff(tie)) == (2.0, 3.0)

    def test_model_transformation_has_no_offset(self) -> None:
        assert _tiepoint_offset(self._geotiff(None, scale=None)) == (0.0, 0.0)


class TestParseNodata:
    def test_absent_or_blank_is_none(self) -> None:
        assert _parse_nodata(None, np.dtype("uint8"), url=URL) is None
        assert _parse_nodata(" \x00", np.dtype("uint8"), url=URL) is None

    def test_values_follow_dtype_kind(self) -> None:
        assert _parse_nodata("-9999\x00", np.dtype("int16"), url=URL) == -9999
        assert isinstance(_parse_nodata("255", np.dtype("uint8"), url=URL), int)
        assert _parse_nodata("-1.5", np.dtype("float32"), url=URL) == -1.5
        assert np.isnan(_parse_nodata("nan", np.dtype("float64"), url=URL))

    @pytest.mark.parametrize("raw", ["-1", "nan", "256", "1.5"])
    def test_unrepresentable_integer_nodata_warns(self, raw: str) -> None:
        with pytest.warns(RuntimeWarning, match=r"not representable as uint8"):
            assert _parse_nodata(raw, np.dtype("uint8"), url=URL) is None
