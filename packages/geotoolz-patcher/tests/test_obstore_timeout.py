"""Timeout + strict-tag behaviour of the obstore COG reader.

These tests exercise `ObstoreCogField.select_many`'s network deadline
and the loud-failure tag parsing (`_dtype_from_ifd`,
`_crs_from_geokeys`) against fake IFD objects, so they need neither the
``obstore`` nor the ``async-tiff`` extra, and touch no network.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from geopatcher._src.fields.obstore_cog import (
    ObstoreCogDomain,
    ObstoreCogField,
    _crs_from_geokeys,
    _dtype_from_ifd,
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


class _FakeTile:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    async def decode(self) -> np.ndarray:
        return self._arr


class _FakeIfd:
    """Minimal IFD: one band, float32, 16x16 tiles over a 32x32 image."""

    tile_width = 16
    tile_height = 16
    image_width = 32
    image_height = 32
    samples_per_pixel = 1
    bits_per_sample = (32,)
    sample_format = (3,)

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay

    async def fetch_tiles(self, coords: list[tuple[int, int]]) -> list[_FakeTile]:
        if self._delay:
            await asyncio.sleep(self._delay)
        tile = np.full((16, 16), 7.0, dtype=np.float32)
        return [_FakeTile(tile) for _ in coords]


def _field(ifd: _FakeIfd, timeout: float | None) -> ObstoreCogField:
    domain = ObstoreCogDomain(
        crs=None,
        transform=None,
        shape=(1, 32, 32),
        bounds=(0.0, 0.0, 32.0, 32.0),
        res=(1.0, 1.0),
    )
    return ObstoreCogField(url=URL, tiff=None, ifd=ifd, domain=domain, timeout=timeout)


class TestSelectManyTimeout:
    def test_stalled_fetch_raises_timeouterror_naming_url_and_batch(self) -> None:
        field = _field(_FakeIfd(delay=30.0), timeout=0.05)
        window = _Window(col_off=0, row_off=0, width=8, height=8)
        with pytest.raises(TimeoutError, match=r"1 tiles.*s3://bucket/scene\.tif"):
            field.select_many([window])  # type: ignore[list-item]

    def test_fast_fetch_completes_within_deadline(self) -> None:
        field = _field(_FakeIfd(), timeout=30.0)
        window = _Window(col_off=0, row_off=0, width=8, height=8)
        out = field.select_many([window])  # type: ignore[list-item]
        assert out[0].shape == (1, 8, 8)
        np.testing.assert_array_equal(out[0], 7.0)

    def test_timeout_none_disables_the_deadline(self) -> None:
        field = _field(_FakeIfd(delay=0.1), timeout=None)
        window = _Window(col_off=0, row_off=0, width=4, height=4)
        out = field.select_many([window])  # type: ignore[list-item]
        assert out[0].shape == (1, 4, 4)

    def test_default_timeout_is_two_minutes(self) -> None:
        field = _field(_FakeIfd(), timeout=120.0)
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


class TestDtypeFromIfdFailsLoud:
    def test_valid_tags_map_to_dtype(self) -> None:
        assert _dtype_from_ifd(_FakeIfd(), url=URL) == np.dtype("float32")

    def test_uint_and_int_formats(self) -> None:
        class _U(_FakeIfd):
            bits_per_sample = (16,)
            sample_format = (1,)

        class _I(_FakeIfd):
            bits_per_sample = (16,)
            sample_format = (2,)

        assert _dtype_from_ifd(_U(), url=URL) == np.dtype("uint16")
        assert _dtype_from_ifd(_I(), url=URL) == np.dtype("int16")

    def test_missing_tags_raise_valueerror_naming_file(self) -> None:
        class _NoTags:
            pass

        with pytest.raises(ValueError, match=r"cannot derive a dtype.*scene\.tif"):
            _dtype_from_ifd(_NoTags(), url=URL)

    def test_unknown_sample_format_raises(self) -> None:
        class _Weird(_FakeIfd):
            sample_format = (4,)  # complex — unsupported

        with pytest.raises(ValueError, match=r"unsupported SampleFormat 4.*scene"):
            _dtype_from_ifd(_Weird(), url=URL)

    def test_unsupported_bit_depth_raises(self) -> None:
        class _Odd(_FakeIfd):
            bits_per_sample = (24,)  # no numpy float24

        with pytest.raises(ValueError, match=r"unsupported BitsPerSample 24"):
            _dtype_from_ifd(_Odd(), url=URL)


class TestCrsFromGeokeysWarns:
    def test_none_geokeys_return_none_silently(self) -> None:
        assert _crs_from_geokeys(None) is None

    def test_valid_epsg_builds_crs(self) -> None:
        pytest.importorskip("pyproj")

        class _Keys:
            projected_type = 32629
            geographic_type = None

        crs = _crs_from_geokeys(_Keys())
        assert "32629" in str(crs)

    def test_bogus_epsg_warns_and_returns_none(self) -> None:
        pytest.importorskip("pyproj")

        class _Keys:
            projected_type = 999999  # not a real EPSG code
            geographic_type = None

        with pytest.warns(RuntimeWarning, match=r"EPSG:999999"):
            assert _crs_from_geokeys(_Keys()) is None
