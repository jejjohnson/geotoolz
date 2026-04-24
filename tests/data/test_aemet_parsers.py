"""Low-level AEMET parser tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from xr_toolz.data._src.aemet.parsers import (
    format_aemet_datetime,
    parse_aemet_datetime,
    parse_dms,
    parse_spanish_float,
)


class TestParseDMS:
    def test_north_positive(self):
        assert parse_dms("402358N") == pytest.approx(40 + 23 / 60 + 58 / 3600)

    def test_south_negative(self):
        assert parse_dms("020000S") == pytest.approx(-2.0)

    def test_east_positive(self):
        # 3-digit degrees (longitude near 100°)
        assert parse_dms("1003015E") == pytest.approx(100 + 30 / 60 + 15 / 3600)

    def test_west_negative(self):
        assert parse_dms("034100W") == pytest.approx(-(3 + 41 / 60))

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_dms("NOT A COORD")
        with pytest.raises(ValueError):
            parse_dms("409999N")  # minutes out of range


class TestParseSpanishFloat:
    def test_comma_decimal(self):
        assert parse_spanish_float("12,5") == 12.5

    def test_passthrough_native(self):
        assert parse_spanish_float(3.14) == 3.14
        assert parse_spanish_float(7) == 7.0

    def test_missing_returns_none(self):
        assert parse_spanish_float(None) is None
        assert parse_spanish_float("") is None
        assert parse_spanish_float("-") is None
        assert parse_spanish_float("ip") is None  # trace precipitation
        assert parse_spanish_float("IP") is None

    def test_garbage_returns_none(self):
        assert parse_spanish_float("not-a-number") is None


class TestAEMETDateTime:
    def test_roundtrip(self):
        dt = datetime(2024, 5, 1, 12, 34, 56, tzinfo=UTC)
        text = format_aemet_datetime(dt)
        assert text == "2024-05-01T12:34:56UTC"
        assert parse_aemet_datetime(text) == dt

    def test_naive_input_treated_as_utc(self):
        naive = datetime(2024, 1, 1, 0, 0, 0)
        out = format_aemet_datetime(naive)
        assert out.endswith("UTC")

    def test_parse_accepts_Z_suffix(self):
        assert parse_aemet_datetime("2024-05-01T12:34:56Z") == datetime(
            2024, 5, 1, 12, 34, 56, tzinfo=UTC
        )

    def test_parse_accepts_no_suffix(self):
        assert parse_aemet_datetime("2024-05-01T12:34:56") == datetime(
            2024, 5, 1, 12, 34, 56, tzinfo=UTC
        )
