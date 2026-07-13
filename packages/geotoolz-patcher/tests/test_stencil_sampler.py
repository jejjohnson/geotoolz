"""Tests for `TemporalStencilSampler` — coordinate-aware anchor placement."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher._src.time.stencils import Stencil
from geopatcher.time import TemporalStencilSampler, TimeStencil


def _hourly_day() -> np.ndarray:
    return np.arange("2020-01-01", "2020-01-02", dtype="datetime64[h]")


class TestNeedsCoordFlag:
    def test_flag_is_true_on_class(self) -> None:
        assert TemporalStencilSampler.needs_coord is True


class TestAnchors:
    def test_requires_coord(self) -> None:
        s = TemporalStencilSampler(stencil=Stencil(-1, 1, 1, closed="both"))
        with pytest.raises(ValueError, match="requires coord="):
            list(s.anchors(time_len=10, coord=None))

    def test_coord_length_must_match_time_len(self) -> None:
        s = TemporalStencilSampler(stencil=Stencil(-1, 1, 1, closed="both"))
        coord = np.arange(10)
        with pytest.raises(ValueError, match="coord length must equal time_len"):
            list(s.anchors(time_len=11, coord=coord))

    def test_returns_integer_indices(self) -> None:
        coord = _hourly_day()
        s = TemporalStencilSampler(
            stencil=TimeStencil("-1h", "1h", "1h", closed="both")
        )
        result = list(s.anchors(time_len=coord.shape[0], coord=coord))
        assert all(isinstance(i, int) for i in result)
        # First valid anchor is index 1 (= 01:00), last is index 22 (= 22:00)
        # so stencil [-1h, +1h] fits in 00:00..23:00.
        assert result[0] == 1
        assert result[-1] == 22

    def test_every_thins_valid_set(self) -> None:
        coord = _hourly_day()
        full = list(
            TemporalStencilSampler(
                stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
                every=1,
            ).anchors(time_len=coord.shape[0], coord=coord)
        )
        thinned = list(
            TemporalStencilSampler(
                stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
                every=4,
            ).anchors(time_len=coord.shape[0], coord=coord)
        )
        assert thinned == full[::4]

    def test_shuffle_is_deterministic_under_seed(self) -> None:
        coord = _hourly_day()
        a = list(
            TemporalStencilSampler(
                stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
                shuffle=True,
                seed=42,
            ).anchors(time_len=coord.shape[0], coord=coord)
        )
        b = list(
            TemporalStencilSampler(
                stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
                shuffle=True,
                seed=42,
            ).anchors(time_len=coord.shape[0], coord=coord)
        )
        assert a == b

    def test_shuffle_differs_from_sorted(self) -> None:
        coord = _hourly_day()
        sorted_anchors = list(
            TemporalStencilSampler(
                stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
            ).anchors(time_len=coord.shape[0], coord=coord)
        )
        shuffled = list(
            TemporalStencilSampler(
                stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
                shuffle=True,
                seed=0,
            ).anchors(time_len=coord.shape[0], coord=coord)
        )
        assert sorted(shuffled) == sorted_anchors
        # Seed 0 is reliably a permutation, not the identity, for this length.
        assert shuffled != sorted_anchors


class TestGetConfig:
    def test_round_trip(self) -> None:
        s = TemporalStencilSampler(
            stencil=TimeStencil("-1h", "1h", "1h", closed="both"),
            every=2,
            shuffle=True,
            seed=7,
        )
        cfg = s.get_config()
        assert cfg["every"] == 2
        assert cfg["shuffle"] is True
        assert cfg["seed"] == 7
        assert cfg["stencil"]["start"] == "-1 hours"
