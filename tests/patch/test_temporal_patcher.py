"""Tests for the temporal axes + `TemporalPatcher`."""

from __future__ import annotations

import numpy as np
import pytest

from geotoolz.patch import (
    TemporalCausalBoxcar,
    TemporalExponentialDecay,
    TemporalFixedLookback,
    TemporalFold,
    TemporalForecast,
    TemporalLookbackHorizon,
    TemporalMean,
    TemporalPatch,
    TemporalPatcher,
    TemporalRegularStride,
)


@pytest.fixture
def series() -> np.ndarray:
    # 100 time steps of a single feature
    return np.arange(100, dtype=np.float64).reshape(100)


class TestTemporalFixedLookback:
    def test_window_shape(self) -> None:
        g = TemporalFixedLookback(length=5)
        w = g.window(time_len=10, anchor=8)
        assert w == slice(4, 9)


class TestTemporalLookbackHorizon:
    def test_window_shape(self) -> None:
        g = TemporalLookbackHorizon(lookback=3, horizon=2)
        w = g.window(time_len=20, anchor=10)
        # lookback ends at anchor+1, horizon extends from there
        assert w == slice(8, 13)


class TestTemporalExponentialDecay:
    def test_recent_step_has_weight_one(self) -> None:
        w = TemporalExponentialDecay(tau=1.0).weights(
            TemporalFixedLookback(length=4), length=4
        )
        assert w[-1] == pytest.approx(1.0)
        assert w[0] < w[-1]


class TestTemporalCausalBoxcar:
    def test_uniform(self) -> None:
        w = TemporalCausalBoxcar().weights(TemporalFixedLookback(length=5), length=5)
        np.testing.assert_array_equal(w, 1.0)


class TestTemporalPatcherSplit:
    def test_yields_lookback_windows(self, series: np.ndarray) -> None:
        tp = TemporalPatcher(
            geometry=TemporalFixedLookback(length=5),
            sampler=TemporalRegularStride(step=10),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(),
        )
        patches = list(tp.split(series))
        assert len(patches) == 10  # 100 / 10
        assert all(isinstance(p, TemporalPatch) for p in patches)
        # Each patch has up to 5 elements
        assert all(p.data.shape[0] <= 5 for p in patches)


class TestTemporalFold:
    def test_state_passing(self, series: np.ndarray) -> None:
        tp = TemporalPatcher(
            geometry=TemporalFixedLookback(length=1),
            sampler=TemporalRegularStride(step=1),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalFold(
                fold_fn=lambda s, p: (s or 0) + int(p.data[0]),
                initial_state=0,
            ),
        )
        result = tp.merge(list(tp.split(series)))
        assert result == sum(range(100))


class TestTemporalMean:
    def test_mean_across_patches(self, series: np.ndarray) -> None:
        # Use stride=length so every patch has the same shape — TemporalMean
        # stacks-and-means, which requires uniform shapes.
        tp = TemporalPatcher(
            geometry=TemporalFixedLookback(length=10),
            sampler=TemporalRegularStride(step=10),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(),
        )
        patches = [p for p in tp.split(series) if p.data.shape[0] == 10]
        result = tp.merge(patches)
        expected = np.mean(np.stack([p.data for p in patches]), axis=0)
        np.testing.assert_allclose(result, expected)


class TestTemporalForecast:
    def test_keeps_horizon_tail(self) -> None:
        tp = TemporalPatcher(
            geometry=TemporalLookbackHorizon(lookback=3, horizon=2),
            sampler=TemporalRegularStride(step=5),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalForecast(horizon=2),
        )
        series = np.arange(20, dtype=np.float64)
        patches = list(tp.split(series))
        result = tp.merge(patches)
        # Each entry should be the last `horizon` elements of the window
        assert all(len(v) == 2 for v in result.values())
