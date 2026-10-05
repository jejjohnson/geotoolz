"""Tests for the temporal axes + `TemporalPatcher`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from geopatcher import (
    TemporalCausalBoxcar,
    TemporalExplicit,
    TemporalExponentialDecay,
    TemporalFixedLookback,
    TemporalFold,
    TemporalForecast,
    TemporalHierarchicalCombine,
    TemporalLookbackHorizon,
    TemporalMean,
    TemporalMultiScale,
    TemporalPatch,
    TemporalPatcher,
    TemporalPhaseWindow,
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
        # Anchors 0, 10, …, 90; anchor 0's 5-step lookback overflows the
        # axis and is dropped (boundary="drop" default).
        assert [p.anchor for p in patches] == list(range(10, 100, 10))
        assert all(isinstance(p, TemporalPatch) for p in patches)
        for p in patches:
            np.testing.assert_array_equal(p.data, series[p.anchor - 4 : p.anchor + 1])

    def test_n_anchors_matches_split_length(self, series: np.ndarray) -> None:
        # ADR-001: `n_anchors` is the cheap len() substitute.
        tp = TemporalPatcher(
            geometry=TemporalFixedLookback(length=5),
            sampler=TemporalRegularStride(step=10),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(),
        )
        n = tp.n_anchors(series)
        assert n == 9  # anchor 0 dropped
        assert n == sum(1 for _ in tp.split(series))

    def test_n_anchors_counts_multi_scale_windows(self, series: np.ndarray) -> None:
        # Regression: TemporalMultiScale.window returns a list[slice], so
        # split yields N anchors * len(scales) patches. n_anchors must
        # count the list, not just the anchors.
        from geopatcher import TemporalMultiScale

        tp = TemporalPatcher(
            geometry=TemporalMultiScale(scales=[5, 20, 50]),
            sampler=TemporalRegularStride(step=10),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(),
        )
        n = tp.n_anchors(series)
        # Anchors 50..90 fit the longest (50-step) scale; 0..40 are dropped.
        assert n == 5 * 3  # 5 anchors x 3 scales
        assert n == sum(1 for _ in tp.split(series))

    def test_n_anchors_does_not_materialise_series(self) -> None:
        # Regression: n_anchors used np.asarray(series) which materialises
        # a list / lazy array. Use np.shape so a generic shape-bearing
        # object suffices.
        class ShapeOnly:
            shape = (100,)

        tp = TemporalPatcher(
            geometry=TemporalFixedLookback(length=5),
            sampler=TemporalRegularStride(step=10),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(),
        )
        assert tp.n_anchors(ShapeOnly()) == 9


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
    def test_reconstructs_series_from_overlapping_windows(
        self, series: np.ndarray
    ) -> None:
        # Every patch is an unmodified slice, so the per-step mean of the
        # overlapping patches is the series itself wherever a patch landed.
        tp = TemporalPatcher(
            geometry=TemporalFixedLookback(length=10),
            sampler=TemporalRegularStride(step=3),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(time_len=100),
        )
        result = tp.merge(list(tp.split(series)))
        # Anchors 9, 12, …, 99 cover [0, 100).
        np.testing.assert_allclose(result, series)

    def test_docstring_configuration_merges(self) -> None:
        # Regression (#189): the TemporalPatcher docstring setup crashed in
        # np.stack because the first length-1 windows were shorter.
        tp = TemporalPatcher(
            TemporalFixedLookback(5, boundary="shrink"),
            TemporalRegularStride(1),
            TemporalCausalBoxcar(),
            TemporalMean(),
        )
        x = np.arange(24.0)
        np.testing.assert_allclose(tp.merge(list(tp.split(x))), x)

    def test_overlap_mean_and_nan_masking(self) -> None:
        patches = [
            TemporalPatch(data=np.array([1.0, 3.0]), anchor=1, indices=slice(0, 2)),
            TemporalPatch(data=np.array([5.0, np.nan]), anchor=2, indices=slice(1, 3)),
        ]
        out = TemporalMean(time_len=4).merge(patches)
        # step 0: 1; step 1: mean(3, 5); step 2: only NaN -> fill; step 3: none
        np.testing.assert_allclose(out, [1.0, 4.0, np.nan, np.nan])

    def test_time_axis_and_feature_dims(self) -> None:
        data = np.arange(6.0).reshape(2, 3)  # (feature, time)
        patch = TemporalPatch(data=data, anchor=4, indices=slice(2, 5))
        out = TemporalMean(time_axis=1).merge([patch])
        assert out.shape == (2, 5)
        np.testing.assert_allclose(out[:, 2:], data)
        assert np.isnan(out[:, :2]).all()

    def test_empty_stream(self) -> None:
        with pytest.raises(ValueError, match="no patches"):
            TemporalMean().merge([])
        out = TemporalMean(time_len=3, fill_value=-1.0).merge([])
        np.testing.assert_array_equal(out, [-1.0, -1.0, -1.0])

    def test_indices_must_match_data(self) -> None:
        patch = TemporalPatch(data=np.ones(3), anchor=0, indices=slice(0, 2))
        with pytest.raises(ValueError, match="indices"):
            TemporalMean().merge([patch])


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
        # Anchors 5, 10, 15 fit (0 overflows the lookback); the horizon is
        # the two steps after each anchor.
        assert sorted(result) == [5, 10, 15]
        for anchor, horizon in result.items():
            np.testing.assert_array_equal(horizon, series[anchor + 1 : anchor + 3])

    def test_time_axis_other_than_zero(self) -> None:
        """When patches carry the time dim at axis != 0, the horizon slice
        must walk that axis. Regression for the bug where ``arr[-horizon:]``
        always sliced axis 0.
        """
        from geopatcher._src.patch import TemporalPatch

        # Build patches whose data is (features=3, time=8) — time at axis 1.
        feats, t = 3, 8
        h = 2
        patches = []
        for anchor in (0, 1):
            data = np.arange(feats * t, dtype=np.float64).reshape(feats, t)
            patches.append(
                TemporalPatch(
                    data=data,
                    anchor=anchor,
                    indices=slice(0, t),
                    weights=None,
                )
            )
        forecast = TemporalForecast(horizon=h, time_axis=1)
        result = forecast.merge(patches)
        for anchor, horizon_arr in result.items():
            assert horizon_arr.shape == (feats, h)
            np.testing.assert_array_equal(
                horizon_arr, patches[anchor].data[:, anchor + 1 : anchor + 1 + h]
            )


class TestTemporalHierarchicalCombine:
    def test_preserves_all_scales(self) -> None:
        """Two scales per anchor must both appear in the merged output —
        regression for the bug where the second scale overwrote the first.
        """
        from geopatcher._src.patch import TemporalPatch
        from geopatcher._src.time.aggregation import (
            TemporalHierarchicalCombine,
        )

        # Anchor 5, two scales (length 2 and length 4)
        p_short = TemporalPatch(
            data=np.array([4, 5], dtype=np.float64),
            anchor=5,
            indices=slice(4, 6),
            weights=None,
        )
        p_long = TemporalPatch(
            data=np.array([2, 3, 4, 5], dtype=np.float64),
            anchor=5,
            indices=slice(2, 6),
            weights=None,
        )
        agg = TemporalHierarchicalCombine(scales=[2, 4])
        out = agg.merge([p_short, p_long])
        # Outer dict keyed by anchor, inner dict keyed by scale length
        assert set(out.keys()) == {5}
        assert set(out[5].keys()) == {2, 4}
        np.testing.assert_array_equal(out[5][2], [4, 5])
        np.testing.assert_array_equal(out[5][4], [2, 3, 4, 5])


# ---------------------------------------------------------------------------
# #189 — temporal boundary policy and the aggregations built on it
# ---------------------------------------------------------------------------

_EDGE_ANCHORS = (0, 1, 4, 21, 22, 23)


def _phase(b: str) -> TemporalPhaseWindow:
    return TemporalPhaseWindow(period=6, phase_width=1, boundary=b)  # type: ignore[arg-type]


# Hand-computed windows on a 24-step axis. ``None`` = the anchor yields no
# patch; a tuple is ``(start, stop)``; a list is one entry per window.
_WINDOW_TABLES: dict[str, tuple[Any, dict[str, dict[int, Any]]]] = {
    "fixed_lookback_5": (
        lambda b: TemporalFixedLookback(5, boundary=b),
        {
            "drop": {
                0: None,
                1: None,
                4: (0, 5),
                21: (17, 22),
                22: (18, 23),
                23: (19, 24),
            },
            "shrink": {
                0: (0, 1),
                1: (0, 2),
                4: (0, 5),
                21: (17, 22),
                22: (18, 23),
                23: (19, 24),
            },
        },
    ),
    "lookback_3_horizon_2": (
        lambda b: TemporalLookbackHorizon(3, 2, boundary=b),
        {
            "drop": {0: None, 1: None, 4: (2, 7), 21: (19, 24), 22: None, 23: None},
            "shrink": {
                0: (0, 3),
                1: (0, 4),
                4: (2, 7),
                21: (19, 24),
                22: (20, 24),
                23: (21, 24),
            },
        },
    ),
    "multi_scale_2_4": (
        lambda b: TemporalMultiScale([2, 4], boundary=b),
        {
            "drop": {
                0: None,
                1: None,
                4: [(3, 5), (1, 5)],
                21: [(20, 22), (18, 22)],
                22: [(21, 23), (19, 23)],
                23: [(22, 24), (20, 24)],
            },
            "shrink": {
                0: [(0, 1), (0, 1)],
                1: [(0, 2), (0, 2)],
                4: [(3, 5), (1, 5)],
                21: [(20, 22), (18, 22)],
                22: [(21, 23), (19, 23)],
                23: [(22, 24), (20, 24)],
            },
        },
    ),
    "phase_period_6_width_1": (
        _phase,
        {
            # Phase 0 (anchor 0): slots centred 0, 6, 12, 18, 24 — the first
            # and last overhang the axis. Phase 1: 1, 7, 13, 19 all fit.
            # Phase 4 (anchors 4, 22): 4, 10, 16, 22 fit (the slot centred
            # at -2 is off-axis). Phase 3 (21): 3, …, 21 fit. Phase 5 (23):
            # the slots centred at -1 and 23 overhang.
            "drop": {
                0: [(5, 8), (11, 14), (17, 20)],
                1: [(0, 3), (6, 9), (12, 15), (18, 21)],
                4: [(3, 6), (9, 12), (15, 18), (21, 24)],
                21: [(2, 5), (8, 11), (14, 17), (20, 23)],
                22: [(3, 6), (9, 12), (15, 18), (21, 24)],
                23: [(4, 7), (10, 13), (16, 19)],
            },
            "shrink": {
                0: [(0, 2), (5, 8), (11, 14), (17, 20), (23, 24)],
                1: [(0, 3), (6, 9), (12, 15), (18, 21)],
                4: [(3, 6), (9, 12), (15, 18), (21, 24)],
                21: [(2, 5), (8, 11), (14, 17), (20, 23)],
                22: [(3, 6), (9, 12), (15, 18), (21, 24)],
                23: [(0, 1), (4, 7), (10, 13), (16, 19), (22, 24)],
            },
        },
    ),
}


def _as_table_entry(window: Any) -> Any:
    if window is None:
        return None
    if isinstance(window, list):
        return [(s.start, s.stop) for s in window]
    return (window.start, window.stop)


@pytest.mark.parametrize("boundary", ["drop", "shrink"])
@pytest.mark.parametrize("name", sorted(_WINDOW_TABLES))
def test_window_tables_24_steps(name: str, boundary: str) -> None:
    make, tables = _WINDOW_TABLES[name]
    geometry = make(boundary)
    got = {a: _as_table_entry(geometry.window(24, a)) for a in _EDGE_ANCHORS}
    assert got == tables[boundary]


@pytest.mark.parametrize("name", sorted(_WINDOW_TABLES))
def test_window_tables_raise_on_overflow(name: str) -> None:
    make, tables = _WINDOW_TABLES[name]
    geometry = make("raise")
    for anchor in _EDGE_ANCHORS:
        if tables["drop"][anchor] == tables["shrink"][anchor]:
            got = _as_table_entry(geometry.window(24, anchor))
            assert got == tables["drop"][anchor]
        else:
            with pytest.raises(ValueError, match="overflows the time axis"):
                geometry.window(24, anchor)


def test_split_follows_window_table() -> None:
    tp = TemporalPatcher(
        TemporalMultiScale([2, 4]),
        TemporalExplicit(times=list(_EDGE_ANCHORS)),
        TemporalCausalBoxcar(),
        TemporalMean(),
    )
    x = np.arange(24.0)
    patches = list(tp.split(x))
    table = _WINDOW_TABLES["multi_scale_2_4"][1]["drop"]
    kept = [a for a in _EDGE_ANCHORS if table[a] is not None]
    assert tp.anchors(x) == kept
    got = [(p.anchor, p.window_index, _as_table_entry(p.indices)) for p in patches]
    assert got == [(a, k, w) for a in kept for k, w in enumerate(table[a])]
    assert tp.patches_at(x, 0) == []
    assert tp.n_anchors(x) == len(patches)


@pytest.mark.parametrize(
    "make",
    [
        lambda: TemporalFixedLookback(3, boundary="pad"),  # type: ignore[arg-type]
        lambda: TemporalFixedLookback(0),
        lambda: TemporalLookbackHorizon(0, 1),
        lambda: TemporalLookbackHorizon(1, -1),
        lambda: TemporalMultiScale([]),
        lambda: TemporalPhaseWindow(period=4, phase_width=2),
    ],
)
def test_geometry_validation(make: Any) -> None:
    with pytest.raises(ValueError):
        make()


def test_forecast_horizon_at_axis_end() -> None:
    # Regression (#189): with a shrunk window at the end of the axis the
    # horizon was the window's tail — lookback data labelled as future.
    x = np.arange(24.0)
    shrink = TemporalPatcher(
        TemporalLookbackHorizon(lookback=3, horizon=2, boundary="shrink"),
        TemporalExplicit(times=[20, 21, 22, 23]),
        TemporalCausalBoxcar(),
        TemporalForecast(horizon=2),
    )
    result = shrink.merge(list(shrink.split(x)))
    # 22's true horizon [23, 24] is half off-axis, 23's does not exist.
    assert sorted(result) == [20, 21]
    np.testing.assert_array_equal(result[20], [21.0, 22.0])
    np.testing.assert_array_equal(result[21], [22.0, 23.0])
    drop = TemporalPatcher(
        TemporalLookbackHorizon(lookback=3, horizon=2),
        TemporalExplicit(times=[20, 21, 22, 23]),
        TemporalCausalBoxcar(),
        TemporalForecast(horizon=2),
    )
    assert sorted(drop.merge(list(drop.split(x)))) == [20, 21]


def test_forecast_accepts_horizon_only_predictions() -> None:
    x = np.arange(24.0)
    tp = TemporalPatcher(
        TemporalLookbackHorizon(lookback=3, horizon=2),
        TemporalExplicit(times=[5, 10]),
        TemporalCausalBoxcar(),
        TemporalForecast(horizon=2),
    )
    preds = [p.with_data(np.asarray(p.data)[-2:] * 10) for p in tp.split(x)]
    result = tp.merge(preds)
    np.testing.assert_array_equal(result[5], [60.0, 70.0])
    np.testing.assert_array_equal(result[10], [110.0, 120.0])
    bad = TemporalPatch(data=np.ones(3), anchor=5, indices=slice(3, 8))
    with pytest.raises(ValueError, match="expected the window's 5 or horizon=2"):
        TemporalForecast(horizon=2).merge([bad])


def test_hierarchical_combine_keeps_all_scales() -> None:
    # Regression (#189): at anchor 1 both scales shrank to slice(0, 2), so
    # keying by the realised slice let scale 4 overwrite scale 2.
    x = np.arange(24.0)
    tp = TemporalPatcher(
        TemporalMultiScale([2, 4], boundary="shrink"),
        TemporalExplicit(times=[1, 10]),
        TemporalCausalBoxcar(),
        TemporalHierarchicalCombine(scales=[2, 4]),
    )
    out = tp.merge(list(tp.split(x)))
    assert {a: set(v) for a, v in out.items()} == {1: {2, 4}, 10: {2, 4}}
    np.testing.assert_array_equal(out[1][2], [0.0, 1.0])
    np.testing.assert_array_equal(out[1][4], [0.0, 1.0])
    np.testing.assert_array_equal(out[10][4], [7.0, 8.0, 9.0, 10.0])
    # Without `scales` the inner key is the window index, never a tuple.
    plain = TemporalHierarchicalCombine().merge(tp.split(x))
    assert set(plain[1]) == {0, 1}


def test_hierarchical_combine_rejects_unknown_scale_index() -> None:
    patch = TemporalPatch(
        data=np.ones(2), anchor=3, indices=slice(2, 4), window_index=2
    )
    with pytest.raises(ValueError, match="no entry in scales"):
        TemporalHierarchicalCombine(scales=[2, 4]).merge([patch])


def test_phase_window_uses_period() -> None:
    # Regression (#189): `period` was never read — the geometry returned
    # the single local slot ``[anchor - w, anchor + w + 1)``.
    x = np.arange(48.0)
    tp = TemporalPatcher(
        TemporalPhaseWindow(period=24, phase_width=1),
        TemporalExplicit(times=[14]),
        TemporalCausalBoxcar(),
        TemporalHierarchicalCombine(),
    )
    patches = list(tp.split(x))
    assert [_as_table_entry(p.indices) for p in patches] == [(13, 16), (37, 40)]
    np.testing.assert_array_equal(patches[1].data, [37.0, 38.0, 39.0])
    assert tp.patch_anchors(x) == [(14, 0), (14, 1)]
    geometry = TemporalPhaseWindow(24, 1)
    assert geometry.window(48, 38) == geometry.window(48, 14)


def test_merge_checks_streaming_safety() -> None:
    from geopatcher import set_strict
    from geopatcher._src.time.aggregation import TemporalAggregation

    class Buffering(TemporalAggregation):
        def merge(self, patches: Any) -> Any:
            return list(patches)

    tp = TemporalPatcher(
        TemporalFixedLookback(2),
        TemporalRegularStride(1),
        TemporalCausalBoxcar(),
        Buffering(),
    )
    with pytest.warns(RuntimeWarning, match="streaming_safe = False") as rec:
        tp.merge([])
    assert rec[0].filename == __file__
    set_strict(True)
    try:
        with pytest.raises(RuntimeError, match="streaming_safe = False"):
            tp.merge([])
    finally:
        set_strict(False)
