"""Tests for the temporal axes + `TemporalPatcher`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from geopatcher import TemporalPatch, TemporalPatcher, temporal


@pytest.fixture
def series() -> np.ndarray:
    # 100 time steps of a single feature
    return np.arange(100, dtype=np.float64).reshape(100)


class TestTemporalFixedLookback:
    def test_window_shape(self) -> None:
        g = temporal.geometry.FixedLookback(length=5)
        w = g.window(time_len=10, anchor=8)
        assert w == slice(4, 9)


class TestTemporalLookbackHorizon:
    def test_window_shape(self) -> None:
        g = temporal.geometry.LookbackHorizon(lookback=3, horizon=2)
        w = g.window(time_len=20, anchor=10)
        # lookback ends at anchor+1, horizon extends from there
        assert w == slice(8, 13)


class TestTemporalExponentialDecay:
    def test_recent_step_has_weight_one(self) -> None:
        w = temporal.window.ExponentialDecay(tau=1.0).weights(
            temporal.geometry.FixedLookback(length=4), length=4
        )
        assert w[-1] == pytest.approx(1.0)
        assert w[0] < w[-1]


class TestTemporalCausalBoxcar:
    def test_uniform(self) -> None:
        w = temporal.window.CausalBoxcar().weights(
            temporal.geometry.FixedLookback(length=5), length=5
        )
        np.testing.assert_array_equal(w, 1.0)


class TestTemporalPatcherSplit:
    def test_yields_lookback_windows(self, series: np.ndarray) -> None:
        tp = TemporalPatcher(
            geometry=temporal.geometry.FixedLookback(length=5),
            sampler=temporal.sampler.RegularStride(step=10),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Mean(),
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
            geometry=temporal.geometry.FixedLookback(length=5),
            sampler=temporal.sampler.RegularStride(step=10),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Mean(),
        )
        n = tp.n_anchors(series)
        assert n == 9  # anchor 0 dropped
        assert n == sum(1 for _ in tp.split(series))

    def test_n_anchors_counts_multi_scale_windows(self, series: np.ndarray) -> None:
        # Regression: temporal.geometry.MultiScale.window returns a list[slice], so
        # split yields N anchors * len(scales) patches. n_anchors must
        # count the list, not just the anchors.

        tp = TemporalPatcher(
            geometry=temporal.geometry.MultiScale(scales=[5, 20, 50]),
            sampler=temporal.sampler.RegularStride(step=10),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Mean(),
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
            geometry=temporal.geometry.FixedLookback(length=5),
            sampler=temporal.sampler.RegularStride(step=10),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Mean(),
        )
        assert tp.n_anchors(ShapeOnly()) == 9


class TestTemporalFold:
    def test_state_passing(self, series: np.ndarray) -> None:
        tp = TemporalPatcher(
            geometry=temporal.geometry.FixedLookback(length=1),
            sampler=temporal.sampler.RegularStride(step=1),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Fold(
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
            geometry=temporal.geometry.FixedLookback(length=10),
            sampler=temporal.sampler.RegularStride(step=3),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Mean(time_len=100),
        )
        result = tp.merge(list(tp.split(series)))
        # Anchors 9, 12, …, 99 cover [0, 100).
        np.testing.assert_allclose(result, series)

    def test_docstring_configuration_merges(self) -> None:
        # Regression (#189): the TemporalPatcher docstring setup crashed in
        # np.stack because the first length-1 windows were shorter.
        tp = TemporalPatcher(
            temporal.geometry.FixedLookback(5, boundary="shrink"),
            temporal.sampler.RegularStride(1),
            temporal.window.CausalBoxcar(),
            temporal.aggregation.Mean(),
        )
        x = np.arange(24.0)
        np.testing.assert_allclose(tp.merge(list(tp.split(x))), x)

    def test_overlap_mean_and_nan_masking(self) -> None:
        patches = [
            TemporalPatch(data=np.array([1.0, 3.0]), anchor=1, indices=slice(0, 2)),
            TemporalPatch(data=np.array([5.0, np.nan]), anchor=2, indices=slice(1, 3)),
        ]
        out = temporal.aggregation.Mean(time_len=4).merge(patches)
        # step 0: 1; step 1: mean(3, 5); step 2: only NaN -> fill; step 3: none
        np.testing.assert_allclose(out, [1.0, 4.0, np.nan, np.nan])

    def test_time_axis_and_feature_dims(self) -> None:
        data = np.arange(6.0).reshape(2, 3)  # (feature, time)
        patch = TemporalPatch(data=data, anchor=4, indices=slice(2, 5))
        out = temporal.aggregation.Mean(time_axis=1).merge([patch])
        assert out.shape == (2, 5)
        np.testing.assert_allclose(out[:, 2:], data)
        assert np.isnan(out[:, :2]).all()

    def test_empty_stream(self) -> None:
        with pytest.raises(ValueError, match="no patches"):
            temporal.aggregation.Mean().merge([])
        out = temporal.aggregation.Mean(time_len=3, fill_value=-1.0).merge([])
        np.testing.assert_array_equal(out, [-1.0, -1.0, -1.0])

    def test_indices_must_match_data(self) -> None:
        patch = TemporalPatch(data=np.ones(3), anchor=0, indices=slice(0, 2))
        with pytest.raises(ValueError, match="indices"):
            temporal.aggregation.Mean().merge([patch])


class TestTemporalForecast:
    def test_keeps_horizon_tail(self) -> None:
        tp = TemporalPatcher(
            geometry=temporal.geometry.LookbackHorizon(lookback=3, horizon=2),
            sampler=temporal.sampler.RegularStride(step=5),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Forecast(horizon=2),
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
        forecast = temporal.aggregation.Forecast(horizon=h, time_axis=1)
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
        agg = temporal.aggregation.HierarchicalCombine(scales=[2, 4])
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


def _phase(b: str) -> temporal.geometry.PhaseWindow:
    return temporal.geometry.PhaseWindow(period=6, phase_width=1, boundary=b)  # type: ignore[arg-type]


# Hand-computed windows on a 24-step axis. ``None`` = the anchor yields no
# patch; a tuple is ``(start, stop)``; a list is one entry per window.
_WINDOW_TABLES: dict[str, tuple[Any, dict[str, dict[int, Any]]]] = {
    "fixed_lookback_5": (
        lambda b: temporal.geometry.FixedLookback(5, boundary=b),
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
        lambda b: temporal.geometry.LookbackHorizon(3, 2, boundary=b),
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
        lambda b: temporal.geometry.MultiScale([2, 4], boundary=b),
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
        temporal.geometry.MultiScale([2, 4]),
        temporal.sampler.Explicit(times=list(_EDGE_ANCHORS)),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Mean(),
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
        lambda: temporal.geometry.FixedLookback(3, boundary="pad"),  # type: ignore[arg-type]
        lambda: temporal.geometry.FixedLookback(0),
        lambda: temporal.geometry.LookbackHorizon(0, 1),
        lambda: temporal.geometry.LookbackHorizon(1, -1),
        lambda: temporal.geometry.MultiScale([]),
        lambda: temporal.geometry.PhaseWindow(period=4, phase_width=2),
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
        temporal.geometry.LookbackHorizon(lookback=3, horizon=2, boundary="shrink"),
        temporal.sampler.Explicit(times=[20, 21, 22, 23]),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Forecast(horizon=2),
    )
    result = shrink.merge(list(shrink.split(x)))
    # 22's true horizon [23, 24] is half off-axis, 23's does not exist.
    assert sorted(result) == [20, 21]
    np.testing.assert_array_equal(result[20], [21.0, 22.0])
    np.testing.assert_array_equal(result[21], [22.0, 23.0])
    drop = TemporalPatcher(
        temporal.geometry.LookbackHorizon(lookback=3, horizon=2),
        temporal.sampler.Explicit(times=[20, 21, 22, 23]),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Forecast(horizon=2),
    )
    assert sorted(drop.merge(list(drop.split(x)))) == [20, 21]


def test_forecast_accepts_horizon_only_predictions() -> None:
    x = np.arange(24.0)
    tp = TemporalPatcher(
        temporal.geometry.LookbackHorizon(lookback=3, horizon=2),
        temporal.sampler.Explicit(times=[5, 10]),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Forecast(horizon=2),
    )
    preds = [p.with_data(np.asarray(p.data)[-2:] * 10) for p in tp.split(x)]
    result = tp.merge(preds)
    np.testing.assert_array_equal(result[5], [60.0, 70.0])
    np.testing.assert_array_equal(result[10], [110.0, 120.0])
    bad = TemporalPatch(data=np.ones(3), anchor=5, indices=slice(3, 8))
    with pytest.raises(ValueError, match="expected the window's 5 or horizon=2"):
        temporal.aggregation.Forecast(horizon=2).merge([bad])


def test_hierarchical_combine_keeps_all_scales() -> None:
    # Regression (#189): at anchor 1 both scales shrank to slice(0, 2), so
    # keying by the realised slice let scale 4 overwrite scale 2.
    x = np.arange(24.0)
    tp = TemporalPatcher(
        temporal.geometry.MultiScale([2, 4], boundary="shrink"),
        temporal.sampler.Explicit(times=[1, 10]),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.HierarchicalCombine(scales=[2, 4]),
    )
    out = tp.merge(list(tp.split(x)))
    assert {a: set(v) for a, v in out.items()} == {1: {2, 4}, 10: {2, 4}}
    np.testing.assert_array_equal(out[1][2], [0.0, 1.0])
    np.testing.assert_array_equal(out[1][4], [0.0, 1.0])
    np.testing.assert_array_equal(out[10][4], [7.0, 8.0, 9.0, 10.0])
    # Without `scales` the inner key is the window index, never a tuple.
    plain = temporal.aggregation.HierarchicalCombine().merge(tp.split(x))
    assert set(plain[1]) == {0, 1}


def test_hierarchical_combine_rejects_unknown_scale_index() -> None:
    patch = TemporalPatch(
        data=np.ones(2), anchor=3, indices=slice(2, 4), window_index=2
    )
    with pytest.raises(ValueError, match="no entry in scales"):
        temporal.aggregation.HierarchicalCombine(scales=[2, 4]).merge([patch])


def test_phase_window_uses_period() -> None:
    # Regression (#189): `period` was never read — the geometry returned
    # the single local slot ``[anchor - w, anchor + w + 1)``.
    x = np.arange(48.0)
    tp = TemporalPatcher(
        temporal.geometry.PhaseWindow(period=24, phase_width=1),
        temporal.sampler.Explicit(times=[14]),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.HierarchicalCombine(),
    )
    patches = list(tp.split(x))
    assert [_as_table_entry(p.indices) for p in patches] == [(13, 16), (37, 40)]
    np.testing.assert_array_equal(patches[1].data, [37.0, 38.0, 39.0])
    assert tp.patch_anchors(x) == [(14, 0), (14, 1)]
    geometry = temporal.geometry.PhaseWindow(24, 1)
    assert geometry.window(48, 38) == geometry.window(48, 14)


def test_merge_checks_streaming_safety() -> None:
    from geopatcher.observe import set_strict

    class Buffering(temporal.aggregation.Aggregation):
        def merge(self, patches: Any) -> Any:
            return list(patches)

    tp = TemporalPatcher(
        temporal.geometry.FixedLookback(2),
        temporal.sampler.RegularStride(1),
        temporal.window.CausalBoxcar(),
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


# ---------------------------------------------------------------------------
# #190 — TemporalPatcher parity with SpatialPatcher
# ---------------------------------------------------------------------------


class _Series:
    """Array-like that records every read and fails on chosen steps."""

    def __init__(
        self, values: np.ndarray, fail_at: set[int] = frozenset(), fails: int = -1
    ) -> None:
        self.values = values
        self.shape = values.shape
        self.ndim = values.ndim
        self.fail_at = fail_at
        self.fails = fails  # remaining failures; -1 = always
        self.reads: list[slice] = []

    def __getitem__(self, idx: Any) -> np.ndarray:
        s = idx[0] if isinstance(idx, tuple) else idx
        self.reads.append(s)
        if any(s.start <= t < s.stop for t in self.fail_at) and self.fails != 0:
            self.fails -= 1
            raise OSError(f"bad read {s}")
        return self.values[idx]

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("the whole series must never be materialised")


def _lookback(n: int = 3, **kwargs: Any) -> TemporalPatcher:
    return TemporalPatcher(
        temporal.geometry.FixedLookback(n),
        temporal.sampler.RegularStride(1),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Mean(),
        **kwargs,
    )


def test_split_reads_only_windows() -> None:
    src = _Series(np.arange(10.0))
    patches = list(_lookback().split(src))
    assert [p.anchor for p in patches] == list(range(2, 10))
    assert src.reads == [slice(a - 2, a + 1) for a in range(2, 10)]
    assert all(isinstance(p.data, np.ndarray) for p in patches)


@pytest.mark.parametrize("prefetch", [0, 2])
def test_on_error_skip(prefetch: int) -> None:
    # Regression (#190): a failing read aborted the whole split, prefetch or not.
    tp = _lookback(on_error="skip")
    src = _Series(np.arange(10.0), fail_at={5})
    patches = list(tp.split(src, prefetch=prefetch))
    # Windows [3,6), [4,7), [5,8) touch step 5.
    assert [p.anchor for p in patches] == [2, 3, 4, 8, 9]
    assert [r.anchor for r in tp.errors] == [5, 6, 7]
    assert {r.kind for r in tp.errors} == {"OSError"}


def test_on_error_raise_is_default() -> None:
    with pytest.raises(OSError, match="bad read"):
        list(_lookback().split(_Series(np.arange(10.0), fail_at={5})))


def test_on_error_mask_yields_nan_window() -> None:
    tp = _lookback(on_error="mask")
    patches = list(tp.split(_Series(np.arange(12.0).reshape(6, 2), fail_at={4})))
    by_anchor = {p.anchor: p for p in patches}
    assert sorted(by_anchor) == [2, 3, 4, 5]
    assert by_anchor[4].data.shape == (3, 2)
    assert np.isnan(by_anchor[4].data).all()
    np.testing.assert_array_equal(by_anchor[3].data, np.arange(2.0, 8.0).reshape(3, 2))
    assert len(tp.errors) == 2  # anchors 4 and 5


def test_on_error_retry_recovers() -> None:
    tp = _lookback(on_error="retry", max_retries=2)
    src = _Series(np.arange(6.0), fail_at={3}, fails=1)
    patches = list(tp.split(src))
    assert [p.anchor for p in patches] == [2, 3, 4, 5]
    assert [(r.anchor, r.retry_count) for r in tp.errors] == [(3, 0)]


def test_on_error_policy_validated() -> None:
    with pytest.raises(ValueError, match="on_error"):
        _lookback(on_error="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_retries"):
        _lookback(max_retries=-1)


def test_hooks_see_skipped_errors() -> None:
    events: list[tuple[str, Any]] = []

    class Hook:
        def on_patch_start(self, anchor: Any, coord_value: Any = None) -> None:
            events.append(("start", (anchor, coord_value)))

        def on_error(self, anchor: Any, exc: Exception) -> None:
            events.append(("error", anchor))

    tp = _lookback(on_error="skip")
    coord = np.arange(100, 106)
    list(tp.split(_Series(np.arange(6.0), fail_at={5}), coord=coord, hooks=[Hook()]))
    assert ("start", (3, 103)) in events
    assert [a for kind, a in events if kind == "error"] == [5]


def test_journal_skips_completed_keys(tmp_path: Any) -> None:
    from geopatcher.observe import PatchJournal

    journal = PatchJournal(str(tmp_path / "run.jsonl"))
    tp = TemporalPatcher(
        temporal.geometry.MultiScale([2, 3]),
        temporal.sampler.Explicit(times=[4, 6]),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Mean(),
    )
    x = np.arange(10.0)
    keys = tp.patch_anchors(x)
    assert keys == [(4, 0), (4, 1), (6, 0), (6, 1)]
    journal.commit((4, 1), status="ok", runtime_s=0.0)
    got = [(p.anchor, p.window_index) for p in tp.split(x, journal=journal)]
    assert got == [(4, 0), (6, 0), (6, 1)]


def test_cache_serves_second_split_without_reads(tmp_path: Any) -> None:
    from geopatcher.run import PatchCache

    cache = PatchCache(tmp_path, field_id="series")
    tp = _lookback()
    first = list(tp.split(_Series(np.arange(8.0)), cache=cache))
    src = _Series(np.arange(8.0))
    second = list(tp.split(src, cache=cache))
    assert src.reads == []
    for a, b in zip(first, second, strict=True):
        assert (a.anchor, a.indices, a.window_index) == (
            b.anchor,
            b.indices,
            b.window_index,
        )
        np.testing.assert_array_equal(a.data, b.data)
        np.testing.assert_array_equal(a.weights, b.weights)
    # time_axis is part of the key: a transposed read misses.
    src_t = _Series(np.arange(16.0).reshape(2, 8))
    list(tp.split(src_t, time_axis=1, cache=cache))
    assert len(src_t.reads) == 6


def test_backpressure_slots_are_owned_and_released() -> None:
    tp = _lookback()
    it = tp.split(np.arange(8.0), max_in_flight=1)
    first = next(it)
    assert first._release is not None
    first.close()
    second = next(it)
    second.close()
    with pytest.raises(ValueError, match="max_in_flight must be >= 1"):
        tp.split(np.arange(8.0), max_in_flight=0)
    with pytest.raises(ValueError, match="exceeding max_in_flight_bytes"):
        list(tp.split(np.arange(8.0), max_in_flight_bytes=8))


def test_reduce_closes_patches_under_backpressure() -> None:
    tp = _lookback()
    x = np.arange(10.0)
    out = tp.reduce(x, temporal.aggregation.Mean(time_len=10), max_in_flight=1)
    np.testing.assert_allclose(out[2:], x[2:])


def test_two_pass_standardises() -> None:

    class MeanStd(temporal.aggregation.Aggregation):
        streaming_safe = True

        def merge(self, patches: Any) -> dict[str, float]:
            vals = np.concatenate([np.asarray(p.data) for p in patches])
            return {"mean": float(vals.mean()), "std": float(vals.std())}

    tp = TemporalPatcher(
        temporal.geometry.FixedLookback(2),
        temporal.sampler.RegularStride(step=2, start=1),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Mean(time_len=8),
    )
    x = np.arange(8.0)
    out = tp.two_pass(
        x, reduce_with=MeanStd(), apply=lambda d, s: (d - s["mean"]) / s["std"]
    )
    np.testing.assert_allclose(out, (x - x.mean()) / x.std())


def test_asplit_and_amerge() -> None:
    import asyncio

    tp = _lookback()
    x = np.arange(10.0)

    async def run() -> Any:
        patches = [p async for p in tp.asplit(x, prefetch=2)]
        assert [p.anchor for p in patches] == list(range(2, 10))

        async def stream() -> Any:
            async for p in tp.asplit(x):
                yield p

        return await TemporalPatcher(
            temporal.geometry.FixedLookback(3),
            temporal.sampler.RegularStride(1),
            temporal.window.CausalBoxcar(),
            temporal.aggregation.Mean(time_len=10),
        ).amerge(stream())

    out = asyncio.run(run())
    np.testing.assert_allclose(out[2:], x[2:])


def test_to_dask_bag_matches_split() -> None:
    pytest.importorskip("dask.bag")
    tp = TemporalPatcher(
        temporal.geometry.MultiScale([2, 3]),
        temporal.sampler.RegularStride(3),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Mean(),
    )
    x = np.arange(20.0).reshape(2, 10)
    got = tp.to_dask_bag(x, time_axis=1).compute()
    ref = list(tp.split(x, time_axis=1))
    assert len(got) == len(ref)
    for a, b in zip(got, ref, strict=True):
        assert (a.anchor, a.indices) == (b.anchor, b.indices)
        np.testing.assert_array_equal(a.data, b.data)
    assert len(tp.to_delayed(x, time_axis=1)) == len(ref)


def test_xarray_field_is_read_through_select() -> None:
    xr = pytest.importorskip("xarray")
    from geopatcher.fields import XarrayField

    da = xr.DataArray(
        np.arange(20.0).reshape(10, 2),
        dims=("time", "band"),
        coords={"time": np.arange(10) * 6},
    )
    field = XarrayField(da)
    selected: list[Any] = []
    real = field.select

    def spy(indexer: Any) -> Any:
        selected.append(indexer)
        return real(indexer)

    field.select = spy  # type: ignore[method-assign]
    patches = list(_lookback().split(field))
    assert selected[0] == {"time": slice(0, 3)}
    assert len(selected) == len(patches) == 8
    np.testing.assert_array_equal(patches[0].data, da.values[0:3])
    assert patches[0].data.dims == ("time", "band")


def test_field_supplies_stencil_coord() -> None:
    xr = pytest.importorskip("xarray")
    from geopatcher.fields import XarrayField
    from geopatcher.temporal.stencils import Stencil

    stencil = Stencil(-12, 0, 6, closed="both")
    da = xr.DataArray(
        np.arange(10.0), dims=("time",), coords={"time": np.arange(10) * 6}
    )
    tp = TemporalPatcher(
        temporal.geometry.StencilGeometry(stencil=stencil),
        temporal.sampler.StencilSampler(stencil=stencil),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Mean(),
    )
    patches = list(tp.split(XarrayField(da)))
    assert [p.indices for p in patches] == [slice(a - 2, a + 1) for a in range(2, 10)]


def test_non_grid_field_rejected() -> None:
    class RasterLike:
        domain = object()

        def select(self, indexer: Any) -> Any:
            return None

    with pytest.raises(TypeError, match="GridDomain"):
        _lookback().n_anchors(RasterLike())


@pytest.mark.parametrize(
    ("coord", "match"),
    [
        (np.arange(10)[::-1], "strictly increasing"),
        (np.zeros((10, 1)), "1-D"),
        (np.arange(9), "coord length"),
    ],
)
def test_coord_validated_once_up_front(coord: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _lookback().split(np.arange(10.0), coord=coord)


def test_coord_accepts_a_list() -> None:
    # Regression (#190): `coord=[...]` raised AttributeError on `.ndim`.
    events: list[Any] = []

    class Hook:
        def on_patch_start(self, anchor: Any, coord_value: Any = None) -> None:
            events.append(coord_value)

    list(_lookback().split(np.arange(4.0), coord=[10, 11, 12, 13], hooks=[Hook()]))
    assert events == [12, 13]


def test_stencil_pipeline_rejects_irregular_coord_at_entry() -> None:
    from geopatcher.temporal.stencils import Stencil

    stencil = Stencil(-1, 0, 1, closed="both")
    tp = TemporalPatcher(
        temporal.geometry.StencilGeometry(stencil=stencil),
        temporal.sampler.StencilSampler(stencil=stencil),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Mean(),
    )
    with pytest.raises(ValueError, match="evenly spaced"):
        tp.split(np.arange(5.0), coord=np.array([0, 1, 2, 4, 5]))


@pytest.mark.parametrize("every", [0, -1, 1.5])
def test_stencil_sampler_every_validated(every: Any) -> None:
    from geopatcher.temporal.stencils import Stencil

    with pytest.raises(ValueError, match="every"):
        temporal.sampler.StencilSampler(
            stencil=Stencil(-1, 0, 1, closed="both"), every=every
        )


def test_time_axis_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        _lookback().split(np.arange(5.0), 0)  # type: ignore[misc]


def test_check_full_scan() -> None:
    from geopatcher.spatial.sampler import IncompleteScanConfiguration

    def patcher(step: int, start: int, boundary: str) -> TemporalPatcher:
        return TemporalPatcher(
            temporal.geometry.FixedLookback(4, boundary=boundary),  # type: ignore[arg-type]
            temporal.sampler.RegularStride(
                step=step, start=start, check_full_scan=True
            ),
            temporal.window.CausalBoxcar(),
            temporal.aggregation.Mean(),
        )

    x = np.arange(12.0)
    # Anchors 3, 7, 11 tile [0, 12) exactly.
    assert len(list(patcher(4, 3, "drop").split(x))) == 3
    # Anchors 0, 4, 8: anchor 0's lookback is dropped and nothing ends at
    # the last step, so steps 0 and 9..11 are never read.
    with pytest.raises(IncompleteScanConfiguration, match=r"first: \[0, 9, 10, 11\]"):
        list(patcher(4, 0, "drop").split(x))
    # On 9 steps the same anchors cover everything once the head shrinks.
    with pytest.raises(IncompleteScanConfiguration, match=r"first: \[0\]"):
        list(patcher(4, 0, "drop").split(x[:9]))
    assert len(list(patcher(4, 0, "shrink").split(x[:9]))) == 3
    # A stride longer than the window skips steps.
    with pytest.raises(ValueError, match="uncovered"):
        list(patcher(5, 3, "drop").split(x))


def test_incomplete_scan_is_a_value_error() -> None:
    from geopatcher.spatial.sampler import IncompleteScanConfiguration

    assert issubclass(IncompleteScanConfiguration, ValueError)


class TestSamplerParity:
    def test_causal_rolling_is_regular_stride(self) -> None:

        assert temporal.sampler.RegularStride is temporal.sampler.RegularStride
        assert list(temporal.sampler.RegularStride(step=3, start=2).anchors(10)) == [
            2,
            5,
            8,
        ]

    def test_event_triggered_is_explicit(self) -> None:

        assert temporal.sampler.Explicit is temporal.sampler.Explicit
        assert list(temporal.sampler.Explicit(times=[7, -1, 3, 40]).anchors(10)) == [
            7,
            3,
        ]

    @pytest.mark.parametrize(
        "make",
        [
            lambda: temporal.sampler.RegularStride(step=0),
            lambda: temporal.sampler.RegularStride(step=-1),
            lambda: temporal.sampler.RegularStride(start=-2),
        ],
    )
    def test_regular_stride_validated(self, make: Any) -> None:
        with pytest.raises(ValueError):
            make()

    def test_random_n_samples(self) -> None:

        draws = list(temporal.sampler.Random(n_samples=50, seed=0).anchors(10))
        assert len(draws) == 50
        assert all(0 <= d < 10 for d in draws)
        assert len(set(draws)) < 50  # with replacement
        assert draws != sorted(draws)  # draw order, not sorted
        with pytest.raises(ValueError, match="n_samples"):
            temporal.sampler.Random(n_samples=-1)


class TestWindowParity:
    def test_exponential_decay_rejects_non_positive_tau(self) -> None:
        for tau in (0.0, -1.0):
            with pytest.raises(ValueError, match="tau"):
                temporal.window.ExponentialDecay(tau=tau)

    def test_tapered_tukey_keeps_newest_step(self) -> None:

        g = temporal.geometry.FixedLookback(4)
        w = temporal.window.TaperedTukey(alpha=0.5).weights(g, 4)
        # x = (k + 1) / 4 = .25, .5, .75, 1: only .25 < alpha tapers.
        np.testing.assert_allclose(w, [0.5, 1.0, 1.0, 1.0])
        full = temporal.window.TaperedTukey(alpha=1.0).weights(g, 4)
        np.testing.assert_allclose(
            full, 0.5 * (1 - np.cos(np.pi * np.array([0.25, 0.5, 0.75, 1.0])))
        )
        assert full[-1] == 1.0
        np.testing.assert_array_equal(
            temporal.window.TaperedTukey(alpha=0.0).weights(g, 3), 1.0
        )
        with pytest.raises(ValueError, match="alpha"):
            temporal.window.TaperedTukey(alpha=1.5)

    def test_periodic_is_a_tagged_boxcar(self) -> None:

        w = temporal.window.Periodic(period=24)
        assert isinstance(w, temporal.window.CausalBoxcar)
        np.testing.assert_array_equal(
            w.weights(temporal.geometry.FixedLookback(3), 3), 1.0
        )
        assert w.get_config() == {"period": 24}
        with pytest.raises(ValueError, match="period"):
            temporal.window.Periodic(period=0)


def test_patches_at_and_patch_at_read_one_anchor() -> None:
    src = _Series(np.arange(10.0))
    tp = TemporalPatcher(
        temporal.geometry.MultiScale([2, 4]),
        temporal.sampler.RegularStride(1),
        temporal.window.CausalBoxcar(),
        temporal.aggregation.Mean(),
    )
    patches = tp.patches_at(src, 6)
    assert [p.indices for p in patches] == [slice(5, 7), slice(3, 7)]
    assert src.reads == [slice(5, 7), slice(3, 7)]
    assert tp.patch_at(src, (6, 1)).indices == slice(3, 7)
    with pytest.raises(ValueError, match=r"pass \(6, k\)"):
        tp.patch_at(src, 6)
