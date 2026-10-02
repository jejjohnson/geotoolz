"""Correctness of the global sketch aggregations (`_SketchAggregation` family)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from geopatcher import (
    SpatialApproxCardinality,
    SpatialApproxMode,
    SpatialApproxQuantile,
    SpatialReservoir,
    SpatialStreamingHistogram,
)
from geopatcher._src.spatial.aggregation import _hll_alpha


def _patch(values: Any) -> SimpleNamespace:
    return SimpleNamespace(data=np.asarray(values))


def _batch(seed: int, n: int = 4) -> list[SimpleNamespace]:
    """Patches whose values are disjoint across ``seed``s (so leaked state shows)."""
    rng = np.random.default_rng(seed)
    lo = 100 * seed
    return [
        _patch(rng.integers(lo, lo + 40, size=50).astype(np.float64)) for _ in range(n)
    ]


def _assert_same(a: Any, b: Any) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert list(a) == list(b)
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


SKETCHES = [
    pytest.param(
        lambda: SpatialApproxQuantile(q=[0.1, 0.5, 0.9], k=32, seed=0), id="quantile"
    ),
    pytest.param(lambda: SpatialApproxCardinality(p=6), id="cardinality"),
    pytest.param(lambda: SpatialApproxMode(k=64), id="mode"),
    pytest.param(lambda: SpatialStreamingHistogram(bins=8), id="histogram"),
    pytest.param(lambda: SpatialReservoir(k=16, seed=0), id="reservoir"),
]


@pytest.mark.parametrize("make", SKETCHES)
def test_sketches_reset_between_merges(make) -> None:
    """A second ``merge`` on one instance equals a fresh instance's ``merge``."""
    reused = make()
    reused.merge(_batch(1), None)
    second = reused.merge(_batch(2), None)
    fresh = make().merge(_batch(2), None)
    _assert_same(second, fresh)


def _filled(cls, items: np.ndarray, k: int, seed: int):
    sketch = cls(k=k, seed=seed)
    sketch.update(_patch(items))
    return sketch


@pytest.mark.parametrize("cls", [SpatialReservoir, SpatialApproxQuantile])
def test_reservoir_merge_state_unbiased(cls) -> None:
    """Merged reservoir is a uniform sample of the union of two unequal streams."""
    n_a, n_b, k, trials = 300, 100, 20, 400
    stream_a = np.arange(n_a, dtype=np.float64)
    stream_b = np.arange(1000, 1000 + n_b, dtype=np.float64)
    inclusion = np.zeros(n_a + n_b)
    for t in range(trials):
        left = _filled(cls, stream_a, k, seed=2 * t)
        right = _filled(cls, stream_b, k, seed=2 * t + 1)
        left.merge_state(right)
        assert left._seen == n_a + n_b
        sample = np.asarray(left._sample)
        assert sample.size == k
        assert np.unique(sample).size == k
        in_a = sample < 1000
        inclusion[sample[in_a].astype(int)] += 1
        inclusion[n_a + (sample[~in_a] - 1000).astype(int)] += 1
    freq = inclusion / trials
    expected = k / (n_a + n_b)  # every item equally likely: 0.05
    share_a = freq[:n_a].sum() / k
    assert share_a == pytest.approx(n_a / (n_a + n_b), abs=0.03)
    assert freq[:n_a].mean() == pytest.approx(expected, rel=0.1)
    assert freq[n_a:].mean() == pytest.approx(expected, rel=0.1)


def test_reservoir_merge_state_small_streams_keeps_everything() -> None:
    left = _filled(SpatialReservoir, np.arange(3.0), k=10, seed=0)
    right = _filled(SpatialReservoir, np.arange(10.0, 14.0), k=10, seed=1)
    left.merge(right)
    assert left._seen == 7
    assert sorted(left.finalize().tolist()) == [0, 1, 2, 10, 11, 12, 13]


def test_reservoir_merge_state_rejects_mismatched_k() -> None:
    with pytest.raises(ValueError, match="different k"):
        SpatialReservoir(k=4).merge_state(SpatialReservoir(k=5))
    with pytest.raises(TypeError, match="cannot merge"):
        SpatialReservoir(k=4).merge_state(SpatialApproxQuantile(k=4))


@pytest.mark.parametrize(
    ("q", "key"), [(1, "1.0"), (0, "0.0"), ([0, 1], None), (0.5, "0.5")]
)
def test_approx_quantile_accepts_integer_q(q, key) -> None:
    out = SpatialApproxQuantile(q=q, k=64, seed=0).merge(
        [_patch(np.arange(11.0))], None
    )
    if key is None:
        assert out == {"0.0": 0.0, "1.0": 10.0}
    else:
        assert key in out


def test_approx_quantile_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError, match=r"q must be in \[0, 1\]"):
        SpatialApproxQuantile(q=2)


def test_approx_quantile_compression_renamed_to_k() -> None:
    with pytest.raises(TypeError):
        SpatialApproxQuantile(compression=32)  # type: ignore[call-arg]
    cfg = SpatialApproxQuantile(k=32).get_config()
    assert cfg["k"] == 32
    assert "compression" not in cfg


@pytest.mark.parametrize("cls", [SpatialApproxQuantile, SpatialReservoir])
def test_stochastic_sketch_seed_defaults_to_none(cls) -> None:
    assert cls().seed is None


@pytest.mark.parametrize(
    ("m", "alpha"),
    [
        (16, 0.673),
        (32, 0.697),
        (64, 0.709),
        (128, 0.7213 / (1 + 1.079 / 128)),
        (1 << 14, 0.7213 / (1 + 1.079 / (1 << 14))),
    ],
)
def test_hll_alpha_standard_constants(m, alpha) -> None:
    assert _hll_alpha(m) == pytest.approx(alpha, rel=1e-12)


@pytest.mark.parametrize("p", [4, 5, 6, 7, 10])
def test_hll_finalize_uses_alpha_for_p(p) -> None:
    """All registers at rank 1 → estimate ``2 * alpha_m * m`` (no linear counting)."""
    sketch = SpatialApproxCardinality(p=p)
    m = 1 << p
    sketch._registers[:] = 1
    expected = {16: 0.673, 32: 0.697, 64: 0.709}.get(m, 0.7213 / (1 + 1.079 / m))
    assert sketch.finalize() == pytest.approx(expected * m * 2, rel=1e-12)


def test_approx_mode_merge_state_sums_counters() -> None:
    left, right = SpatialApproxMode(k=2), SpatialApproxMode(k=2)
    left.update(_patch([1] * 10))
    right.update(_patch([1] * 5 + [2] * 3))
    left.merge(right)
    assert left.finalize() == {1: 15, 2: 3}


def test_approx_mode_merge_state_prunes_to_k() -> None:
    left, right = SpatialApproxMode(k=1), SpatialApproxMode(k=1)
    left.update(_patch([1] * 10))
    right.update(_patch([2] * 3))
    left.merge(right)
    assert left.finalize() == {1: 7}


def test_streaming_histogram_merge_state_keeps_total_count() -> None:
    left, right = SpatialStreamingHistogram(bins=4), SpatialStreamingHistogram(bins=4)
    left.update(_patch(np.arange(20.0)))
    right.update(_patch(np.arange(100.0, 130.0)))
    left.merge(right)
    out = left.finalize()
    assert out["counts"].sum() == 50
    assert out["centers"].size <= 4
