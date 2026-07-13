"""Per-merge determinism for the reservoir-backed sketch aggregations.

`SpatialReservoir` and `SpatialApproxQuantile` rebuild their reservoir
state and RNG from ``seed`` at each ``merge(patches)`` entry, so reusing
one instance across multiple ``merge()`` / ``reduce()`` calls is
reproducible — the same convention as the samplers, which rebuild
``default_rng(seed)`` per call.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from geopatcher import SpatialApproxQuantile, SpatialReservoir


def _patches() -> list[SimpleNamespace]:
    """Deterministic patch batch — enough values to overflow the reservoirs."""
    rng = np.random.default_rng(42)
    return [SimpleNamespace(data=rng.normal(size=64)) for _ in range(4)]


class TestReservoirDeterminism:
    def test_reservoir_same_instance_repeated_merge(self) -> None:
        agg = SpatialReservoir(k=16, seed=0)
        first = agg.merge(_patches(), None)
        second = agg.merge(_patches(), None)
        np.testing.assert_array_equal(first, second)

    def test_approx_quantile_same_instance_repeated_merge(self) -> None:
        agg = SpatialApproxQuantile(q=[0.1, 0.5, 0.9], compression=32, seed=0)
        first = agg.merge(_patches(), None)
        second = agg.merge(_patches(), None)
        assert first == second
