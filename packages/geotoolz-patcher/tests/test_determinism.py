"""Determinism contract for stochastic samplers — issue #18.

Pins down the behavior #21 (the Hypothesis round-trip suite) relies on:
**given the same integer seed and the same (domain, geometry), every
stochastic sampler returns bit-identical anchors across calls and
across instances.** Without that, Hypothesis can't shrink a failing
example and the round-trip properties can't be verified.

The four stochastic samplers covered here:

- `SpatialJitteredStride`
- `SpatialRandom`
- `SpatialPoissonDisk`
- `TemporalRandom`

The contract:

| ``seed`` value          | Determinism                                                |
| ----------------------- | ---------------------------------------------------------- |
| ``int``                 | Bit-identical anchors across calls and across instances.   |
| ``None`` (default)      | Re-seeded from OS entropy each call — anchors will differ. |

(`SpatialExplicit` and `SpatialRegularStride` are deterministic by
construction; not exercised here.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from hypothesis import given, settings, strategies as st

from geopatcher import (
    RasterField,
    SpatialJitteredStride,
    SpatialPoissonDisk,
    SpatialRandom,
    SpatialRectangular,
    TemporalRandom,
)


@pytest.fixture
def domain() -> RasterField:
    arr = np.zeros((64, 64), dtype=np.float32)
    return RasterField(
        GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
    )


@pytest.fixture
def rect() -> SpatialRectangular:
    return SpatialRectangular(size=(16, 16))


# ---------------------------------------------------------------------------
# Per-sampler determinism — explicit cases
# ---------------------------------------------------------------------------


class TestSpatialJitteredStrideDeterminism:
    def test_same_seed_same_anchors_across_calls(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        s = SpatialJitteredStride(step=16, jitter=0.5, seed=42)
        first = list(s.anchors(domain.domain, rect))
        second = list(s.anchors(domain.domain, rect))
        assert first == second

    def test_same_seed_same_anchors_across_instances(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        a = list(
            SpatialJitteredStride(step=16, jitter=0.5, seed=42).anchors(
                domain.domain, rect
            )
        )
        b = list(
            SpatialJitteredStride(step=16, jitter=0.5, seed=42).anchors(
                domain.domain, rect
            )
        )
        assert a == b

    def test_different_seeds_differ(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        a = list(
            SpatialJitteredStride(step=16, jitter=0.5, seed=0).anchors(
                domain.domain, rect
            )
        )
        b = list(
            SpatialJitteredStride(step=16, jitter=0.5, seed=1).anchors(
                domain.domain, rect
            )
        )
        assert a != b


class TestSpatialRandomDeterminism:
    def test_same_seed_same_anchors_across_calls(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        s = SpatialRandom(n_samples=20, seed=7)
        assert list(s.anchors(domain.domain, rect)) == list(
            s.anchors(domain.domain, rect)
        )

    def test_same_seed_same_anchors_across_instances(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        a = list(SpatialRandom(n_samples=20, seed=7).anchors(domain.domain, rect))
        b = list(SpatialRandom(n_samples=20, seed=7).anchors(domain.domain, rect))
        assert a == b

    def test_seed_none_constructs_a_fresh_rng_each_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        domain: RasterField,
        rect: SpatialRectangular,
    ) -> None:
        # Behavioral proxy for "seed=None is non-deterministic": prove
        # that each `anchors()` call reaches for `np.random.default_rng`
        # with the same `None` argument, which is what makes successive
        # calls draw fresh OS entropy. A probabilistic
        # `first != second` check would be technically flaky on
        # deterministic CI entropy sources; this is bulletproof.
        seen_seeds: list[Any] = []
        real = np.random.default_rng

        def tracking_default_rng(seed=None, *args, **kwargs):
            seen_seeds.append(seed)
            return real(seed, *args, **kwargs)

        monkeypatch.setattr(np.random, "default_rng", tracking_default_rng)

        s = SpatialRandom(n_samples=20, seed=None)
        list(s.anchors(domain.domain, rect))
        list(s.anchors(domain.domain, rect))

        # The sampler must have re-entered default_rng(None) at least
        # twice — once per anchors() call. Other call sites with
        # non-None seeds are allowed (we don't constrain them).
        assert seen_seeds.count(None) >= 2


class TestSpatialPoissonDiskDeterminism:
    def test_same_seed_same_anchors_across_calls(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        s = SpatialPoissonDisk(min_dist=6.0, seed=11)
        assert list(s.anchors(domain.domain, rect)) == list(
            s.anchors(domain.domain, rect)
        )

    def test_same_seed_same_anchors_across_instances(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        a = list(SpatialPoissonDisk(min_dist=6.0, seed=11).anchors(domain.domain, rect))
        b = list(SpatialPoissonDisk(min_dist=6.0, seed=11).anchors(domain.domain, rect))
        assert a == b


class TestTemporalRandomDeterminism:
    def test_same_seed_same_anchors_across_calls(self) -> None:
        s = TemporalRandom(n=5, seed=3)
        first = list(s.anchors(time_len=100))
        second = list(s.anchors(time_len=100))
        assert first == second

    def test_same_seed_same_anchors_across_instances(self) -> None:
        a = list(TemporalRandom(n=5, seed=3).anchors(time_len=100))
        b = list(TemporalRandom(n=5, seed=3).anchors(time_len=100))
        assert a == b


# ---------------------------------------------------------------------------
# Hypothesis: for any seed, anchors are bit-identical across calls
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_spatial_random_bit_identical_for_any_int_seed(seed: int) -> None:
    # Property: independent of the seed value, two calls return the
    # same anchor sequence. Catches any accidental reliance on global
    # RNG state that example-based tests might miss.
    arr = np.zeros((32, 32), dtype=np.float32)
    field = RasterField(
        GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
    )
    geom = SpatialRectangular(size=(8, 8))
    s = SpatialRandom(n_samples=10, seed=seed)
    assert list(s.anchors(field.domain, geom)) == list(s.anchors(field.domain, geom))


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_jittered_stride_bit_identical_for_any_int_seed(seed: int) -> None:
    arr = np.zeros((64, 64), dtype=np.float32)
    field = RasterField(
        GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
    )
    geom = SpatialRectangular(size=(8, 8))
    s = SpatialJitteredStride(step=8, jitter=0.5, seed=seed)
    assert list(s.anchors(field.domain, geom)) == list(s.anchors(field.domain, geom))
