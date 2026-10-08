"""Golden tests: streaming aggregations match their in-memory closed
form — issue #22.

Concrete content of #22 for the v0.x aggregation family:

- The only aggregation with two distinct code paths is
  `spatial.aggregation.OverlapAdd`: `_merge_in_memory` (numpy accumulators) vs
  `_merge_streaming` (zarr-backed accumulators on disk). The bulk of
  this file is the equality contract between those two paths under
  varied zarr chunk shapes (1, 7 prime, 16 block-aligned, full).

- The other streaming-safe aggregations (`Sum`, `Max`, `Min`,
  `WeightedSum`, `Mean`, `Variance`) are monoidal folds with a single
  implementation. The "golden" property we can still verify on them is
  **permutation invariance**: feeding the same patches in any order
  must yield the same result. That catches the same class of
  bookkeeping bugs the streaming/in-memory comparison would.

- `spatial.aggregation.Variance` uses Welford specifically because it's more
  accurate than a naive two-pass on ill-conditioned data (large mean,
  small variance, near-cancellation in `E[x²] - E[x]²`). One test pins
  that claim down — Welford error must not exceed naive error on a
  deliberately ill-conditioned fixture.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Iterator

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geopatcher import Patch, spatial


def _needs_zarr() -> None:
    # Only the streaming overlap-add tests need zarr (the `streaming`
    # extra); the permutation / Welford goldens below run on a slim
    # install too, so the skip is per test, not per module.
    pytest.importorskip("zarr")


# ---------------------------------------------------------------------------
# Fixture: deterministic patch set on a 64x64 domain
# ---------------------------------------------------------------------------


@pytest.fixture
def domain() -> GeoTensor:
    return GeoTensor(
        values=np.zeros((64, 64), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )


@pytest.fixture
def overlapping_patches() -> list[Patch]:
    # Deterministic (16x16) patches on a 64x64 grid, anchored on a
    # 12-pixel stride so they overlap (4-px overlap region per pair) —
    # the regime where OverlapAdd's streaming and in-memory paths
    # actually have something to disagree about. Kept small (5x5
    # lattice = 25 patches) because the streaming path's per-patch
    # zarr RMW dominates wall time; the equality property doesn't need
    # bulk to surface.
    rng = np.random.default_rng(seed=0)
    patches: list[Patch] = []
    anchors = [(r, c) for r in range(0, 49, 12) for c in range(0, 49, 12)]
    for r, c in anchors:
        data = rng.normal(loc=10.0, scale=2.0, size=(16, 16)).astype(np.float64)
        weights = rng.uniform(0.1, 1.0, size=(16, 16))
        patches.append(
            Patch(
                data=data,
                anchor=(r, c),
                indices=Window(col_off=c, row_off=r, width=16, height=16),
                weights=weights,
            )
        )
    return patches


# ---------------------------------------------------------------------------
# OverlapAdd: streaming vs in-memory, across zarr chunk shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunks",
    [
        (8, 8),  # patch-misaligned small chunk (16 / 8 = 2)
        (7, 7),  # prime, deliberately misaligned with patches
        (16, 16),  # patch-aligned
        (64, 64),  # whole-domain (single chunk)
    ],
)
def test_overlap_add_streaming_matches_in_memory(
    domain: GeoTensor,
    overlapping_patches: list[Patch],
    tmp_path,
    chunks: tuple[int, int],
) -> None:
    _needs_zarr()
    in_mem = spatial.aggregation.OverlapAdd().merge(overlapping_patches, domain)

    streamed_agg = spatial.aggregation.OverlapAdd(
        streaming=True,
        target_path=str(tmp_path),
        chunks=chunks,
    )
    streamed = np.asarray(streamed_agg.merge(overlapping_patches, domain)[:])

    # The streaming path stores float32 accumulators (the default
    # `dtype="float32"`); the in-memory path uses float64. The float32
    # rtol of 1e-6 is the right ceiling.
    np.testing.assert_allclose(streamed, in_mem, rtol=1e-6, atol=1e-6)


def test_overlap_add_streaming_empty_patches_returns_fill_array(
    domain: GeoTensor, tmp_path
) -> None:
    # No patches at all should yield a fill-valued zarr of the domain's shape.
    _needs_zarr()
    agg = spatial.aggregation.OverlapAdd(
        streaming=True, target_path=str(tmp_path), chunks=(16, 16)
    )
    result = np.asarray(agg.merge([], domain)[:])
    assert result.shape == (64, 64)
    assert np.isnan(result).all()


def test_overlap_add_streaming_chunk_size_invariant(
    domain: GeoTensor,
    overlapping_patches: list[Patch],
    tmp_path,
) -> None:
    # Stronger version of the parametrized test: every chunk shape must
    # produce the same numerical result *to each other*, not just to
    # the in-memory path. Catches any bug where blocking introduces
    # per-chunk drift.
    _needs_zarr()
    results = []
    for i, chunks in enumerate([(7, 7), (8, 8), (16, 16), (64, 64)]):
        agg = spatial.aggregation.OverlapAdd(
            streaming=True,
            target_path=str(tmp_path / f"run_{i}"),
            chunks=chunks,
        )
        results.append(np.asarray(agg.merge(overlapping_patches, domain)[:]))
    for r in results[1:]:
        np.testing.assert_allclose(r, results[0], rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Permutation invariance — every streaming-safe monoidal aggregation
# ---------------------------------------------------------------------------
#
# The permutation tests share the `overlapping_patches` fixture
# (12-pixel stride on 16x16 patches). Earlier drafts used a disjoint
# tiling that touched every cell exactly once, which made the tests
# vacuous: `spatial.aggregation.Sum/Max/Min/Mean/WeightedSum` collapse to a single
# write per cell and the shuffle is a no-op, and `spatial.aggregation.Variance`
# returns 0 everywhere because Welford's count never exceeds 1.
# Overlap is the only regime where ordering can matter (Welford's
# intermediate `mean` updates differ; fp summation has ULP drift).
# This is the regime the test must cover.


@pytest.mark.parametrize(
    "agg",
    [
        spatial.aggregation.Max(),
        spatial.aggregation.Min(),
    ],
    ids=lambda a: type(a).__name__,
)
def test_exactly_commutative_aggregations_are_bit_identical_under_permutation(
    domain: GeoTensor, overlapping_patches: list[Patch], agg
) -> None:
    # `Max` and `Min` are exactly commutative-and-associative on
    # float64, so reordering the inputs must yield the bit-identical
    # result — no floating-point slack. A `rtol > 0` here would hide
    # a real bug where an accumulator threaded patches through a
    # non-commutative op.
    forward = agg.merge(overlapping_patches, domain)
    rng = np.random.default_rng(seed=2)
    shuffled = list(overlapping_patches)
    rng.shuffle(shuffled)
    permuted = agg.merge(shuffled, domain)
    np.testing.assert_array_equal(forward, permuted)


@pytest.mark.parametrize(
    "agg",
    [
        spatial.aggregation.Sum(),
        spatial.aggregation.Mean(),
        spatial.aggregation.WeightedSum(),
    ],
    ids=lambda a: type(a).__name__,
)
def test_summation_aggregations_are_permutation_invariant_to_fp(
    domain: GeoTensor, overlapping_patches: list[Patch], agg
) -> None:
    # `Sum`, `Mean`, `WeightedSum` are mathematically commutative but
    # IEEE-754 float64 addition isn't bit-associative, so the order of
    # the accumulator updates can shift the result by a handful of ULPs.
    # A tight `rtol=1e-12` is the right ceiling — order-sensitivity
    # below that is the cost of doing fp arithmetic; above it is a
    # genuine bookkeeping bug.
    forward = agg.merge(overlapping_patches, domain)
    rng = np.random.default_rng(seed=2)
    shuffled = list(overlapping_patches)
    rng.shuffle(shuffled)
    permuted = agg.merge(shuffled, domain)
    np.testing.assert_allclose(forward, permuted, rtol=1e-12, atol=0)


def test_variance_permutation_invariant_under_overlap(
    domain: GeoTensor, overlapping_patches: list[Patch]
) -> None:
    # Welford's running mean update is the order-sensitive step:
    # `mean += (x - mean) / count` and `M2 += delta * (x - new_mean)`
    # both depend on the partial-sum-so-far. Under the overlapping
    # fixture, `count > 1` on the overlap regions, so the order
    # actually exercises the update path. The looser tolerance
    # reflects that — the final result is mathematically order-
    # invariant; ULP drift is expected.
    forward = spatial.aggregation.Variance().merge(overlapping_patches, domain)
    rng = np.random.default_rng(seed=3)
    shuffled = list(overlapping_patches)
    rng.shuffle(shuffled)
    permuted = spatial.aggregation.Variance().merge(shuffled, domain)
    # Sanity check the fixture: at least some cells must be touched
    # more than once for the Welford code path to run at all.
    assert (forward > 0.0).any(), (
        "overlapping_patches fixture failed to produce >1 sample at "
        "any cell — variance test would be vacuous"
    )
    np.testing.assert_allclose(forward, permuted, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Welford accuracy claim: streaming variance error <= naive two-pass error
# ---------------------------------------------------------------------------


def test_welford_variance_no_worse_than_naive_on_ill_conditioned_data(
    domain: GeoTensor,
) -> None:
    # Classic ill-conditioned fixture: very large mean, very small
    # spread. `Sum(x^2)/N - (Sum(x)/N)^2` (the naive two-pass form, in
    # float32 to surface the cancellation problem) loses precision in
    # the subtraction. Welford's running update is robust to this. The
    # framework's `spatial.aggregation.Variance` runs in float64 internally, so we
    # compare it against a deliberately-stressed float32 naive
    # reference to make the accuracy gap visible.
    rng = np.random.default_rng(seed=7)
    true_mean = 1e8
    true_std = 1e-3
    # Many patches at the same anchor so the variance is computed over
    # 100 noisy samples at every cell — enough samples for the
    # cancellation regime to bite.
    patches: list[Patch] = []
    samples_per_cell = 100
    for i in range(samples_per_cell):
        del i  # only the count matters here
        data = rng.normal(loc=true_mean, scale=true_std, size=(16, 16))
        patches.append(
            Patch(
                data=data,
                anchor=(0, 0),
                indices=Window(col_off=0, row_off=0, width=16, height=16),
                weights=None,
            )
        )

    welford = spatial.aggregation.Variance().merge(patches, domain)

    # Naive two-pass in float32 — explicit cancellation regime.
    stacked32 = np.stack([np.asarray(p.data, dtype=np.float32) for p in patches])
    sum_x = stacked32.sum(axis=0)
    sum_xx = (stacked32 * stacked32).sum(axis=0)
    n = float(samples_per_cell)
    naive_var32 = (sum_xx - (sum_x * sum_x) / n) / (n - 1.0)

    interior_welford = welford[:16, :16]
    interior_naive = naive_var32[:16, :16]
    true_var = true_std**2

    welford_err = float(np.max(np.abs(interior_welford - true_var)))
    naive_err = float(np.max(np.abs(interior_naive - true_var)))
    assert welford_err <= naive_err, (
        f"Welford error {welford_err:.3e} exceeded naive float32 "
        f"two-pass error {naive_err:.3e} on ill-conditioned input — "
        "the whole reason spatial.aggregation.Variance uses Welford is that this "
        "inequality should hold."
    )


# ---------------------------------------------------------------------------
# #193 — streaming store hygiene and bounded memory
# ---------------------------------------------------------------------------


class _ShapeDomain:
    """A dense domain that is only a shape — no backing array to count."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def _hann_patches(n: int, size: int = 16, step: int = 8) -> Iterator[Patch]:
    """Lazily generated Hann-weighted patches covering an ``n x n`` domain.

    ``n`` is deliberately not a multiple of the zarr chunk, and the last
    anchor overhangs the domain (pad-style), so edge chunks are partial.
    """

    weights = spatial.window.Hann().weights(
        spatial.geometry.Rectangular(size=(size, size))
    )
    for r in range(0, n, step):
        for c in range(0, n, step):
            rows = np.arange(r, r + size, dtype=np.float64)[:, None]
            cols = np.arange(c, c + size, dtype=np.float64)[None, :]
            yield Patch(
                data=np.sin(rows / 7.0) + np.cos(cols / 5.0),
                anchor=(r, c),
                indices=Window(col_off=c, row_off=r, width=size, height=size),
                weights=weights,
            )


def test_streaming_matches_in_memory_on_partial_chunks(tmp_path) -> None:
    _needs_zarr()
    domain = _ShapeDomain((70, 60))  # chunk 32 → partial edge chunks
    in_mem = spatial.aggregation.OverlapAdd().merge(_hann_patches(70), domain)
    agg = spatial.aggregation.OverlapAdd(
        streaming=True, target_path=str(tmp_path), chunks=(32, 32)
    )
    streamed = np.asarray(agg.merge(_hann_patches(70), domain)[:])
    # Same NaN fill (the Σw = 0 leading ring) and the same values.
    np.testing.assert_array_equal(np.isnan(streamed), np.isnan(in_mem))
    assert np.isnan(streamed[0]).all() and np.isnan(streamed[:, 0]).all()
    np.testing.assert_allclose(streamed, in_mem, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_streaming_peak_memory(tmp_path) -> None:
    # The final normalisation used to read both accumulators whole and
    # build a third full-size result (>= 3x the domain in RAM). Chunk-wise
    # normalisation keeps the peak at O(patch + chunk). 1000 is not a
    # multiple of the 128 chunk, so edge chunks are partial and the last
    # patches overhang the domain.
    _needs_zarr()
    n = 1000
    domain = _ShapeDomain((n, n))
    domain_bytes = n * n * np.dtype("float32").itemsize

    def patches() -> Iterator[Patch]:
        return _hann_patches(n, size=128, step=128)

    # Warm zarr's lazy imports / codec registry outside the measurement.
    spatial.aggregation.OverlapAdd(
        streaming=True, target_path=str(tmp_path / "warm"), chunks=(8, 8)
    ).merge(_hann_patches(16, size=8, step=8), _ShapeDomain((16, 16)))
    agg = spatial.aggregation.OverlapAdd(
        streaming=True, target_path=str(tmp_path / "run"), chunks=(128, 128)
    )
    tracemalloc.start()
    try:
        out = agg.merge(patches(), domain)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < domain_bytes / 2, f"peak {peak} B vs domain {domain_bytes} B"
    in_mem = spatial.aggregation.OverlapAdd().merge(patches(), domain)
    streamed = np.asarray(out[:])
    np.testing.assert_array_equal(np.isnan(streamed), np.isnan(in_mem))
    np.testing.assert_allclose(streamed, in_mem, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_store_not_overwritten(tmp_path) -> None:
    _needs_zarr()
    domain = _ShapeDomain((32, 32))
    first = [Patch(data=np.ones((16, 16)), anchor=(0, 0), indices=Window(0, 0, 16, 16))]
    second = [
        Patch(data=np.full((16, 16), 2.0), anchor=(0, 0), indices=Window(0, 0, 16, 16))
    ]
    agg = spatial.aggregation.OverlapAdd(
        streaming=True, target_path=str(tmp_path), chunks=(16, 16)
    )
    agg.merge(first, domain)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        agg.merge(second, domain)
    import zarr

    kept = np.asarray(zarr.open_array(str(tmp_path / "rec.zarr"), mode="r")[:])
    assert kept[0, 0] == 1.0
    replaced = spatial.aggregation.OverlapAdd(
        streaming=True, target_path=str(tmp_path), chunks=(16, 16), overwrite=True
    ).merge(second, domain)
    assert np.asarray(replaced[:])[0, 0] == 2.0


def test_streaming_requires_chunks(tmp_path) -> None:
    # The chunk shape used to come from the first patch — a shrunk 2x2
    # edge chip made the whole store 2x2-chunked.
    agg = spatial.aggregation.OverlapAdd(streaming=True, target_path=str(tmp_path))
    with pytest.raises(ValueError, match="needs chunks="):
        agg.merge([], _ShapeDomain((8, 8)))


def test_streaming_dtype_knob(tmp_path) -> None:
    _needs_zarr()
    domain = _ShapeDomain((3, 16, 16))  # chunks right-align: band axis whole
    patch = Patch(
        data=np.full((3, 16, 16), 1.0 / 3.0),
        anchor=(0, 0),
        indices=Window(0, 0, 16, 16),
    )
    agg = spatial.aggregation.OverlapAdd(
        streaming=True, target_path=str(tmp_path), chunks=(8, 8), dtype="float64"
    )
    out = agg.merge([patch], domain)
    assert out.dtype == np.float64
    assert out.chunks == (3, 8, 8)
    assert np.asarray(out[:])[0, 0, 0] == 1.0 / 3.0
    with pytest.raises(ValueError, match="floating dtype"):
        spatial.aggregation.OverlapAdd(dtype="int16")


def test_streaming_rejects_extra_patch_dims(tmp_path) -> None:
    # A patch with more dims than the domain used to fail inside zarr with
    # a different error than the in-RAM path; both now raise the same one.
    patch = Patch(data=np.ones((2, 4, 4)), anchor=(0, 0), indices=Window(0, 0, 4, 4))
    with pytest.raises(ValueError, match="patch data has 3 dims"):
        spatial.aggregation.OverlapAdd().merge([patch], _ShapeDomain((8, 8)))
    _needs_zarr()
    agg = spatial.aggregation.OverlapAdd(
        streaming=True, target_path=str(tmp_path), chunks=(4, 4)
    )
    with pytest.raises(ValueError, match="patch data has 3 dims"):
        agg.merge([patch], _ShapeDomain((8, 8)))
