"""`SpatialPatcher.merge` takes batched carriers (#530): parity with the 0.13 path.

The same scene is split, scored and stitched three ways (patch by patch,
through the deprecated `BatchedPatch`, and as `PatchBatch` carriers mapped
with `geoarray.map_over`); all three agree, for Hann and Tukey windows
(#461). Gate GA-G15: 64 patches run in one trace and stitch the same.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from types import ModuleType

import geoarray
import numpy as np
import pytest
from geoarray import GeoArray

from geopatcher import RasterField, SpatialPatcher, spatial
from geopatcher.run import BatchedPatch, PatchBatch, batch_split, unbatch


UTM = (10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)
WINDOWS = {"hann": spatial.window.Hann(), "tukey": spatial.window.Tukey(alpha=0.5)}


def _patcher(window: str) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8), boundary="pad"),
        sampler=spatial.sampler.RegularStride(step=4),
        window=WINDOWS[window],
        aggregation=spatial.aggregation.OverlapAdd(),
    )


@pytest.fixture
def field() -> RasterField:
    data = np.random.default_rng(3).random((2, 30, 30)).astype(np.float32)
    return RasterField(
        GeoArray(data, UTM, "EPSG:32630", fill_value=np.nan, dims="band y x")
    )


def _model(values: np.ndarray) -> np.ndarray:
    """A per-pixel model: (…, band, y, x) → same shape."""
    return 2.0 * values + values[..., ::-1, :, :]


def _score(patch: GeoArray) -> GeoArray:
    return patch.with_data(_model(np.asarray(patch.data)))


def _patch_by_patch(patcher: SpatialPatcher, field: RasterField) -> np.ndarray:
    scored = [p.with_data(_score(p.data)) for p in patcher.split(field)]
    return np.asarray(patcher.merge(scored, field.domain))


@pytest.mark.parametrize("window", sorted(WINDOWS))
def test_batched_carriers_stitch_like_batched_patch(
    field: RasterField, window: str
) -> None:
    patcher = _patcher(window)
    with pytest.deprecated_call():
        legacy = [
            p
            for b in BatchedPatch.split(patcher, field, batch_size=16)
            for p in unbatch(b, _model(b.data))
        ]
    old = np.asarray(patcher.merge(legacy, field.domain))

    batches = (
        PatchBatch(geoarray.map_over(b.patches, _score, "patch"), b.valid)
        for b in batch_split(patcher, field, batch_size=16)
    )
    new = np.asarray(patcher.merge(batches, field.domain))

    np.testing.assert_allclose(new, old, rtol=1e-6)
    np.testing.assert_allclose(new, _patch_by_patch(patcher, field), rtol=1e-6)


def test_merge_to_field_takes_one_batch(field: RasterField) -> None:
    patcher = _patcher("hann")
    (batch,) = batch_split(patcher, field, batch_size=128)  # one padded batch
    assert not batch.valid.all()
    scores = PatchBatch(geoarray.map_over(batch.patches, _score, "patch"), batch.valid)

    out = patcher.merge_to_field(scores, field)

    assert isinstance(out, GeoArray) and out.grid == field.reader.grid
    np.testing.assert_allclose(
        np.asarray(out), _patch_by_patch(patcher, field), rtol=1e-6
    )


def test_a_batched_geoarray_merges_with_every_row_real(field: RasterField) -> None:
    patcher = _patcher("hann")
    batches = [
        b.patches for b in batch_split(patcher, field, batch_size=7, pad_last=False)
    ]

    merged = patcher.merge(batches, field.domain)

    np.testing.assert_allclose(merged, np.asarray(field.reader), rtol=1e-6)


def test_patches_and_batches_mix(field: RasterField) -> None:
    patcher = _patcher("tukey")
    patches = list(patcher.split(field))
    (head,) = batch_split(_HeadPatcher(patcher, 10), field, batch_size=10)

    merged = patcher.merge([head, *patches[10:]], field.domain)

    np.testing.assert_allclose(merged, np.asarray(field.reader), rtol=1e-6)


def test_amerge_takes_an_async_stream_of_batches(field: RasterField) -> None:
    patcher = _patcher("hann")

    async def stream():
        for b in batch_split(patcher, field, batch_size=9):
            yield b

    merged = asyncio.run(patcher.amerge(stream(), field.domain))

    np.testing.assert_allclose(merged, np.asarray(field.reader), rtol=1e-6)


@pytest.mark.parametrize(
    ("row", "column"), [(0, 11.0), (4, 11.0), (1, 0.5)], ids=["size", "size-y", "rot"]
)
def test_a_row_off_the_grid_resolution_is_rejected_by_index(
    field: RasterField, row: int, column: float
) -> None:
    patcher = _patcher("hann")
    batch = next(batch_split(patcher, field, batch_size=6))
    transforms = np.array(batch.patches.transform)
    transforms[2, row] = column if row != 4 else -column
    bad = GeoArray(
        batch.patches.data,
        transforms,
        batch.patches.crs,
        fill_value=batch.patches.fill_value,
        dims=batch.patches.dims,
        coords=batch.patches.coords,
    )

    with pytest.raises(ValueError, match=r"patch 2 of the batch.*never resampled"):
        patcher.merge(PatchBatch(bad, batch.valid), field.domain)


@pytest.fixture
def x64() -> Iterator[ModuleType]:
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)


def test_gate_g15_sixty_four_patches_one_trace(
    field: RasterField, x64: ModuleType
) -> None:
    """GA-G15: 64 patches with different transforms, one trace, same stitch."""
    jax = x64
    patcher = _patcher("hann")
    traces = []

    def score(patch: GeoArray) -> GeoArray:
        traces.append(1)
        values = patch.data
        return patch.with_data(2.0 * values + values[::-1])

    step = jax.jit(lambda patches: geoarray.map_over(patches, score, "patch"))
    batches = list(batch_split(patcher, field, batch_size=64))
    assert {b.patches.shape[0] for b in batches} == {64}
    scored = [PatchBatch(step(b.patches.to_jax()).to_numpy(), b.valid) for b in batches]

    assert len(traces) == 1
    np.testing.assert_allclose(
        patcher.merge(scored, field.domain), _patch_by_patch(patcher, field), rtol=1e-5
    )


class _HeadPatcher:
    """The first ``n`` patches of another patcher's split."""

    def __init__(self, patcher: SpatialPatcher, n: int) -> None:
        self._patcher, self._n = patcher, n

    def split(self, field: RasterField) -> Iterator:
        for i, patch in enumerate(self._patcher.split(field)):
            if i == self._n:
                return
            yield patch
