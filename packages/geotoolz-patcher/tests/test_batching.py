"""`geopatcher.run.batch_split`: a batch is one `GeoArray` with a ``patch`` dim.

ADR-013 (#529): NumPy in, NumPy out; padding copies the last real patch;
`unbatch` places patches from their transform rows.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from types import ModuleType

import geoarray
import numpy as np
import pytest
from _helpers import ArrayField
from geoarray import GeoArray

from geopatcher import RasterField, SpatialPatcher, spatial
from geopatcher.run import BatchedPatch, PatchBatch, batch_split, unbatch


UTM = (10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)


def _patcher(boundary: str = "drop", size: int = 8, step: int | None = None):
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(size, size), boundary=boundary),
        sampler=spatial.sampler.RegularStride(step=step or size),
        window=spatial.window.Hann() if step else spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


def _scene(size: int = 32) -> RasterField:
    data = np.random.default_rng(0).random((3, size, size)).astype(np.float32)
    return RasterField(
        GeoArray(data, UTM, "EPSG:32630", fill_value=np.nan, dims="band y x")
    )


@pytest.fixture
def field() -> RasterField:
    return _scene()


@pytest.fixture
def x64() -> Iterator[ModuleType]:
    """JAX with ``jax_enable_x64`` on (transforms are float64), restored after."""
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)


def test_a_batch_is_one_geoarray_with_a_patch_dim(field: RasterField) -> None:
    patcher = _patcher()
    expected = list(patcher.split(field))

    first = next(batch_split(patcher, field, batch_size=4))

    assert isinstance(first, PatchBatch)
    patches, valid = first
    assert isinstance(patches, GeoArray) and patches.backend == "numpy"
    assert patches.dims == ("patch", "band", "y", "x")
    assert patches.batch_dims == ("patch",)
    assert patches.shape == (4, 3, 8, 8) and patches.transform.shape == (4, 6)
    assert patches.crs == field.reader.crs and patches.coords == field.reader.coords
    np.testing.assert_array_equal(valid, [True] * 4)
    for i, want in enumerate(expected[:4]):
        item = geoarray.isel(patches, patch=i)
        np.testing.assert_array_equal(item.transform, want.data.transform)
        np.testing.assert_array_equal(np.asarray(item), np.asarray(want.data))


def test_padded_final_batch_copies_the_last_real_patch(field: RasterField) -> None:
    patcher = _patcher()  # 16 patches → 5, 5, 5, 1 (+ 4 padding)

    *_, (patches, valid) = batch_split(patcher, field, batch_size=5)

    np.testing.assert_array_equal(valid, [True, False, False, False, False])
    assert patches.shape[0] == 5
    assert np.isfinite(patches.transform).all()
    for i in range(1, 5):
        np.testing.assert_array_equal(patches.transform[i], patches.transform[0])
        np.testing.assert_array_equal(patches.data[i], patches.data[0])


def test_padded_edge_chips_batch_with_interior_ones() -> None:
    """A float field with no fill: pad edge chips declare NaN (review on #567)."""
    data = np.random.default_rng(1).random((20, 20))
    field = RasterField(GeoArray(data, UTM, "EPSG:32630", dims="y x"))
    patcher = _patcher("pad")  # 3 x 3 patches, the edge ones padded

    ((patches, valid),) = batch_split(patcher, field, batch_size=9)

    assert valid.all() and np.isnan(patches.fill_value)
    np.testing.assert_array_equal(patches.data[0], data[:8, :8])


def test_patches_with_different_fills_are_rejected() -> None:
    class TwoFills(RasterField):
        def select(self, window):  # type: ignore[no-untyped-def]
            chip = super().select(window)
            fill = -1.0 if window.col_off else -2.0
            return GeoArray(
                chip.data, chip.transform, chip.crs, fill_value=fill, dims=chip.dims
            )

    field = TwoFills(GeoArray(np.zeros((16, 16)), UTM, "EPSG:32630", dims="y x"))
    with pytest.raises(ValueError, match="one fill"):
        next(batch_split(_patcher(), field, batch_size=4))


def test_pad_last_false_yields_a_short_final_batch(field: RasterField) -> None:
    *_, (patches, valid) = batch_split(_patcher(), field, batch_size=5, pad_last=False)
    assert patches.shape[0] == 1 and valid.tolist() == [True]


def test_batch_split_works_without_jax() -> None:
    """No JAX installed: a NumPy carrier, and JAX is never imported."""
    script = """
import sys

class NoJax:  # an install without JAX: `import jax` fails
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("jax", "jaxlib"):
            raise ModuleNotFoundError(f"No module named {name!r}")

sys.meta_path.insert(0, NoJax())
import numpy as np
from geoarray import GeoArray
from geopatcher import RasterField, SpatialPatcher, spatial
from geopatcher.run import batch_split
scene = GeoArray(np.zeros((16, 16)), (1, 0, 0, 0, -1, 16), 32630, dims="y x")
field = RasterField(scene)
patcher = SpatialPatcher(
    geometry=spatial.geometry.Rectangular(size=(8, 8)),
    sampler=spatial.sampler.RegularStride(step=8),
    window=spatial.window.Boxcar(),
    aggregation=spatial.aggregation.OverlapAdd(),
)
(patches, valid), = batch_split(patcher, field, batch_size=4)
assert patches.backend == "numpy" and type(patches.data) is np.ndarray, patches
assert patches.dims == ("patch", "y", "x")
assert "jax" not in sys.modules
print("ok")
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_unbatch_places_patches_from_their_transforms(field: RasterField) -> None:
    """Overlapping Hann-weighted patches merge back to the field."""
    patcher = _patcher("pad", size=8, step=4)
    expected = list(patcher.split(field))

    patches = [
        p
        for b in batch_split(patcher, field, batch_size=7)
        for p in unbatch(b, patcher=patcher, domain=field.domain)
    ]

    assert [p.anchor for p in patches] == [p.anchor for p in expected]
    assert [p.indices for p in patches] == [p.indices for p in expected]
    for got, want in zip(patches, expected, strict=True):
        np.testing.assert_array_equal(got.weights, want.weights)
        assert got.data.grid == want.data.grid
    merged = patcher.merge(patches, field.domain)
    np.testing.assert_allclose(merged, np.asarray(field.reader), rtol=1e-6)


def test_unbatch_takes_a_map_over_output(field: RasterField) -> None:
    patcher = _patcher()
    batch = next(batch_split(patcher, field, batch_size=4))

    def score(patch: GeoArray) -> GeoArray:
        return patch.with_data(np.asarray(patch).sum(axis=0), dims="y x")

    scores = geoarray.map_over(batch.patches, score, "patch")
    patches = unbatch(batch, scores, patcher=patcher, domain=field.domain)

    assert patches[1].data.dims == ("y", "x")
    assert patches[1].data.grid == geoarray.isel(batch.patches, patch=1).grid
    np.testing.assert_allclose(patches[1].data.data, batch.patches.data[1].sum(0))


def test_unbatch_wraps_array_rows_that_keep_the_grid(field: RasterField) -> None:
    patcher = _patcher()
    batch = next(batch_split(patcher, field, batch_size=4))
    data = batch.patches.data
    per_pixel = np.stack([data[:, 0], -data[:, 0]], axis=1)  # (B, 2, H, W)
    logits = data.reshape(data.shape[0], -1)[:, :3]  # (B, 3)

    wrapped = unbatch(batch, per_pixel, patcher=patcher, domain=field.domain)
    bare = unbatch(batch, logits, patcher=patcher, domain=field.domain)

    assert isinstance(wrapped[0].data, GeoArray) and wrapped[0].data.shape == (2, 8, 8)
    assert wrapped[0].data.grid == geoarray.isel(batch.patches, patch=0).grid
    assert not isinstance(bare[0].data, GeoArray) and bare[0].data.shape == (3,)


def test_unbatch_needs_the_patcher_and_domain(field: RasterField) -> None:
    batch = next(batch_split(_patcher(), field, batch_size=4))
    with pytest.raises(TypeError, match="patcher= and domain="):
        unbatch(batch)


def test_unbatch_rejects_a_domain_the_patches_are_not_on(field: RasterField) -> None:
    patcher = _patcher()
    batch = next(batch_split(patcher, field, batch_size=4))
    shifted = GeoArray(
        np.zeros((32, 32)), (10.0, 0, 500_003.0, 0, -10.0, 4.6e6), 32630, dims="y x"
    )
    with pytest.raises(ValueError, match="not on its grid"):
        unbatch(batch, patcher=patcher, domain=shifted)


def test_unbatch_places_patches_on_a_bare_grid(field: RasterField) -> None:
    """A bare `geoarray.Grid` is a raster domain (review on #567)."""
    patcher = _patcher("pad", size=8, step=4)
    grid = field.domain.grid
    patches = [
        p
        for b in batch_split(patcher, field, batch_size=7)
        for p in unbatch(b, patcher=patcher, domain=grid)
    ]
    assert [p.indices for p in patches] == [p.indices for p in patcher.split(field)]
    np.testing.assert_allclose(
        patcher.merge(patches, grid), np.asarray(field.reader), rtol=1e-6
    )


def test_unbatch_rejects_patches_clipped_by_shrink() -> None:
    """A shrunk edge chip cannot be placed from its transform (review on #567)."""
    field = _scene(20)  # 8-px patches → 4-px edge chips
    patcher = _patcher("shrink")
    *_, edge = batch_split(patcher, field, batch_size=1)
    assert edge.patches.shape[-2:] == (4, 4)
    with pytest.raises(ValueError, match="boundary='shrink'"):
        unbatch(edge, patcher=patcher, domain=field.domain)


def test_batch_split_rejects_mixed_shapes_with_clear_error() -> None:
    """#196: shrunk edge chips fail with a message, not numpy's stack error."""
    field = _scene(20)  # 8-px patches → 4-px edge chips
    with pytest.raises(ValueError, match=r"equal-shaped patches.*boundary='pad'"):
        list(batch_split(_patcher("shrink"), field, batch_size=9))


def test_plain_array_patches_name_the_deprecated_path() -> None:
    field = ArrayField(np.arange(256, dtype=np.float32).reshape(16, 16))
    with pytest.raises(TypeError, match=r"BatchedPatch\.split"):
        next(batch_split(_patcher(), field, batch_size=2))


def test_batched_patch_split_is_deprecated_and_numpy() -> None:
    field = ArrayField(np.arange(256, dtype=np.float32).reshape(16, 16))
    patcher = _patcher()

    with pytest.deprecated_call(match="BatchedPatch.split"):
        batches = list(BatchedPatch.split(patcher, field, batch_size=3))

    assert all(isinstance(b, BatchedPatch) for b in batches)
    assert type(batches[-1].data) is np.ndarray
    np.testing.assert_array_equal(batches[-1].valid, [True, False, False])
    patches = [p for b in batches for p in unbatch(b)]
    assert [p.anchor for p in patches] == [p.anchor for p in patcher.split(field)]


def test_a_grid_dataarray_field_batches_as_a_geoarray() -> None:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")
    import rasterio

    from geopatcher.fields import RioXarrayField

    arr = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    da = xr.DataArray(arr, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:32630")
    da = da.rio.write_transform(rasterio.Affine(10.0, 0, 5e5, 0, -10.0, 4.6e6))
    da = da.assign_coords(
        x=5e5 + 5.0 + 10.0 * np.arange(16), y=4.6e6 - 5.0 - 10.0 * np.arange(16)
    )
    field = RioXarrayField(da)
    patcher = _patcher()
    expected = list(patcher.split(field))

    (patches, valid), *_ = batch_split(patcher, field, batch_size=3)

    assert patches.dims == ("patch", "y", "x") and valid.all()
    for i, want in enumerate(expected[:3]):
        item = geoarray.isel(patches, patch=i)
        assert item.grid.affine == want.data.rio.transform()
        np.testing.assert_array_equal(item.data, want.data.values)


def test_a_jitted_step_on_a_batch_compiles_once(
    field: RasterField, x64: ModuleType
) -> None:
    jax = x64
    patcher = _patcher()  # 16 patches → 5, 5, 5, 1 + 4 padding
    traces = []

    @jax.jit
    def step(batch: PatchBatch):
        traces.append(1)
        patches, valid = batch
        per_patch = patches.data.mean(axis=(1, 2, 3)) + patches.transform[:, 2]
        return (per_patch * valid).sum()

    for batch in batch_split(patcher, field, batch_size=5):
        step(PatchBatch(batch.patches.to_jax(), batch.valid))

    assert len(traces) == 1


def test_batched_patch_is_a_pytree_once_jax_loads() -> None:
    jax = pytest.importorskip("jax")
    field = ArrayField(np.arange(256, dtype=np.float32).reshape(16, 16))
    with pytest.deprecated_call():
        (batch, *_) = BatchedPatch.split(_patcher(), field, batch_size=3)

    doubled = jax.tree_util.tree_map(lambda x: x * 2, batch)

    assert len(jax.tree_util.tree_leaves(batch)) == 2  # data and valid
    assert isinstance(doubled, BatchedPatch) and doubled.anchors == batch.anchors
    np.testing.assert_array_equal(doubled.data, batch.data * 2)
