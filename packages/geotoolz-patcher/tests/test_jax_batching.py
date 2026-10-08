"""`geopatcher.jax` batching: carriers, shape pre-check, pytree registration."""

from __future__ import annotations

import numpy as np
import pytest
from georeader.geotensor import GeoTensor

from geopatcher import RasterField, SpatialPatcher, spatial


jax = pytest.importorskip("jax")

from geopatcher.run import BatchedPatch, batch_split, unbatch


def _patcher(boundary: str = "drop", size: int = 8) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(size, size), boundary=boundary),
        sampler=spatial.sampler.RegularStride(step=size),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


@pytest.fixture
def field(raster_field_factory) -> RasterField:
    return raster_field_factory(32)


def test_unbatch_round_trips_geotensor_carriers(field: RasterField) -> None:
    """#196: unbatch rebuilds GeoTensor chips with their georeferencing."""
    patcher = _patcher()
    expected = list(patcher.split(field))
    batches = list(batch_split(patcher, field, batch_size=5))

    patches = [p for b in batches for p in unbatch(b)]

    assert [p.anchor for p in patches] == [p.anchor for p in expected]
    for got, want in zip(patches, expected, strict=True):
        assert isinstance(got.data, GeoTensor)
        assert got.data.transform == want.data.transform
        assert got.data.crs == want.data.crs
        assert got.data.fill_value_default == want.data.fill_value_default
        np.testing.assert_array_equal(np.asarray(got.data), np.asarray(want.data))
    merged = patcher.merge(patches, field.domain)
    np.testing.assert_array_equal(merged, np.asarray(field.reader.values))


def test_unbatch_wraps_model_output_with_new_band_axis(field: RasterField) -> None:
    """A per-pixel output with a band axis keeps the chip's georeferencing."""
    patcher = _patcher()
    (batch, *_) = batch_split(patcher, field, batch_size=4)
    out = jax.numpy.stack([batch.data, -batch.data], axis=1)  # (B, 2, H, W)

    patches = unbatch(batch, out)

    first = next(iter(patcher.split(field)))
    assert isinstance(patches[0].data, GeoTensor)
    assert patches[0].data.shape == (2, 8, 8)
    assert patches[0].data.transform == first.data.transform


def test_unbatch_leaves_non_spatial_output_bare(field: RasterField) -> None:
    patcher = _patcher()
    (batch, *_) = batch_split(patcher, field, batch_size=4)
    logits = batch.data.reshape(batch.data.shape[0], -1)[:, :3]

    patches = unbatch(batch, logits)

    assert not isinstance(patches[0].data, GeoTensor)
    assert patches[0].data.shape == (3,)


def test_batch_split_rejects_mixed_shapes_with_clear_error(
    raster_field_factory,
) -> None:
    """#196: shrunk edge chips fail with a message, not numpy's stack error."""
    field = raster_field_factory(20)  # 8-px patches → 4-px edge chips
    with pytest.raises(ValueError, match=r"equal-shaped patches.*boundary='pad'"):
        list(batch_split(_patcher("shrink"), field, batch_size=9))


def test_batched_patch_is_a_pytree(field: RasterField) -> None:
    patcher = _patcher()
    (batch, *_) = batch_split(patcher, field, batch_size=3)

    leaves = jax.tree_util.tree_leaves(batch)
    doubled = jax.tree_util.tree_map(lambda x: x * 2, batch)

    assert len(leaves) == 2  # data and valid
    assert isinstance(doubled, BatchedPatch)
    assert doubled.anchors == batch.anchors
    np.testing.assert_array_equal(doubled.data, batch.data * 2)
    placed = jax.device_put(batch)
    assert isinstance(placed, BatchedPatch)
    assert isinstance(unbatch(placed)[0].data, GeoTensor)


def test_padded_entries_are_dropped(field: RasterField) -> None:
    patcher = _patcher()  # 16 patches → 5, 5, 5, 1 (+ 4 padding)
    last = list(batch_split(patcher, field, batch_size=5))[-1]
    assert last.data.shape[0] == 5
    assert last.carriers[1:] == [None] * 4
    assert len(unbatch(last)) == 1


def test_unbatch_round_trips_dataarray_carriers() -> None:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")
    import rasterio

    from geopatcher.fields import RioXarrayField

    arr = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": np.arange(16) + 0.5})
    da = da.rio.write_crs("EPSG:32630")
    da = da.rio.write_transform(rasterio.Affine(10.0, 0, 5e5, 0, -10.0, 4.6e6))
    field = RioXarrayField(da)
    patcher = _patcher()
    expected = list(patcher.split(field))

    patches = [p for b in batch_split(patcher, field, batch_size=3) for p in unbatch(b)]

    for got, want in zip(patches, expected, strict=True):
        assert isinstance(got.data, xr.DataArray)
        xr.testing.assert_identical(got.data, want.data)
        assert got.data.rio.transform() == want.data.rio.transform()
