"""The QA decoders on every carrier (#533).

The bit, class and field decoders are element-wise kernels run through
pipekit-array's ``KernelOp``, so each mask operator must give the same
boolean mask on a NumPy `GeoArray`, a JAX one (eagerly and under
``jax.jit``), its ``to_xarray()`` (eagerly and dask-backed) and a plain
array.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import toy_geoarray
from geoarray import GeoArray

import geoops.qa as gq


jax = pytest.importorskip("jax")
xr = pytest.importorskip("xarray")
pytest.importorskip("dask")

MASK_OPS = [
    pytest.param(gq.MaskClouds(qa_band=1, bits=[3, 4]), id="MaskClouds-bits"),
    pytest.param(gq.MaskClouds(qa_band=1, bits=[3, 4], mode="all"), id="all"),
    pytest.param(gq.MaskCirrus(qa_band=1, bits=[2]), id="MaskCirrus"),
    pytest.param(gq.MaskSnow(qa_band=1, values=[11], invert=True), id="values"),
    pytest.param(gq.MaskWater(qa_band=1, bits=[7]), id="MaskWater"),
    pytest.param(gq.MaskCloudShadow(qa_band=1, bits=[4]), id="MaskCloudShadow"),
    pytest.param(gq.S2QA60(qa_band=1), id="S2QA60"),
    pytest.param(gq.S2SCL(qa_band=1), id="S2SCL"),
    pytest.param(gq.LandsatQA_PIXEL(qa_band=1), id="LandsatQA_PIXEL"),
    pytest.param(gq.MODISStateQA(qa_band=1), id="MODISStateQA"),
]


@pytest.fixture(autouse=True)
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def scene() -> GeoArray:
    """A reflectance band and a ``uint16`` QA band, 6 x 5."""
    rng = np.random.default_rng(5)
    values = np.stack(
        [
            rng.integers(0, 10_000, size=(6, 5)),
            rng.integers(0, 2**16, size=(6, 5)),
        ]
    ).astype(np.uint16)
    values[1, :2, :2] = [[0, 11], [3, 8]]  # SCL-style classes
    return toy_geoarray(values, fill_value=None)


@pytest.mark.parametrize("op", MASK_OPS)
def test_mask_agrees_on_every_carrier(op: object, scene: GeoArray) -> None:
    want = op(scene)
    assert isinstance(want, GeoArray)
    assert want.dims == ("y", "x")
    assert want.dtype == np.bool_
    assert want.fill_value is False

    jitted = jax.jit(lambda g: op(g))(scene.to_jax())
    da = scene.to_xarray()
    on_dask = op(da.chunk({"y": 3}))
    plain = op(np.asarray(scene))

    assert jitted.backend == "jax"
    assert hasattr(on_dask.data, "dask")
    for got in (op(scene.to_jax()), jitted, op(da), on_dask.compute(), plain):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_whole_carrier_is_the_qa_band(scene: GeoArray) -> None:
    """``qa_band=None`` decodes every element and keeps the carrier's dims."""
    op = gq.MaskClouds(bits=[3])
    want = op(scene)
    assert want.dims == scene.dims
    got = jax.jit(lambda g: op(g))(scene.to_jax())
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_time_stack_keeps_a_singleton_band_dim(scene: GeoArray) -> None:
    stack = GeoArray(
        np.stack([np.asarray(scene)] * 3),
        scene.transform,
        scene.crs,
        dims=("time", "band", "y", "x"),
    )
    out = gq.S2SCL(qa_band=1)(stack)
    assert out.shape == (3, 1, 6, 5)
    np.testing.assert_array_equal(
        np.asarray(out)[1, 0], np.asarray(gq.S2SCL(qa_band=1)(scene))
    )


def test_unknown_target_fails_before_any_compute(scene: GeoArray) -> None:
    lazy = scene.to_xarray().chunk({"y": 3})
    with pytest.raises(ValueError, match="unknown s2_scl QA target"):
        gq.S2SCL(qa_band=1, targets=("nope",))(lazy)
