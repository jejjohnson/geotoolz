"""The column diagnostics on diagnostix's kernels (#773, #794)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrops import ocn


DEPTH = np.array([0.0, 5.0, 12.0, 25.0, 40.0, 70.0, 120.0])


def _columns(**coords: np.ndarray) -> xr.Dataset:
    """Four columns: stratified, mixed, land, and one whose bottom is at 40 m."""
    rho = np.stack(
        [
            1025.0 + 0.002 * DEPTH,
            np.full(7, 1025.0),
            np.full(7, np.nan),
            np.full(7, 1025.0),
        ],
        axis=-1,
    )
    rho[5:, 3] = np.nan
    return xr.Dataset({"rho": (("depth", "x"), rho)}, coords={"depth": DEPTH, **coords})


def test_mixed_layer_depth_on_land_is_nan_and_at_the_bottom_where_mixed() -> None:
    mld = ocn.mixed_layer_depth(_columns())["mld"]
    # ρ(z) = ρ(10 m) + 0.03 at 10 + 0.03 / 0.002 m
    np.testing.assert_allclose(mld, [25.0, 120.0, np.nan, 40.0], rtol=1e-9)
    assert mld.attrs["units"] == "m"


def test_mixed_layer_depth_reads_a_column_downwards() -> None:
    ds = _columns()
    xr.testing.assert_allclose(
        ocn.mixed_layer_depth(ds.isel(depth=slice(None, None, -1))),
        ocn.mixed_layer_depth(ds),
    )


@pytest.mark.parametrize(
    "call",
    [
        ocn.mixed_layer_depth,
        ocn.brunt_vaisala_frequency,
        lambda ds: ocn.lapse_rate(ds, temperature="rho"),
    ],
    ids=["mld", "n2", "lapse"],
)
def test_dask_chunked_along_the_column_equals_numpy_and_stays_lazy(call) -> None:
    pytest.importorskip("dask.array")
    ds = _columns()
    lazy = call(ds.chunk({"depth": 2, "x": 1}))
    assert all(v.chunks is not None for v in lazy.data_vars.values())
    xr.testing.assert_allclose(lazy.compute(), call(ds))


def test_the_level_dim_stays_where_it_is() -> None:
    ds = _columns().transpose("x", "depth")
    out = ocn.brunt_vaisala_frequency(ds)["n_squared"]
    assert out.dims == ("x", "depth")
    xr.testing.assert_identical(out["depth"], ds["depth"])
    xr.testing.assert_allclose(
        out.transpose("depth", "x"),
        ocn.brunt_vaisala_frequency(_columns())["n_squared"],
    )


def test_n_squared_of_a_quadratic_is_exact_at_inner_levels() -> None:
    rho = xr.DataArray(
        1025.0 + 1e-4 * DEPTH**2, dims="depth", coords={"depth": DEPTH}, name="rho"
    )
    out = ocn.brunt_vaisala_frequency(rho.to_dataset(), rho0=1025.0, g=9.81)
    expected = 9.81 / 1025.0 * 2e-4 * DEPTH
    np.testing.assert_allclose(out["n_squared"][1:-1], expected[1:-1], rtol=1e-8)


def test_positive_up_and_down_agree() -> None:
    ds = _columns().isel(x=[0])
    height = ds.assign_coords(depth=-DEPTH)
    np.testing.assert_allclose(
        ocn.brunt_vaisala_frequency(height, positive="up")["n_squared"],
        ocn.brunt_vaisala_frequency(ds)["n_squared"],
    )


def test_float32_stays_float32() -> None:
    ds = _columns().astype("float32")
    assert ocn.brunt_vaisala_frequency(ds)["n_squared"].dtype == np.float32
    assert ocn.mixed_layer_depth(ds)["mld"].dtype == np.float32


def test_repeated_levels_raise() -> None:
    ds = _columns().assign_coords(depth=[0.0, 5.0, 5.0, 25.0, 40.0, 70.0, 120.0])
    with pytest.raises(ValueError, match="strictly monotonic"):
        ocn.brunt_vaisala_frequency(ds)
