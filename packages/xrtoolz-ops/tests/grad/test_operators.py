"""`Gradient` and `Difference`: the grad functions as operators (#364)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xrops
from xrops import grad


def _ssh() -> xr.DataArray:
    lon = np.arange(0.0, 40.0, 2.0)
    lat = np.linspace(-20.0, 20.0, 11)
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.normal(size=(3, 11, 20)),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1, 2], "lat": lat, "lon": lon},
        name="ssh",
        attrs={"units": "m"},
    )


@pytest.mark.parametrize("nan_policy", ["propagate", "adaptive"])
def test_gradient_stacks_the_partials_on_component(nan_policy) -> None:
    ssh = _ssh()
    op = xrops.Gradient(
        dims=("lat", "lon"), geometry="spherical", nan_policy=nan_policy
    )
    g = op(ssh)
    assert g.dims == ("component", "time", "lat", "lon")
    assert list(g["component"].values) == ["d/dlat", "d/dlon"]
    kw = dict(geometry="spherical", nan_policy=nan_policy)
    xr.testing.assert_allclose(
        g.sel(component="d/dlon", drop=True),
        grad.partial(ssh, "lon", **kw).rename("ssh"),
    )
    assert g.attrs == {"units": "m"} and g.name == "ssh"


def test_difference_is_partial_and_takes_datasets() -> None:
    ssh = _ssh()
    ds = xr.Dataset({"ssh": ssh, "sst": ssh * 2.0})
    out = xrops.Difference("lat", geometry="spherical", accuracy=2)(ds)
    want = grad.partial(ds["sst"], "lat", geometry="spherical", accuracy=2)
    xr.testing.assert_allclose(out["sst"], want.rename("sst"))


def test_config_round_trip_and_bad_arguments() -> None:
    op = xrops.Gradient(dims=("lat", "lon"), geometry="spherical", ellipsoid="WGS84")
    assert type(op)(**op.get_config()).get_config() == op.get_config()
    with pytest.raises(ValueError, match="sequence"):
        xrops.Gradient(dims="lat")
    with pytest.raises(ValueError, match="geometry='spherical'"):
        xrops.Difference("x", ellipsoid="WGS84")


def test_one_periodic_dim_may_be_a_string() -> None:
    ssh = _ssh().assign_coords(lon=np.arange(0.0, 360.0, 18.0))
    one = xrops.Gradient(dims=("lat", "lon"), geometry="spherical", periodic="lon")
    many = xrops.Gradient(dims=("lat", "lon"), geometry="spherical", periodic=["lon"])
    assert one.periodic == ("lon",)
    xr.testing.assert_identical(one(ssh), many(ssh))
    wrapped = grad.partial(ssh, "lon", geometry="spherical", periodic=True).rename(
        "ssh"
    )
    xr.testing.assert_allclose(one(ssh).sel(component="d/dlon", drop=True), wrapped)
