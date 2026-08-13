from __future__ import annotations

import json

import numpy as np
import pytest
import rioxarray  # noqa: F401  — needed so ds.rio is populated
import xarray as xr

from xrtoolz.geo import ReprojectMatch, reproject_match


def _raster(n: int, *, crs: str = "EPSG:4326", scale: float = 1.0) -> xr.DataArray:
    """North-up raster of ``n``x``n`` cells covering x,y in [0, n*scale)."""
    return xr.DataArray(
        np.arange(float(n * n)).reshape(n, n),
        dims=("y", "x"),
        coords={
            "y": scale * np.arange(n - 1.0, -1.0, -1.0),
            "x": scale * np.arange(float(n)),
        },
    ).rio.write_crs(crs)


def test_lands_exactly_on_target_grid_across_crs() -> None:
    source = _raster(8, crs="EPSG:4326")
    target = _raster(3, crs="EPSG:3857", scale=2.0)

    out = reproject_match(source, target)

    assert out.shape == target.shape
    assert out.rio.crs == target.rio.crs
    assert out.rio.transform() == target.rio.transform()
    np.testing.assert_allclose(out.x.values, target.x.values)
    np.testing.assert_allclose(out.y.values, target.y.values)


def test_same_crs_downsample_keeps_target_shape() -> None:
    source = _raster(8)
    target = _raster(4, scale=2.0)

    out = reproject_match(source, target)

    assert out.shape == target.shape
    np.testing.assert_allclose(out.x.values, target.x.values)


def test_dataset_input_returns_dataset() -> None:
    source = _raster(8).to_dataset(name="sst")
    target = _raster(4, scale=2.0)

    out = reproject_match(source, target)

    assert isinstance(out, xr.Dataset)
    assert out["sst"].shape == target.shape


def test_unknown_resampling_raises() -> None:
    with pytest.raises(ValueError, match="Unknown resampling"):
        reproject_match(_raster(4), _raster(2, scale=2.0), resampling="bogus")


def test_operator_matches_function_and_config_round_trips() -> None:
    source, target = _raster(8), _raster(4, scale=2.0)
    op = ReprojectMatch(target, resampling="nearest")

    xr.testing.assert_allclose(
        op(source), reproject_match(source, target, resampling="nearest")
    )
    config = op.get_config()
    assert config == {"target": "<DataArray>", "resampling": "nearest"}
    assert json.loads(json.dumps(config)) == config


def test_operator_forbids_yaml_serialisation() -> None:
    # Holds a live target raster, so it must not be YAML round-trippable.
    assert ReprojectMatch.forbid_in_yaml is True


def test_operator_maps_over_a_two_leaf_datatree() -> None:
    target = _raster(4, scale=2.0)
    tree = xr.DataTree.from_dict(
        {
            "coarse": _raster(8).to_dataset(name="sst"),
            "fine": (_raster(8) * 2).to_dataset(name="sst"),
        }
    )

    out = ReprojectMatch(target)(tree)

    assert isinstance(out, xr.DataTree)
    assert set(out.children) == {"coarse", "fine"}
    for leaf in ("coarse", "fine"):
        assert out[leaf].dataset["sst"].shape == target.shape
        assert out[leaf].dataset.rio.crs == target.rio.crs


def test_output_signature_replaces_source_spatial_dims_with_targets() -> None:
    from xrcore import Signature

    target = _raster(4, scale=2.0)

    signature = ReprojectMatch(target).compute_output_signature(
        Signature({"time": 5, "lat": 8, "lon": 8}, dtype=np.dtype("float32"))
    )

    # reproject_match returns the *target's* spatial names, so the source's
    # must not survive alongside them as a phantom third/fourth spatial dim.
    assert signature.dims == {"time": 5, "y": 4, "x": 4}


def test_differing_spatial_dim_names_take_the_target_names() -> None:
    target = _raster(4, scale=2.0)
    source = _raster(8).rename({"y": "lat", "x": "lon"})
    source = source.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs(
        "EPSG:4326"
    )

    out = ReprojectMatch(target)(source)

    assert out.dims == ("y", "x")
    assert out.shape == target.shape
