"""Fit-time feature layout: validation, reordering, target alignment, fidelity."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from xrsklearn import SklearnOp, XarrayEstimator


def _cube(seed: int = 0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    da = xr.DataArray(
        rng.normal(size=(20, 3, 4)),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(20),
            "lat": [10.0, 0.0, -10.0],
            "lon": np.arange(4.0),
        },
        name="ssh",
        attrs={"units": "m"},
    )
    da["lat"].attrs["units"] = "degrees_north"
    return da


# ---------- feature validation + reordering (X) ---------------------------


def test_transposed_input_gives_identical_output() -> None:
    da = _cube()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    a = wrap.transform(da)
    b = wrap.transform(da.transpose("time", "lon", "lat"))

    xr.testing.assert_allclose(a, b.transpose(*a.dims))


def test_permuted_coordinate_is_reordered_to_fit_order() -> None:
    da = _cube()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    flipped = da.isel(lat=slice(None, None, -1))
    out = wrap.transform(flipped)

    xr.testing.assert_allclose(out.sel(lat=da.lat), wrap.transform(da))


def test_different_region_same_shape_raises() -> None:
    da = _cube()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    with pytest.raises(ValueError, match="coordinate 'lat' does not match"):
        wrap.transform(da.assign_coords(lat=[50.0, 60.0, 70.0]))


def test_different_feature_dims_raise() -> None:
    da = _cube()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    with pytest.raises(ValueError, match="feature dims"):
        wrap.transform(da.rename(lon="x"))


def test_different_size_raises() -> None:
    da = _cube()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    with pytest.raises(ValueError, match="has size 2"):
        wrap.transform(da.isel(lon=slice(0, 2)))


def test_input_without_coords_matches_by_size() -> None:
    da = _cube()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    bare = da.drop_vars(["lat", "lon"])

    np.testing.assert_allclose(wrap.transform(bare).values, wrap.transform(da).values)


def test_fit_on_dataarray_transform_dataset_raises() -> None:
    da = _cube()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    with pytest.raises(TypeError, match="fit on a DataArray"):
        wrap.transform(da.to_dataset())


def test_dataset_variables_must_match_fit_time() -> None:
    ds = xr.Dataset({"u": _cube(0), "v": _cube(1)})
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(ds)

    with pytest.raises(ValueError, match="data variables"):
        wrap.transform(ds[["u"]])


def test_dataset_variable_order_does_not_matter() -> None:
    ds = xr.Dataset({"u": _cube(0), "v": _cube(1)})
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(ds)

    swapped = xr.Dataset({"v": ds["v"], "u": ds["u"]})

    xr.testing.assert_allclose(wrap.transform(swapped), wrap.transform(ds))


def test_dataset_variable_without_sample_dim_is_named_in_error() -> None:
    ds = xr.Dataset({"ssh": _cube(), "mask": _cube().isel(time=0, drop=True)})

    with pytest.raises(ValueError, match="variable 'mask'"):
        XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(ds)


# ---------- sample_dim is resolved once ----------------------------------


def test_inferred_sample_dim_is_recorded_at_fit() -> None:
    da = _cube()
    wrap = XarrayEstimator(StandardScaler()).fit(da)

    out = wrap.transform(da.transpose("lat", "time", "lon"))

    assert wrap.sample_dim_ == "time"
    assert out.dims == ("lat", "time", "lon")
    xr.testing.assert_allclose(out.transpose(*da.dims), wrap.transform(da))


def test_refit_re_resolves_sample_dim() -> None:
    wrap = XarrayEstimator(StandardScaler())
    wrap.fit(_cube())
    wrap.fit(_cube().transpose("lon", "time", "lat"))

    assert wrap.sample_dim_ == "lon"


# ---------- y alignment ---------------------------------------------------


def test_y_is_aligned_to_x_samples() -> None:
    da = _cube()
    y = da.mean(("lat", "lon"))

    a = XarrayEstimator(LinearRegression(), sample_dim="time").fit(da, y)
    b = XarrayEstimator(LinearRegression(), sample_dim="time").fit(
        da, y.isel(time=slice(None, None, -1))
    )

    np.testing.assert_allclose(a.coef_, b.coef_)


def test_y_with_other_samples_raises() -> None:
    da = _cube()
    y = da.mean(("lat", "lon")).assign_coords(time=np.arange(100, 120))

    with pytest.raises(ValueError, match="does not match X's"):
        XarrayEstimator(LinearRegression(), sample_dim="time").fit(da, y)


def test_y_with_wrong_length_raises() -> None:
    da = _cube()
    y = da.mean(("lat", "lon")).isel(time=slice(0, 10))

    with pytest.raises(ValueError, match="10 samples"):
        XarrayEstimator(LinearRegression(), sample_dim="time").fit(da, y)


def test_score_aligns_y() -> None:
    da = _cube()
    y = da.mean(("lat", "lon"))
    wrap = XarrayEstimator(LinearRegression(), sample_dim="time").fit(da, y)

    assert wrap.score(da, y.isel(time=slice(None, None, -1))) == pytest.approx(
        wrap.score(da, y)
    )


# ---------- parameter validation + method gating --------------------------


def test_unknown_nan_policy_raises_at_fit() -> None:
    wrap = XarrayEstimator(StandardScaler(), nan_policy="drop")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="nan_policy must be one of"):
        wrap.fit(_cube())


def test_non_string_new_feature_dim_raises() -> None:
    wrap = XarrayEstimator(PCA(n_components=2), new_feature_dim=3)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="new_feature_dim"):
        wrap.fit(_cube())


def test_methods_absent_on_delegate_are_absent_on_wrapper() -> None:
    scaler = XarrayEstimator(StandardScaler(), sample_dim="time")
    regressor = XarrayEstimator(LinearRegression(), sample_dim="time")

    assert not hasattr(scaler, "predict")
    assert not hasattr(scaler, "predict_proba")
    assert hasattr(scaler, "inverse_transform")
    assert not hasattr(regressor, "transform")
    assert hasattr(regressor, "predict")
    with pytest.raises(AttributeError, match="StandardScaler does not implement"):
        scaler.predict_proba  # noqa: B018


# ---------- output fidelity ----------------------------------------------


def test_dim_order_and_coord_attrs_are_restored() -> None:
    da = _cube().transpose("lat", "lon", "time")

    out = XarrayEstimator(StandardScaler(), sample_dim="time").fit_transform(da)

    assert out.dims == da.dims
    assert out["lat"].attrs == {"units": "degrees_north"}
    assert out.attrs == da.attrs


def test_auxiliary_coords_are_kept() -> None:
    rng = np.random.default_rng(1)
    da = _cube().assign_coords(
        nav_lat=(("lat", "lon"), rng.normal(size=(3, 4))),
        month=("time", np.arange(20) % 12),
    )

    out = XarrayEstimator(StandardScaler(), sample_dim="time").fit_transform(da)

    xr.testing.assert_equal(out["nav_lat"], da["nav_lat"])
    xr.testing.assert_equal(out["month"], da["month"])


def test_inverse_transform_restores_auxiliary_coords() -> None:
    rng = np.random.default_rng(1)
    da = _cube().assign_coords(nav_lat=(("lat", "lon"), rng.normal(size=(3, 4))))
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time")

    recon = wrap.inverse_transform(wrap.fit_transform(da))

    xr.testing.assert_equal(recon["nav_lat"], da["nav_lat"])
    assert recon["lat"].attrs == {"units": "degrees_north"}


def test_no_coordinates_are_invented() -> None:
    da = xr.DataArray(np.random.default_rng(0).normal(size=(5, 3)), dims=("t", "x"))

    out = XarrayEstimator(StandardScaler()).fit_transform(da)

    assert list(out.coords) == []


def test_layout_does_not_retain_training_data() -> None:
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(_cube())

    assert not wrap.layout_.feature_coords.to_dataset().data_vars
    assert "time" not in wrap.layout_.feature_coords


# ---------- accessor + SklearnOp settings --------------------------------


def test_accessor_rejects_conflicting_override_on_fitted_wrapper() -> None:
    da = _cube()
    fitted = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    with pytest.raises(ValueError, match="nan_policy='raise'"):
        da.sklearn.transform(fitted, nan_policy="raise")
    # Restating the wrapper's own value is fine.
    da.sklearn.transform(fitted, sample_dim="time")


def test_accessor_fit_with_wrapper_fits_a_clone() -> None:
    da = _cube()
    template = XarrayEstimator(StandardScaler(), sample_dim="time")

    fitted = da.sklearn.fit(template, nan_policy="raise")

    assert fitted is not template
    assert "estimator_" not in template.__dict__
    assert isinstance(fitted.estimator_, StandardScaler)  # not a nested wrapper
    assert fitted.nan_policy == "raise"


def test_sklearn_op_fit_transform_does_not_mutate_wrapper() -> None:
    da = _cube()
    template = XarrayEstimator(StandardScaler(), sample_dim="time")

    SklearnOp(template, method="fit_transform")(da)

    assert "estimator_" not in template.__dict__


def test_sklearn_op_rejects_conflicting_setting_on_fitted_wrapper() -> None:
    da = _cube()
    fitted = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    with pytest.raises(ValueError, match="sample_dim='lat'"):
        SklearnOp(fitted, sample_dim="lat")(da)


def test_sklearn_op_accepts_falsy_variable_names() -> None:
    ds = xr.Dataset({0: _cube()})
    fitted = XarrayEstimator(StandardScaler(), sample_dim="time").fit(ds[0])

    out = SklearnOp(fitted, variable=0)(ds)

    assert list(out.data_vars) == [0]


# ---------- review follow-ups (#329) --------------------------------------


def test_different_auxiliary_grid_with_same_indexes_raises() -> None:
    rng = np.random.default_rng(1)
    da = _cube().assign_coords(nav_lat=(("lat", "lon"), rng.normal(size=(3, 4))))
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    other = da.assign_coords(nav_lat=da["nav_lat"] + 1.0)
    with pytest.raises(ValueError, match="auxiliary coordinate 'nav_lat'"):
        wrap.transform(other)
    # Same grid passes, including after a permutation, and so does an input
    # that does not carry the auxiliary coordinate at all.
    wrap.transform(da.transpose("time", "lon", "lat"))
    wrap.transform(da.isel(lat=slice(None, None, -1)))
    wrap.transform(da.drop_vars("nav_lat"))


def test_forward_output_keeps_the_input_metadata() -> None:
    da = _cube()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time").fit(da)

    out = wrap.transform(da.rename("sst").assign_attrs(units="K"))

    assert out.name == "sst"
    assert out.attrs == {"units": "K"}


def test_inverse_transform_restores_training_metadata() -> None:
    da = _cube()
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time")

    recon = wrap.inverse_transform(wrap.fit_transform(da).rename("scores"))

    assert recon.name == "ssh"
    assert recon.attrs == {"units": "m"}


def test_restating_an_inferred_sample_dim_is_not_a_conflict() -> None:
    da = _cube()
    fitted = XarrayEstimator(StandardScaler()).fit(da)
    assert fitted.sample_dim_ == "time"

    da.sklearn.transform(fitted, sample_dim="time")
    SklearnOp(fitted, method="transform", variable="ssh", sample_dim="time")(
        da.to_dataset()
    )
    with pytest.raises(ValueError, match="sample_dim='lat'"):
        da.sklearn.transform(fitted, sample_dim="lat")


def test_sample_coord_named_like_new_feature_dim_is_dropped() -> None:
    da = _cube().assign_coords(component=("time", np.arange(20) * 10))
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time")

    scores = wrap.fit_transform(da)

    assert scores.dims == ("time", "component")
    np.testing.assert_array_equal(scores["component"].values, [0, 1])
