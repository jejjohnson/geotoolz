"""`xrops.learn.XarrayEstimator`: DataArray, Dataset and DataTree on pipekit-array."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


pytest.importorskip("sklearn")
import pipekit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from xrops.learn import XarrayEstimator


def _cube(seed: int = 0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(10, 3, 4)),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(10),
            "lat": [10.0, 20.0, 30.0],
            "lon": [0.0, 1.0, 2.0, 3.0],
        },
        name="ssh",
        attrs={"units": "m"},
    )


def _table(da: xr.DataArray) -> np.ndarray:
    """The hand-flattened table: one row per time, lat-lon cells as columns."""
    return da.transpose("time", "lat", "lon").values.reshape(da.sizes["time"], -1)


def test_pca_matches_sklearn_on_the_flattened_table() -> None:
    da = _cube()
    est = XarrayEstimator(PCA(3), sample_dims="time", feature_dim="component")
    scores = est.fit(da).transform(da)
    assert scores.dims == ("time", "component")
    np.testing.assert_array_equal(scores["time"].values, da["time"].values)
    want = PCA(3).fit(_table(da)).transform(_table(da))
    np.testing.assert_allclose(scores.values, want, rtol=1e-10, atol=1e-12)
    recon = est.inverse_transform(scores)
    assert recon.dims == da.dims and recon.name == "ssh" and recon.attrs == da.attrs
    np.testing.assert_array_equal(recon["lat"].values, da["lat"].values)
    np.testing.assert_allclose(
        _table(recon), PCA(3).fit(_table(da)).inverse_transform(want), atol=1e-12
    )


def test_a_transposed_input_gives_the_same_scores() -> None:
    da = _cube()
    est = XarrayEstimator(PCA(2), sample_dims="time", feature_dim="mode").fit(da)
    same = est.transform(da.transpose("lon", "time", "lat"))
    np.testing.assert_allclose(same.values, est.transform(da).values)


def test_a_one_to_one_transform_keeps_the_grid_and_metadata() -> None:
    da = _cube()
    out = XarrayEstimator(StandardScaler(), sample_dims="time").fit_transform(da)
    xr.testing.assert_allclose(out, (da - da.mean("time")) / da.std("time"))
    assert out.name == "ssh" and out.attrs == {"units": "m"}


def test_dataset_variables_are_side_by_side_features() -> None:
    da = _cube()
    ds = xr.Dataset({"u": da, "v": da.isel(lat=0) * 2.0})
    est = XarrayEstimator(PCA(2), sample_dims="time", feature_dim="component").fit(ds)
    table = np.concatenate([_table(ds["u"]), ds["v"].values], axis=1)
    np.testing.assert_allclose(
        est.transform(ds).values, PCA(2).fit(table).transform(table), atol=1e-12
    )
    back = est.inverse_transform(est.transform(ds))
    assert isinstance(back, xr.Dataset) and set(back.data_vars) == {"u", "v"}
    assert back["v"].dims == ("time", "lon")


def test_dataset_one_to_one_returns_a_dataset() -> None:
    ds = xr.Dataset({"u": _cube(), "v": _cube(1).isel(lat=0)})
    out = XarrayEstimator(StandardScaler(), sample_dims="time").fit_transform(ds)
    assert isinstance(out, xr.Dataset)
    xr.testing.assert_allclose(
        out["v"], (ds["v"] - ds["v"].mean("time")) / ds["v"].std("time")
    )


def test_several_sample_dims() -> None:
    da = _cube()
    labels = (
        XarrayEstimator(
            KMeans(2, n_init=1, random_state=0),
            sample_dims=("lat", "lon"),
            method="predict",
        )
        .fit(da)
        .predict(da)
    )
    assert labels.dims == ("lat", "lon")
    pixels = da.transpose("lat", "lon", "time").values.reshape(12, 10)
    want = KMeans(2, n_init=1, random_state=0).fit(pixels).predict(pixels)
    np.testing.assert_array_equal(labels.values.ravel(), want)


def test_supervised_fit_and_predict_proba() -> None:
    da = _cube()
    y = (da.mean(("lat", "lon")) > 0).astype(int)
    est = XarrayEstimator(LogisticRegression(), sample_dims="time").fit(da, y)
    proba = est.predict_proba(da)
    assert proba.dims == ("time", "class") and proba.sizes["class"] == 2
    reg = XarrayEstimator(LinearRegression(), sample_dims="time").fit(
        da, da.mean(("lat", "lon"))
    )
    np.testing.assert_allclose(
        reg.predict(da).values, da.mean(("lat", "lon")).values, atol=1e-8
    )


def test_mask_samples_puts_gaps_back_as_nan() -> None:
    da = _cube().copy()
    da[3, 1, 1] = np.nan
    est = XarrayEstimator(
        PCA(2), sample_dims="time", feature_dim="component", nan_policy="mask_samples"
    )
    scores = est.fit_transform(da)
    assert (
        bool(scores[3].isnull().all())
        and int(scores.notnull().all("component").sum()) == 9
    )


def test_mask_features_drops_land_and_restores_it() -> None:
    da = _cube().copy()
    da[:, 0, 0] = np.nan
    est = XarrayEstimator(
        PCA(2), sample_dims="time", feature_dim="component", nan_policy="mask_features"
    )
    recon = est.inverse_transform(est.fit_transform(da))
    assert bool(recon[:, 0, 0].isnull().all()) and bool(recon[:, 1:, :].notnull().all())


def test_mismatched_grids_and_kinds_name_the_problem() -> None:
    da = _cube()
    est = XarrayEstimator(StandardScaler(), sample_dims="time").fit(da)
    with pytest.raises(ValueError, match="fitted on features"):
        est.transform(da.isel(lon=slice(0, 2)))
    with pytest.raises(TypeError, match="fitted on a DataArray"):
        est.transform(xr.Dataset({"ssh": da}))
    with pytest.raises(ValueError, match="without the sample dim"):
        est.transform(da.isel(time=0))
    with pytest.raises(ValueError, match="not fitted"):
        XarrayEstimator(PCA(2), sample_dims="time").transform(da)
    with pytest.raises(ValueError, match="tree_mode"):
        XarrayEstimator(PCA(2), sample_dims="time", tree_mode="concat_features")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nan_policy"):
        XarrayEstimator(PCA(2), sample_dims="time", nan_policy="mask")  # type: ignore[arg-type]
    ds = xr.Dataset({"u": da, "v": da})
    fitted = XarrayEstimator(StandardScaler(), sample_dims="time").fit(ds)
    with pytest.raises(ValueError, match="fitted on variables"):
        fitted.transform(ds[["u"]])


def test_datatree_per_node_fits_each_grid() -> None:
    tree = xr.DataTree.from_dict(
        {
            "coarse": xr.Dataset({"ssh": _cube().isel(lon=slice(0, 2))}),
            "fine": xr.Dataset({"ssh": _cube(1)}),
        }
    )
    est = XarrayEstimator(PCA(2), sample_dims="time", feature_dim="component").fit(tree)
    assert est.estimators_ is not None and set(est.estimators_) == {"/coarse", "/fine"}
    scores = est.transform(tree)
    assert scores["fine"]["transformed"].dims == ("time", "component")
    recon = est.inverse_transform(scores)
    assert recon["coarse"]["ssh"].shape == (10, 3, 2)


def test_datatree_pool_samples_fits_one_estimator() -> None:
    a, b = _cube(), _cube(1)
    tree = xr.DataTree.from_dict(
        {"a": xr.Dataset({"ssh": a}), "b": xr.Dataset({"ssh": b})}
    )
    est = XarrayEstimator(
        PCA(2), sample_dims="time", feature_dim="component", tree_mode="pool_samples"
    ).fit(tree)
    pooled = np.concatenate([_table(a), _table(b)])
    ref = PCA(2).fit(pooled)
    np.testing.assert_allclose(
        est.transform(tree)["b"]["transformed"].values,
        ref.transform(_table(b)),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="fitted on features"):
        est.fit(
            xr.DataTree.from_dict(
                {"a": xr.Dataset({"ssh": a}), "b": xr.Dataset({"ssh": b.isel(lon=[0])})}
            )
        )


def test_dask_input_is_computed_for_the_fit() -> None:
    pytest.importorskip("dask")
    da = _cube()
    lazy = da.chunk({"time": 5})
    est = XarrayEstimator(StandardScaler(), sample_dims="time")
    xr.testing.assert_allclose(est.fit_transform(lazy), est.fit_transform(da))


def test_it_is_an_operator_that_composes() -> None:
    da = _cube()
    est = XarrayEstimator(PCA(2), sample_dims="time", feature_dim="component").fit(da)
    assert isinstance(est, pipekit.FittableTransformer) and est.forbid_in_yaml
    xr.testing.assert_allclose(pipekit.Sequential([est])(da), est.transform(da))


def test_the_accessor_forwards() -> None:
    import xrops.learn  # noqa: F401  (registers .sklearn)

    da = _cube()
    est = da.sklearn.fit(PCA(2), sample_dims="time", feature_dim="component")
    xr.testing.assert_allclose(da.sklearn.transform(est), est.transform(da))
    with pytest.raises(TypeError, match=r"returned by \.sklearn\.fit"):
        da.sklearn.transform(PCA(2))
    ds = xr.Dataset({"u": da})
    assert isinstance(
        ds.sklearn.fit_transform(StandardScaler(), sample_dims="time"), xr.Dataset
    )


def test_a_stacked_multiindex_sample_dim_survives() -> None:
    ps = _cube().stack(point=("lat", "lon")).transpose("point", "time")
    est = XarrayEstimator(PCA(2), sample_dims="point", feature_dim="component").fit(ps)
    patches = [est.transform(ps.isel(point=s)) for s in (slice(0, 6), slice(6, 12))]
    scores = xr.concat(patches, dim="point")
    assert scores.indexes["point"].equals(ps.indexes["point"])
    np.testing.assert_allclose(scores.values, est.transform(ps).values)


def test_permuted_feature_labels_are_matched_by_label() -> None:
    da = _cube()
    est = XarrayEstimator(PCA(2), sample_dims="time", feature_dim="component").fit(da)
    shuffled = da.isel(lon=[2, 0, 3, 1], lat=[1, 2, 0])
    np.testing.assert_allclose(est.transform(shuffled).values, est.transform(da).values)
    with pytest.raises(ValueError, match="other 'lon' labels"):
        est.transform(da.assign_coords(lon=da["lon"] + 10.0))


def test_a_target_in_another_sample_order_is_matched_by_label() -> None:
    da = _cube()
    y = da.mean(("lat", "lon"))
    est = XarrayEstimator(LinearRegression(), sample_dims="time")
    shuffled = y.isel(time=np.random.default_rng(1).permutation(10))
    np.testing.assert_allclose(
        est.fit(da, shuffled).predict(da).values, y.values, atol=1e-8
    )
    with pytest.raises(ValueError, match="other 'time' labels"):
        est.fit(da, y.assign_coords(time=y["time"] + 100))


def test_one_to_one_outputs_invert_on_their_grids() -> None:
    da = _cube()
    est = XarrayEstimator(StandardScaler(), sample_dims="time").fit(da)
    xr.testing.assert_allclose(est.inverse_transform(est.transform(da)), da)
    ds = xr.Dataset({"u": da, "v": da.isel(lat=0) * 2.0})
    est = XarrayEstimator(StandardScaler(), sample_dims="time").fit(ds)
    xr.testing.assert_allclose(est.inverse_transform(est.transform(ds)), ds)
    tree = xr.DataTree.from_dict({"a": xr.Dataset({"u": da, "v": da * 2.0})})
    est = XarrayEstimator(StandardScaler(), sample_dims="time").fit(tree)
    back = est.inverse_transform(est.transform(tree))
    xr.testing.assert_allclose(back["a"].to_dataset(), tree["a"].to_dataset())


@pytest.mark.parametrize("tree_mode", ["per_node", "pool_samples"])
def test_tree_scores_invert_even_when_a_variable_is_named_like_them(tree_mode) -> None:
    tree = xr.DataTree.from_dict({"a": xr.Dataset({"transformed": _cube()})})
    est = XarrayEstimator(
        PCA(2), sample_dims="time", feature_dim="component", tree_mode=tree_mode
    ).fit(tree)
    back = est.inverse_transform(est.transform(tree))
    assert back["a"]["transformed"].dims == ("time", "lat", "lon")


def test_scalar_coordinates_come_from_the_input_not_the_fit() -> None:
    ens = xr.concat([_cube(0), _cube(1)], dim="member").assign_coords(member=[0, 1])
    est = XarrayEstimator(StandardScaler(), sample_dims="time").fit(ens.sel(member=0))
    assert int(est.transform(ens.sel(member=1))["member"]) == 1
    pca = XarrayEstimator(PCA(2), sample_dims="time", feature_dim="component")
    scores = pca.fit(ens.sel(member=0)).transform(ens.sel(member=1))
    assert int(scores["member"]) == 1
    assert int(pca.inverse_transform(scores)["member"]) == 1


@pytest.mark.parametrize("tree_mode", ["per_node", "pool_samples"])
def test_tree_outputs_keep_metadata_only_nodes(tree_mode) -> None:
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"title": "run"}),
            "/a": xr.Dataset({"ssh": _cube()}, attrs={"region": "a"}),
            "/meta": xr.Dataset(attrs={"note": "kept"}),
        }
    )
    est = XarrayEstimator(StandardScaler(), sample_dims="time", tree_mode=tree_mode)
    out = est.fit(tree).transform(tree)
    assert out.attrs == {"title": "run"} and out["meta"].attrs == {"note": "kept"}
    assert out["a"].attrs == {"region": "a"}
    assert out["a"]["ssh"].dims == ("time", "lat", "lon")


def test_predict_proba_is_labelled_by_class() -> None:
    da = _cube()
    y = xr.where(da.mean(("lat", "lon")) > 0, "wet", "dry")
    est = XarrayEstimator(LogisticRegression(), sample_dims="time").fit(da, y)
    proba = est.predict_proba(da)
    assert list(proba["class"].values) == ["dry", "wet"]
    np.testing.assert_array_equal(proba.idxmax("class").values, est.predict(da).values)
    tree = xr.DataTree.from_dict({"a": xr.Dataset({"ssh": da})})
    fitted = XarrayEstimator(LogisticRegression(), sample_dims="time").fit(
        tree, {"/a": y}
    )
    assert list(fitted.predict_proba(tree)["a"]["class"].values) == ["dry", "wet"]


def test_another_curvilinear_grid_is_refused() -> None:
    da = _cube().assign_coords(nav_lat=(("lat", "lon"), np.arange(12.0).reshape(3, 4)))
    est = XarrayEstimator(StandardScaler(), sample_dims="time").fit(da)
    est.transform(da)  # the same grid
    other = da.assign_coords(nav_lat=da["nav_lat"] + 1.0)
    with pytest.raises(ValueError, match="another 'nav_lat' coordinate"):
        est.transform(other)
    tree = xr.DataTree.from_dict(
        {"a": xr.Dataset({"ssh": da}), "b": xr.Dataset({"ssh": other})}
    )
    with pytest.raises(ValueError, match="another 'nav_lat' coordinate"):
        XarrayEstimator(
            StandardScaler(), sample_dims="time", tree_mode="pool_samples"
        ).fit(tree)


def test_dataset_attrs_survive_rebuilding() -> None:
    ds = xr.Dataset({"u": _cube()}, attrs={"Conventions": "CF-1.8"})
    est = XarrayEstimator(StandardScaler(), sample_dims="time").fit(ds)
    assert est.transform(ds).attrs == {"Conventions": "CF-1.8"}
    pca = XarrayEstimator(PCA(2), sample_dims="time", feature_dim="component").fit(ds)
    assert pca.inverse_transform(pca.transform(ds)).attrs == {"Conventions": "CF-1.8"}


def test_scores_named_like_the_feature_dim_still_invert() -> None:
    da = _cube().isel(lat=0)  # (time, lon) with four lon
    est = XarrayEstimator(PCA(2), sample_dims="time", feature_dim="lon").fit(da)
    scores = est.transform(da)
    assert scores.sizes["lon"] == 2
    assert est.inverse_transform(scores).sizes["lon"] == 4


def test_calling_the_operator_on_a_tree_uses_the_fitted_tree() -> None:
    tree = xr.DataTree.from_dict({"a": xr.Dataset({"ssh": _cube()})})
    est = XarrayEstimator(StandardScaler(), sample_dims="time").fit(tree)
    xr.testing.assert_allclose(
        est(tree)["a"].to_dataset(), est.transform(tree)["a"].to_dataset()
    )
