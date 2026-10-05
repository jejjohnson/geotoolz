"""DataTree support: per_node, pool_samples, concat_features, accessor, SklearnOp."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pipekit import Sequential
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from xrsklearn import SklearnOp, XarrayEstimator


def _grid(ny: int, nx: int, seed: int, *, with_time: bool = True) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    coords: dict[str, object] = {
        "lat": np.arange(ny, dtype=float),
        "lon": np.arange(nx, dtype=float),
    }
    if with_time:
        coords["time"] = np.arange(12)
    return xr.Dataset(
        {"ssh": (("time", "lat", "lon"), rng.normal(size=(12, ny, nx)))}, coords=coords
    )


@pytest.fixture
def multires() -> xr.DataTree:
    """Coarse + fine grids sharing a root ``time`` coord, plus a metadata node."""
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"time": np.arange(12)}),
            "coarse": _grid(3, 4, 0, with_time=False),
            "fine": _grid(6, 8, 1, with_time=False),
            "meta": xr.Dataset({"land": (("lat", "lon"), np.zeros((3, 4)))}),
        }
    )


@pytest.fixture
def ensemble() -> xr.DataTree:
    """Three members on the same grid."""
    return xr.DataTree.from_dict({f"member{i}": _grid(3, 4, i) for i in range(3)})


# ---------- per_node -------------------------------------------------------


def test_per_node_fits_one_estimator_per_node(multires: xr.DataTree) -> None:
    wrap = XarrayEstimator(
        PCA(n_components=2), sample_dim="time", tree_paths=["coarse", "fine"]
    )

    scores = wrap.fit_transform(multires)

    assert set(wrap.estimators_) == {"/coarse", "/fine"}
    assert wrap.estimators_["/coarse"].components_.shape == (2, 12)
    assert wrap.estimators_["/fine"].components_.shape == (2, 48)
    assert scores["coarse"]["transformed"].dims == ("time", "component")
    np.testing.assert_array_equal(scores["fine"]["time"], np.arange(12))


def test_per_node_default_selection_skips_nodes_without_sample_dim(
    multires: xr.DataTree,
) -> None:
    # "meta" has data but no time axis, so every node must be named or it fails.
    with pytest.raises(ValueError, match="variable 'land'"):
        XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(multires)


def test_per_node_inverse_transform_restores_each_grid(multires: xr.DataTree) -> None:
    wrap = XarrayEstimator(
        PCA(n_components=2), sample_dim="time", tree_paths=["coarse", "fine"]
    )
    scores = wrap.fit_transform(multires)

    recon = wrap.inverse_transform(scores)

    assert recon["coarse"]["ssh"].shape == (12, 3, 4)
    assert recon["fine"]["ssh"].shape == (12, 6, 8)


def test_per_node_attributes_are_per_node(multires: xr.DataTree) -> None:
    wrap = XarrayEstimator(
        PCA(n_components=2), sample_dim="time", tree_paths=["coarse", "fine"]
    ).fit(multires)

    with pytest.raises(AttributeError, match=r"estimators_\[path\]"):
        wrap.components_  # noqa: B018
    with pytest.raises(TypeError, match="fit per DataTree node"):
        wrap.transform(multires["coarse"].to_dataset(inherit=True))


def test_per_node_targets_and_scores(multires: xr.DataTree) -> None:
    paths = ["coarse", "fine"]
    y = {
        p: multires[p].to_dataset(inherit=True)["ssh"].mean(("lat", "lon"))
        for p in paths
    }
    wrap = XarrayEstimator(LinearRegression(), sample_dim="time", tree_paths=paths)

    wrap.fit(multires, y)
    scores = wrap.score(multires, y)
    pred = wrap.predict(multires)

    assert set(scores) == {"/coarse", "/fine"}
    assert all(s > 0.99 for s in scores.values())
    assert pred["coarse"]["ssh"].dims == ("time",)


def test_per_node_numpy_target_is_rejected(multires: xr.DataTree) -> None:
    wrap = XarrayEstimator(LinearRegression(), sample_dim="time", tree_paths=["coarse"])

    with pytest.raises(TypeError, match="mapping from node path"):
        wrap.fit(multires, np.zeros(12))


def test_per_node_learns_a_land_mask_per_node(multires: xr.DataTree) -> None:
    tree = multires.copy()
    coarse = tree["coarse"].to_dataset()
    coarse["ssh"][:, 0, 0] = np.nan
    tree["coarse"].dataset = coarse

    wrap = XarrayEstimator(
        PCA(n_components=2),
        sample_dim="time",
        nan_policy="mask",
        tree_paths=["coarse", "fine"],
    ).fit(tree)

    assert wrap.estimators_["/coarse"].feature_mask_.sum() == 11
    assert wrap.estimators_["/fine"].feature_mask_ is None


def test_transform_needs_the_fitted_nodes(multires: xr.DataTree) -> None:
    wrap = XarrayEstimator(
        PCA(n_components=2), sample_dim="time", tree_paths=["coarse", "fine"]
    ).fit(multires)
    pruned = xr.DataTree.from_dict(
        {"coarse": multires["coarse"].to_dataset(inherit=True)}
    )

    with pytest.raises(KeyError, match="/fine"):
        wrap.transform(pruned)


# ---------- pool_samples --------------------------------------------------


def test_pool_samples_fits_one_estimator_on_every_member(ensemble: xr.DataTree) -> None:
    wrap = XarrayEstimator(
        PCA(n_components=2), sample_dim="time", tree_mode="pool_samples"
    )

    scores = wrap.fit_transform(ensemble)

    pooled = np.concatenate(
        [ensemble[f"member{i}"]["ssh"].values.reshape(12, -1) for i in range(3)]
    )
    expected = PCA(n_components=2).fit(pooled)
    np.testing.assert_allclose(np.abs(wrap.components_), np.abs(expected.components_))
    assert scores["member1"]["transformed"].sizes == {"time": 12, "component": 2}
    xr.testing.assert_allclose(
        scores["member1"]["transformed"],
        wrap.transform(ensemble)["member1"]["transformed"],
    )


def test_pool_samples_rejects_mismatched_grids(multires: xr.DataTree) -> None:
    wrap = XarrayEstimator(
        PCA(n_components=2),
        sample_dim="time",
        tree_mode="pool_samples",
        tree_paths=["coarse", "fine"],
    )

    with pytest.raises(ValueError, match="node '/fine'"):
        wrap.fit(multires)


def test_pool_samples_pools_targets_and_score(ensemble: xr.DataTree) -> None:
    y = xr.DataTree.from_dict(
        {
            p: ensemble[p].to_dataset()["ssh"].mean(("lat", "lon")).to_dataset(name="y")
            for p in ("member0", "member1", "member2")
        }
    )
    wrap = XarrayEstimator(
        LinearRegression(), sample_dim="time", tree_mode="pool_samples"
    ).fit(ensemble, y)

    assert isinstance(wrap.score(ensemble, y), float)
    assert wrap.predict(ensemble)["member2"]["y"].dims == ("time",)


# ---------- concat_features -----------------------------------------------


def test_concat_features_joins_nodes_column_wise(multires: xr.DataTree) -> None:
    wrap = XarrayEstimator(
        PCA(n_components=3),
        sample_dim="time",
        tree_mode="concat_features",
        tree_paths=["coarse", "fine"],
    )

    scores = wrap.fit_transform(multires)
    recon = wrap.inverse_transform(scores)

    assert wrap.components_.shape == (3, 12 + 48)
    assert isinstance(scores, xr.DataArray)
    assert scores.dims == ("time", "component")
    assert isinstance(recon, xr.DataTree)
    assert recon["fine"]["ssh"].shape == (12, 6, 8)


def test_concat_features_one_to_one_round_trips(multires: xr.DataTree) -> None:
    wrap = XarrayEstimator(
        StandardScaler(),
        sample_dim="time",
        tree_mode="concat_features",
        tree_paths=["coarse", "fine"],
    )

    scaled = wrap.fit_transform(multires)
    recon = wrap.inverse_transform(scaled)

    assert isinstance(scaled, xr.DataTree)
    np.testing.assert_allclose(scaled["fine"]["ssh"].mean("time"), 0.0, atol=1e-12)
    xr.testing.assert_allclose(
        recon["coarse"]["ssh"], multires["coarse"].to_dataset(inherit=True)["ssh"]
    )


def test_concat_features_supervised(multires: xr.DataTree) -> None:
    y = multires["fine"].to_dataset(inherit=True)["ssh"].mean(("lat", "lon"))
    wrap = XarrayEstimator(
        LinearRegression(),
        sample_dim="time",
        tree_mode="concat_features",
        tree_paths=["coarse", "fine"],
    ).fit(multires, y)

    assert wrap.predict(multires).dims == ("time",)


def test_concat_features_requires_a_shared_sample_axis() -> None:
    tree = xr.DataTree.from_dict(
        {"a": _grid(3, 4, 0), "b": _grid(3, 4, 1).isel(time=slice(0, 6))}
    )

    with pytest.raises(ValueError, match="share the sample axis"):
        XarrayEstimator(PCA(n_components=2), tree_mode="concat_features").fit(tree)


def test_unknown_tree_mode_raises(ensemble: xr.DataTree) -> None:
    with pytest.raises(ValueError, match="tree_mode must be one of"):
        XarrayEstimator(PCA(n_components=2), tree_mode="stack").fit(ensemble)  # type: ignore[arg-type]


# ---------- accessor ------------------------------------------------------


def test_datatree_accessor(ensemble: xr.DataTree) -> None:
    wrap = ensemble.sklearn.fit(
        PCA(n_components=2), sample_dim="time", tree_mode="pool_samples"
    )

    scores = ensemble.sklearn.transform(wrap)

    assert wrap.components_.shape == (2, 12)
    assert set(scores.children) == {"member0", "member1", "member2"}


# ---------- SklearnOp -----------------------------------------------------


def test_sklearn_op_fit_transform_per_node_writes_into_each_node(
    multires: xr.DataTree,
) -> None:
    op = SklearnOp(
        PCA(n_components=2),
        variable="ssh",
        output_variable="pcs",
        sample_dim="time",
        method="fit_transform",
    )

    out = op(multires)

    assert out["coarse"]["pcs"].dims == ("time", "component")
    assert out["fine"]["pcs"].dims == ("time", "component")
    xr.testing.assert_identical(out["meta"].to_dataset(), multires["meta"].to_dataset())
    assert "ssh" in out["fine"].data_vars  # input kept


def test_sklearn_op_with_pooled_wrapper(ensemble: xr.DataTree) -> None:
    fitted = XarrayEstimator(
        StandardScaler(), sample_dim="time", tree_mode="pool_samples"
    ).fit(ensemble)

    out = SklearnOp(fitted, variable="ssh")(ensemble)

    assert out["member0"]["ssh"].dims == ("time", "lat", "lon")
    assert not np.allclose(out["member0"]["ssh"], ensemble["member0"]["ssh"])


def test_sklearn_op_raw_estimator_skips_nodes_without_variable(
    multires: xr.DataTree,
) -> None:
    fitted = XarrayEstimator(StandardScaler(), sample_dim="time").fit(
        multires["coarse"].to_dataset(inherit=True)["ssh"]
    )
    tree = multires.copy()
    del tree["fine"]

    out = SklearnOp(fitted, variable="ssh", output_variable="z")(tree)

    assert "z" in out["coarse"].data_vars
    assert "z" not in out["meta"].data_vars


def test_sklearn_op_concat_features_reduced_result_goes_to_root(
    multires: xr.DataTree,
) -> None:
    fitted = XarrayEstimator(
        PCA(n_components=2),
        sample_dim="time",
        tree_mode="concat_features",
        tree_paths=["coarse", "fine"],
    ).fit(multires.filter(lambda n: n.path != "/meta"))

    out = SklearnOp(fitted, variable="ssh", output_variable="pcs")(multires)

    assert out["pcs"].dims == ("time", "component")


def test_sklearn_op_in_sequential_on_datatree(ensemble: xr.DataTree) -> None:
    fitted = XarrayEstimator(
        PCA(n_components=2), sample_dim="time", tree_mode="pool_samples"
    ).fit(ensemble)
    pipeline = Sequential(
        [
            SklearnOp(
                StandardScaler(),
                variable="ssh",
                sample_dim="time",
                method="fit_transform",
            ),
            SklearnOp(fitted, variable="ssh", output_variable="pcs"),
        ]
    )

    out = pipeline(ensemble)

    assert all(out[f"member{i}"]["pcs"].sizes["component"] == 2 for i in range(3))


def test_sklearn_op_tree_without_variable_or_output_raises(
    ensemble: xr.DataTree,
) -> None:
    with pytest.raises(ValueError, match="neither `variable` nor `output_variable`"):
        SklearnOp(StandardScaler(), method="fit_transform")(ensemble)


# ---------- review follow-ups (#333) --------------------------------------


def test_unselected_nodes_are_kept_in_the_output(multires: xr.DataTree) -> None:
    wrap = XarrayEstimator(
        StandardScaler(), sample_dim="time", tree_paths=["coarse", "fine"]
    )

    out = wrap.fit_transform(multires)
    again = wrap.transform(multires)

    for tree in (out, again):
        assert {n.path for n in tree.subtree} == {n.path for n in multires.subtree}
        xr.testing.assert_identical(
            tree["meta"].to_dataset(), multires["meta"].to_dataset()
        )


def test_pooled_numpy_targets_in_a_mapping(ensemble: xr.DataTree) -> None:
    rng = np.random.default_rng(0)
    y = {f"member{i}": rng.normal(size=12) for i in range(3)}
    wrap = XarrayEstimator(
        LinearRegression(), sample_dim="time", tree_mode="pool_samples"
    )

    wrap.fit(ensemble, y)

    assert wrap.coef_.shape == (12,)


def test_pooled_targets_are_conformed_not_overridden(ensemble: xr.DataTree) -> None:
    rng = np.random.default_rng(0)

    def target(order: list[str]) -> xr.DataArray:
        return xr.DataArray(
            rng.normal(size=(12, 2)),
            dims=("time", "out"),
            coords={"time": np.arange(12), "out": order},
        )

    wrap = XarrayEstimator(
        LinearRegression(), sample_dim="time", tree_mode="pool_samples"
    )
    permuted = {
        "member0": target(["a", "b"]),
        "member1": target(["b", "a"]),
        "member2": target(["a", "b"]),
    }
    wrap.fit(ensemble, permuted)  # same labels, other order: reordered

    bad = {**permuted, "member2": target(["a", "c"])}
    with pytest.raises(ValueError, match="member2"):
        wrap.fit(ensemble, bad)


def test_pooled_fit_transform_keeps_each_nodes_metadata(ensemble: xr.DataTree) -> None:
    tree = ensemble.copy()
    for i in range(3):
        node = tree[f"member{i}"]
        node.dataset = node.to_dataset().assign_coords(member=i).assign_attrs(run=i)
    wrap = XarrayEstimator(
        StandardScaler(), sample_dim="time", tree_mode="pool_samples"
    )

    out = wrap.fit_transform(tree)

    for i in range(3):
        node = out[f"member{i}"]
        assert int(node["member"]) == i
        assert node.attrs == {"run": i}


def test_accessor_forwards_tree_paths(multires: xr.DataTree) -> None:
    scores = multires.sklearn.fit_transform(
        PCA(n_components=2), sample_dim="time", tree_paths=["coarse"]
    )

    assert scores["coarse"]["transformed"].dims == ("time", "component")
    fitted = multires.sklearn.fit(
        StandardScaler(), sample_dim="time", tree_paths=["fine"]
    )
    assert set(fitted.estimators_) == {"/fine"}
