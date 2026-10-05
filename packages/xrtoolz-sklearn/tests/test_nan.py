"""NaN policies: feature masks (land), sample masks (gaps), y / fit-param masking."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from xrsklearn import SklearnOp, XarrayEstimator
from xrsklearn._src.nan import (
    fill_dtype,
    learn_feature_mask,
    mask_fit_params,
    missing_mask,
    restore_rows,
)


def _ocean(seed: int = 0) -> xr.DataArray:
    """(time, lat, lon) with a fixed land cell at (0, 0) and (2, 3)."""
    rng = np.random.default_rng(seed)
    da = xr.DataArray(
        rng.normal(size=(30, 3, 4)),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(30),
            "lat": [10.0, 0.0, -10.0],
            "lon": np.arange(4.0),
        },
        name="ssh",
    )
    da[:, 0, 0] = np.nan
    da[:, 2, 3] = np.nan
    return da


LAND = np.zeros((3, 4), dtype=bool)
LAND[0, 0] = LAND[2, 3] = True


# ---------- feature masks (land) -----------------------------------------


def test_mask_samples_alone_cannot_handle_land() -> None:
    wrap = XarrayEstimator(
        PCA(n_components=2), sample_dim="time", nan_policy="mask_samples"
    )

    with pytest.raises(ValueError, match="removed all sample rows"):
        wrap.fit(_ocean())


@pytest.mark.parametrize("policy", ["mask", "mask_features"])
def test_land_mask_pca_round_trip(policy: str) -> None:
    da = _ocean()
    wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time", nan_policy=policy)  # type: ignore[arg-type]

    scores = wrap.fit_transform(da)
    recon = wrap.inverse_transform(scores)

    assert wrap.feature_mask_.sum() == 10
    assert wrap.components_.shape == (2, 10)
    assert bool(np.isfinite(scores).all())
    assert recon.dims == da.dims
    assert bool(recon.isnull().all("time").values[LAND].all())
    assert bool(recon.notnull().all("time").values[~LAND].all())


def test_land_mask_scaler_returns_grid_with_land_restored() -> None:
    da = _ocean()

    out = XarrayEstimator(
        StandardScaler(), sample_dim="time", nan_policy="mask"
    ).fit_transform(da)

    xr.testing.assert_equal(out.isnull(), da.isnull())
    np.testing.assert_allclose(out.mean("time").values[~LAND], 0.0, atol=1e-12)


def test_fitted_feature_mask_is_reused_at_transform() -> None:
    da = _ocean()
    wrap = XarrayEstimator(
        PCA(n_components=2), sample_dim="time", nan_policy="mask"
    ).fit(da)

    # Values over land at transform time are ignored, not fed to the estimator.
    filled = da.fillna(99.0)

    xr.testing.assert_allclose(wrap.transform(filled), wrap.transform(da))


def test_land_plus_gaps_drops_rows_and_columns() -> None:
    da = _ocean()
    da[5, 1, 1] = np.nan  # a gap in an ocean cell

    out = XarrayEstimator(
        StandardScaler(), sample_dim="time", nan_policy="mask"
    ).fit_transform(da)

    assert bool(out.isel(time=5).isnull().all())
    assert bool(out.drop_sel(time=5).notnull().values[:, ~LAND].all())


def test_scaler_inverse_transform_on_land_grid() -> None:
    da = _ocean()
    wrap = XarrayEstimator(StandardScaler(), sample_dim="time", nan_policy="mask")

    recon = wrap.inverse_transform(wrap.fit_transform(da))

    xr.testing.assert_allclose(recon, da)


def test_land_mask_on_dataset_spans_variables() -> None:
    ds = xr.Dataset({"ssh": _ocean(0), "sst": _ocean(1).fillna(0.0)})

    out = XarrayEstimator(
        StandardScaler(), sample_dim="time", nan_policy="mask"
    ).fit_transform(ds)

    xr.testing.assert_equal(out["ssh"].isnull(), ds["ssh"].isnull())
    assert bool(out["sst"].notnull().all())


def test_sklearn_op_handles_land_masked_field() -> None:
    ds = xr.Dataset({"ssh": _ocean()})

    out = SklearnOp(
        PCA(n_components=2),
        variable="ssh",
        output_variable="pcs",
        sample_dim="time",
        method="fit_transform",
        nan_policy="mask",
    )(ds)

    assert out["pcs"].dims == ("time", "component")
    assert bool(out["pcs"].notnull().all())


# ---------- sample masks: y, fit params, inf -----------------------------


def _regression() -> tuple[xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(0)
    x = xr.DataArray(
        rng.normal(size=(40, 3)), dims=("time", "f"), coords={"time": np.arange(40)}
    )
    y = (x * [1.0, 2.0, 3.0]).sum("f") + 0.01 * rng.normal(size=40)
    return x, y


def test_nan_in_y_drops_the_row() -> None:
    x, y = _regression()
    y_gappy = y.copy()
    y_gappy[[3, 7]] = np.nan

    wrap = XarrayEstimator(LinearRegression(), sample_dim="time", nan_policy="mask")
    wrap.fit(x, y_gappy)

    keep = np.isfinite(y_gappy.values)
    expected = LinearRegression().fit(x.values[keep], y.values[keep])
    np.testing.assert_allclose(wrap.coef_, expected.coef_)


def test_sample_weight_is_masked_with_the_rows() -> None:
    x, y = _regression()
    x_gappy = x.copy()
    x_gappy[4, 0] = np.nan
    weights = np.linspace(1.0, 2.0, 40)

    wrap = XarrayEstimator(LinearRegression(), sample_dim="time", nan_policy="mask")
    wrap.fit(x_gappy, y, sample_weight=weights)

    keep = np.ones(40, dtype=bool)
    keep[4] = False
    expected = LinearRegression().fit(
        x.values[keep], y.values[keep], sample_weight=weights[keep]
    )
    np.testing.assert_allclose(wrap.coef_, expected.coef_)


def test_xarray_sample_weight_is_aligned_to_x() -> None:
    x, y = _regression()
    weights = xr.DataArray(
        np.linspace(1.0, 2.0, 40), dims="time", coords={"time": x.time}
    )

    a = XarrayEstimator(LinearRegression(), sample_dim="time").fit(
        x, y, sample_weight=weights
    )
    b = XarrayEstimator(LinearRegression(), sample_dim="time").fit(
        x, y, sample_weight=weights.isel(time=slice(None, None, -1))
    )

    np.testing.assert_allclose(a.coef_, b.coef_)


def test_inf_is_only_missing_with_nonfinite() -> None:
    x, y = _regression()
    x_inf = x.copy()
    x_inf[2, 1] = np.inf

    with pytest.raises(ValueError, match="infinity"):
        XarrayEstimator(LinearRegression(), sample_dim="time", nan_policy="mask").fit(
            x_inf, y
        )

    wrap = XarrayEstimator(
        LinearRegression(), sample_dim="time", nan_policy="mask", missing="nonfinite"
    ).fit(x_inf, y)
    assert bool(np.isnan(wrap.predict(x_inf)[2]))


# ---------- raise ---------------------------------------------------------


def test_raise_counts_missing_values_in_x() -> None:
    x, y = _regression()
    x[[1, 2], 0] = np.nan

    with pytest.raises(ValueError, match="X contains 2 missing"):
        XarrayEstimator(LinearRegression(), sample_dim="time", nan_policy="raise").fit(
            x, y
        )


def test_raise_checks_y_too() -> None:
    x, y = _regression()
    y[0] = np.nan

    with pytest.raises(ValueError, match="y contains 1 missing"):
        XarrayEstimator(LinearRegression(), sample_dim="time", nan_policy="raise").fit(
            x, y
        )


def test_raise_accepts_object_dtype() -> None:
    x, _ = _regression()

    XarrayEstimator(StandardScaler(), sample_dim="time", nan_policy="raise").fit(
        x.astype(object)
    )


def test_raise_with_nonfinite_flags_inf() -> None:
    x, y = _regression()
    x[0, 0] = np.inf

    with pytest.raises(ValueError, match="NaN/inf"):
        XarrayEstimator(
            LinearRegression(),
            sample_dim="time",
            nan_policy="raise",
            missing="nonfinite",
        ).fit(x, y)


# ---------- fill values ---------------------------------------------------


def test_string_labels_are_restored_as_missing_not_the_string_nan() -> None:
    x, y = _regression()
    labels = xr.where(y > 0, "warm", "cold")
    x_gappy = x.copy()
    x_gappy[3, 0] = np.nan
    wrap = XarrayEstimator(
        RandomForestClassifier(n_estimators=5, random_state=0),
        sample_dim="time",
        nan_policy="mask",
    ).fit(x, labels)

    pred = wrap.predict(x_gappy)

    assert pred.dtype == object
    assert pred.isnull().values.tolist() == [i == 3 for i in range(40)]
    assert "nan" not in pred.values.tolist()


def test_integer_labels_are_promoted_to_float() -> None:
    x, _ = _regression()
    x_gappy = x.copy()
    x_gappy[0, 0] = np.nan
    wrap = XarrayEstimator(
        KMeans(n_clusters=2, n_init=1, random_state=0),
        sample_dim="time",
        nan_policy="mask",
    ).fit(x_gappy)

    labels = wrap.predict(x_gappy)

    assert labels.dtype == np.float64
    assert bool(np.isnan(labels[0]))


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float32, np.float32),
        (np.int64, np.float64),
        (np.bool_, np.float64),
        ("datetime64[ns]", "datetime64[ns]"),
        ("<U4", object),
    ],
)
def test_fill_dtype(dtype: object, expected: object) -> None:
    assert fill_dtype(np.dtype(dtype))[0] == np.dtype(expected)


def test_restore_rows_uses_nat_for_datetimes() -> None:
    out = restore_rows(
        np.array(["2020-01-01"], dtype="datetime64[ns]"), np.array([False, True])
    )

    assert np.isnat(out[0])
    assert out[1] == np.datetime64("2020-01-01")


# ---------- NumPy inputs follow the policy too ---------------------------


def test_numpy_input_respects_policy() -> None:
    arr = np.random.default_rng(0).normal(size=(10, 3))
    arr[2, 1] = np.nan

    with pytest.raises(ValueError, match="missing"):
        XarrayEstimator(StandardScaler(), nan_policy="raise").fit(arr)
    out = XarrayEstimator(StandardScaler(), nan_policy="mask").fit_transform(arr)
    assert isinstance(out, np.ndarray)
    assert bool(np.isnan(out[2]).all())
    assert bool(np.isfinite(np.delete(out, 2, axis=0)).all())


def test_unknown_missing_kind_raises() -> None:
    with pytest.raises(ValueError, match="missing must be one of"):
        XarrayEstimator(StandardScaler(), missing="inf").fit(np.ones((3, 2)))  # type: ignore[arg-type]


# ---------- helpers -------------------------------------------------------


def test_missing_mask_by_dtype() -> None:
    assert missing_mask(np.array([1.0, np.nan])).tolist() == [False, True]
    assert missing_mask(np.array([1, 2])).tolist() == [False, False]
    assert missing_mask(np.array(["a", "b"])).tolist() == [False, False]
    assert missing_mask(np.array([1.0, None, np.nan], dtype=object)).tolist() == [
        False,
        True,
        True,
    ]
    assert missing_mask(
        np.array([np.inf, 1.0], dtype=object), "nonfinite"
    ).tolist() == [
        True,
        False,
    ]
    assert missing_mask(
        np.array(["NaT", "2020-01-01"], dtype="datetime64[D]")
    ).tolist() == [
        True,
        False,
    ]


def test_learn_feature_mask_returns_none_when_nothing_is_empty() -> None:
    assert learn_feature_mask(np.array([[1.0, np.nan], [np.nan, 2.0]])) is None


def test_mask_fit_params_only_touches_per_sample_arrays() -> None:
    keep = np.array([True, False, True])

    out = mask_fit_params(
        {"sample_weight": [1, 2, 3], "tag": "abc", "alpha": 0.5, "classes": [0, 1]},
        keep,
    )

    assert list(out["sample_weight"]) == [1, 3]
    assert out["tag"] == "abc"
    assert out["alpha"] == 0.5
    assert out["classes"] == [0, 1]


# ---------- review follow-ups (#332) --------------------------------------


def test_feature_mask_ignores_rows_dropped_for_missing_y() -> None:
    x = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 7.0]])
    y = np.array([0.0, 1.0, np.nan])
    wrap = XarrayEstimator(LinearRegression(), nan_policy="mask").fit(x, y)

    np.testing.assert_array_equal(wrap.feature_mask_, [True, False])
    assert wrap.coef_.shape == (1,)


@pytest.mark.parametrize("policy", ["mask", "mask_features"])
def test_raw_estimator_rejects_feature_masking(policy: str) -> None:
    da = _ocean()
    fitted = StandardScaler().fit(
        np.nan_to_num(da.values.reshape(da.sizes["time"], -1))
    )

    with pytest.raises(ValueError, match="feature mask learned at fit time"):
        da.sklearn.transform(fitted, sample_dim="time", nan_policy=policy)


@pytest.mark.parametrize("policy", ["mask", "mask_features"])
def test_numpy_inverse_transform_reapplies_the_feature_mask(policy: str) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3))
    x[:, 1] = np.nan
    wrap = XarrayEstimator(StandardScaler(), nan_policy=policy)

    scaled = wrap.fit_transform(x)
    assert scaled.shape == (10, 3)
    recon = wrap.inverse_transform(scaled)

    np.testing.assert_allclose(recon[:, [0, 2]], x[:, [0, 2]])
    assert np.isnan(recon[:, 1]).all()


def test_mask_fit_params_subsets_sparse_matrices() -> None:
    sparse = pytest.importorskip("scipy.sparse")
    keep = np.array([True, False, True])
    weights = sparse.csr_matrix(np.arange(6.0).reshape(3, 2))

    out = mask_fit_params({"groups": weights}, keep)

    assert sparse.issparse(out["groups"])
    np.testing.assert_array_equal(out["groups"].toarray(), [[0, 1], [4, 5]])
