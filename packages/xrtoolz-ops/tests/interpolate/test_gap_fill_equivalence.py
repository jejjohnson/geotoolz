"""The gap fills and IDW on functax equal their ``b3855d7`` outputs (xrtoolz#395).

``fixtures/gap_fill_b3855d7.npz`` holds the inputs and the outputs of the
NumPy/SciPy/scikit-learn/scikit-image implementations at ``b3855d7``
(``fixtures/make_gap_fill_fixtures.py``). Each case replays the inputs and
compares at the tolerance the issue sets: 1e-10 for IDW and RBF, ``10 · tol``
for the Laplace fill (SOR stopped at ``tol``; functax solves to √ε).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.spatial.distance import cdist

from xrops.interpolate import (
    Grid,
    fillnan_biharmonic,
    fillnan_idw,
    fillnan_laplacian,
    fillnan_rbf,
    idw_to_grid,
    idw_to_points,
)


pytest.importorskip("functax.project")

FIXTURES = np.load(Path(__file__).parent / "fixtures" / "gap_fill_b3855d7.npz")


def _cube() -> xr.DataArray:
    return xr.DataArray(
        FIXTURES["cube"],
        dims=("time", "lat", "lon"),
        coords={
            "time": [0, 1],
            "lat": FIXTURES["cube_lat"],
            "lon": FIXTURES["cube_lon"],
        },
    )


def _assert_close(actual: np.ndarray, expected: np.ndarray, atol: float) -> None:
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)


#: case -> (function, keyword arguments, absolute tolerance)
CUBE_CASES = {
    # SOR stopped at tol = 1e-4; measured 3.5e-4
    "laplace_default": (fillnan_laplacian, {}, 1e-3),
    # tol = 1e-9; measured 2.0e-9
    "laplace_tight": (
        fillnan_laplacian,
        {"max_iter": 20000, "tol": 1e-9, "relaxation": 1.5},
        1e-8,
    ),
    # still SOR on NumPy (functax has no periodic fill), so exact
    "laplace_wrap": (
        fillnan_laplacian,
        {"max_iter": 20000, "tol": 1e-9, "relaxation": 1.5, "boundary": "wrap"},
        0.0,
    ),
    # scikit-image's solve of the same system; measured 2.1e-10
    "biharmonic_split": (fillnan_biharmonic, {}, 1e-9),
    "biharmonic_coupled": (fillnan_biharmonic, {"split_into_regions": False}, 1e-9),
    "rbf_tps_global": (fillnan_rbf, {"neighbors": None}, 1e-10),
    "rbf_linear_global": (fillnan_rbf, {"kernel": "linear", "neighbors": None}, 1e-10),
    "rbf_cubic_global": (fillnan_rbf, {"kernel": "cubic", "neighbors": None}, 1e-10),
    # the quintic system is ill-conditioned: measured 3.8e-9 against SciPy's LU
    "rbf_quintic_global": (
        fillnan_rbf,
        {"kernel": "quintic", "neighbors": None},
        1e-8,
    ),
    # still SciPy (functax has no local RBF), so exact
    "rbf_tps_local": (fillnan_rbf, {"neighbors": 16}, 0.0),
    # still scikit-learn (functax weighs by chord, not great-circle angle)
    "idw_haversine": (
        fillnan_idw,
        {"k": 6, "metric": "haversine", "max_distance": float(np.deg2rad(12.0))},
        0.0,
    ),
}


@pytest.mark.parametrize("case", sorted(CUBE_CASES))
def test_cube_fill_matches_b3855d7(case: str) -> None:
    function, kwargs, atol = CUBE_CASES[case]

    out = function(_cube(), **kwargs)

    _assert_close(out.values, FIXTURES[case], atol)


@pytest.mark.parametrize(
    ("case", "k", "kwargs"),
    [
        ("idw_euclidean", 8, {}),
        ("idw_euclidean_cutoff", 6, {"k": 6, "power": 1.0, "max_distance": 2.5}),
    ],
)
def test_fillnan_idw_euclidean_matches_b3855d7_off_ties(
    case: str, k: int, kwargs: dict
) -> None:
    # On the integer index grid many holes have a tie at the k-th neighbour
    # distance; scikit-learn's KDTree (b3855d7) and SciPy's cKDTree (inside
    # functax) break it differently. Off the ties the weights must agree.
    cube = _cube()
    out = fillnan_idw(cube, **kwargs).values
    rows, cols = np.indices(cube.shape[1:])
    compared = 0
    for t, field in enumerate(cube.values):
        finite = np.isfinite(field)
        sources = np.column_stack([cols[finite], rows[finite]])
        holes = np.column_stack([cols[~finite], rows[~finite]])
        distances = np.sort(cdist(holes, sources), axis=1)
        # a tie beyond the cutoff picks no neighbour, so it does not count
        cutoff = kwargs.get("max_distance", np.inf)
        tied = np.isclose(distances[:, k - 1], distances[:, k]) & (
            distances[:, k - 1] <= cutoff
        )
        filled, expected = out[t][~finite], FIXTURES[case][t][~finite]
        _assert_close(filled[~tied], expected[~tied], 1e-10)
        # a tied hole has neighbours, so it is filled whichever tree breaks it
        assert np.isfinite(filled[tied]).all()
        compared += int((~tied).sum())
    np.testing.assert_array_equal(
        out[np.isfinite(cube.values)], cube.values[np.isfinite(cube.values)]
    )
    assert compared > 0


@pytest.mark.parametrize(
    ("case", "kwargs", "atol"),
    [
        ("points_euclidean", {}, 1e-10),
        ("points_cutoff", {"k": 5, "power": 1.0, "max_distance": 1.5}, 1e-10),
        ("points_exact", {"k": 4, "eps": 0.0}, 1e-10),
        # still scikit-learn, so exact
        ("points_haversine", {"k": 6, "metric": "haversine"}, 0.0),
    ],
)
def test_idw_to_points_matches_b3855d7(case: str, kwargs: dict, atol: float) -> None:
    out = idw_to_points(
        FIXTURES["src_lon"],
        FIXTURES["src_lat"],
        FIXTURES["src_val"],
        FIXTURES["dst_lon"],
        FIXTURES["dst_lat"],
        **kwargs,
    )

    _assert_close(out, FIXTURES[case], atol)


def test_idw_to_grid_matches_b3855d7() -> None:
    grid = Grid(lon=FIXTURES["grid_lon"], lat=FIXTURES["grid_lat"])

    out = idw_to_grid(
        FIXTURES["src_lon"], FIXTURES["src_lat"], FIXTURES["src_val"], grid, k=6
    )

    _assert_close(out.values, FIXTURES["grid_euclidean"], 1e-10)


@pytest.mark.dask
@pytest.mark.parametrize(
    "case", ["laplace_default", "biharmonic_split", "rbf_tps_global"]
)
def test_dask_cube_fills_slice_by_slice(case: str) -> None:
    pytest.importorskip("dask.array")
    function, kwargs, atol = CUBE_CASES[case]

    out = function(_cube().chunk({"time": 1}), **kwargs)

    assert hasattr(out.data, "chunks")
    _assert_close(out.compute().values, FIXTURES[case], atol)


@pytest.mark.parametrize("module", ["functax.project", "functax.interp"])
def test_a_missing_functax_names_the_extra(monkeypatch, module) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=r"xrtoolz-ops\[functax\]"):
        if module == "functax.project":
            fillnan_laplacian(_cube())
        else:
            fillnan_idw(_cube())


def test_biharmonic_without_a_finite_pixel_outside_the_mask_raises() -> None:
    # b3855d7 raised scikit-image's ValueError here; functax has nothing to
    # anchor the solve on either
    da = xr.DataArray(np.full((4, 4), np.nan), dims=("lat", "lon"))
    mask = xr.zeros_like(da, dtype=bool)
    mask.values[1, 1] = True

    with pytest.raises(ValueError, match="finite pixel outside"):
        fillnan_biharmonic(da, mask=mask)
