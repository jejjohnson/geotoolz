"""Record the gap-fill and IDW outputs of xrtoolz ``b3855d7`` as fixtures.

The fills in :mod:`xrops.interpolate` moved onto functax plans (xrtoolz#395).
``gap_fill_b3855d7.npz`` holds the inputs of each case and the output the
NumPy/SciPy/scikit-learn/scikit-image implementations gave at ``b3855d7``;
``tests/interpolate/test_gap_fill_equivalence.py`` replays the inputs through
today's functions and compares. The script is kept for provenance: run it from
a checkout of ``b3855d7`` (with the ``image`` and ``sklearn`` extras) to
rebuild the file. Run on later code, it records that code instead.

    uv run python tests/interpolate/fixtures/make_gap_fill_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import cyclopts
import numpy as np
import xarray as xr
from loguru import logger

from xrops.interpolate import (
    Grid,
    fillnan_biharmonic,
    fillnan_idw,
    fillnan_laplacian,
    fillnan_rbf,
    idw_to_grid,
    idw_to_points,
)


HERE = Path(__file__).resolve().parent
app = cyclopts.App(help=__doc__)


def gap_cube() -> xr.DataArray:
    """A (time, lat, lon) cube of a smooth field with blocks, edge gaps and specks."""
    rng = np.random.default_rng(395)
    lat = np.linspace(-30.0, 30.0, 18)
    lon = np.linspace(0.0, 63.0, 22)
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")
    base = np.sin(np.deg2rad(2 * lon_grid)) * np.cos(np.deg2rad(3 * lat_grid))
    field = np.stack([base, 0.5 * base + 0.02 * lat_grid])
    field = field + 0.05 * rng.standard_normal(field.shape)
    field[0, 5:9, 7:12] = np.nan
    field[0, 0:3, 0:2] = np.nan
    field[0, rng.integers(0, 18, 6), rng.integers(0, 22, 6)] = np.nan
    field[1, 10:15, 15:20] = np.nan
    field[1, 6:9, 0:3] = np.nan
    field[1, 4:6, 19:22] = np.nan
    return xr.DataArray(
        field,
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": lat, "lon": lon},
    )


def scattered() -> dict[str, np.ndarray]:
    """Scattered sources and targets on lon/lat degrees, some targets on sources."""
    rng = np.random.default_rng(396)
    src_lon = rng.uniform(0.0, 10.0, 60)
    src_lat = rng.uniform(-5.0, 5.0, 60)
    src_val = np.sin(src_lon) + np.cos(src_lat) + 0.1 * rng.standard_normal(60)
    dst_lon = np.concatenate([rng.uniform(-1.0, 11.0, 37), src_lon[:3]])
    dst_lat = np.concatenate([rng.uniform(-6.0, 6.0, 37), src_lat[:3]])
    return {
        "src_lon": src_lon,
        "src_lat": src_lat,
        "src_val": src_val,
        "dst_lon": dst_lon,
        "dst_lat": dst_lat,
    }


GRID = Grid(lon=np.linspace(0.0, 10.0, 11), lat=np.linspace(-5.0, 5.0, 9))

#: name -> (function, keyword arguments) on the gap cube
CUBE_CASES = {
    "laplace_default": (fillnan_laplacian, {}),
    "laplace_tight": (
        fillnan_laplacian,
        {"max_iter": 20000, "tol": 1e-9, "relaxation": 1.5},
    ),
    "laplace_wrap": (
        fillnan_laplacian,
        {"max_iter": 20000, "tol": 1e-9, "relaxation": 1.5, "boundary": "wrap"},
    ),
    "biharmonic_split": (fillnan_biharmonic, {}),
    "biharmonic_coupled": (fillnan_biharmonic, {"split_into_regions": False}),
    "rbf_tps_global": (fillnan_rbf, {"neighbors": None}),
    "rbf_linear_global": (fillnan_rbf, {"kernel": "linear", "neighbors": None}),
    "rbf_cubic_global": (fillnan_rbf, {"kernel": "cubic", "neighbors": None}),
    "rbf_quintic_global": (fillnan_rbf, {"kernel": "quintic", "neighbors": None}),
    "rbf_tps_local": (fillnan_rbf, {"neighbors": 16}),
    "idw_euclidean": (fillnan_idw, {}),
    "idw_euclidean_cutoff": (
        fillnan_idw,
        {"k": 6, "power": 1.0, "max_distance": 2.5},
    ),
    "idw_haversine": (
        fillnan_idw,
        {"k": 6, "metric": "haversine", "max_distance": float(np.deg2rad(12.0))},
    ),
}

#: name -> keyword arguments of idw_to_points / idw_to_grid on the scattered set
POINT_CASES = {
    "points_euclidean": {},
    "points_cutoff": {"k": 5, "power": 1.0, "max_distance": 1.5},
    "points_exact": {"k": 4, "eps": 0.0},
    "points_haversine": {"k": 6, "metric": "haversine"},
}


@app.default
def main(out: Path = HERE / "gap_fill_b3855d7.npz") -> None:
    """Run every case on the current code and save inputs and outputs."""
    cube = gap_cube()
    arrays: dict[str, np.ndarray] = {
        "cube": cube.values,
        "cube_lat": cube.lat.values,
        "cube_lon": cube.lon.values,
        "grid_lon": GRID.lon,
        "grid_lat": GRID.lat,
    }
    for name, (function, kwargs) in CUBE_CASES.items():
        arrays[name] = function(cube, **kwargs).values
        logger.info("{}: {} NaN left", name, int(np.isnan(arrays[name]).sum()))
    points = scattered()
    arrays.update(points)
    for name, kwargs in POINT_CASES.items():
        arrays[name] = idw_to_points(
            points["src_lon"],
            points["src_lat"],
            points["src_val"],
            points["dst_lon"],
            points["dst_lat"],
            **kwargs,
        )
        logger.info("{}: {} NaN", name, int(np.isnan(arrays[name]).sum()))
    arrays["grid_euclidean"] = idw_to_grid(
        points["src_lon"], points["src_lat"], points["src_val"], GRID, k=6
    ).values
    np.savez_compressed(out, **arrays)
    logger.info("wrote {} arrays to {}", len(arrays), out)


if __name__ == "__main__":
    app()
