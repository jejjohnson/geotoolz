"""Repo-wide numpy/dask parity sweep.

Each :class:`Case` runs one operator against an eager numpy input and the
same input chunked with dask, then asserts the two results match. Where an
operator is expected to *stay lazy* (its kernel is naturally per-chunk),
``lazy=True`` additionally asserts the dask output is not eagerly computed.

Operators that cannot yet accept dask input are registered with an ``xfail``
mark so the matrix stays complete; the corresponding remediation flips them
to passing.

See ``docs/`` (the Dask page) for the per-operator tier table.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.decomposition import PCA

import xrtoolz.calc as calc
from xrtoolz.geo.operators import (
    AddLandMask,
    AddOceanMask,
    CalculateClimatology,
    RemoveMean,
    ValidateLongitude,
)
from xrtoolz.interpolate import Grid
from xrtoolz.interpolate.operators import (
    Bin2D,
    Coarsen,
    FillNaNBiharmonic,
    FillNaNIDW,
    FillNaNLaplacian,
    FillNaNRBF,
    FillNaNSpatial,
    GaussianSmooth,
    Histogram2D,
    MovingAverage,
    RegridLike,
)
from xrtoolz.metrics.operators import MAE, RMSE, Bias, Correlation
from xrtoolz.ocn.operators import (
    Divergence,
    GeostrophicVelocities,
    KineticEnergy,
    OkuboWeiss,
    RelativeVorticity,
    Streamfunction,
    VelocityMagnitude,
)
from xrtoolz.transforms.operators import PowerSpectrum
from xrtoolz.utils import XarrayEstimator


pytestmark = pytest.mark.dask
pytest.importorskip("dask.array")

# Chunk the leading (time) axis and keep each 2-D spatial slice whole — the
# realistic pattern for per-slice kernels.
CHUNKS = {"time": 1, "lat": -1, "lon": -1}


def _grid(*, gap: bool = False) -> xr.Dataset:
    """Deterministic ``(time, lat, lon)`` dataset with standard var names."""
    rng = np.random.default_rng(0)
    t, na, no = 6, 24, 32
    lat = np.linspace(-10.0, 10.0, na)
    lon = np.linspace(-50.0, -30.0, no)
    time = pd.date_range("2020-01-01", periods=t, freq="D")

    def field(scale: float = 1.0, offset: float = 0.0) -> xr.DataArray:
        data = rng.standard_normal((t, na, no)) * scale + offset
        return xr.DataArray(
            data,
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
        )

    ds = xr.Dataset(
        {
            "ssh": field(0.1),
            "u": field(),
            "v": field(),
            "sst": field(5.0, 15.0),
        }
    )
    if gap:
        holed = ds["ssh"].copy()
        holed.values[:, 12, 16] = np.nan
        ds["ssh"] = holed
    return ds


def _grid_gap() -> xr.Dataset:
    return _grid(gap=True)


def _scattered() -> xr.DataArray:
    """1-D scattered observations (an ``obs`` dim with lon/lat coords)."""
    rng = np.random.default_rng(1)
    n = 200
    lon = rng.uniform(-10.0, 10.0, n)
    lat = rng.uniform(-5.0, 5.0, n)
    return xr.DataArray(
        rng.standard_normal(n),
        dims=("obs",),
        coords={"lon": ("obs", lon), "lat": ("obs", lat)},
        name="obs",
    )


# A coarse target grid for regridding / binning cases.
_TARGET = xr.Dataset(
    coords={"lat": np.linspace(-9.0, 9.0, 9), "lon": np.linspace(-48.0, -32.0, 11)}
)
_BIN_GRID = Grid.from_bounds((-10.0, 10.0), (-5.0, 5.0), resolution=2.0)


@dataclass(frozen=True)
class Case:
    """A single operator invocation to check for numpy/dask parity."""

    id: str
    run: Callable[..., xr.DataArray | xr.Dataset]
    data: Callable[[], xr.DataArray | xr.Dataset] = _grid
    chunks: Mapping[str, int] = field(default_factory=lambda: dict(CHUNKS))
    needs_ref: bool = False
    lazy: bool = False
    xfail: str | None = None


CASES: list[Case] = [
    # -- calc: works under dask, materialises (finitediffx kernels) ----------
    Case("calc.gradient", lambda d: calc.gradient(d["ssh"], dims=("lat", "lon"))),
    Case("calc.laplacian", lambda d: calc.laplacian(d["ssh"], dims=("lat", "lon"))),
    # -- geo: lazy --------------------------------------------------------------
    Case("geo.RemoveMean", RemoveMean("time"), lazy=True),
    Case("geo.ValidateLongitude", ValidateLongitude(), lazy=True),
    Case("geo.CalculateClimatology", CalculateClimatology(), lazy=True),
    # -- ocn: a mix of lazy and (correct) eager --------------------------------
    Case("ocn.KineticEnergy", KineticEnergy(), lazy=True),
    Case("ocn.VelocityMagnitude", VelocityMagnitude(), lazy=True),
    Case("ocn.Streamfunction", Streamfunction(), lazy=True),
    Case("ocn.RelativeVorticity", RelativeVorticity()),
    Case("ocn.Divergence", Divergence()),
    Case("ocn.GeostrophicVelocities", GeostrophicVelocities()),
    Case("ocn.OkuboWeiss", OkuboWeiss()),
    # -- transforms: lazy -------------------------------------------------------
    Case("transforms.PowerSpectrum", PowerSpectrum("ssh", ("lat", "lon")), lazy=True),
    # -- interpolate: smoothing works (eager); coarsen lazy --------------------
    Case("interpolate.GaussianSmooth", GaussianSmooth(dim="lat", sigma=1.0), lazy=True),
    Case("interpolate.MovingAverage", MovingAverage(dim="lon", window=3), lazy=True),
    Case("interpolate.Coarsen", Coarsen({"lat": 2, "lon": 2}), lazy=True),
    # -- metrics: lazy ----------------------------------------------------------
    Case("metrics.RMSE", RMSE("ssh", ("lat", "lon")), needs_ref=True, lazy=True),
    Case("metrics.MAE", MAE("ssh", ("lat", "lon")), needs_ref=True, lazy=True),
    Case("metrics.Bias", Bias("ssh", ("lat", "lon")), needs_ref=True, lazy=True),
    Case(
        "metrics.Correlation",
        Correlation("ssh", ("lat", "lon")),
        needs_ref=True,
        lazy=True,
    ),
    # -- gap-fill: per-slice apply_ufunc, lazy via dask="parallelized" ----------
    Case("interpolate.FillNaNLaplacian", FillNaNLaplacian(), data=_grid_gap, lazy=True),
    Case(
        "interpolate.FillNaNSpatial",
        FillNaNSpatial(method="linear"),
        data=_grid_gap,
        lazy=True,
    ),
    Case("interpolate.FillNaNRBF", FillNaNRBF(), data=_grid_gap, lazy=True),
    Case(
        "interpolate.FillNaNBiharmonic", FillNaNBiharmonic(), data=_grid_gap, lazy=True
    ),
    Case("interpolate.FillNaNIDW", FillNaNIDW(), data=_grid_gap, lazy=True),
    # -- Tier-3: lazy (regionmask masks, xarray-interp regrid) -----------------
    Case("geo.AddLandMask", AddLandMask(), lazy=True),
    Case("geo.AddOceanMask", AddOceanMask(), lazy=True),
    Case("interpolate.RegridLike", RegridLike(_TARGET), lazy=True),
    # -- Tier-3: dask-safe but eager (materialise internally) ------------------
    Case(
        "interpolate.Bin2D",
        Bin2D(grid=_BIN_GRID),
        data=_scattered,
        chunks={"obs": 50},
    ),
    Case(
        "interpolate.Histogram2D",
        Histogram2D(grid=_BIN_GRID),
        data=_scattered,
        chunks={"obs": 50},
    ),
    Case(
        "utils.XarrayEstimator(PCA)",
        lambda d: XarrayEstimator(PCA(n_components=2), sample_dim="time").fit_transform(
            d["ssh"]
        ),
    ),
]


def _params() -> list[Any]:
    out = []
    for c in CASES:
        marks = (pytest.mark.xfail(reason=c.xfail, strict=True),) if c.xfail else ()
        out.append(pytest.param(c, id=c.id, marks=marks))
    return out


def _compute(obj: Any) -> Any:
    return obj.compute() if hasattr(obj, "compute") else obj


def _is_lazy(obj: xr.DataArray | xr.Dataset) -> bool:
    values = obj.data_vars.values() if isinstance(obj, xr.Dataset) else [obj]
    return any(getattr(v, "chunks", None) is not None for v in values)


@pytest.mark.parametrize("case", _params())
def test_operator_dask_parity(case: Case) -> None:
    """Every operator must produce identical results on numpy and dask input."""
    data = case.data()
    args_np: tuple[xr.DataArray | xr.Dataset, ...] = (
        (data, data * 1.01) if case.needs_ref else (data,)
    )
    args_dk = tuple(a.chunk(case.chunks) for a in args_np)

    out_np = case.run(*args_np)
    out_dk = case.run(*args_dk)

    if case.lazy:
        assert _is_lazy(out_dk), f"{case.id}: expected a lazy (dask) result"

    xr.testing.assert_allclose(_compute(out_np), _compute(out_dk))
