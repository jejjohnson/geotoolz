"""`load_xarray` selects what the slice asks for; footprints cover pixels (#218)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyproj
import pytest


xr = pytest.importorskip("xarray")

import numpy as np

from geocatalog import GeoSlice, build_xarray_catalog, load_xarray


def _write(
    path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    times: pd.DatetimeIndex,
) -> Path:
    data = np.arange(len(times) * len(y) * len(x), dtype=np.float32).reshape(
        len(times), len(y), len(x)
    )
    xr.Dataset(
        {"v": (("time", "y", "x"), data)},
        coords={"time": times, "y": y, "x": x},
    ).to_netcdf(path)
    return path


def _iv(a: str, b: str, tz: str | None = None) -> pd.Interval:
    return pd.Interval(pd.Timestamp(a, tz=tz), pd.Timestamp(b, tz=tz), closed="both")


TIMES = pd.date_range("2024-01-01", periods=3, freq="D")
# Pixel centres at 0.5 .. 9.5, north-up (descending y).
X = np.arange(10) + 0.5
Y_DESC = (np.arange(10) + 0.5)[::-1]


def test_descending_y_selection_is_not_empty(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.nc", x=X, y=Y_DESC, times=TIMES)
    cat = build_xarray_catalog([path], target_crs="EPSG:32629")
    out = load_xarray(
        cat,
        GeoSlice(
            (2.0, 2.0, 5.0, 5.0),
            _iv("2024-01-01", "2024-01-03"),
            (1.0, 1.0),
            "EPSG:32629",
        ),
    )
    assert out.sizes["y"] == 3
    assert out.sizes["x"] == 3
    assert list(out["y"].values) == [4.5, 3.5, 2.5]


def test_footprint_covers_pixel_edges(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.nc", x=X, y=Y_DESC, times=TIMES)
    cat = build_xarray_catalog([path], target_crs="EPSG:32629")
    assert cat.gdf.geometry.iloc[0].bounds == (0.0, 0.0, 10.0, 10.0)


def test_foreign_crs_slice_selects_same_window(tmp_path: Path) -> None:
    x0, y0 = 500_000.0, 4_200_000.0
    x = x0 + (np.arange(20) + 0.5) * 100
    y = (y0 + (np.arange(20) + 0.5) * 100)[::-1]
    path = _write(tmp_path / "utm.nc", x=x, y=y, times=TIMES)
    cat = build_xarray_catalog([path], target_crs="EPSG:32629")

    native = (x0 + 500, y0 + 500, x0 + 1500, y0 + 1500)
    to_ll = pyproj.Transformer.from_crs(32629, 4326, always_xy=True)
    ll = to_ll.transform_bounds(*native)
    iv = _iv("2024-01-01", "2024-01-03")
    out = load_xarray(cat, GeoSlice(ll, iv, (0.001, 0.001), "EPSG:4326"))
    ref = load_xarray(cat, GeoSlice(native, iv, (100.0, 100.0), "EPSG:32629"))
    assert out.sizes["x"] > 0 and out.sizes["y"] > 0
    # transform_bounds grows the box a little; it must contain the native window.
    assert set(ref["x"].values) <= set(out["x"].values)
    assert set(ref["y"].values) <= set(out["y"].values)


def test_tz_aware_slice_interval(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.nc", x=X, y=Y_DESC, times=TIMES)
    cat = build_xarray_catalog([path], target_crs="EPSG:32629")
    out = load_xarray(
        cat,
        GeoSlice(
            (0.0, 0.0, 10.0, 10.0),
            _iv("2024-01-02", "2024-01-03", tz="UTC"),
            (1.0, 1.0),
            "EPSG:32629",
        ),
    )
    assert out.sizes["time"] == 2


def test_concat_sorts_time_and_drops_duplicates(tmp_path: Path) -> None:
    late = _write(
        tmp_path / "late.nc",
        x=X,
        y=Y_DESC,
        times=pd.date_range("2024-01-02", periods=2),
    )
    early = _write(
        tmp_path / "early.nc",
        x=X,
        y=Y_DESC,
        times=pd.date_range("2024-01-01", periods=2),
    )
    cat = build_xarray_catalog([late, early], target_crs="EPSG:32629")
    out = load_xarray(
        cat,
        GeoSlice(
            (0.0, 0.0, 10.0, 10.0),
            _iv("2024-01-01", "2024-01-03"),
            (1.0, 1.0),
            "EPSG:32629",
        ),
    )
    assert list(pd.DatetimeIndex(out["time"].values)) == list(
        pd.date_range("2024-01-01", periods=3)
    )


def test_crs_recovered_without_caller_importing_rioxarray(tmp_path: Path) -> None:
    pytest.importorskip("rioxarray")
    import rioxarray  # noqa: F401

    ds = xr.Dataset(
        {"v": (("y", "x"), np.zeros((10, 10), dtype=np.float32))},
        coords={"y": Y_DESC, "x": X},
    ).rio.write_crs("EPSG:32629")
    path = tmp_path / "crs.nc"
    ds.to_netcdf(path)
    # Fresh interpreter: rioxarray is not imported unless geocatalog does it.
    code = (
        "import sys; from geocatalog import build_xarray_catalog; "
        f"c = build_xarray_catalog([{str(path)!r}]); "
        "print('rioxarray' in sys.modules, c.gdf['crs'].iloc[0])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out[0] == "True"
    assert "32629" in out[1]
