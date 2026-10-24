"""xarray converters with a CF spatial_ref and GeoTransform (#537; GF1–GF4, GH2)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import geoarray
from geoarray import GeoArray, GeoPoints, GeoSwath, GridMismatchError, pad_points
from geoarray.coords import (
    SpectralAxis,
    TimeAxis,
    VariableAxis,
    Vertical,
    VerticalAxis,
)
from geoarray.types import GeoMapSeries, GeoSwathImage, matches


xr = pytest.importorskip("xarray")
from geoarray.interop.xarray import from_xarray, to_xarray


T = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
ROTATED = (19.5, 4.33, 600000.0, 4.33, -19.5, 3550000.0)
CRS = "EPSG:32613"
DATES = TimeAxis(
    np.datetime64("2026-06-01", "ns") + np.arange(3) * np.timedelta64(5, "D")
)


def _emit() -> GeoArray:
    wavelengths = np.linspace(381.0, 2493.0, 285)
    bands = SpectralAxis(
        tuple(f"b{k:03d}" for k in range(285)), wavelengths, np.full(285, 8.5)
    )
    data = np.random.default_rng(0).random((285, 3, 4)).astype(np.float32)
    data[0, 0, 0] = np.nan
    return GeoArray(data, T, CRS, np.nan, dims="band y x", coords=(bands,))


def _fixtures() -> dict[str, Any]:
    rng = np.random.default_rng(1)
    depth = Vertical("depth", "m", "down", datum="mean sea level")
    lat, lon = np.meshgrid(
        np.linspace(31, 32, 5), np.linspace(-104, -103, 6), indexing="ij"
    )
    hours = TimeAxis(
        np.datetime64("2026-06-01", "ns") + np.arange(2) * np.timedelta64(1, "h")
    )
    return {
        "nan fill": GeoArray(rng.random((2, 4, 5)), T, CRS, np.nan, dims="band y x"),
        "int fill": GeoArray(
            rng.integers(0, 500, (3, 4, 5)).astype(np.uint16),
            T,
            CRS,
            999,
            dims="time y x",
            coords=(DATES,),
            attrs={"sensor": "S2A", "gain": 2.5},
        ),
        "emit": _emit(),
        "rotated": GeoArray(rng.random((4, 5)), ROTATED, CRS, dims="y x"),
        "one pixel": GeoArray(rng.random((1, 1)), T, CRS, dims="y x"),
        "levels": GeoArray(
            rng.random((3, 4, 5)),
            T,
            CRS,
            dims="level y x",
            coords=(VerticalAxis((5.0, 15.0, 30.0), depth),),
        ),
        "no crs": GeoArray(rng.random((4, 5)), T, None, dims="y x"),
        "0.2 attrs": GeoArray(
            rng.random((2, 4, 5)),
            T,
            CRS,
            dims="band y x",
            attrs={"band_names": ["B11", "B12"], "wavelengths": [1610.0, 2190.0]},
        ),
        "swath": GeoSwath(
            rng.random((2, 5, 6)),
            lat,
            lon,
            dims="band row col",
            coords=(SpectralAxis(("a", "b")),),
            fill_value=np.nan,
        ),
        "tracks": pad_points(
            GeoPoints(
                rng.random((1, 4)),
                xy=rng.uniform(-30, 30, (4, 2)),
                time=np.datetime64("2026-06-01", "ns")
                + np.arange(4) * np.timedelta64(1, "s"),
                dims="variable obs",
                coords=(VariableAxis(("sla",), ("m",)),),
                fill_value=np.nan,
            ),
            6,
        ),
        "profiles": GeoPoints(
            rng.random((2, 3)),
            xy=rng.uniform(-30, 30, (3, 2)),
            z=[5.0, 10.0, 50.0],
            vertical=depth,
            dims="variable obs",
            coords=(VariableAxis(("temp", "psal"), ("degC", "1")),),
        ),
        "stations": GeoPoints(
            rng.random((2, 2, 3)),
            xy=rng.uniform(-30, 30, (3, 2)),
            dims="time variable obs",
            coords=(hours, VariableAxis(("t2m", "psfc"))),
        ),
        "plain points": GeoPoints(
            rng.random(3), xy=rng.uniform(0, 1, (3, 2)), dims="obs", crs=CRS
        ),
    }


@pytest.mark.parametrize("name", list(_fixtures()))
def test_gf1_round_trip_is_identical(name: str) -> None:
    carrier = _fixtures()[name]
    back = from_xarray(to_xarray(carrier))
    assert type(back) is type(carrier)
    assert back.identical(carrier), (back, carrier)
    assert type(carrier).from_xarray(carrier.to_xarray()).identical(carrier)


@pytest.mark.parametrize("seed", range(4))
def test_gf1_random_windows(seed: int) -> None:
    """GA-G20: 200 random windows of a scene round-trip bit for bit."""
    rng = np.random.default_rng(seed)
    scene = GeoArray(
        rng.random((2, 60, 50)),
        T,
        CRS,
        np.nan,
        dims="band y x",
        coords=(SpectralAxis(("B11", "B12")),),
    )
    for _ in range(50):
        r0, c0 = int(rng.integers(0, 59)), int(rng.integers(0, 49))
        r1, c1 = int(rng.integers(r0 + 1, 61)), int(rng.integers(c0 + 1, 51))
        window = scene.isel(y=slice(r0, r1, int(rng.integers(1, 3))), x=slice(c0, c1))
        assert from_xarray(to_xarray(window)).identical(window)


def test_the_encoding() -> None:
    da = to_xarray(_fixtures()["int fill"])
    ref = da.coords["spatial_ref"].attrs
    assert ref["GeoTransform"] == "600000.0 20.0 0.0 3550000.0 0.0 -20.0"
    assert "crs_wkt" in ref and ref["spatial_ref"] == ref["crs_wkt"]
    assert ref["grid_mapping_name"] == "transverse_mercator"
    assert da.encoding["grid_mapping"] == "spatial_ref"
    assert da.encoding["_FillValue"] == 999 and da.attrs["fill_value"] == 999
    # centres: xⱼ = c + a (j + ½)
    assert float(da.x[0]) == 600010.0 and float(da.y[0]) == 3549990.0
    assert "x" not in to_xarray(_fixtures()["rotated"]).coords


def test_wavelength_is_an_indexed_coordinate() -> None:
    """LL-F2: sel by wavelength on the xarray side."""
    da = to_xarray(_emit())
    assert "wavelength" in da.indexes
    assert da.sel(wavelength=381.0).dims == ("y", "x")
    assert da.coords["wavelength"].attrs["units"] == "nm"


def test_vertical_is_cf() -> None:
    level = to_xarray(_fixtures()["levels"]).coords["level"]
    assert level.attrs["positive"] == "down" and level.attrs["units"] == "m"
    assert level.attrs["standard_name"] == "depth"


@pytest.mark.parametrize(
    ("name", "feature"),
    [
        ("tracks", "trajectory"),
        ("profiles", "profile"),
        ("stations", "timeSeries"),
        ("plain points", "point"),
    ],
)
def test_points_are_a_cf_dsg_dataset(name: str, feature: str) -> None:
    ds = to_xarray(_fixtures()[name])
    assert isinstance(ds, xr.Dataset)
    assert ds.attrs["featureType"] == feature
    assert "obs" in ds.dims


def test_profiles_split_variables() -> None:
    ds = to_xarray(_fixtures()["profiles"])
    assert set(ds.data_vars) == {"temp", "psal"}
    assert ds["temp"].attrs["units"] == "degC"
    assert ds.coords["z"].attrs["positive"] == "down"
    assert {"lon", "lat"} <= set(ds.coords)


def test_gf2_netcdf_opens_in_rioxarray(tmp_path: Path) -> None:
    rioxarray = pytest.importorskip("rioxarray")
    g = _fixtures()["nan fill"]
    path = tmp_path / "scene.nc"
    to_xarray(g).to_dataset(name="data").to_netcdf(path)
    with xr.open_dataset(path, decode_coords="all") as ds:
        opened = ds["data"]
        assert rioxarray is not None
        theirs = np.array(tuple(opened.rio.transform())[:6])
        np.testing.assert_allclose(theirs, T, rtol=0, atol=1e-9 * 20.0)
        assert opened.rio.crs.to_epsg() == 32613
        assert from_xarray(opened).identical(g)


def test_gf2_rotated_netcdf_round_trips(tmp_path: Path) -> None:
    g = _fixtures()["rotated"]
    path = tmp_path / "rotated.nc"
    to_xarray(g).to_dataset(name="data").to_netcdf(path)
    with xr.open_dataset(path, decode_coords="all") as ds:
        assert from_xarray(ds["data"]).identical(g)


def test_gf3_centres_off_the_transform_raise() -> None:
    da = to_xarray(_fixtures()["nan fill"])
    a = 20.0
    shifted = da.assign_coords(x=da.x + 2e-9 * a)
    with pytest.raises(GridMismatchError, match="1e-9 pixel"):
        from_xarray(shifted)
    fine = da.assign_coords(x=da.x + 0.5e-9 * a)
    np.testing.assert_array_equal(from_xarray(fine).transform, T)


def test_463_centres_only_give_the_edge_origin() -> None:
    da = to_xarray(_fixtures()["nan fill"]).drop_vars("spatial_ref")
    g = from_xarray(da)
    # the edge, not the first centre
    np.testing.assert_array_equal(g.transform, T)
    assert g.crs is None


def test_464_edge_coordinates_under_a_centre_transform_raise() -> None:
    da = to_xarray(_fixtures()["nan fill"])
    edges = da.assign_coords(x=da.x - 10.0, y=da.y + 10.0)
    with pytest.raises(GridMismatchError):
        from_xarray(edges)


def test_one_pixel_needs_a_geotransform() -> None:
    da = to_xarray(_fixtures()["one pixel"])
    assert from_xarray(da).identical(_fixtures()["one pixel"])
    with pytest.raises(ValueError, match="one-pixel"):
        from_xarray(da.drop_vars("spatial_ref"))


def test_georeader_crs_attr_is_read_with_a_warning() -> None:
    g = _fixtures()["nan fill"]
    da = to_xarray(g).drop_vars("spatial_ref").assign_attrs(crs="EPSG:32613")
    with pytest.warns(DeprecationWarning, match="attrs\\['crs'\\]"):
        back = from_xarray(da)
    assert back.crs == g.crs and "crs" not in back.attrs


def test_lat_lon_grids_read_as_y_x() -> None:
    lat = np.linspace(40.0, 39.0, 5)
    lon = np.linspace(-3.0, -2.0, 6)
    da = xr.DataArray(
        np.zeros((3, 5, 6)),
        dims=("time", "lat", "lon"),
        coords={"time": DATES.stamps, "lat": lat, "lon": lon},
    )
    g = from_xarray(da)
    assert g.dims == ("time", "y", "x")
    assert g.labels("time") == DATES.labels
    np.testing.assert_allclose(g.transform, [0.2, 0, -3.1, 0, -0.25, 40.125])


def test_gf4_datatree_of_three_grids() -> None:
    grids = {
        res: GeoArray(
            np.zeros((2, 60 // k, 60 // k)),
            (10.0 * k, 0, 600000, 0, -10.0 * k, 3550000),
            CRS,
            dims="band y x",
        )
        for res, k in (("r10m", 1), ("r20m", 2), ("r60m", 6))
    }
    tree = geoarray.to_datatree(grids)
    back = geoarray.from_datatree(tree)
    assert sorted(back) == ["/r10m", "/r20m", "/r60m"]
    for name, g in grids.items():
        assert back[f"/{name}"][name].grid == g.grid


def test_dataset_of_one_grid() -> None:
    g = _fixtures()["nan fill"]
    ds = geoarray.to_dataset(
        {"radiance": g, "mask": g.with_data(g.data > 0.5, fill_value=False)}
    )
    back = geoarray.from_dataset(ds)
    assert back["radiance"].identical(g)
    assert back["mask"].dtype == bool
    with pytest.raises(GridMismatchError):
        geoarray.to_dataset({"a": g, "b": g.shift_origin((1.0, 0.0))})


def test_batched_carriers_do_not_convert() -> None:
    t = np.stack([np.array(T)] * 2)
    with pytest.raises(ValueError, match="one grid"):
        to_xarray(GeoArray(np.zeros((2, 4, 4)), t, CRS, dims="patch y x"))


# -- readable types on DataArrays ----------------------------------------------------


def test_a_lazy_lat_lon_cube_is_a_map_series() -> None:
    da_mod = pytest.importorskip("dask.array")
    cube = xr.DataArray(
        da_mod.zeros((3, 5, 6), chunks=(1, 5, 6)),
        dims=("time", "lat", "lon"),
        coords={"lat": np.arange(5.0), "lon": np.arange(6.0)},
    )
    assert matches(cube, GeoMapSeries)
    assert isinstance(cube.data, da_mod.Array)  # nothing computed or renamed


def test_curvilinear_dataarray_is_a_swath_not_a_grid() -> None:
    swath = _fixtures()["swath"]
    da = to_xarray(swath).rename({"row": "y", "col": "x"})
    assert not matches(da, GeoMapSeries)
    assert matches(to_xarray(swath), GeoSwathImage)


# -- installs ------------------------------------------------------------------------


def test_gh2_import_without_xarray() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "xarray":
                    raise ImportError(name)

        sys.meta_path.insert(0, Block())
        import geoarray, numpy as np
        g = geoarray.GeoArray(np.zeros((2, 2)), (1, 0, 0, 0, -1, 0), None, dims="y x")
        for call in (g.to_xarray, lambda: geoarray.to_dataset({"a": g})):
            try:
                call()
            except ImportError as exc:
                print(exc)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.count("geotoolz-array[xarray]") == 2


# -- review fixes (Codex, #551) -------------------------------------------------------


def test_to_dataset_refuses_carriers_with_different_band_labels() -> None:
    red = GeoArray(
        np.zeros((1, 2, 2)), T, CRS, dims="band y x", coords=(SpectralAxis(("red",)),)
    )
    nir = red.with_data(red.data, coords=(SpectralAxis(("nir",)),))
    with pytest.raises(ValueError, match="coordinates differ"):
        geoarray.to_dataset({"red": red, "nir": nir})


def test_batched_swath_round_trips() -> None:
    lat, lon = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
    s = GeoSwath(
        np.zeros((2, 3, 4)),
        np.stack([lat, lat + 1]),
        np.stack([lon, lon + 1]),
        dims="time row col",
    )
    back = GeoSwath.from_xarray(s.to_xarray())
    assert back.dims == ("time", "row", "col")
    np.testing.assert_array_equal(back.lat, s.lat)


def test_bool_mask_writes_to_netcdf(tmp_path: Path) -> None:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    mask = GeoArray(np.zeros((2, 2), dtype=bool), T, CRS, fill_value=False, dims="y x")
    path = tmp_path / "mask.nc"
    mask.to_xarray().to_dataset(name="mask").to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path, decode_coords="all") as ds:
        back = GeoArray.from_xarray(ds["mask"].astype(bool))
    assert back.fill_value is False
