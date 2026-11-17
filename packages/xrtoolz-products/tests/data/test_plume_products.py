"""Plume-complex GeoJSON and the sensitivity / uncertainty companions.

Every file is synthetic and written to ``tmp_path``: a 20 × 20 scene with
its two companions (EMIT and AVIRIS-3 naming) and a two-plume GeoJSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import xrproducts
import xrproducts.types
from xrproducts import (
    LocalL2Source,
    MissingPropertyError,
    ProductGridError,
    open_av3_ch4_l2b,
    open_ch4_plume_complexes,
    open_emit_ch4_l2b,
    plume_outline,
)


N = 20
FILL = -9999.0

# ---- synthetic writers ----------------------------------------------------


def _lonlat() -> tuple[np.ndarray, np.ndarray]:
    lon, lat = np.meshgrid(np.linspace(-104.0, -103.8, N), np.linspace(32.5, 32.3, N))
    return lon, lat


def _enhancement() -> np.ndarray:
    enh = np.arange(N * N, dtype=np.float32).reshape(N, N)
    enh[0, 0] = FILL
    return enh


def _sensitivity() -> np.ndarray:
    sens = np.full((N, N), 0.8, dtype=np.float32)
    sens[1, 1] = FILL
    return sens


def _uncertainty() -> np.ndarray:
    unc = np.full((N, N), 150.0, dtype=np.float32)
    unc[2, 2] = FILL
    return unc


def _raster(
    name: str,
    values: np.ndarray,
    *,
    dims: tuple[str, str] = ("downtrack", "crosstrack"),
    lonlat: bool = True,
    lon_shift: float = 0.0,
) -> xr.Dataset:
    data_vars: dict[str, tuple] = {name: (dims, values)}
    if lonlat:
        lon, lat = _lonlat()
        data_vars["longitude"] = (dims, lon + lon_shift)
        data_vars["latitude"] = (dims, lat)
    return xr.Dataset(data_vars, attrs={"time_coverage_start": "2023-10-06T19:01:53Z"})


@pytest.fixture
def emit_scene(tmp_path: Path) -> dict[str, Path]:
    """EMIT scene with sensitivity and uncertainty in their own files."""
    files = {
        "enh": tmp_path / "EMIT_L2B_CH4ENH_002_20231006T190153_2327912_020.nc",
        "sens": tmp_path / "EMIT_L2B_CH4SENS_002_20231006T190153_2327912_020.nc",
        "unc": tmp_path / "EMIT_L2B_CH4UNCERT_002_20231006T190153_2327912_020.nc",
    }
    _raster("ch4_enhancement", _enhancement()).to_netcdf(files["enh"])
    _raster("ch4_sensitivity", _sensitivity()).to_netcdf(files["sens"])
    _raster("ch4_uncertainty", _uncertainty()).to_netcdf(files["unc"])
    return files


@pytest.fixture
def av3_scene(tmp_path: Path) -> dict[str, Path]:
    """AVIRIS-3 scene as converted GeoTIFFs: one generic band on (band, y, x)."""
    prefix = tmp_path / "AV320231006t190153_020_L2B_GHG_6395d427"
    files = {
        "enh": Path(f"{prefix}_CH4_ORT.nc"),
        "sens": Path(f"{prefix}_CH4_SNS_ORT.nc"),
        "unc": Path(f"{prefix}_CH4_UNC_ORT.nc"),
    }
    x = 500_000.0 + 5.0 * np.arange(N)
    y = 3_600_000.0 - 5.0 * np.arange(N)
    for key, values in (
        ("enh", _enhancement()),
        ("sens", _sensitivity()),
        ("unc", _uncertainty()),
    ):
        band = values[None].copy()
        if key != "enh":
            band[0, 3, 3] = -1.0  # flare marker
        xr.Dataset(
            {"band_data": (("band", "y", "x"), band)},
            coords={"band": [1], "y": y, "x": x},
        ).to_netcdf(files[key])
    return files


def _feature(
    plume_id: str | None,
    ring: list[list[float]],
    **properties: object,
) -> dict:
    props = dict(properties)
    if plume_id is not None:
        props["Plume ID"] = plume_id
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": props,
    }


RING_A = [[-103.9, 32.40], [-103.88, 32.40], [-103.88, 32.42], [-103.9, 32.40]]
RING_B = [[10.0, 50.0], [10.1, 50.0], [10.1, 50.1], [10.0, 50.1], [10.0, 50.0]]


def _write_geojson(path: Path, features: list[dict]) -> Path:
    doc = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def two_plumes(tmp_path: Path) -> Path:
    """A DAAC-style plume (no origin) and a JPL-list plume with an origin."""
    return _write_geojson(
        tmp_path / "plumes.json",
        [
            _feature(
                "CH4_PlumeComplex-1234",
                RING_A,
                **{
                    "UTC Time Observed": "2023-10-06T19:01:53Z",
                    "Scene FIDs": ["emit20231006t190153"],
                },
            ),
            _feature(
                "CH4_PlumeComplex-77",
                RING_B,
                fids=["AV320241104t180240_001", "AV320241104t180240_002"],
                **{
                    "Psuedo-Origin": json.dumps(
                        {"type": "Point", "coordinates": [10.05, 50.05]}
                    )
                },
            ),
            # The plume portal pairs each outline with its max point: skipped.
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [10.05, 50.05]},
                "properties": {"Plume ID": "CH4_PlumeComplex-77"},
            },
        ],
    )


# ---- companions on the EMIT opener ----------------------------------------


def test_emit_companions_share_one_grid_with_fill_as_nan(emit_scene):
    ds = open_emit_ch4_l2b(
        emit_scene["enh"],
        sensitivity_path=emit_scene["sens"],
        uncertainty_path=emit_scene["unc"],
    )
    names = ("ch4_enhancement", "ch4_sensitivity", "ch4_uncertainty")
    for name in names:
        assert ds[name].dims == ("time", "y", "x")
        assert ds[name].shape == (1, N, N)
        assert np.issubdtype(ds[name].dtype, np.floating)
        assert ds[name].attrs["source_fill_value"] == FILL
        assert not (ds[name].values == FILL).any()
    assert np.isnan(ds["ch4_enhancement"].values[0, 0, 0])
    assert np.isnan(ds["ch4_sensitivity"].values[0, 1, 1])
    assert np.isnan(ds["ch4_uncertainty"].values[0, 2, 2])
    # The enhancement is returned as stored, never divided by the sensitivity.
    np.testing.assert_array_equal(
        ds["ch4_enhancement"].values[0].ravel()[1:], np.arange(1, N * N, dtype=float)
    )
    assert ds["ch4_sensitivity"].attrs["units"] == "1"
    assert ds["ch4_uncertainty"].attrs["units"] == "ppm m"
    assert ds["ch4_uncertainty"].attrs["uncertainty_of"] == "truth"
    assert ds["lon"].dims == ("y", "x")


def test_emit_without_companions_keeps_todays_variables(emit_scene):
    ds = open_emit_ch4_l2b(emit_scene["enh"])
    assert set(ds.data_vars) == {"ch4_enhancement"}
    assert ds["ch4_enhancement"].dims == ("time", "y", "x")
    assert ds["ch4_enhancement"].attrs["units"] == "ppm m"
    assert set(ds.coords) == {"time", "lon", "lat"}


@pytest.mark.parametrize("which", ["sens", "unc"])
def test_companion_of_another_shape_raises_naming_the_file(tmp_path, emit_scene, which):
    small = tmp_path / f"small_{which}.nc"
    _raster("band_data", np.zeros((N - 1, N)), lonlat=False).to_netcdf(small)
    kwargs = {"sensitivity_path" if which == "sens" else "uncertainty_path": small}
    with pytest.raises(ProductGridError, match=small.name):
        open_emit_ch4_l2b(emit_scene["enh"], **kwargs)


def test_companion_with_shifted_coordinates_raises(tmp_path, emit_scene):
    shifted = tmp_path / "shifted.nc"
    _raster("ch4_sensitivity", _sensitivity(), lon_shift=1e-5).to_netcdf(shifted)
    with pytest.raises(ProductGridError, match="lon"):
        open_emit_ch4_l2b(emit_scene["enh"], sensitivity_path=shifted)


def test_companion_within_tolerance_is_accepted(tmp_path, emit_scene):
    nudged = tmp_path / "nudged.nc"
    _raster("ch4_sensitivity", _sensitivity(), lon_shift=1e-7).to_netcdf(nudged)
    ds = open_emit_ch4_l2b(emit_scene["enh"], sensitivity_path=nudged)
    assert "ch4_sensitivity" in ds.data_vars


def test_companion_with_two_rasters_raises(tmp_path, emit_scene):
    ambiguous = tmp_path / "ambiguous.nc"
    xr.Dataset(
        {
            "a": (("downtrack", "crosstrack"), np.zeros((N, N))),
            "b": (("downtrack", "crosstrack"), np.zeros((N, N))),
        }
    ).to_netcdf(ambiguous)
    with pytest.raises(KeyError, match="ch4_sensitivity"):
        open_emit_ch4_l2b(emit_scene["enh"], sensitivity_path=ambiguous)


def test_glt_orthorectifies_all_three_to_the_same_nan_cells(tmp_path, emit_scene):
    gy, gx = np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1), indexing="ij")
    gx = gx[:, ::-1].copy()
    gx[5, 5] = 0
    gy[5, 5] = 0
    glt = tmp_path / "glt.nc"
    xr.Dataset(
        {
            "glt_x": (("ortho_y", "ortho_x"), gx.astype("int32")),
            "glt_y": (("ortho_y", "ortho_x"), gy.astype("int32")),
        },
        attrs={"geotransform": [-104.0, 0.01, 0.0, 32.5, 0.0, -0.01]},
    ).to_netcdf(glt)
    # Same lookup on every raster, so put no-data at the same raw pixel.
    files = {}
    for key, name in (
        ("enh", "ch4_enhancement"),
        ("sens", "ch4_sensitivity"),
        ("unc", "ch4_uncertainty"),
    ):
        values = np.ones((N, N), dtype=np.float32)
        values[7, 3] = FILL
        files[key] = tmp_path / f"{key}.nc"
        _raster(name, values, lonlat=False).to_netcdf(files[key])
    ds = open_emit_ch4_l2b(
        files["enh"],
        sensitivity_path=files["sens"],
        uncertainty_path=files["unc"],
        glt_path=glt,
    )
    nan = np.isnan(ds["ch4_enhancement"].values[0])
    assert nan[5, 5]  # the GLT's no-data cell
    assert nan[7, N - 1 - 3]  # the raw fill, through the mirrored columns
    assert nan.sum() == 2
    for name in ("ch4_sensitivity", "ch4_uncertainty"):
        assert ds[name].dims == ("time", "y", "x")
        np.testing.assert_array_equal(np.isnan(ds[name].values[0]), nan)


# ---- AVIRIS-3 --------------------------------------------------------------


def test_av3_reads_generic_bands_masks_flares_and_takes_flight_time(av3_scene):
    ds = open_av3_ch4_l2b(
        av3_scene["enh"],
        sensitivity_path=av3_scene["sens"],
        uncertainty_path=av3_scene["unc"],
    )
    assert ds["time"].values[0] == np.datetime64("2023-10-06T19:01:53")
    for name in ("ch4_sensitivity", "ch4_uncertainty"):
        assert np.isnan(ds[name].values[0, 3, 3])
        assert ds[name].attrs["flare_value"] == -1.0
    assert ds["ch4_enhancement"].values[0, 3, 3] == 3 * N + 3
    assert "band" not in ds.dims
    assert np.isnan(ds["ch4_enhancement"].values[0, 0, 0])


def test_emit_and_av3_return_the_same_names_dims_and_units(emit_scene, av3_scene):
    emit = open_emit_ch4_l2b(
        emit_scene["enh"],
        sensitivity_path=emit_scene["sens"],
        uncertainty_path=emit_scene["unc"],
    )
    av3 = open_av3_ch4_l2b(
        av3_scene["enh"],
        sensitivity_path=av3_scene["sens"],
        uncertainty_path=av3_scene["unc"],
    )
    assert set(emit.data_vars) == set(av3.data_vars)
    for name in emit.data_vars:
        assert emit[name].dims == av3[name].dims
        assert emit[name].attrs["units"] == av3[name].attrs["units"]


def test_av3_companion_on_another_grid_raises(tmp_path, av3_scene):
    other = tmp_path / "other.nc"
    xr.Dataset(
        {"band_data": (("y", "x"), np.zeros((N, N)))},
        coords={"y": np.arange(N, dtype=float), "x": np.arange(N, dtype=float)},
    ).to_netcdf(other)
    with pytest.raises(ProductGridError, match=r"other\.nc"):
        open_av3_ch4_l2b(av3_scene["enh"], uncertainty_path=other)


def test_source_opens_av3_and_refuses_shared_companions(av3_scene):
    src = LocalL2Source()
    ds = src.open("av3.ch4", path=av3_scene["enh"], sensitivity_path=av3_scene["sens"])
    assert "ch4_sensitivity" in ds.data_vars
    with pytest.raises(ValueError, match="sensitivity_path"):
        src.open(
            "av3.ch4",
            paths=[av3_scene["enh"], av3_scene["enh"]],
            sensitivity_path=av3_scene["sens"],
        )


# ---- plume complexes -------------------------------------------------------


def test_plume_complexes_layout_and_outline_round_trip(two_plumes):
    plumes = open_ch4_plume_complexes(two_plumes)
    assert plumes.sizes == {"plume": 2, "vertex": len(RING_A) + len(RING_B)}
    assert list(plumes["plume"].values) == [
        "CH4_PlumeComplex-1234",
        "CH4_PlumeComplex-77",
    ]
    assert list(plumes["plume_id"].values) == list(plumes["plume"].values)
    assert plumes.attrs["crs"] == "EPSG:4326"
    assert plumes.attrs["geometry_type"] == "polygon"
    assert plumes["vertex_count"].attrs["sample_dimension"] == "vertex"
    np.testing.assert_array_equal(
        plume_outline(plumes, "CH4_PlumeComplex-1234"), RING_A
    )
    np.testing.assert_array_equal(plume_outline(plumes, "CH4_PlumeComplex-77"), RING_B)
    first = plumes.sel(plume="CH4_PlumeComplex-1234")
    assert first["time"].values == np.datetime64("2023-10-06T19:01:53")
    assert first["scene_ids"].item() == "emit20231006t190153"
    assert np.isnan(first["origin_lon"].item())
    second = plumes.sel(plume="CH4_PlumeComplex-77")
    assert np.isnat(second["time"].values)
    assert second["scene_ids"].item() == "AV320241104t180240_001,AV320241104t180240_002"
    assert second["origin_lon"].item() == pytest.approx(10.05)
    assert second["origin_lat"].item() == pytest.approx(50.05)


def test_plume_outline_unknown_id_raises(two_plumes):
    with pytest.raises(KeyError, match="nope"):
        plume_outline(open_ch4_plume_complexes(two_plumes), "nope")


def test_feature_without_plume_id_raises(tmp_path):
    path = _write_geojson(tmp_path / "p.json", [_feature(None, RING_A)])
    with pytest.raises(MissingPropertyError, match="Plume ID"):
        open_ch4_plume_complexes(path)


def test_feature_without_geometry_raises(tmp_path):
    feature = {"type": "Feature", "geometry": None, "properties": {"Plume ID": "x"}}
    path = _write_geojson(tmp_path / "p.json", [feature])
    with pytest.raises(MissingPropertyError, match="geometry"):
        open_ch4_plume_complexes(path)


def test_line_geometry_and_empty_file_raise(tmp_path):
    line = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": RING_A},
        "properties": {"Plume ID": "x"},
    }
    with pytest.raises(ValueError, match="LineString"):
        open_ch4_plume_complexes(_write_geojson(tmp_path / "l.json", [line]))
    with pytest.raises(ValueError, match="no Polygon"):
        open_ch4_plume_complexes(_write_geojson(tmp_path / "e.json", []))


@pytest.mark.parametrize(
    ("coordinates", "order"),
    [([10.05, 50.05], "lon_lat"), ([50.05, 10.05], "lat_lon")],
)
def test_pseudo_origin_in_either_axis_order(tmp_path, coordinates, order):
    origin = json.dumps({"type": "Point", "coordinates": coordinates})
    path = _write_geojson(
        tmp_path / "p.json", [_feature("p", RING_B, **{"Psuedo-Origin": origin})]
    )
    plumes = open_ch4_plume_complexes(path)
    assert plumes["origin_lon"].item() == pytest.approx(10.05)
    assert plumes["origin_lat"].item() == pytest.approx(50.05)
    assert plumes.attrs["origin_axis_order"] == order


def test_mixed_origin_orders_are_recorded(tmp_path):
    def origin(c: list[float]) -> str:
        return json.dumps({"type": "Point", "coordinates": c})

    path = _write_geojson(
        tmp_path / "p.json",
        [
            _feature("a", RING_B, **{"Psuedo-Origin": origin([10.05, 50.05])}),
            _feature("b", RING_B, **{"Psuedo-Origin": origin([50.05, 10.05])}),
        ],
    )
    assert open_ch4_plume_complexes(path).attrs["origin_axis_order"] == "mixed"


def test_multipolygon_keeps_the_largest_ring(tmp_path):
    small = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.0]]
    hole = [[10.02, 50.02], [10.03, 50.02], [10.03, 50.03], [10.02, 50.02]]
    feature = {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": [[small], [RING_B, hole]]},
        "properties": {"Plume ID": "multi"},
    }
    plumes = open_ch4_plume_complexes(_write_geojson(tmp_path / "m.json", [feature]))
    np.testing.assert_array_equal(plume_outline(plumes, "multi"), RING_B)
    assert plumes.attrs["dropped_polygons"] == 1


# ---- public API ------------------------------------------------------------


def test_public_api_lists_the_new_names():
    for name in (
        "open_av3_ch4_l2b",
        "open_ch4_plume_complexes",
        "plume_outline",
        "ProductGridError",
        "MissingPropertyError",
    ):
        assert name in xrproducts.__all__
        assert hasattr(xrproducts, name)
    assert "CH4_SENSITIVITY" in xrproducts.types.__all__
    assert xrproducts.resolve("ch4_sensitivity") is xrproducts.types.CH4_SENSITIVITY
    assert issubclass(ProductGridError, ValueError)
    assert issubclass(MissingPropertyError, KeyError)
