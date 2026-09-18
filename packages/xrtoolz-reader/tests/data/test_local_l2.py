"""LocalL2Source + TROPOMI / EMIT / GHGSat openers on synthetic files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrreader import (
    CATALOG,
    BBox,
    DatasetKind,
    LocalL2Source,
    TimeRange,
    apply_cf_attrs,
    describe,
    open_emit_ch4_l2b,
    open_ghgsat_ch4_l2,
    open_tropomi_ch4_l2,
    resolve,
)


# ---- synthetic writers ----------------------------------------------------

TROPOMI_CANONICAL = {
    "methane_mixing_ratio": "xch4",
    "methane_mixing_ratio_bias_corrected": "xch4_bias_corrected",
    "methane_mixing_ratio_precision": "xch4_precision",
    "column_averaging_kernel": "column_averaging_kernel",
    "methane_profile_apriori": "ch4_profile_apriori",
    "dry_air_subcolumns": "dry_air_subcolumns",
    "qa_value": "qa_value",
    "surface_pressure": "sp",
}
LOW_QA_PIXELS = [(0, 0), (2, 1)]  # (scanline, ground_pixel) planted below 0.5
REF_TIME = "2019-06-01T00:00:00"


def write_synthetic_tropomi(
    path: Path,
    n_scan: int = 4,
    n_pix: int = 3,
    n_layer: int = 12,
    *,
    lon0: float = 10.0,
    absolute_delta_time: bool = True,
) -> Path:
    """Write the four TROPOMI groups with the real variable names.

    The swath spans ``lon0 .. lon0 + n_pix - 1`` along ``ground_pixel`` and
    ``40 .. 40 + n_scan - 1`` along ``scanline``; scanlines are one second
    apart. ``qa_value`` is 0.9 everywhere except :data:`LOW_QA_PIXELS`
    (0.2). With ``absolute_delta_time=False`` the ``delta_time`` offsets
    are written as bare milliseconds (no CF reference date).
    """
    swath = ("time", "scanline", "ground_pixel")
    lon = np.broadcast_to(lon0 + np.arange(n_pix), (1, n_scan, n_pix)).astype(float)
    lat = np.broadcast_to(40.0 + np.arange(n_scan)[:, None], (1, n_scan, n_pix)).astype(
        float
    )
    qa = np.full((1, n_scan, n_pix), 0.9)
    for s, p in LOW_QA_PIXELS:
        qa[0, s, p] = 0.2
    xch4 = 1800.0 + np.arange(n_scan * n_pix, dtype=float).reshape(1, n_scan, n_pix)
    delta_attrs = (
        {"units": f"milliseconds since {REF_TIME.replace('T', ' ')}"}
        if absolute_delta_time
        else {"units": "ms", "long_name": "offset from reference start time"}
    )
    product = xr.Dataset(
        {
            "latitude": (swath, lat, {"standard_name": "latitude"}),
            "longitude": (swath, lon, {"standard_name": "longitude"}),
            "delta_time": (
                ("time", "scanline"),
                (np.arange(n_scan) * 1000).reshape(1, n_scan).astype("int32"),
                delta_attrs,
            ),
            "qa_value": (swath, qa),
            "methane_mixing_ratio": (swath, xch4, {"units": "1e-9"}),
            "methane_mixing_ratio_precision": (swath, np.full(xch4.shape, 5.0)),
            "methane_mixing_ratio_bias_corrected": (swath, xch4 - 10.0),
        },
        coords={
            "time": (
                "time",
                np.array(
                    [
                        int(
                            pd.Timestamp(REF_TIME).timestamp()
                            - pd.Timestamp("2010-01-01").timestamp()
                        )
                    ],
                    dtype="int32",
                ),
                {"units": "seconds since 2010-01-01 00:00:00"},
            ),
            "scanline": np.arange(n_scan),
            "ground_pixel": np.arange(n_pix),
            "layer": np.arange(n_layer),
            "level": np.arange(n_layer + 1),
            "corner": np.arange(4),
        },
    )
    layered = (*swath, "layer")
    detailed = xr.Dataset(
        {
            "column_averaging_kernel": (
                layered,
                np.ones((1, n_scan, n_pix, n_layer)),
            ),
            "surface_albedo_SWIR": (swath, np.full((1, n_scan, n_pix), 0.3)),
        }
    )
    inputs = xr.Dataset(
        {
            "methane_profile_apriori": (
                layered,
                np.full((1, n_scan, n_pix, n_layer), 0.01),
            ),
            "dry_air_subcolumns": (
                layered,
                np.full((1, n_scan, n_pix, n_layer), 300.0),
            ),
            "surface_pressure": (swath, np.full((1, n_scan, n_pix), 101300.0)),
            "pressure_interval": (swath, np.full((1, n_scan, n_pix), 8400.0)),
            "altitude_levels": (
                (*swath, "level"),
                np.zeros((1, n_scan, n_pix, n_layer + 1)),
            ),
        }
    )
    geo = xr.Dataset(
        {
            "latitude_bounds": ((*swath, "corner"), np.zeros((1, n_scan, n_pix, 4))),
            "longitude_bounds": ((*swath, "corner"), np.zeros((1, n_scan, n_pix, 4))),
            "solar_zenith_angle": (swath, np.full((1, n_scan, n_pix), 30.0)),
        }
    )
    tree = xr.DataTree.from_dict(
        {
            "PRODUCT": product,
            "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS": detailed,
            "PRODUCT/SUPPORT_DATA/INPUT_DATA": inputs,
            "PRODUCT/SUPPORT_DATA/GEOLOCATIONS": geo,
        }
    )
    tree.attrs["title"] = "synthetic S5P CH4 L2"
    tree.to_netcdf(path)
    return path


def write_synthetic_emit(path: Path, glt_path: Path, ny: int = 3, nx: int = 4) -> Path:
    """Write a raw ``(y, x)`` enhancement raster plus a GLT for it."""
    enh = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    raster = xr.Dataset(
        {
            "ch4_enhancement": (("y", "x"), enh, {"units": "ppm m"}),
            "ch4_uncertainty": (("y", "x"), enh / 10.0, {"units": "ppm m"}),
        },
        attrs={"time_coverage_start": "2023-08-01T10:00:00Z"},
    )
    raster.to_netcdf(path)
    # Orthorectified grid: same shape, transposed lookup + one no-data cell.
    gy, gx = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1), indexing="ij")
    gx = gx[:, ::-1].copy()  # mirror columns so the lookup is non-trivial
    gx[0, 0] = 0
    gy[0, 0] = 0
    glt = xr.Dataset(
        {
            "glt_x": (("ortho_y", "ortho_x"), gx.astype("int32")),
            "glt_y": (("ortho_y", "ortho_x"), gy.astype("int32")),
        },
        attrs={"geotransform": [-100.0, 0.01, 0.0, 35.0, 0.0, -0.01]},
    )
    glt.to_netcdf(glt_path)
    return path


def write_synthetic_ghgsat(path: Path, ny: int = 3, nx: int = 3) -> Path:
    """Write a per-plume raster with ``xch4`` and 2-D latitude/longitude."""
    lon, lat = np.meshgrid(np.linspace(5.0, 5.2, nx), np.linspace(50.0, 50.2, ny))
    ds = xr.Dataset(
        {
            "xch4": (("y", "x"), np.full((ny, nx), 1900.0), {"units": "ppb"}),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        attrs={"time_coverage_start": "2022-03-04T05:06:07"},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture
def tropomi_file(tmp_path: Path) -> Path:
    return write_synthetic_tropomi(tmp_path / "S5P_synthetic_L2__CH4____a.nc")


@pytest.fixture
def emit_files(tmp_path: Path) -> tuple[Path, Path]:
    raster = tmp_path / "EMIT_L2B_CH4ENH.nc"
    glt = tmp_path / "EMIT_GLT.nc"
    write_synthetic_emit(raster, glt)
    return raster, glt


# ---- TROPOMI opener -------------------------------------------------------


def test_open_tropomi_flattens_groups_and_renames(tropomi_file: Path):
    ds = open_tropomi_ch4_l2(tropomi_file, qa_min=None)

    assert ds["xch4"].dims == ("time", "scanline", "ground_pixel")
    assert ds.sizes == {
        "time": 1,
        "scanline": 4,
        "ground_pixel": 3,
        "layer": 12,
        "level": 13,
        "corner": 4,
    }
    assert ds["column_averaging_kernel"].dims == (
        "time",
        "scanline",
        "ground_pixel",
        "layer",
    )
    for canonical in TROPOMI_CANONICAL.values():
        assert canonical in ds.data_vars
    for raw in TROPOMI_CANONICAL:
        if raw not in TROPOMI_CANONICAL.values():
            assert raw not in ds.variables
    # Support-group variables without a registry entry keep their names.
    assert "pressure_interval" in ds.data_vars
    assert "surface_albedo_SWIR" in ds.data_vars
    assert "solar_zenith_angle" in ds.data_vars
    # 2-D lon/lat coordinates.
    assert ds["lon"].dims == ("time", "scanline", "ground_pixel")
    assert "lon" in ds.coords and "lat" in ds.coords
    assert "latitude" not in ds.variables
    assert ds.attrs["title"] == "synthetic S5P CH4 L2"


def test_open_tropomi_decodes_time(tropomi_file: Path):
    ds = open_tropomi_ch4_l2(tropomi_file, qa_min=None)

    assert np.issubdtype(ds["time"].dtype, np.datetime64)
    assert ds["time"].values[0] == np.datetime64(REF_TIME)
    assert ds["scanline_time"].dims == ("time", "scanline")
    assert np.issubdtype(ds["scanline_time"].dtype, np.datetime64)
    expected = np.datetime64(REF_TIME) + np.arange(4) * np.timedelta64(1, "s")
    np.testing.assert_array_equal(ds["scanline_time"].values[0], expected)
    assert "delta_time" not in ds.variables


def test_open_tropomi_decodes_relative_delta_time(tmp_path: Path):
    path = write_synthetic_tropomi(tmp_path / "rel.nc", absolute_delta_time=False)
    ds = open_tropomi_ch4_l2(path, qa_min=None)

    expected = np.datetime64(REF_TIME) + np.arange(4) * np.timedelta64(1, "s")
    np.testing.assert_array_equal(ds["scanline_time"].values[0], expected)


def test_open_tropomi_keep_groups_keeps_raw_names(tropomi_file: Path):
    ds = open_tropomi_ch4_l2(tropomi_file, keep_groups=True)

    assert "methane_mixing_ratio" in ds.data_vars
    assert "xch4" not in ds.data_vars
    assert "latitude" in ds.variables and "lat" not in ds.variables
    # Flattening and time decoding still happen.
    assert "column_averaging_kernel" in ds.data_vars
    assert "scanline_time" in ds.coords


def test_open_tropomi_qa_min_masks_planted_pixels(tropomi_file: Path):
    ds = open_tropomi_ch4_l2(tropomi_file, qa_min=0.5)

    masked = np.isnan(ds["xch4"].values[0])
    expected = np.zeros((4, 3), dtype=bool)
    for s, p in LOW_QA_PIXELS:
        expected[s, p] = True
    np.testing.assert_array_equal(masked, expected)
    # Layered variables are masked on the same pixels.
    ak_masked = np.isnan(ds["column_averaging_kernel"].values[0]).all(axis=-1)
    np.testing.assert_array_equal(ak_masked, expected)
    # Shape is preserved (mask, not drop).
    assert ds.sizes["scanline"] == 4 and ds.sizes["ground_pixel"] == 3


def test_open_tropomi_default_qa_is_half(tropomi_file: Path):
    ds = open_tropomi_ch4_l2(tropomi_file)
    assert int(np.isnan(ds["xch4"].values).sum()) == len(LOW_QA_PIXELS)


def test_open_tropomi_qa_none_keeps_all(tropomi_file: Path):
    ds = open_tropomi_ch4_l2(tropomi_file, qa_min=None)
    assert not np.isnan(ds["xch4"].values).any()


def test_tropomi_renamed_variables_round_trip_registry(tropomi_file: Path):
    ds = open_tropomi_ch4_l2(tropomi_file, qa_min=None)
    for raw, canonical in TROPOMI_CANONICAL.items():
        var = resolve(canonical)
        assert var.for_source("tropomi") == raw
        stamped = apply_cf_attrs(ds[canonical], canonical)
        assert stamped.attrs["long_name"] == var.long_name
        if var.units is not None:
            assert stamped.attrs["units"] == var.units


# ---- EMIT / GHGSat openers ------------------------------------------------


def test_open_emit_with_glt_orthorectifies(emit_files: tuple[Path, Path]):
    raster, glt = emit_files
    ds = open_emit_ch4_l2b(raster, glt_path=glt)

    assert ds["ch4_enhancement"].dims == ("time", "y", "x")
    assert ds["ch4_enhancement"].attrs["units"] == "ppm m"
    assert ds["lon"].dims == ("y", "x") and ds["lat"].dims == ("y", "x")
    enh = ds["ch4_enhancement"].values[0]
    raw = np.arange(12, dtype=float).reshape(3, 4)
    # Columns mirrored by the GLT; cell (0, 0) is no-data.
    assert np.isnan(enh[0, 0])
    np.testing.assert_array_equal(enh[1:], raw[1:, ::-1])
    np.testing.assert_array_equal(enh[0, 1:], raw[0, ::-1][1:])
    # Uncertainty rides along and the geotransform gives cell centres.
    assert "ch4_uncertainty" in ds.data_vars
    assert ds["lon"].values[0, 0] == pytest.approx(-100.0 + 0.005)
    assert ds["lat"].values[0, 0] == pytest.approx(35.0 - 0.005)
    assert ds["time"].values[0] == np.datetime64("2023-08-01T10:00:00")


def test_open_emit_without_glt_keeps_raw_grid(emit_files: tuple[Path, Path]):
    raster, _ = emit_files
    ds = open_emit_ch4_l2b(raster)
    assert ds["ch4_enhancement"].dims == ("time", "y", "x")
    assert "lon" not in ds.coords


def test_open_emit_glt_lonlat_variables_take_precedence(tmp_path: Path):
    raster, glt_path = tmp_path / "r.nc", tmp_path / "g.nc"
    write_synthetic_emit(raster, glt_path)
    glt = xr.open_dataset(glt_path).load()
    glt = glt.assign(
        lon=(("ortho_y", "ortho_x"), np.full((3, 4), 7.0)),
        lat=(("ortho_y", "ortho_x"), np.full((3, 4), 8.0)),
    )
    glt.to_netcdf(tmp_path / "g2.nc")
    ds = open_emit_ch4_l2b(raster, glt_path=tmp_path / "g2.nc")
    assert float(ds["lon"].values[1, 1]) == 7.0


def test_open_emit_glt_without_geolocation_raises(tmp_path: Path):
    raster, glt_path = tmp_path / "r.nc", tmp_path / "g.nc"
    write_synthetic_emit(raster, glt_path)
    glt = xr.open_dataset(glt_path).load()
    glt.attrs = {}
    glt.to_netcdf(tmp_path / "g2.nc")
    with pytest.raises(ValueError, match="geotransform"):
        open_emit_ch4_l2b(raster, glt_path=tmp_path / "g2.nc")


def test_open_emit_missing_variable_raises(tmp_path: Path):
    xr.Dataset({"other": (("y", "x"), np.zeros((2, 2)))}).to_netcdf(tmp_path / "e.nc")
    with pytest.raises(KeyError, match="ch4_enhancement"):
        open_emit_ch4_l2b(tmp_path / "e.nc")


def test_open_emit_accepts_methane_plume_complex(emit_files: tuple[Path, Path]):
    # The distributed EMIT_L2B_CH4ENH files name the raster this way.
    raster, glt = emit_files
    renamed = raster.with_name("plume.nc")
    xr.open_dataset(raster).load().rename(
        ch4_enhancement="methane_plume_complex"
    ).to_netcdf(renamed)
    ds = open_emit_ch4_l2b(renamed, glt_path=glt)
    assert "ch4_enhancement" in ds.data_vars
    assert "methane_plume_complex" not in ds.variables
    assert ds["ch4_enhancement"].attrs["units"] == "ppm m"


def test_open_emit_geotransform_applies_rotation(emit_files: tuple[Path, Path]):
    raster, glt_path = emit_files
    glt = xr.open_dataset(glt_path).load()
    # GDAL order [ulx, a, b, uly, d, e]; b and d are the rotation terms.
    glt.attrs["geotransform"] = [-100.0, 0.01, 0.002, 35.0, 0.003, -0.01]
    rotated = glt_path.with_name("rot.nc")
    glt.to_netcdf(rotated)
    ds = open_emit_ch4_l2b(raster, glt_path=rotated)
    # Cell (row jj=1, col ii=2), centre offsets 1.5 / 2.5.
    assert ds["lon"].values[1, 2] == pytest.approx(-100.0 + 2.5 * 0.01 + 1.5 * 0.002)
    assert ds["lat"].values[1, 2] == pytest.approx(35.0 + 2.5 * 0.003 - 1.5 * 0.01)


def test_open_emit_glt_keeps_scalar_time_variable(emit_files: tuple[Path, Path]):
    raster, glt = emit_files
    ds = xr.open_dataset(raster).load()
    ds.attrs.pop("time_coverage_start")  # only the scalar variable is left
    ds = ds.assign(time=np.datetime64("2023-08-02T11:00:00"))
    timed = raster.with_name("timed.nc")
    ds.to_netcdf(timed)
    out = open_emit_ch4_l2b(timed, glt_path=glt)
    assert out.sizes["time"] == 1
    assert out["ch4_enhancement"].dims == ("time", "y", "x")
    assert out["time"].values[0] == np.datetime64("2023-08-02T11:00:00")


def test_open_ghgsat_promotes_lonlat(tmp_path: Path):
    path = write_synthetic_ghgsat(tmp_path / "plume.nc")
    ds = open_ghgsat_ch4_l2(path)

    assert ds["xch4"].dims == ("time", "y", "x")
    assert ds["lon"].dims == ("y", "x") and "lon" in ds.coords
    assert "latitude" not in ds.variables
    assert ds["xch4"].attrs["standard_name"] == resolve("xch4").standard_name
    assert ds["xch4"].attrs["units"] == "ppb"  # existing attrs win
    assert ds["time"].values[0] == np.datetime64("2022-03-04T05:06:07")


def test_open_ghgsat_missing_variables_raises(tmp_path: Path):
    xr.Dataset({"other": (("y", "x"), np.zeros((2, 2)))}).to_netcdf(tmp_path / "g.nc")
    with pytest.raises(KeyError, match="xch4"):
        open_ghgsat_ch4_l2(tmp_path / "g.nc")


# ---- LocalL2Source --------------------------------------------------------


def test_source_describe_and_catalog():
    src = LocalL2Source()
    info = src.describe("tropomi.ch4")
    assert info.kind is DatasetKind.SWATH
    assert info.source == "local"
    assert src.describe("emit.ch4").kind is DatasetKind.SCENE
    assert {d.dataset_id for d in src.list_datasets()} == {
        "tropomi.ch4",
        "emit.ch4",
        "ghgsat.ch4",
    }
    for name in ("tropomi.ch4", "emit.ch4", "ghgsat.ch4"):
        assert CATALOG[name].source == "local"
        assert describe(name) is src.describe(name)
    with pytest.raises(KeyError):
        src.describe("nope")


def test_source_download_raises(tmp_path: Path):
    with pytest.raises(NotImplementedError):
        LocalL2Source().download("tropomi.ch4", tmp_path / "out.nc")


def test_source_open_requires_exactly_one_path(tropomi_file: Path):
    src = LocalL2Source()
    with pytest.raises(ValueError, match="exactly one"):
        src.open("tropomi.ch4")
    with pytest.raises(ValueError, match="exactly one"):
        src.open("tropomi.ch4", path=tropomi_file, paths=[tropomi_file])
    with pytest.raises(KeyError):
        src.open("nope", path=tropomi_file)


def test_source_open_bbox_uses_2d_coords(tropomi_file: Path):
    # Swath: lon = 10, 11, 12 along ground_pixel; lat = 40..43 along scanline.
    ds = LocalL2Source().open(
        "tropomi.ch4",
        path=tropomi_file,
        bbox=BBox(lon_min=10.5, lon_max=12.5, lat_min=41.0, lat_max=42.0),
        qa_min=0.0,  # None would leave the opener's 0.5 default in force
    )
    assert ds.sizes["ground_pixel"] == 2
    assert ds.sizes["scanline"] == 2
    np.testing.assert_array_equal(np.unique(ds["lon"].values), [11.0, 12.0])
    np.testing.assert_array_equal(np.unique(ds["lat"].values), [41.0, 42.0])
    assert not np.isnan(ds["xch4"].values).any()


def test_source_open_bbox_rejects_antimeridian(tropomi_file: Path):
    with pytest.raises(ValueError, match="antimeridian"):
        LocalL2Source().open(
            "tropomi.ch4",
            path=tropomi_file,
            bbox=BBox(lon_min=170.0, lon_max=-170.0, lat_min=0.0, lat_max=1.0),
        )


def test_source_open_bbox_360_box_on_180_dataset(tmp_path: Path):
    # Swath lon = -12, -11, -10; a 350..355 box means -10..-5 here.
    path = write_synthetic_tropomi(tmp_path / "west.nc", lon0=-12.0)
    ds = LocalL2Source().open(
        "tropomi.ch4",
        path=path,
        bbox=BBox(lon_min=350.0, lon_max=355.0, lat_min=40.0, lat_max=43.0),
        qa_min=0.0,
    )
    assert ds.sizes["ground_pixel"] == 1
    np.testing.assert_array_equal(np.unique(ds["lon"].values), [-10.0])


def test_source_open_bbox_180_box_on_360_dataset(tmp_path: Path):
    # Swath lon = 350, 351, 352; a -9..-8 box means 351..352 here.
    path = write_synthetic_tropomi(tmp_path / "east.nc", lon0=350.0)
    ds = LocalL2Source().open(
        "tropomi.ch4",
        path=path,
        bbox=BBox(lon_min=-9.0, lon_max=-8.0, lat_min=40.0, lat_max=43.0),
        qa_min=0.0,
    )
    np.testing.assert_array_equal(np.unique(ds["lon"].values), [351.0, 352.0])


def test_source_open_forwards_qa_and_variables(tropomi_file: Path):
    ds = LocalL2Source().open(
        "tropomi.ch4", path=tropomi_file, variables=["xch4", "qa_value"], qa_min=0.5
    )
    assert set(ds.data_vars) == {"xch4", "qa_value"}
    assert int(np.isnan(ds["xch4"].values).sum()) == len(LOW_QA_PIXELS)
    assert "lon" in ds.coords


def test_source_open_time_masks_scanlines(tropomi_file: Path):
    ds = LocalL2Source().open(
        "tropomi.ch4",
        path=tropomi_file,
        time=TimeRange.parse("2019-06-01T00:00:01", "2019-06-01T00:00:02"),
        qa_min=None,
    )
    assert ds.sizes["scanline"] == 2
    assert ds["scanline_time"].values[0, 0] == np.datetime64("2019-06-01T00:00:01")


def test_source_open_paths_concatenates_along_time(tmp_path: Path):
    a = write_synthetic_tropomi(tmp_path / "a.nc", lon0=10.0)
    b = write_synthetic_tropomi(tmp_path / "b.nc", lon0=20.0)
    ds = LocalL2Source().open("tropomi.ch4", paths=[b, a], qa_min=None)

    assert ds.sizes["time"] == 2
    assert float(ds["lon"].values[0, 0, 0]) == 20.0  # order as given
    assert float(ds["lon"].values[1, 0, 0]) == 10.0
    assert ds["scanline_time"].shape == (2, 4)


def test_source_open_emit_scenes_concatenate(tmp_path: Path):
    r1, g1 = tmp_path / "r1.nc", tmp_path / "g1.nc"
    r2, g2 = tmp_path / "r2.nc", tmp_path / "g2.nc"
    write_synthetic_emit(r1, g1)
    write_synthetic_emit(r2, g2)
    ds = LocalL2Source().open("emit.ch4", paths=[r1, r2], glt_paths=[g1, g2])
    assert ds["ch4_enhancement"].dims == ("time", "y", "x")
    assert ds.sizes["time"] == 2
    assert ds["lon"].dims == ("y", "x")  # both scenes went through their GLT
    # Scenes are sliced on the time index.
    sub = LocalL2Source().open(
        "emit.ch4",
        path=r1,
        glt_path=g1,
        time=TimeRange.parse("2023-08-01", "2023-08-02"),
    )
    assert sub.sizes["time"] == 1
    empty = LocalL2Source().open(
        "emit.ch4",
        path=r1,
        glt_path=g1,
        time=TimeRange.parse("2020-01-01", "2020-01-02"),
    )
    assert empty.sizes["time"] == 0


def test_source_open_glt_paths_must_line_up_with_paths(tmp_path: Path):
    r1, g1 = tmp_path / "r1.nc", tmp_path / "g1.nc"
    r2, g2 = tmp_path / "r2.nc", tmp_path / "g2.nc"
    write_synthetic_emit(r1, g1)
    write_synthetic_emit(r2, g2)
    src = LocalL2Source()
    with pytest.raises(ValueError, match="glt_paths has 1 entries but paths has 2"):
        src.open("emit.ch4", paths=[r1, r2], glt_paths=[g1])
    # One lookup table cannot serve several scenes.
    with pytest.raises(ValueError, match="pass glt_paths="):
        src.open("emit.ch4", paths=[r1, r2], glt_path=g1)
    with pytest.raises(ValueError, match="not both"):
        src.open("emit.ch4", paths=[r1], glt_path=g1, glt_paths=[g1])
    # A single scene still takes glt_path= as before.
    assert "lon" in src.open("emit.ch4", paths=[r1], glt_path=g1).coords


def test_source_open_scene_time_masks_out_of_order_paths(tmp_path: Path):
    early, glt = tmp_path / "early.nc", tmp_path / "g.nc"
    write_synthetic_emit(early, glt)
    late = tmp_path / "late.nc"
    ds = xr.open_dataset(early).load()
    ds.attrs["time_coverage_start"] = "2023-09-15T10:00:00Z"
    ds.to_netcdf(late)
    # Later scene first: the ``time`` index is not monotonic, so a label
    # slice would fail — the window is a mask on the coordinate instead.
    out = LocalL2Source().open(
        "emit.ch4",
        paths=[late, early],
        glt_paths=[glt, glt],
        time=TimeRange.parse("2023-08-01", "2023-08-02"),
    )
    assert out.sizes["time"] == 1
    assert out["time"].values[0] == np.datetime64("2023-08-01T10:00:00")
