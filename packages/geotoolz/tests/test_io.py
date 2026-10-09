"""Tests for geotoolz.io operators."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from pipekit import Identity, Operator, Sequential
from pyproj import CRS
from rasterio.transform import array_bounds, from_origin
from rasterio.windows import Window
from shapely.geometry import box

import geotoolz as gz
from geotoolz import io
from geotoolz.io._src import operators as io_operators


def _sample_geotensor() -> GeoTensor:
    values = np.arange(2 * 4 * 5, dtype=np.int16).reshape(2, 4, 5)
    transform = from_origin(100.0, 200.0, 10.0, 10.0)
    return GeoTensor(
        values, transform=transform, crs="EPSG:32631", fill_value_default=-9999
    )


def _cog_test_geotensor() -> GeoTensor:
    values = np.arange(64 * 64, dtype=np.int16).reshape(1, 64, 64)
    transform = from_origin(100.0, 740.0, 10.0, 10.0)
    return GeoTensor(
        values, transform=transform, crs="EPSG:32631", fill_value_default=-9999
    )


def test_io_module_is_exported() -> None:
    assert gz.io is io
    assert io.ReadBounds is not None
    assert io.ReadHDF is not None
    assert io.ReadNetCDF is not None


def test_write_geotiff_then_read_bounds_roundtrips(
    tmp_path: Path,
) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"

    assert io.WriteGeoTIFF(path=path)(gt) is None
    bounds = array_bounds(gt.shape[-2], gt.shape[-1], gt.transform)
    out = io.ReadBounds(src=path, bounds=bounds, crs="EPSG:32631", indexes=[2, 1])()

    np.testing.assert_array_equal(out.values, gt.values[[1, 0]])
    assert out.shape == (2, 4, 5)
    assert out.transform == gt.transform
    assert out.crs == gt.crs
    assert out.fill_value_default == -9999


def test_read_bounds_without_indexes_reads_all_bands_in_order(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    bounds = array_bounds(gt.shape[-2], gt.shape[-1], gt.transform)
    out = io.ReadBounds(src=path, bounds=bounds, crs="EPSG:32631")()

    np.testing.assert_array_equal(out.values, gt.values)


def test_read_hdf5_reads_dataset_indexes_and_metadata(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sample.h5"
    values = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("EV_1KM_RefSB", data=values)
        dataset.attrs["_FillValue"] = -9999
        file.create_dataset("Latitude", data=np.ones((3, 4), dtype=np.float32))
        file.create_dataset("Longitude", data=np.zeros((3, 4), dtype=np.float32))
        metadata = file.create_group("metadata")
        metadata.attrs["sensor"] = "MODIS"

    out = io.ReadHDF(
        path=path,
        dataset="EV_1KM_RefSB",
        indexes=[2],
        geolocation=("Latitude", "Longitude"),
        metadata_groups=["metadata"],
    )()

    np.testing.assert_array_equal(out.values, values[1:2])
    assert out.fill_value_default == -9999
    assert out.attrs["metadata"]["metadata"]["sensor"] == "MODIS"
    np.testing.assert_array_equal(
        out.attrs["geolocation"]["latitude"], np.ones((3, 4), dtype=np.float32)
    )


def test_read_hdf5_missing_dependency_has_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n")
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "h5py":
            raise ImportError("missing h5py")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=r"geotoolz\[hdf5\]"):
        io.ReadHDF(path=path, dataset="data")()


def test_read_netcdf_decodes_cf_and_recovers_grid_mapping(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "sample.nc"
    with netcdf4.Dataset(path, "w") as root:
        group = root.createGroup("PRODUCT")
        group.createDimension("band", 2)
        group.createDimension("y", 2)
        group.createDimension("x", 3)
        crs_var = group.createVariable("crs", "i4")
        crs_var.crs_wkt = CRS.from_epsg(4326).to_wkt()
        # GDAL's netCDF driver writes GeoTransform on the grid_mapping variable.
        crs_var.GeoTransform = "10 1 0 20 0 -1"
        variable = group.createVariable(
            "methane_mixing_ratio_bias_corrected",
            "i2",
            ("band", "y", "x"),
            fill_value=-9999,
        )
        variable.scale_factor = 0.5
        variable.add_offset = 10.0
        variable.grid_mapping = "crs"
        variable.set_auto_maskandscale(False)
        variable[:] = np.arange(12, dtype=np.int16).reshape(2, 2, 3)

    out = io.ReadNetCDF(
        path=path,
        group="PRODUCT",
        variable="methane_mixing_ratio_bias_corrected",
        indexes=[2],
    )()

    np.testing.assert_allclose(out.values, np.arange(6, 12).reshape(1, 2, 3) * 0.5 + 10)
    assert out.crs == CRS.from_epsg(4326)
    assert tuple(out.transform)[:6] == (1.0, 0.0, 10.0, 0.0, -1.0, 20.0)
    # Decoded values are scaled floats, so the fill sentinel is NaN, not -9999.
    assert np.isnan(out.fill_value_default)


def test_read_netcdf_decodes_fill_value_to_nan(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "fill.nc"
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("band", 1)
        root.createDimension("y", 2)
        root.createDimension("x", 2)
        variable = root.createVariable(
            "values",
            "i2",
            ("band", "y", "x"),
            fill_value=-9999,
        )
        variable.scale_factor = 0.5
        variable.add_offset = 1.0
        variable.set_auto_maskandscale(False)
        variable[:] = np.array([[[1, -9999], [3, 4]]], dtype=np.int16)

    out = io.ReadNetCDF(path=path, variable="values", indexes=[1])()

    assert out.values.dtype.kind == "f"
    assert np.isnan(out.values[0, 0, 1])
    assert out.values[0, 0, 0] == 1 * 0.5 + 1.0
    assert out.values[0, 1, 0] == 3 * 0.5 + 1.0
    assert out.values[0, 1, 1] == 4 * 0.5 + 1.0


def test_read_netcdf_allows_indexes_on_two_dim_variable(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "twod.nc"
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("y", 2)
        root.createDimension("x", 3)
        variable = root.createVariable("values", "f4", ("y", "x"))
        variable[:] = np.arange(6, dtype=np.float32).reshape(2, 3)

    out = io.ReadNetCDF(path=path, variable="values", indexes=[1])()
    np.testing.assert_array_equal(
        out.values, np.arange(6, dtype=np.float32).reshape(2, 3)
    )


def test_read_netcdf_missing_dependency_has_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.nc"
    path.write_bytes(b"not a real netcdf")
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "netCDF4":
            raise ImportError("missing netCDF4")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=r"geotoolz\[netcdf\]"):
        io.ReadNetCDF(path=path, variable="data")()


def test_source_operator_can_start_sequential_without_input(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    out = Sequential(
        [
            io.ReadWindow(src=path, window=Window(1, 1, 2, 2), indexes=[1]),
            Identity(),
        ]
    )()

    np.testing.assert_array_equal(out.values, gt.values[:1, 1:3, 1:3])


def test_read_window_accepts_reader_source_and_rejects_indexed_objects(
    tmp_path: Path,
) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)
    reader = RasterioReader(str(path))

    out = io.ReadWindow(src=reader, window=Window(0, 0, 2, 2), indexes=[2])()
    np.testing.assert_array_equal(out.values, gt.values[1:2, :2, :2])

    with pytest.raises(io.GeoToolzIOError, match="indexes are only supported"):
        io.ReadWindow(src=object(), window=Window(0, 0, 1, 1), indexes=[1])()


def test_read_window_delegates_custom_sources_without_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gt = _sample_geotensor()
    source = object()
    window = Window(0, 0, 1, 1)

    def fake_read_from_window(src, window_arg, boundless=True):
        assert src is source
        assert window_arg == window
        assert boundless is True
        return gt

    monkeypatch.setattr(io_operators.read, "read_from_window", fake_read_from_window)

    assert io.ReadWindow(src=source, window=window)() is gt


def test_read_window_outside_source_raises_clear_error(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    with pytest.raises(io.GeoToolzIOError, match="does not intersect"):
        io.ReadWindow(
            src=path,
            window=Window(100, 100, 2, 2),
            boundless=False,
        )()


def test_read_window_accepts_tuple_config(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    op = io.ReadWindow(src=path, window=(1, 1, 2, 2), indexes=[1])
    out = op()

    np.testing.assert_array_equal(out.values, gt.values[:1, 1:3, 1:3])
    assert op.get_config()["window"] == (1, 1, 2, 2)


def test_read_center_coords_and_polygon(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    centered = io.ReadCenterCoords(
        src=path,
        center=(120.0, 180.0),
        shape=(2, 2),
        crs="EPSG:32631",
        indexes=[1],
    )()
    polygon = io.ReadPolygon(
        src=path,
        polygon=box(110.0, 170.0, 130.0, 190.0),
        crs="EPSG:32631",
        indexes=[1],
    )()

    np.testing.assert_array_equal(centered.values, gt.values[:1, 1:3, 1:3])
    np.testing.assert_array_equal(polygon.values, gt.values[:1, 1:3, 1:3])


def test_reprojecting_readers_match_reference_grid(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)
    bounds = array_bounds(gt.shape[-2], gt.shape[-1], gt.transform)

    like = io.ReadReprojectLike(src=path, like=gt, indexes=[1])()
    to_crs = io.ReadToCRS(
        src=path,
        dst_crs="EPSG:32631",
        bounds=bounds,
        resolution=(10.0, 10.0),
        indexes=[1],
    )()
    whole = io.ReadToCRS(src=path, dst_crs="EPSG:32631", indexes=[1])()

    assert like.shape == (1, 4, 5)
    assert like.transform == gt.transform
    assert like.crs == gt.crs
    assert to_crs.crs == gt.crs
    assert to_crs.shape[-2:] == gt.shape[-2:]
    assert whole.crs == gt.crs
    assert whole.shape[-2:] == gt.shape[-2:]


def test_write_cog_writes_readable_cog(tmp_path: Path) -> None:
    gt = _cog_test_geotensor()
    path = tmp_path / "sample_cog.tif"

    assert io.WriteCOG(path=path, compress="deflate")(gt) is None
    out = io.ReadBounds(
        src=path,
        bounds=array_bounds(gt.shape[-2], gt.shape[-1], gt.transform),
        crs="EPSG:32631",
    )()

    np.testing.assert_array_equal(out.values, gt.values)


def test_write_geotiff_handles_2d_data_profile_and_invalid_shapes(
    tmp_path: Path,
) -> None:
    gt = _sample_geotensor()
    two_dim = GeoTensor(
        gt.values[0],
        transform=gt.transform,
        crs=gt.crs,
        fill_value_default=None,
    )
    path = tmp_path / "two_dim.tif"

    io.WriteGeoTIFF(path=path, profile={"compress": "lzw"})(two_dim)
    out = io.ReadBounds(
        src=path,
        bounds=array_bounds(two_dim.shape[-2], two_dim.shape[-1], two_dim.transform),
        crs="EPSG:32631",
    )()
    np.testing.assert_array_equal(out.values, two_dim.values[np.newaxis, ...])

    invalid = SimpleNamespace(
        values=np.zeros((1, 1, 1, 1), dtype=np.uint8),
        crs=gt.crs,
        transform=gt.transform,
        fill_value_default=None,
    )
    with pytest.raises(io.GeoToolzIOError, match="expects 2D or 3D"):
        io.WriteGeoTIFF(path=tmp_path / "invalid.tif")(invalid)


def test_sink_operator_is_only_valid_at_end_of_sequential() -> None:
    assert io.WriteGeoTIFF(path="out.tif")._terminal is True
    with pytest.raises(TypeError, match="terminal operator"):
        Sequential([io.WriteGeoTIFF(path="out.tif"), Identity()])


def test_missing_source_raises_geotoolz_io_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.tif"

    with pytest.raises(io.GeoToolzIOError, match="Unable to read raster source"):
        io.ReadBounds(src=missing, bounds=(0.0, 0.0, 1.0, 1.0))()


def test_load_from_stac_reads_asset_href(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "asset.tif"
    io.WriteGeoTIFF(path=path)(gt)
    item = SimpleNamespace(assets={"visual": SimpleNamespace(href=str(path))})

    out = io.LoadFromSTAC(item=item, asset_key="visual")()

    np.testing.assert_array_equal(out.values, gt.values)
    assert out.transform == gt.transform


def test_operator_configs_are_serializable_for_common_values() -> None:
    """Path / bounds / window / asset-ID configs survive a JSON state round-trip.

    A source given as a runtime object is the opposite case; see
    ``test_io_source_given_as_object_is_refused_by_from_state``.
    """
    polygon = box(0.0, 0.0, 1.0, 1.0)
    ops = [
        io.ReadBounds(src="x.tif", bounds=(0.0, 0.0, 1.0, 1.0)),
        io.ReadCenterCoords(src="x.tif", center=(0.5, 0.5), shape=(2, 2)),
        io.ReadTile(src="x.tif", tile=(1, 0, 0)),
        io.ReadPolygon(src="x.tif", polygon=polygon),
        io.ReadToCRS(src="x.tif", dst_crs="EPSG:4326"),
        io.WriteCOG(path="x.tif"),
        io.WriteGeoTIFF(path="x.tif"),
        io.WriteZarr(store="x.zarr", group="data", chunks={"y": 16, "x": 16}),
        io.LoadFromEE(
            image_id="LANDSAT/LC08/C02/T1_L2/LC08_001001_20200101",
            bounds=(0.0, 0.0, 1.0, 1.0),
            crs="EPSG:4326",
            scale=30.0,
            bands=["B4"],
        ),
    ]

    for op in ops:
        state = json.loads(json.dumps(op.state, allow_nan=False))
        clone = Operator.from_state(state)
        assert type(clone) is type(op)
        assert clone.get_config() == op.get_config()


def test_write_zarr_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gt = _sample_geotensor()

    import builtins
    from typing import Any

    real_import = builtins.__import__

    def _raise_for_zarr(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "zarr" or name.startswith("zarr."):
            raise ImportError("simulated missing zarr")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raise_for_zarr)

    with pytest.raises(io.GeoToolzIOError, match="optional zarr dependency"):
        io.WriteZarr(store="memory://out.zarr")(gt)


@pytest.mark.skipif(
    importlib.util.find_spec("ee") is not None,
    reason="earthengine-api installed (workspace [gee] extra) — the "
    "missing-dependency error path is not exercisable",
)
def test_load_from_ee_reports_missing_optional_dependencies() -> None:
    op = io.LoadFromEE(
        image_id="asset",
        bounds=(0.0, 0.0, 1.0, 1.0),
        crs="EPSG:4326",
        scale=30.0,
        bands=["B4"],
    )

    with pytest.raises(io.GeoToolzIOError, match="Earth Engine dependencies"):
        op()


# ---------------------------------------------------------------------------
# Round-trip discipline: forbid_in_yaml flags + hydra-zen builds round-trip
# ---------------------------------------------------------------------------


_RUNTIME_IO_OPERATOR_CLASSES = (io.ReadReprojectLike, io.LoadFromSTAC)


@pytest.mark.parametrize(
    "op_cls",
    [
        io.ReadWindow,
        io.ReadBounds,
        io.ReadCenterCoords,
        io.ReadTile,
        io.ReadPolygon,
        io.ReadToCRS,
        io.ReadHDF,
        io.ReadNetCDF,
        io.WriteCOG,
        io.WriteGeoTIFF,
        io.WriteZarr,
        io.LoadFromEE,
    ],
)
def test_path_configured_io_operators_round_trip(op_cls: type) -> None:
    """Paths, bounds, windows and asset IDs are plain JSON (#140)."""
    assert op_cls.forbid_in_yaml is False


@pytest.mark.parametrize("op_cls", _RUNTIME_IO_OPERATOR_CLASSES)
def test_runtime_io_operators_are_forbid_in_yaml(op_cls: type) -> None:
    """A STAC item or a reference grid is a runtime object."""
    assert op_cls.forbid_in_yaml is True


def test_io_source_given_as_object_is_refused_by_from_state() -> None:
    reader = object()
    op = io.ReadWindow(src=reader, window=(0, 0, 4, 4))
    with pytest.raises(RuntimeError, match="non-primitive"):
        Operator.from_state(op.state)


def test_write_operators_are_terminal() -> None:
    assert io.WriteCOG._terminal is True
    assert io.WriteGeoTIFF._terminal is True
    assert io.WriteZarr._terminal is True


def test_write_cog_passes_its_options_to_geocloud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`WriteCOG` forwards every option to `geocloud.cog.write_cog`."""
    import geocloud.cog

    captured: dict[str, object] = {}

    def fake_write_cog(data, dest, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs, dest=dest)

    monkeypatch.setattr(geocloud.cog, "write_cog", fake_write_cog)

    io.WriteCOG(
        path="s3://bucket/out.tif",
        compress="zstd",
        descriptions=["b1", "b2"],
        tags={"source": "test"},
        creation_options={"NUM_THREADS": "2"},
    )(_sample_geotensor())

    assert captured["dest"] == "s3://bucket/out.tif"  # a URI is not mangled
    assert captured["compress"] == "zstd"
    assert captured["descriptions"] == ["b1", "b2"]
    assert captured["tags"] == {"source": "test"}
    assert captured["creation_options"] == {"NUM_THREADS": "2"}
    assert captured["nodata"] == "auto"


def test_write_geotiff_passes_blocksize_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`WriteGeoTIFF` should forward blocksize / descriptions / tags to
    `save.save_tiled_geotiff` rather than re-implement IO via rasterio."""
    captured: dict[str, object] = {}

    def fake_save_tiled(
        data,  # type: ignore[no-untyped-def]
        path,
        *,
        profile_arg,
        descriptions,
        tags,
        blocksize,
    ):
        captured["blocksize"] = blocksize
        captured["descriptions"] = descriptions
        captured["tags"] = tags
        captured["profile_arg"] = profile_arg

    monkeypatch.setattr(io_operators.save, "save_tiled_geotiff", fake_save_tiled)

    gt = _sample_geotensor()
    io.WriteGeoTIFF(
        path=tmp_path / "out.tif",
        blocksize=512,
        descriptions=["b1", "b2"],
        tags={"k": "v"},
        profile={"compress": "zstd"},
    )(gt)

    assert captured["blocksize"] == 512
    assert captured["descriptions"] == ["b1", "b2"]
    assert captured["tags"] == {"k": "v"}
    assert captured["profile_arg"] == {"compress": "zstd"}


try:
    import hydra_zen
except ImportError:  # pragma: no cover - exercised via the [hydra] extra
    hydra_zen = None  # type: ignore[assignment]


_HYDRA_ZEN_OPERATORS: list[gz.Operator] = [
    io.ReadWindow(src="x.tif", window=(0, 0, 4, 4)),
    io.ReadBounds(src="x.tif", bounds=(0.0, 0.0, 1.0, 1.0)),
    io.ReadCenterCoords(src="x.tif", center=(0.5, 0.5), shape=(2, 2)),
    io.ReadTile(src="x.tif", tile=(1, 0, 0)),
    io.ReadToCRS(src="x.tif", dst_crs="EPSG:4326"),
    io.ReadHDF(path="x.h5", dataset="data", indexes=[1]),
    io.ReadNetCDF(path="x.nc", variable="data", group="PRODUCT"),
    io.WriteCOG(path="x.tif"),
    io.WriteGeoTIFF(path="x.tif"),
    io.WriteZarr(store="x.zarr", group="data", chunks={"y": 16, "x": 16}),
    io.LoadFromEE(
        image_id="LANDSAT/LC08/C02/T1_L2/LC08_001001_20200101",
        bounds=(0.0, 0.0, 1.0, 1.0),
        crs="EPSG:4326",
        scale=30.0,
        bands=["B4"],
    ),
]


@pytest.mark.skipif(hydra_zen is None, reason="requires hydra-zen extra")
@pytest.mark.parametrize("op", _HYDRA_ZEN_OPERATORS)
def test_io_hydra_zen_builds_roundtrip(op: gz.Operator) -> None:
    """Operator config dicts must accept ``hydra_zen.builds`` /
    ``instantiate``. Operators whose config includes runtime objects
    (STAC items, shapely geometries, GeoTensor references) are excluded
    because their config is intentionally debug-only —
    ``forbid_in_yaml`` documents that contract."""
    cfg = hydra_zen.builds(type(op), **op.get_config())  # type: ignore[attr-defined]
    restored = hydra_zen.instantiate(cfg)
    assert type(restored) is type(op)
    assert restored.get_config() == op.get_config()  # type: ignore[attr-defined]


def test_sink_mapping_options_round_trip() -> None:
    """Mapping options are emitted as pairs (#140 review)."""
    import json

    ops = [
        io.WriteCOG(
            path="out.tif", creation_options={"BIGTIFF": "YES"}, tags={"a": "1"}
        ),
        io.WriteGeoTIFF(path="out.tif", profile={"nodata": 0}, tags={"a": "1"}),
        io.WriteZarr(store="out.zarr", chunks={"y": 256, "x": 256}),
    ]
    for op in ops:
        clone = Operator.from_state(json.loads(json.dumps(op.state)))
        assert clone.get_config() == op.get_config()
    assert ops[2].get_config()["chunks"] == [["y", 256], ["x", 256]]
    assert Operator.from_state(ops[0].state).creation_options == {"BIGTIFF": "YES"}


# ---------------------------------------------------------------------------
# #127 — non-intersecting boundless=False reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_op",
    [
        pytest.param(
            lambda path: io.ReadBounds(
                src=path, bounds=(1e6, 1e6, 1e6 + 10, 1e6 + 10), boundless=False
            ),
            id="ReadBounds",
        ),
        pytest.param(
            lambda path: io.ReadPolygon(
                src=path, polygon=box(1e6, 1e6, 1e6 + 10, 1e6 + 10), boundless=False
            ),
            id="ReadPolygon",
        ),
        pytest.param(
            lambda path: io.ReadCenterCoords(
                src=path, center=(1e6, 1e6), shape=(2, 2), boundless=False
            ),
            id="ReadCenterCoords",
        ),
    ],
)
def test_read_bounds_outside_raises_io_error(tmp_path: Path, make_op) -> None:
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(_sample_geotensor())

    with pytest.raises(io.GeoToolzIOError, match="does not intersect"):
        make_op(path)()


# ---------------------------------------------------------------------------
# #128 — LoadFromEE bounds are in ``crs``
# ---------------------------------------------------------------------------


def test_load_from_ee_projected_crs_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """UTM bounds must be interpreted in the UTM ``crs``, not EPSG:4326."""
    ee = pytest.importorskip("ee")
    pytest.importorskip("georeader.readers.ee_image")
    import rasterio
    from rasterio.io import MemoryFile

    requests: list[dict] = []

    def fake_get_pixels(params: dict) -> bytes:
        # Stand-in for the Earth Engine server: return a GeoTIFF on the
        # requested grid.
        requests.append(params)
        grid = params["grid"]
        affine = grid["affineTransform"]
        transform = rasterio.Affine(
            affine["scaleX"],
            affine["shearX"],
            affine["translateX"],
            affine["shearY"],
            affine["scaleY"],
            affine["translateY"],
        )
        height, width = grid["dimensions"]["height"], grid["dimensions"]["width"]
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="uint16",
                crs=grid["crsCode"],
                transform=transform,
            ) as dst:
                dst.write(np.ones((1, height, width), dtype=np.uint16))
            return memfile.read()

    monkeypatch.setattr(ee.data, "getPixels", fake_get_pixels)

    bounds = (500_000.0, 4_500_000.0, 500_300.0, 4_500_600.0)
    out = io.LoadFromEE(
        image_id="asset", bounds=bounds, crs="EPSG:32631", scale=30.0, bands=["B4"]
    )()

    (params,) = requests
    # The request grid is anchored at the UTM upper-left corner. georeader's
    # ``window_surrounding=True`` may add one trailing pixel per axis.
    dims = params["grid"]["dimensions"]
    assert dims["height"] in (20, 21)
    assert dims["width"] in (10, 11)
    assert params["grid"]["affineTransform"]["translateX"] == 500_000.0
    assert params["grid"]["affineTransform"]["translateY"] == 4_500_600.0
    assert out.shape == (1, dims["height"], dims["width"])
    assert out.crs == CRS.from_epsg(32631)
    assert tuple(out.transform)[:6] == (30.0, 0.0, 500_000.0, 0.0, -30.0, 4_500_600.0)


# ---------------------------------------------------------------------------
# #130 — reader edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("on_grid_mapping", [True, False])
def test_read_netcdf_geotransform_on_grid_mapping(
    tmp_path: Path, on_grid_mapping: bool
) -> None:
    """GDAL writes GeoTransform on the grid_mapping variable; the data
    variable is only a fallback."""
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "gdal.nc"
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("y", 2)
        root.createDimension("x", 3)
        crs_var = root.createVariable("spatial_ref", "i4")
        crs_var.crs_wkt = CRS.from_epsg(32631).to_wkt()
        variable = root.createVariable("values", "f4", ("y", "x"))
        variable.grid_mapping = "spatial_ref"
        target = crs_var if on_grid_mapping else variable
        target.GeoTransform = "500000 10 0 4500000 0 -10"
        variable[:] = np.arange(6, dtype=np.float32).reshape(2, 3)

    out = io.ReadNetCDF(path=path, variable="values")()

    assert out.crs == CRS.from_epsg(32631)
    assert tuple(out.transform)[:6] == (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)


def test_read_hdf_array_fill_value_scalar(tmp_path: Path) -> None:
    """A ``(1,)``-shaped ``_FillValue`` (netCDF4/xarray style) is a scalar fill."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fill.h5"
    values = np.array([[[1, -9999], [3, 4]]], dtype=np.int16)
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("data", data=values)
        dataset.attrs["_FillValue"] = np.array([-9999], dtype=np.int16)

    out = io.ReadHDF(path=path, dataset="data")()

    assert out.fill_value_default == -9999
    assert np.ndim(out.fill_value_default) == 0
    np.testing.assert_array_equal(out.validmask().values, values != -9999)


def test_read_netcdf_nan_fill_consistent(tmp_path: Path) -> None:
    """decode_cf NaN-fills masked values, so the fill sentinel must be NaN;
    undecoded reads keep the raw ``_FillValue``."""
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "fill.nc"
    raw = np.array([[[1, -9999], [3, 4]]], dtype=np.int16)
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("band", 1)
        root.createDimension("y", 2)
        root.createDimension("x", 2)
        variable = root.createVariable(
            "values", "i2", ("band", "y", "x"), fill_value=-9999
        )
        variable.scale_factor = 0.5
        variable.set_auto_maskandscale(False)
        variable[:] = raw

    decoded = io.ReadNetCDF(path=path, variable="values")()
    assert np.isnan(decoded.fill_value_default)
    np.testing.assert_array_equal(np.isnan(decoded.values), raw == -9999)

    undecoded = io.ReadNetCDF(path=path, variable="values", decode_cf=False)()
    assert undecoded.fill_value_default == -9999
    np.testing.assert_array_equal(undecoded.values, raw)


@pytest.mark.parametrize("indexes", [None, [2]])
def test_read_window_open_dataset_source(
    tmp_path: Path, indexes: list[int] | None
) -> None:
    import rasterio

    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    with rasterio.open(path) as dataset:
        out = io.ReadWindow(src=dataset, window=Window(0, 0, 2, 2), indexes=indexes)()

    expected = gt.values if indexes is None else gt.values[[i - 1 for i in indexes]]
    np.testing.assert_array_equal(out.values, expected[:, :2, :2])


def test_read_reproject_like_path(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    out = io.ReadReprojectLike(src=path, like=str(path), indexes=[1])()

    assert out.shape == (1, 4, 5)
    assert out.transform == gt.transform
    assert out.crs == gt.crs
    np.testing.assert_array_equal(out.values, gt.values[:1])


def test_read_reproject_like_missing_path_raises_io_error(tmp_path: Path) -> None:
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(_sample_geotensor())

    with pytest.raises(io.GeoToolzIOError, match=r"missing\.tif"):
        io.ReadReprojectLike(src=path, like=str(tmp_path / "missing.tif"))()


# ---------------------------------------------------------------------------
# #131 — WriteZarr
# ---------------------------------------------------------------------------


def test_write_zarr_two_groups_coexist(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    store = str(tmp_path / "out.zarr")
    gt = _sample_geotensor()
    other = GeoTensor(
        gt.values[:1] + 100,
        transform=gt.transform,
        crs=gt.crs,
        fill_value_default=-1,
    )

    io.WriteZarr(store=store, group="a")(gt)
    io.WriteZarr(store=store, group="b", chunks={"y": 2, "x": 2})(other)

    root = zarr.open_group(store, mode="r")
    np.testing.assert_array_equal(root["a/values"][...], gt.values)
    np.testing.assert_array_equal(root["b/values"][...], other.values)
    assert root["b/values"].chunks == (1, 2, 2)
    assert root["a"].attrs["crs"] == str(gt.crs)
    assert tuple(root["a"].attrs["transform"]) == tuple(gt.transform)
    assert root["a"].attrs["fill_value_default"] == -9999
    assert root["b"].attrs["fill_value_default"] == -1


def test_write_zarr_overwrites_same_group(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    store = str(tmp_path / "out.zarr")
    gt = _sample_geotensor()
    doubled = GeoTensor(
        gt.values[:1] * 2,
        transform=gt.transform,
        crs=gt.crs,
        fill_value_default=np.int16(-1),
    )

    io.WriteZarr(store=store)(gt)
    io.WriteZarr(store=store)(doubled)

    root = zarr.open_group(store, mode="r")
    np.testing.assert_array_equal(root["values"][...], doubled.values)
    # numpy-scalar fills are stored as JSON scalars.
    assert root.attrs["fill_value_default"] == -1


def test_write_zarr_4d(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    store = str(tmp_path / "cube.zarr")
    values = np.arange(3 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5)
    gt = GeoTensor(
        values,
        transform=from_origin(100.0, 200.0, 10.0, 10.0),
        crs="EPSG:32631",
        fill_value_default=np.nan,
    )

    io.WriteZarr(store=store, chunks={"time": 1, "band": 1, "y": 2})(gt)

    array = zarr.open_group(store, mode="r")["values"]
    np.testing.assert_array_equal(array[...], values)
    assert array.chunks == (1, 1, 2, 5)


def test_write_zarr_rejects_unsupported_ndim(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    store = tmp_path / "bad.zarr"
    invalid = SimpleNamespace(
        values=np.zeros((1, 1, 1, 1, 1), dtype=np.uint8),
        crs="EPSG:32631",
        transform=from_origin(0.0, 0.0, 1.0, 1.0),
        fill_value_default=None,
    )

    with pytest.raises(io.GeoToolzIOError, match="2D to 4D"):
        io.WriteZarr(store=str(store))(invalid)
    assert not store.exists()


def test_4d_time_stack(tmp_path: Path) -> None:
    """GeoTIFF sinks reject a stack with GeoToolzIOError; Zarr keeps it (#147)."""
    from _helpers import time_stack

    stack = time_stack()
    for sink in (
        io.WriteCOG(path=tmp_path / "stack.tif"),
        io.WriteGeoTIFF(path=tmp_path / "stack_tiled.tif"),
    ):
        with pytest.raises(
            io.GeoToolzIOError, match=rf"{type(sink).__name__} expects 2D or 3D"
        ):
            sink(stack)
    zarr = pytest.importorskip("zarr")
    store = str(tmp_path / "stack.zarr")
    io.WriteZarr(store=store)(stack)
    np.testing.assert_array_equal(
        zarr.open_group(store, mode="r")["values"][...], np.asarray(stack)
    )
