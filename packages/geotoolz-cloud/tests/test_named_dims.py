"""Reads name their dims (#526): COG bands, NetCDF CF dims, write_cog."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from geoarray import GeoArray
from geoarray.coords import SpectralAxis, TimeAxis, Vertical, VerticalAxis


UTM = (10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)


def _tiff(path: Path, count: int, descriptions: tuple[str, ...] | None) -> Path:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=count,
        dtype="float32",
        crs="EPSG:32630",
        transform=Affine(*UTM),
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(np.arange(count * 32 * 32, dtype="float32").reshape(count, 32, 32))
        if descriptions:
            dst.descriptions = descriptions
    return path


def _open(path: Path):
    pytest.importorskip("async_geotiff")
    from obstore.store import LocalStore

    from geocloud.cog import CogSource

    return CogSource.open(
        url=f"file://{path}", store=LocalStore(prefix=str(path.parent)), path=path.name
    )


def test_a_cog_read_names_its_bands(tmp_path: Path) -> None:
    source = _open(_tiff(tmp_path / "a.tif", 2, ("red", "nir")))

    with warnings.catch_warnings():
        warnings.filterwarnings("error", "GeoArray dims read from", DeprecationWarning)
        chip = source.read_window(_window())
        whole = source.load()

    assert chip.dims == whole.dims == ("band", "y", "x")
    assert source.domain.band_names == ("red", "nir")
    assert chip.coord("band") == SpectralAxis(("red", "nir"))
    assert [n for n in whole.coord("band").names] == ["red", "nir"]


def _window():
    from geoarray import PixelWindow

    return PixelWindow(col_off=0, row_off=0, width=16, height=16)


def test_a_cog_without_descriptions_has_no_band_coordinate(tmp_path: Path) -> None:
    source = _open(_tiff(tmp_path / "b.tif", 3, None))

    chip = source.read_window(_window())

    assert chip.dims == ("band", "y", "x") and chip.coord("band") is None
    assert source.domain.band_names is None


def test_partial_or_duplicate_descriptions_are_dropped() -> None:
    from geocloud._src.cog_source import _band_names

    item = '<Item name="DESCRIPTION" sample="{}" role="description">{}</Item>'
    xml = "<GDALMetadata>{}</GDALMetadata>"
    assert _band_names(xml.format(item.format(0, "a")), 2) is None  # band 1 missing
    both = item.format(0, "a") + item.format(1, "a")
    assert _band_names(xml.format(both), 2) is None  # duplicate
    assert _band_names("<not xml", 1) is None
    assert _band_names(None, 1) is None


def test_write_cog_round_trips_band_names(tmp_path: Path) -> None:
    from geocloud.cog import write_cog

    scene = GeoArray(
        np.ones((2, 32, 32), dtype=np.float32),
        UTM,
        "EPSG:32630",
        dims="band y x",
        coords=(SpectralAxis(("swir1", "swir2")),),
    )
    out = tmp_path / "scene.tif"
    write_cog(scene, out, blocksize=16)

    back = _open(out).load()

    assert back.dims == ("band", "y", "x")
    assert back.coord("band") == SpectralAxis(("swir1", "swir2"))


def test_write_cog_refuses_a_time_dim(tmp_path: Path) -> None:
    from geocloud.cog import write_cog

    series = GeoArray(
        np.ones((3, 32, 32), dtype=np.float32), UTM, "EPSG:32630", dims="time y x"
    )
    with pytest.raises(ValueError, match=r"one scene.*isel"):
        write_cog(series, tmp_path / "series.tif")


def _cf_file(path: Path, dims: dict[str, tuple[int, dict]]) -> Path:
    """A NetCDF file with one variable ``v`` over ``dims`` (name → size, attrs)."""
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(path, "w") as ds:
        for name, (size, attrs) in dims.items():
            ds.createDimension(name, size)
            if attrs is not None:
                var = ds.createVariable(name, "f8", (name,))
                var[:] = attrs.pop("values", np.arange(size, dtype=float) + 0.5)
                for key, value in attrs.items():
                    var.setncattr(key, value)
        v = ds.createVariable("v", "f4", tuple(dims))
        v[:] = np.ones(tuple(size for size, _ in dims.values()), dtype="f4")
        v.setncattr("GeoTransform", "500000 10 0 4600000 0 -10")
    return path


def test_read_netcdf_names_cf_depth_as_level(tmp_path: Path) -> None:
    from geocloud.hdf import read_netcdf

    path = _cf_file(
        tmp_path / "ocean.nc",
        {
            "depth": (
                3,
                {"units": "m", "positive": "down", "values": [5.0, 15.0, 25.0]},
            ),
            "y": (4, {}),
            "x": (5, {}),
        },
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", "GeoArray dims read from", DeprecationWarning)
        g = read_netcdf(path, "v")

    assert g.dims == ("level", "y", "x")
    assert g.coord("level") == VerticalAxis(
        (5.0, 15.0, 25.0), Vertical(kind="depth", units="m", positive="down")
    )


def test_read_netcdf_names_time_and_pressure(tmp_path: Path) -> None:
    from geocloud.hdf import read_netcdf

    path = _cf_file(
        tmp_path / "atmos.nc",
        {
            "time": (2, {"units": "days since 2026-01-01", "values": [0.0, 10.0]}),
            "plev": (3, {"units": "hPa", "values": [1000.0, 850.0, 500.0]}),
            "y": (4, {}),
            "x": (5, {}),
        },
    )

    g = read_netcdf(path, "v")

    assert g.dims == ("time", "level", "y", "x")
    stamps = g.coord("time")
    assert isinstance(stamps, TimeAxis)
    assert [str(s)[:10] for s in stamps.labels] == ["2026-01-01", "2026-01-11"]
    vertical = g.coord("level").vertical
    assert (vertical.kind, vertical.positive, vertical.units) == (
        "pressure",
        "down",
        "hPa",
    )


def test_read_netcdf_keeps_other_dim_names_and_subsets_coordinates(
    tmp_path: Path,
) -> None:
    from geocloud.hdf import read_netcdf

    path = _cf_file(
        tmp_path / "ens.nc",
        {
            "lev": (4, {"values": [1.0, 2.0, 3.0, 4.0]}),
            "y": (4, {}),
            "x": (5, {}),
        },
    )
    members = _cf_file(
        tmp_path / "members.nc", {"member": (3, None), "y": (4, {}), "x": (5, {})}
    )

    picked = read_netcdf(path, "v", indexes=[4, 2])
    ensemble = read_netcdf(members, "v")

    assert picked.dims == ("level", "y", "x")
    assert picked.coord("level").values == (4.0, 2.0)
    assert ensemble.dims == ("member", "y", "x") and ensemble.coord("member") is None


def test_read_reproject_keeps_the_band_names(tmp_path: Path) -> None:
    """georeader's warp keeps the bands; the reader carries their names over."""
    pytest.importorskip("async_geotiff")
    import asyncio

    from obstore.store import LocalStore

    from geocloud import cog

    path = _tiff(tmp_path / "c.tif", 2, ("red", "nir"))
    reader = asyncio.run(
        cog.AsyncCogReader.open(
            f"file://{path}", store=LocalStore(prefix=str(tmp_path)), path=path.name
        )
    )
    bounds = (500_000.0, 4_599_680.0, 500_320.0, 4_600_000.0)  # the whole scene

    out = asyncio.run(
        cog.read_reproject(
            reader, bounds=bounds, dst_crs="EPSG:32630", resolution_dst_crs=20.0
        )
    )

    assert out.dims == ("band", "y", "x")
    assert out.coord("band") == SpectralAxis(("red", "nir"))
