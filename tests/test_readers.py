"""Tests for sensor reader framework."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

import geotoolz as gz
from geotoolz.readers import SensorReader, modis
from geotoolz.readers.modis import constants as modis_constants


class MissingReader(SensorReader):
    """Incomplete reader used to verify ABC enforcement."""


def test_readers_module_is_exported() -> None:
    assert gz.readers is not None
    assert gz.SensorReader is SensorReader
    assert modis.Reader is not None


def test_sensor_reader_abc_enforces_required_surface() -> None:
    with pytest.raises(TypeError):
        MissingReader()  # type: ignore[abstract]


def test_modis_reader_passes_geodata_conformance() -> None:
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    reader = modis.Reader(
        "synthetic-modis",
        data=data,
        transform=Affine.translation(100, 200) * Affine.scale(10, -10),
        crs="EPSG:32631",
        fill_value=-9999.0,
    )

    assert isinstance(reader, SensorReader)
    assert reader.track == "A"
    assert reader.shape == data.shape
    assert reader.bands == ("blue", "green", "red", "nir")

    tile = reader.read_from_window(Window(1, 2, 3, 2))
    assert isinstance(tile, GeoTensor)
    np.testing.assert_array_equal(tile.values, data[:, 2:4, 1:4])
    assert tile.attrs["band_names"] == reader.bands

    boundless = reader.read_from_center_coords(95, 205, width=3, height=3)
    assert boundless.shape == (4, 3, 3)
    assert np.all(boundless.values[:, 0, 0] == -9999.0)


def test_modis_constants_are_lazy_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_load_csv(package: str, resource: str) -> tuple[dict[str, str], ...]:
        calls.append(f"{package}:{resource}")
        return ({"name": "red"},)

    monkeypatch.setattr(modis_constants, "_CACHE", {})
    monkeypatch.setattr(modis_constants, "load_csv", fake_load_csv)

    assert calls == []
    assert modis_constants.BANDS == ({"name": "red"},)
    assert modis_constants.BANDS == ({"name": "red"},)
    assert calls == ["geotoolz.readers.modis:data/bands.csv"]


def test_shared_csv_loader_caches_package_data() -> None:
    from geotoolz.readers._constants import load_csv

    load_csv.cache_clear()
    bands = load_csv("geotoolz.readers.modis", "data/bands.csv")
    assert bands[2]["name"] == "red"
    assert bands is load_csv("geotoolz.readers.modis", "data/bands.csv")


def test_viirs_missing_optional_extra_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "h5py":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    from geotoolz.readers import viirs

    with pytest.raises(ImportError, match=r"h5py.*geotoolz\[viirs\]"):
        viirs.Reader("scene.h5")


def test_modis_ndvi_preset_matches_generic_operator() -> None:
    op = modis.NDVI()
    expected = gz.indices.NDVI(red="red", nir="nir")

    assert op.get_config() == expected.get_config()


def test_modis_package_data_is_in_wheel() -> None:
    wheelhouse = Path("dist")
    wheels = sorted(wheelhouse.glob("geotoolz-*.whl"))
    if not wheels:
        pytest.skip("wheel has not been built")

    with ZipFile(wheels[-1]) as zf:
        names = set(zf.namelist())

    assert "geotoolz/readers/modis/data/bands.csv" in names
    assert "geotoolz/readers/modis/data/solar_irradiance.csv" in names
