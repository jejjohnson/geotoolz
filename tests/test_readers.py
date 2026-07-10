"""Tests for sensor reader framework."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

import geotoolz as gz
from geotoolz.readers import SensorReader, toy_sensor
from geotoolz.readers._base import require_optional_dependency
from geotoolz.readers.toy_sensor import constants as toy_constants


class MissingReader(SensorReader):
    """Incomplete reader used to verify ABC enforcement."""


def test_readers_module_is_exported() -> None:
    assert gz.readers is not None
    assert gz.SensorReader is SensorReader
    assert toy_sensor.Reader is not None


def test_sensor_reader_abc_enforces_required_surface() -> None:
    with pytest.raises(TypeError):
        MissingReader()  # type: ignore[abstract]


def test_toy_sensor_reader_passes_geodata_conformance() -> None:
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    reader = toy_sensor.Reader(
        "synthetic-toy",
        data=data,
        transform=Affine.translation(100, 200) * Affine.scale(10, -10),
        crs="EPSG:32631",
        fill_value_default=-9999.0,
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


def test_toy_sensor_constants_are_lazy_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_load_csv(package: str, resource: str) -> tuple[dict[str, str], ...]:
        calls.append(f"{package}:{resource}")
        return ({"name": "red"},)

    monkeypatch.setattr(toy_constants, "_CACHE", {})
    monkeypatch.setattr(toy_constants, "load_csv", fake_load_csv)

    assert calls == []
    assert toy_constants.BANDS == ({"name": "red"},)
    assert toy_constants.BANDS == ({"name": "red"},)
    assert toy_constants.CONSTANTS == {"solar_irradiance": ({"name": "red"},)}
    assert toy_constants.CONSTANTS == {"solar_irradiance": ({"name": "red"},)}
    assert calls == [
        "geotoolz.readers.toy_sensor:data/bands.csv",
        "geotoolz.readers.toy_sensor:data/solar_irradiance.csv",
    ]


def test_shared_csv_loader_caches_package_data() -> None:
    from geotoolz.readers._constants import load_csv

    load_csv.cache_clear()
    bands = load_csv("geotoolz.readers.toy_sensor", "data/bands.csv")
    assert bands[2]["name"] == "red"
    assert bands is load_csv("geotoolz.readers.toy_sensor", "data/bands.csv")


def test_optional_extra_guard_accepts_available_package() -> None:
    require_optional_dependency("json", extra="toy_sensor")


def test_toy_sensor_ndvi_preset_matches_generic_operator() -> None:
    op = toy_sensor.NDVI()
    expected = gz.indices.NDVI(red="red", nir="nir")

    assert op.get_config() == expected.get_config()

    gt = GeoTensor(
        np.stack(
            [
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
                np.full((2, 2), 3.0, dtype=np.float32),
            ]
        ),
        transform=Affine.identity(),
        crs="EPSG:4326",
        attrs={"band_names": ("blue", "green", "red", "nir")},
    )
    np.testing.assert_allclose(op(gt).values, 0.5)


@pytest.mark.slow
def test_toy_sensor_package_data_is_in_wheel() -> None:
    wheelhouse = Path("dist")
    wheels = sorted(wheelhouse.glob("geotoolz-*.whl"))
    if not wheels:
        pytest.skip("wheel has not been built")

    with ZipFile(wheels[-1]) as zf:
        names = set(zf.namelist())

    assert "geotoolz/readers/toy_sensor/data/bands.csv" in names
    assert "geotoolz/readers/toy_sensor/data/solar_irradiance.csv" in names
