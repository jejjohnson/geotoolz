"""``geoproducts.himawari.presets`` — geotoolz operators bound to AHI bands."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor

from geoproducts import himawari
from geoproducts.himawari import constants, recipes


def _stack(values: dict[str, float]) -> GeoTensor:
    data = np.stack([np.full((4, 4), v, dtype=np.float32) for v in values.values()])
    return GeoTensor(
        data,
        transform=Affine(0.01, 0.0, 140.0, 0.0, -0.01, 35.0),
        crs="EPSG:4326",
        fill_value_default=np.nan,
        attrs={"band_names": tuple(values)},
    )


def test_ndvi_reads_red_b03_and_nir_b04() -> None:
    pytest.importorskip("geotoolz")
    out = himawari.NDVI()(_stack({"B03": 0.1, "B04": 0.5}))
    np.testing.assert_allclose(np.asarray(out), (0.5 - 0.1) / (0.5 + 0.1), rtol=1e-6)


def test_hybrid_green_mixes_in_the_near_infrared() -> None:
    pytest.importorskip("geotoolz")
    scene = _stack({"B02": 0.2, "B04": 0.6})
    out = np.asarray(himawari.HybridGreen()(scene))
    f = constants.HYBRID_GREEN_NIR_FRACTION
    np.testing.assert_allclose(out, (1 - f) * 0.2 + f * 0.6, rtol=1e-6)
    pure = np.asarray(himawari.HybridGreen(nir_fraction=0.0)(scene))
    np.testing.assert_allclose(pure, 0.2, rtol=1e-6)
    with pytest.raises(ValueError, match="nir_fraction"):
        himawari.HybridGreen(nir_fraction=1.5)


def test_parallax_correct_binds_the_himawari_geometry() -> None:
    gz = pytest.importorskip("geotoolz")
    op = himawari.ParallaxCorrect()
    assert isinstance(op, gz.geom.GeostationaryParallaxCorrect)
    config = op.get_config()
    assert config["satellite_lon_deg"] == constants.HIMAWARI_LON_DEG
    assert config["satellite_height_m"] == constants.SATELLITE_HEIGHT_M
    scene = _stack({"B13": 280.0})
    assert op(scene) is scene  # zero height is an exact identity


def test_mask_clouds_reads_the_cloud_mask() -> None:
    pytest.importorskip("geotoolz")
    data = np.stack(
        [
            np.full((4, 4), constants.CLOUD_MASK_BINARY_CLOUDY, np.float32),
            np.full((4, 4), constants.CLOUD_MASK_CLOUDY, np.float32),
        ]
    )
    data[0, 0, 0] = constants.CLOUD_MASK_BINARY_CLEAR
    data[1, 0, 0] = constants.CLOUD_MASK_PROBABLY_CLEAR
    scene = GeoTensor(
        data,
        transform=Affine(0.01, 0.0, 140.0, 0.0, -0.01, 35.0),
        crs="EPSG:4326",
        attrs={"band_names": ("CloudMaskBinary", "CloudMask")},
    )
    default = np.asarray(himawari.MaskClouds()(scene))
    assert default.dtype == bool and not default[0, 0] and default.sum() == 15
    assert np.asarray(himawari.MaskClouds(conservative=True)(scene)).all()


@pytest.mark.parametrize(
    "preset",
    [
        "NDVI",
        "HybridGreen",
        "ParallaxCorrect",
        "MaskClouds",
        "TrueColor",
        "NaturalColor",
        "DayCloudPhase",
        "FireTemperature",
    ],
)
def test_presets_name_the_operators_extra(monkeypatch, preset) -> None:
    for name in [m for m in sys.modules if m.split(".")[0] == "geotoolz"]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "geotoolz", None)
    with pytest.raises(ImportError, match=rf"{preset}.*geotoolz-products\[operators\]"):
        getattr(himawari.presets, preset)()


@pytest.mark.parametrize("name", sorted(recipes.RECIPES))
def test_recipe_presets_follow_the_recipe_table(name: str) -> None:
    gz = pytest.importorskip("geotoolz")
    spec = recipes.RECIPES[name]
    op = himawari.presets.Recipe(name)
    assert isinstance(op, gz.viz.RGBRecipe)
    config = op.get_config()
    assert (config["red"], config["green"], config["blue"]) == (
        spec.red,
        spec.green,
        spec.blue,
    )
    # Every band a recipe reads is declared in ``channels`` and is an AHI band.
    assert set(spec.channels) <= set(constants.CHANNELS)
    assert op(_stack({ch: 0.5 for ch in spec.channels})).shape == (3, 4, 4)


def test_named_presets_match_recipe() -> None:
    pytest.importorskip("geotoolz")
    pairs = {
        "true_color": himawari.TrueColor,
        "natural_color": himawari.NaturalColor,
        "day_cloud_phase": himawari.DayCloudPhase,
        "fire_temperature": himawari.FireTemperature,
    }
    for name, preset in pairs.items():
        assert preset().get_config() == himawari.presets.Recipe(name).get_config()
    with pytest.raises(KeyError):
        himawari.presets.Recipe("air_mass")


def test_true_color_uses_the_hybrid_green() -> None:
    pytest.importorskip("geotoolz")
    rgb = np.asarray(
        himawari.TrueColor()(_stack({"B01": 0.2, "B02": 0.3, "B03": 0.4, "B04": 0.6}))
    )
    f = constants.HYBRID_GREEN_NIR_FRACTION
    np.testing.assert_allclose(
        rgb[:, 0, 0],
        np.array([0.4, (1 - f) * 0.3 + f * 0.6, 0.2]) ** (1 / 2.2),
        rtol=1e-5,
    )


def test_band_table() -> None:
    assert [row["name"] for row in himawari.BANDS] == list(constants.CHANNELS)
    resolutions = {row["name"]: float(row["resolution_km"]) for row in himawari.BANDS}
    assert resolutions["B03"] == 0.5 and resolutions["B04"] == 1.0
    assert resolutions["B13"] == 2.0
    with pytest.raises(AttributeError):
        himawari.NOPE  # noqa: B018
