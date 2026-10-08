"""``geoproducts.goes.presets`` — geotoolz operators bound to ABI channels."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor

from geoproducts import goes
from geoproducts.goes import constants, recipes


def _stack(values: dict[str, float], *, crs: str = "EPSG:4326") -> GeoTensor:
    data = np.stack(
        [np.full((4, 4), v, dtype=np.asarray(v).dtype) for v in values.values()]
    ).astype(np.float32)
    return GeoTensor(
        data,
        transform=Affine(0.01, 0.0, -100.0, 0.0, -0.01, 40.0),
        crs=crs,
        fill_value_default=np.nan,
        attrs={"band_names": tuple(values)},
    )


def test_ndvi_reads_red_c02_and_nir_c03() -> None:
    pytest.importorskip("geotoolz")
    out = goes.NDVI()(_stack({"C01": 0.05, "C02": 0.1, "C03": 0.5}))
    np.testing.assert_allclose(np.asarray(out), (0.5 - 0.1) / (0.5 + 0.1), rtol=1e-6)


def test_synthetic_green_is_the_cimss_mix() -> None:
    pytest.importorskip("geotoolz")
    out = goes.SyntheticGreen()(_stack({"C01": 0.2, "C02": 0.4, "C03": 0.6}))
    assert np.asarray(out).shape[-2:] == (4, 4)
    np.testing.assert_allclose(
        np.asarray(out), 0.45 * 0.4 + 0.10 * 0.6 + 0.45 * 0.2, rtol=1e-6
    )
    assert sum(constants.SYNTHETIC_GREEN_WEIGHTS.values()) == pytest.approx(1.0)


def test_parallax_correct_binds_the_goes_geometry() -> None:
    gz = pytest.importorskip("geotoolz")
    op = goes.ParallaxCorrect(
        satellite_lon_deg=constants.GOES_WEST_LON_DEG, target_height_m=0.0
    )
    assert isinstance(op, gz.geom.GeostationaryParallaxCorrect)
    config = op.get_config()
    assert config["satellite_lon_deg"] == constants.GOES_WEST_LON_DEG
    assert config["satellite_height_m"] == constants.SATELLITE_HEIGHT_M
    scene = _stack({"C13": 280.0})
    assert op(scene) is scene  # zero height is an exact identity
    default = goes.ParallaxCorrect().get_config()
    assert default["satellite_lon_deg"] == constants.GOES_EAST_LON_DEG


@pytest.mark.parametrize(
    "preset",
    [
        "NDVI",
        "SyntheticGreen",
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
        getattr(goes.presets, preset)()


def test_mask_clouds_reads_the_clear_sky_mask() -> None:
    pytest.importorskip("geotoolz")
    stack = _stack({"BCM": 1.0, "ACM": float(constants.ACM_CLOUDY)})
    data = np.asarray(stack).copy()
    data[0, 0, 0] = constants.BCM_CLEAR
    data[1, 0, 0] = constants.ACM_PROBABLY_CLEAR
    stack = GeoTensor(data, transform=stack.transform, crs=stack.crs, attrs=stack.attrs)
    default = np.asarray(goes.MaskClouds()(stack))
    assert default.dtype == bool
    assert not default[0, 0]
    assert default.sum() == 15
    strict = np.asarray(goes.MaskClouds(conservative=True)(stack))
    assert strict.all()  # probably clear counts as cloudy too


@pytest.mark.parametrize("name", sorted(recipes.RECIPES))
def test_recipe_presets_follow_the_recipe_table(name: str) -> None:
    gz = pytest.importorskip("geotoolz")
    spec = recipes.RECIPES[name]
    op = goes.presets.Recipe(name)
    assert isinstance(op, gz.viz.RGBRecipe)
    config = op.get_config()
    assert (config["red"], config["green"], config["blue"]) == (
        spec.red,
        spec.green,
        spec.blue,
    )
    assert tuple(config["vmin"]) == spec.vmin
    assert tuple(config["gamma"]) == spec.gamma
    # Every channel a recipe reads is declared in ``channels``.
    names = set(spec.channels)
    stack = _stack({ch: 0.5 for ch in sorted(names)})
    assert op(stack).shape == (3, 4, 4)


def test_true_color_mixes_the_synthetic_green() -> None:
    pytest.importorskip("geotoolz")
    rgb = np.asarray(goes.TrueColor()(_stack({"C01": 0.2, "C02": 0.4, "C03": 0.6})))
    green = 0.45 * 0.4 + 0.1 * 0.6 + 0.45 * 0.2
    np.testing.assert_allclose(
        rgb[:, 0, 0], np.array([0.4, green, 0.2]) ** (1 / 2.2), rtol=1e-5
    )


def test_day_cloud_phase_inverts_the_brightness_temperature() -> None:
    pytest.importorskip("geotoolz")
    cold = goes.DayCloudPhase()(_stack({"C02": 0.5, "C05": 0.3, "C13": 219.65}))
    warm = goes.DayCloudPhase()(_stack({"C02": 0.5, "C05": 0.3, "C13": 280.65}))
    assert np.asarray(cold)[0, 0, 0] == pytest.approx(1.0)
    assert np.asarray(warm)[0, 0, 0] == pytest.approx(0.0, abs=1e-6)


def test_named_presets_match_recipe() -> None:
    pytest.importorskip("geotoolz")
    pairs = {
        "true_color": goes.TrueColor,
        "natural_color": goes.NaturalColor,
        "day_cloud_phase": goes.DayCloudPhase,
        "fire_temperature": goes.FireTemperature,
    }
    for name, preset in pairs.items():
        assert preset().get_config() == goes.presets.Recipe(name).get_config()
    with pytest.raises(KeyError):
        goes.presets.Recipe("air_mass")
