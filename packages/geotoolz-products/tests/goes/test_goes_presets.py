"""``geoproducts.goes.presets`` — geotoolz operators bound to ABI channels."""

from __future__ import annotations

import builtins

import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor

from geoproducts import goes
from geoproducts.goes import constants


def _stack(values: dict[str, float], *, crs: str = "EPSG:4326") -> GeoTensor:
    data = np.stack([np.full((4, 4), v, dtype=np.float32) for v in values.values()])
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


@pytest.mark.parametrize("preset", ["NDVI", "SyntheticGreen", "ParallaxCorrect"])
def test_presets_name_the_operators_extra(monkeypatch, preset) -> None:
    real_import = builtins.__import__

    def no_geotoolz(name, *args, **kwargs):
        if name.split(".")[0] == "geotoolz":
            raise ImportError("No module named 'geotoolz'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_geotoolz)
    with pytest.raises(ImportError, match=r"geotoolz-products\[operators\]"):
        getattr(goes.presets, preset)()
