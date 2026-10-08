"""Himawari AHI-aware geotoolz operator presets.

Presets bind geotoolz operators to AHI band names (``"B01"`` … ``"B16"``,
the ``band_names`` of :class:`geoproducts.himawari.Reader` outputs and of
the L2 cloud-mask variables) and to the RGB recipes in
:mod:`geoproducts.himawari.recipes`. geotoolz is an optional dependency
(``pip install 'geotoolz-products[operators]'``), imported when a preset
is called, so the readers install without the operator library.

The multi-band presets expect the bands on one grid: readers put
together with :func:`geoproducts.stack` (B03 is at 0.5 km, B01 / B02 /
B04 at 1 km, the rest at 2 km).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geoproducts._src.presets import geotoolz_module, parallax_correct, rgb_recipe
from geoproducts.himawari import constants, recipes


if TYPE_CHECKING:
    from pipekit import Operator


__all__ = [
    "NDVI",
    "DayCloudPhase",
    "FireTemperature",
    "HybridGreen",
    "MaskClouds",
    "NaturalColor",
    "ParallaxCorrect",
    "Recipe",
    "TrueColor",
]


def _geotoolz(module: str, preset: str) -> Any:
    return geotoolz_module(module, f"himawari.presets.{preset}")


def NDVI() -> Operator:
    """NDVI from AHI reflectance: ``red="B03"``, ``nir="B04"``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    return _geotoolz("indices", "NDVI").NDVI(red="B03", nir="B04")


def HybridGreen(
    *, nir_fraction: float = constants.HYBRID_GREEN_NIR_FRACTION
) -> Operator:
    """The hybrid green band, ``(1 - f) · B02 + f · B04``.

    AHI's green band (0.51 µm) sits below the vegetation reflectance peak,
    so vegetation looks too dark in true colour; mixing in a little
    near-infrared restores it. Output is one band, ``(1, H, W)``.

    Args:
        nir_fraction: Share ``f`` of the 0.86 µm band. Default ``0.07``.

    Raises:
        ValueError: ``nir_fraction`` is outside ``[0, 1]``.
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    if not 0.0 <= nir_fraction <= 1.0:
        raise ValueError(f"nir_fraction must be in [0, 1]; got {nir_fraction}.")
    expression = f"{1.0 - nir_fraction:g} * B02 + {nir_fraction:g} * B04"
    return _geotoolz("spectral", "HybridGreen").BandMath(expression=expression)


def MaskClouds(*, conservative: bool = False) -> Operator:
    """Cloud mask from the NOAA L2 cloud mask (``CMSK``), ``True`` = cloudy.

    Reads the ``CloudMaskBinary`` band of a ``himawari.L2Reader`` CMSK read.
    ``conservative=True`` reads the four-level ``CloudMask`` band instead and
    also flags *probably clear* pixels. Pair with ``gz.mask.ApplyMask`` once
    the mask is on the data's grid.

    Args:
        conservative: Flag probably-clear pixels as well.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    qa = _geotoolz("qa", "MaskClouds")
    if conservative:
        values = [
            constants.CLOUD_MASK_PROBABLY_CLEAR,
            constants.CLOUD_MASK_PROBABLY_CLOUDY,
            constants.CLOUD_MASK_CLOUDY,
        ]
        return qa.MaskClouds(qa_band="CloudMask", values=values)
    return qa.MaskClouds(
        qa_band="CloudMaskBinary", values=[constants.CLOUD_MASK_BINARY_CLOUDY]
    )


def ParallaxCorrect(
    *,
    satellite_lon_deg: float = constants.HIMAWARI_LON_DEG,
    target_height_m: Any = 0.0,
    method: str = "bilinear",
) -> Operator:
    """Geostationary parallax correction from the Himawari viewpoint (140.7°E).

    Binds ``gz.geom.GeostationaryParallaxCorrect`` to the Himawari height;
    the input must already be on an EPSG:4326 grid. Pass the cloud-top
    height (on the same grid) as ``target_height_m`` to move clouds to where
    they are.

    Args:
        satellite_lon_deg: Sub-satellite longitude. Default 140.7°E.
        target_height_m: Height of the observed feature (metres): a scalar,
            a same-grid array or a same-grid GeoTensor.
        method: ``"nearest"`` or ``"bilinear"``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    return parallax_correct(
        "himawari.presets.ParallaxCorrect",
        satellite_lon_deg=satellite_lon_deg,
        satellite_height_m=constants.SATELLITE_HEIGHT_M,
        target_height_m=target_height_m,
        method=method,
    )


def Recipe(recipe: recipes.Recipe | str) -> Operator:
    """A ``gz.viz.RGBRecipe`` for one of :data:`himawari.recipes.RECIPES`.

    Args:
        recipe: A :class:`~geoproducts.himawari.recipes.Recipe`, or its name
            (``"true_color"``, ``"natural_color"``, ``"day_cloud_phase"``,
            ``"fire_temperature"``).

    Returns:
        An operator mapping a band-named stack to a ``(3, H, W)``
        ``float32`` RGB in ``[0, 1]``.

    Raises:
        KeyError: Unknown recipe name.
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    spec = recipes.RECIPES[recipe] if isinstance(recipe, str) else recipe
    return rgb_recipe(spec, "himawari.presets.Recipe")


def TrueColor() -> Operator:
    """True colour (B03, hybrid green, B01), gamma 2.2. Reads B01-B04."""
    return rgb_recipe(recipes.TRUE_COLOR, "himawari.presets.TrueColor")


def NaturalColor() -> Operator:
    """Day land cloud "natural colour" (B05, B04, B03)."""
    return rgb_recipe(recipes.NATURAL_COLOR, "himawari.presets.NaturalColor")


def DayCloudPhase() -> Operator:
    """Day cloud phase distinction (B13 BT, B03, B05)."""
    return rgb_recipe(recipes.DAY_CLOUD_PHASE, "himawari.presets.DayCloudPhase")


def FireTemperature() -> Operator:
    """Fire temperature (B07 BT, B06, B05)."""
    return rgb_recipe(recipes.FIRE_TEMPERATURE, "himawari.presets.FireTemperature")
