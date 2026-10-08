"""GOES-R ABI-aware geotoolz operator presets.

Presets bind geotoolz operators to ABI channel names (``"C01"`` …
``"C16"``, the ``band_names`` of :class:`geoproducts.goes.Reader` and
:class:`geoproducts.goes.L2Reader` outputs) and to the RGB recipes in
:mod:`geoproducts.goes.recipes`. geotoolz is an optional dependency
(``pip install 'geotoolz-products[operators]'``), imported when a preset
is called, so the readers install without the operator library.

The multi-channel presets expect the channels on one grid: an MCMIP file
(all 16 channels at 2 km), or L1b readers put together with
:func:`geoproducts.stack`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geoproducts._src.extras import require
from geoproducts.goes import constants, recipes


if TYPE_CHECKING:
    from pipekit import Operator


__all__ = [
    "NDVI",
    "DayCloudPhase",
    "FireTemperature",
    "MaskClouds",
    "NaturalColor",
    "ParallaxCorrect",
    "Recipe",
    "SyntheticGreen",
    "TrueColor",
]


def _geotoolz(module: str, preset: str) -> Any:
    return require(f"geotoolz.{module}", f"goes.presets.{preset}", "operators")


def NDVI() -> Operator:
    """NDVI from ABI reflectance: ``red="C02"``, ``nir="C03"``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    return _geotoolz("indices", "NDVI").NDVI(red="C02", nir="C03")


def SyntheticGreen() -> Operator:
    """The CIMSS synthetic green band, ``0.45 C02 + 0.10 C03 + 0.45 C01``.

    ABI has no green channel; this linear mix of red, "veggie" near-IR and
    blue reflectance stands in for it in true-colour imagery. Output is one
    band, ``(1, H, W)``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    expression = " + ".join(
        f"{weight} * {channel}"
        for channel, weight in constants.SYNTHETIC_GREEN_WEIGHTS.items()
    )
    return _geotoolz("spectral", "SyntheticGreen").BandMath(expression=expression)


def MaskClouds(*, conservative: bool = False) -> Operator:
    """Cloud mask from the L2 clear sky mask (ACM), ``True`` = cloudy.

    Reads the ``BCM`` band of a ``goes.L2Reader`` ACM read (cloudy or
    probably cloudy). ``conservative=True`` reads the four-level ``ACM``
    band instead and also flags *probably clear* pixels, for analyses that
    need confidently clear sky. Pair with ``gz.mask.ApplyMask`` once the
    mask is on the data's grid (``geoproducts.stack`` resamples it with ``mode``).

    Args:
        conservative: Flag probably-clear pixels as well.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    qa = _geotoolz("qa", "MaskClouds")
    if conservative:
        values = [
            constants.ACM_PROBABLY_CLEAR,
            constants.ACM_PROBABLY_CLOUDY,
            constants.ACM_CLOUDY,
        ]
        return qa.MaskClouds(qa_band="ACM", values=values)
    return qa.MaskClouds(qa_band="BCM", values=[constants.BCM_CLOUDY])


def ParallaxCorrect(
    *,
    satellite_lon_deg: float = constants.GOES_EAST_LON_DEG,
    target_height_m: Any = 0.0,
    method: str = "bilinear",
) -> Operator:
    """Geostationary parallax correction from a GOES-R viewpoint.

    Binds ``gz.geom.GeostationaryParallaxCorrect`` to the GOES geostationary
    height; the input must already be on an EPSG:4326 grid. Pass
    ``reader.satellite_lon_deg`` (or ``constants.GOES_WEST_LON_DEG``) for a
    satellite other than GOES-East, and the ACHA cloud-top height (on the
    same grid) as ``target_height_m`` to move clouds to where they are.

    Args:
        satellite_lon_deg: Sub-satellite longitude. Default GOES-East.
        target_height_m: Height of the observed feature (metres): a scalar,
            a same-grid array or a same-grid GeoTensor.
        method: ``"nearest"`` or ``"bilinear"``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    return _geotoolz("geom", "ParallaxCorrect").GeostationaryParallaxCorrect(
        satellite_lon_deg=satellite_lon_deg,
        satellite_height_m=constants.SATELLITE_HEIGHT_M,
        target_height_m=target_height_m,
        method=method,
    )


def Recipe(recipe: recipes.Recipe | str) -> Operator:
    """A ``gz.viz.RGBRecipe`` for one of :data:`goes.recipes.RECIPES`.

    Args:
        recipe: A :class:`~geoproducts.goes.recipes.Recipe`, or its name
            (``"true_color"``, ``"natural_color"``, ``"day_cloud_phase"``,
            ``"fire_temperature"``).

    Returns:
        An operator mapping a channel-named stack to a ``(3, H, W)``
        ``float32`` RGB in ``[0, 1]``.

    Raises:
        KeyError: Unknown recipe name.
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    spec = recipes.RECIPES[recipe] if isinstance(recipe, str) else recipe
    return _rgb_recipe(spec, "Recipe")


def _rgb_recipe(spec: recipes.Recipe, preset: str) -> Operator:
    return _geotoolz("viz", preset).RGBRecipe(
        red=spec.red,
        green=spec.green,
        blue=spec.blue,
        vmin=spec.vmin,
        vmax=spec.vmax,
        gamma=spec.gamma,
    )


def TrueColor() -> Operator:
    """True colour (C02, synthetic green, C01), gamma 2.2. Reads C01-C03."""
    return _rgb_recipe(recipes.TRUE_COLOR, "TrueColor")


def NaturalColor() -> Operator:
    """CIRA day land cloud "natural colour" (C05, C03, C02). Reads C02/C03/C05."""
    return _rgb_recipe(recipes.NATURAL_COLOR, "NaturalColor")


def DayCloudPhase() -> Operator:
    """CIRA day cloud phase distinction (C13 BT, C02, C05)."""
    return _rgb_recipe(recipes.DAY_CLOUD_PHASE, "DayCloudPhase")


def FireTemperature() -> Operator:
    """CIRA fire temperature (C07 BT, C06, C05)."""
    return _rgb_recipe(recipes.FIRE_TEMPERATURE, "FireTemperature")
