"""Shared building blocks of the per-product geotoolz presets.

Every product's ``presets`` module binds geotoolz operators to its own
band names. geotoolz is optional (the ``[operators]`` extra), so it is
imported when a preset is called, through :func:`geotoolz_module`; RGB
composites are plain data (:class:`Recipe`) turned into a
``gz.viz.RGBRecipe`` by :func:`rgb_recipe`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from geoproducts._src.extras import require


if TYPE_CHECKING:
    from pipekit import Operator


__all__ = ["Recipe", "geotoolz_module", "parallax_correct", "rgb_recipe"]


@dataclass(frozen=True)
class Recipe:
    """One RGB recipe: per-channel expression, stretch and gamma.

    Attributes:
        name: Short identifier.
        red: Red-channel band name or band-math expression.
        green: Green-channel band name or expression.
        blue: Blue-channel band name or expression.
        vmin: Per-channel value mapped to 0 (above ``vmax`` inverts).
        vmax: Per-channel value mapped to 1.
        gamma: Per-channel display gamma.
        channels: Bands the expressions read.
        description: What the composite shows.
    """

    name: str
    red: str
    green: str
    blue: str
    vmin: tuple[float, float, float]
    vmax: tuple[float, float, float]
    gamma: tuple[float, float, float]
    channels: tuple[str, ...]
    description: str


def geotoolz_module(module: str, feature: str) -> Any:
    """Import ``geotoolz.<module>`` for a preset, naming the extra if missing.

    Args:
        module: geotoolz family (``"viz"``, ``"indices"``, …).
        feature: The preset, for the error message (``"goes.presets.NDVI"``).

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    return require(f"geotoolz.{module}", feature, "operators")


def rgb_recipe(spec: Recipe, feature: str) -> Operator:
    """A ``gz.viz.RGBRecipe`` operator for ``spec``.

    Args:
        spec: The recipe.
        feature: The preset, for the missing-extra message.

    Returns:
        An operator mapping a band-named stack to a ``(3, H, W)``
        ``float32`` RGB in ``[0, 1]``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    return geotoolz_module("viz", feature).RGBRecipe(
        red=spec.red,
        green=spec.green,
        blue=spec.blue,
        vmin=spec.vmin,
        vmax=spec.vmax,
        gamma=spec.gamma,
    )


def parallax_correct(
    feature: str,
    *,
    satellite_lon_deg: float,
    satellite_height_m: float,
    target_height_m: Any,
    method: str,
) -> Operator:
    """``gz.geom.GeostationaryParallaxCorrect`` bound to one satellite's view.

    Args:
        feature: The preset, for the missing-extra message.
        satellite_lon_deg: Sub-satellite longitude (degrees east).
        satellite_height_m: Satellite height above the equator (metres).
        target_height_m: Height of the observed feature (metres): a scalar,
            a same-grid array or a same-grid GeoTensor.
        method: ``"nearest"`` or ``"bilinear"``.

    Raises:
        ImportError: geotoolz is not installed (the ``[operators]`` extra).
    """
    return geotoolz_module("geom", feature).GeostationaryParallaxCorrect(
        satellite_lon_deg=satellite_lon_deg,
        satellite_height_m=satellite_height_m,
        target_height_m=target_height_m,
        method=method,
    )
