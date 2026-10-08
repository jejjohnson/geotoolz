"""The standard RGB recipes on AHI bands, as data.

AHI and ABI share their instrument heritage, so the operational RGBs keep
the ABI quick-guide stretches (CIRA / JMA) on the equivalent AHI bands.
Inputs are AHI band names (``band_names`` of ``himawari.Reader``
outputs): reflectance factor (0-1) for B01-B06, brightness temperature
(K) for B07-B16, i.e. ``calibration="reflectance"`` /
``"brightness_temperature"`` reads put on one grid with
:func:`geoproducts.stack`.

``himawari.presets`` turns a recipe into a ``geotoolz.viz.RGBRecipe``
operator; the recipes themselves need no extra.
"""

from __future__ import annotations

from geoproducts._src.presets import Recipe
from geoproducts.himawari.constants import HYBRID_GREEN_NIR_FRACTION


__all__ = [
    "DAY_CLOUD_PHASE",
    "FIRE_TEMPERATURE",
    "HYBRID_GREEN",
    "NATURAL_COLOR",
    "RECIPES",
    "TRUE_COLOR",
    "Recipe",
]

#: AHI's green band (0.51 µm) plus a share of the near-infrared (0.86 µm).
HYBRID_GREEN = (
    f"{1.0 - HYBRID_GREEN_NIR_FRACTION:g} * B02 + {HYBRID_GREEN_NIR_FRACTION:g} * B04"
)

TRUE_COLOR = Recipe(
    name="true_color",
    red="B03",
    green=HYBRID_GREEN,
    blue="B01",
    vmin=(0.0, 0.0, 0.0),
    vmax=(1.0, 1.0, 1.0),
    gamma=(2.2, 2.2, 2.2),
    channels=("B01", "B02", "B03", "B04"),
    description="Daytime true colour with the hybrid (green + near-infrared) green.",
)

NATURAL_COLOR = Recipe(
    name="natural_color",
    red="B05",
    green="B04",
    blue="B03",
    vmin=(0.0, 0.0, 0.0),
    vmax=(0.975, 1.086, 1.0),
    gamma=(1.0, 1.0, 1.0),
    channels=("B03", "B04", "B05"),
    description=(
        "Day land cloud: vegetation green, bare soil brown, water clouds "
        "white, ice clouds and snow cyan."
    ),
)

DAY_CLOUD_PHASE = Recipe(
    name="day_cloud_phase",
    red="B13",
    green="B03",
    blue="B05",
    # Red runs from 7.5 °C (dark) to -53.5 °C (bright): cold tops glow red.
    vmin=(280.65, 0.0, 0.01),
    vmax=(219.65, 0.78, 0.59),
    gamma=(1.0, 1.0, 1.0),
    channels=("B03", "B05", "B13"),
    description=(
        "Day cloud phase distinction: glaciating and ice tops red / orange, "
        "liquid low clouds cyan, snow green."
    ),
)

FIRE_TEMPERATURE = Recipe(
    name="fire_temperature",
    red="B07",
    green="B06",
    blue="B05",
    vmin=(273.15, 0.0, 0.0),
    vmax=(333.15, 1.0, 0.75),
    gamma=(0.4, 1.0, 1.0),
    channels=("B05", "B06", "B07"),
    description=(
        "Fire temperature: hot fires red through yellow to white as the 2.3 "
        "and 1.6 µm bands saturate."
    ),
)

RECIPES: dict[str, Recipe] = {
    r.name: r for r in (TRUE_COLOR, NATURAL_COLOR, DAY_CLOUD_PHASE, FIRE_TEMPERATURE)
}
