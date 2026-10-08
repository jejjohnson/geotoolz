"""The standard ABI RGB recipes, as data.

Each operational RGB is three band expressions, each linearly stretched
between a fixed ``vmin`` and ``vmax`` and gamma-corrected. The numbers
follow the published quick guides (CIRA / NOAA / CIMSS). Inputs are ABI
channel names (``band_names`` of ``goes.Reader`` / ``goes.L2Reader``
outputs): reflectance factor (0-1) for C01-C06, brightness temperature
(K) for C07-C16, i.e. ``calibration="reflectance"`` /
``"brightness_temperature"`` L1b reads or MCMIP / CMIP imagery.

``goes.presets`` turns a recipe into a ``geotoolz.viz.RGBRecipe``
operator; the recipes themselves need no extra.
"""

from __future__ import annotations

from geoproducts._src.presets import Recipe


__all__ = [
    "DAY_CLOUD_PHASE",
    "FIRE_TEMPERATURE",
    "NATURAL_COLOR",
    "RECIPES",
    "TRUE_COLOR",
    "Recipe",
]


TRUE_COLOR = Recipe(
    name="true_color",
    red="C02",
    # CIMSS synthetic green: ABI has no green band.
    green="0.45 * C02 + 0.1 * C03 + 0.45 * C01",
    blue="C01",
    vmin=(0.0, 0.0, 0.0),
    vmax=(1.0, 1.0, 1.0),
    gamma=(2.2, 2.2, 2.2),
    channels=("C01", "C02", "C03"),
    description="Daytime true colour with the CIMSS synthetic green band.",
)

NATURAL_COLOR = Recipe(
    name="natural_color",
    red="C05",
    green="C03",
    blue="C02",
    vmin=(0.0, 0.0, 0.0),
    vmax=(0.975, 1.086, 1.0),
    gamma=(1.0, 1.0, 1.0),
    channels=("C02", "C03", "C05"),
    description=(
        "CIRA day land cloud: vegetation green, bare soil brown, water clouds "
        "white, ice clouds and snow cyan."
    ),
)

DAY_CLOUD_PHASE = Recipe(
    name="day_cloud_phase",
    red="C13",
    green="C02",
    blue="C05",
    # Red runs from 7.5 °C (dark) to -53.5 °C (bright): cold tops glow red.
    vmin=(280.65, 0.0, 0.01),
    vmax=(219.65, 0.78, 0.59),
    gamma=(1.0, 1.0, 1.0),
    channels=("C02", "C05", "C13"),
    description=(
        "CIRA day cloud phase distinction: glaciating and ice tops red / "
        "orange, liquid low clouds cyan, snow green."
    ),
)

FIRE_TEMPERATURE = Recipe(
    name="fire_temperature",
    red="C07",
    green="C06",
    blue="C05",
    vmin=(273.15, 0.0, 0.0),
    vmax=(333.15, 1.0, 0.75),
    gamma=(0.4, 1.0, 1.0),
    channels=("C05", "C06", "C07"),
    description=(
        "CIRA fire temperature: hot fires red through yellow to white as "
        "the 2.2 and 1.6 µm channels saturate."
    ),
)

RECIPES: dict[str, Recipe] = {
    r.name: r for r in (TRUE_COLOR, NATURAL_COLOR, DAY_CLOUD_PHASE, FIRE_TEMPERATURE)
}
