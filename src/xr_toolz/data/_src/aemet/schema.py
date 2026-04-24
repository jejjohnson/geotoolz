"""AEMET field-name → canonical :class:`Variable` mappings.

The daily and hourly endpoints overlap conceptually but use different
field names for the same physical quantity (e.g. ``tmed`` daily mean vs
``ta`` instantaneous air temperature). Each endpoint therefore gets its
own mapping so ``AemetSource`` can translate without guessing.
"""

from __future__ import annotations

from collections.abc import Mapping

from xr_toolz.types import (
    AIR_TEMPERATURE,
    AIR_TEMPERATURE_DAILY_MAX,
    AIR_TEMPERATURE_DAILY_MEAN,
    AIR_TEMPERATURE_DAILY_MIN,
    AIR_TEMPERATURE_MAX,
    AIR_TEMPERATURE_MIN,
    DEW_POINT_TEMPERATURE,
    MEAN_SEA_LEVEL_PRESSURE_HPA,
    PRECIPITATION_AMOUNT,
    RELATIVE_HUMIDITY,
    SOIL_TEMPERATURE_5CM,
    SOIL_TEMPERATURE_20CM,
    SUNSHINE_DURATION,
    SUNSHINE_DURATION_DAILY,
    SURFACE_PRESSURE_HPA,
    SURFACE_PRESSURE_MAX_HPA,
    SURFACE_PRESSURE_MIN_HPA,
    SURFACE_SNOW_THICKNESS,
    VISIBILITY,
    WIND_FROM_DIRECTION,
    WIND_FROM_DIRECTION_DAILY,
    WIND_FROM_DIRECTION_OF_GUST,
    WIND_SPEED,
    WIND_SPEED_DAILY_MEAN,
    WIND_SPEED_OF_GUST,
    WIND_SPEED_OF_GUST_DAILY,
    Variable,
)


HOURLY_FIELDS: Mapping[str, Variable] = {
    "ta": AIR_TEMPERATURE,
    "tamin": AIR_TEMPERATURE_MIN,
    "tamax": AIR_TEMPERATURE_MAX,
    "tpr": DEW_POINT_TEMPERATURE,
    "hr": RELATIVE_HUMIDITY,
    "prec": PRECIPITATION_AMOUNT,
    "pres": SURFACE_PRESSURE_HPA,
    "pres_nmar": MEAN_SEA_LEVEL_PRESSURE_HPA,
    "vv": WIND_SPEED,
    "dv": WIND_FROM_DIRECTION,
    "vmax": WIND_SPEED_OF_GUST,
    "dmax": WIND_FROM_DIRECTION_OF_GUST,
    "inso": SUNSHINE_DURATION,
    "vis": VISIBILITY,
    "nieve": SURFACE_SNOW_THICKNESS,
    "tss5cm": SOIL_TEMPERATURE_5CM,
    "tss20cm": SOIL_TEMPERATURE_20CM,
}


DAILY_FIELDS: Mapping[str, Variable] = {
    "tmed": AIR_TEMPERATURE_DAILY_MEAN,
    "tmin": AIR_TEMPERATURE_DAILY_MIN,
    "tmax": AIR_TEMPERATURE_DAILY_MAX,
    "prec": PRECIPITATION_AMOUNT,
    "velmedia": WIND_SPEED_DAILY_MEAN,
    "racha": WIND_SPEED_OF_GUST_DAILY,
    "dir": WIND_FROM_DIRECTION_DAILY,
    "presMax": SURFACE_PRESSURE_MAX_HPA,
    "presMin": SURFACE_PRESSURE_MIN_HPA,
    "sol": SUNSHINE_DURATION_DAILY,
}


# Daily endpoint extras that are useful but not numeric (hour-of-extreme
# strings). Kept as a separate set so the loader can pass them through
# without attempting float conversion.
DAILY_PASSTHROUGH_FIELDS: frozenset[str] = frozenset(
    {"horatmin", "horatmax", "horaracha", "horaPresMax", "horaPresMin"}
)


# Monthly-annual endpoint — same physical quantities as daily but
# aggregated; AEMET uses a different field vocabulary. We surface only
# the most common fields via the Variable registry and leave the rest
# accessible as raw attrs (aggregations like ``np_*`` percentile ranks).
MONTHLY_FIELDS: Mapping[str, Variable] = {
    # ``tm_mes`` is the monthly mean; ``tm_min`` / ``tm_max`` are the
    # *means of daily* min/max over the month (the usual climate
    # quantities). AEMET also exposes ``ta_min`` / ``ta_max`` — the
    # *absolute* extremes annotated with day-of-month (e.g.
    # ``"-1.8(23)"``) — which we keep in the raw attrs since they're
    # not time-series-scalar.
    "tm_mes": AIR_TEMPERATURE_DAILY_MEAN,
    "tm_min": AIR_TEMPERATURE_DAILY_MIN,
    "tm_max": AIR_TEMPERATURE_DAILY_MAX,
    "p_mes": PRECIPITATION_AMOUNT,
    "w_med": WIND_SPEED_DAILY_MEAN,
    "w_racha": WIND_SPEED_OF_GUST_DAILY,
    "q_med": SURFACE_PRESSURE_HPA,
    "q_max": SURFACE_PRESSURE_MAX_HPA,
    "q_min": SURFACE_PRESSURE_MIN_HPA,
    "inso": SUNSHINE_DURATION_DAILY,
}


def canonical_for(endpoint: str, field: str) -> Variable | None:
    """Return the canonical :class:`Variable` for ``field`` in ``endpoint``.

    ``endpoint`` is one of ``"hourly"``, ``"daily"``, ``"monthly"``.
    Returns ``None`` for fields that aren't mapped (station metadata,
    hour-of-extreme columns, AEMET-internal extras).
    """
    match endpoint:
        case "hourly":
            return HOURLY_FIELDS.get(field)
        case "daily":
            return DAILY_FIELDS.get(field)
        case "monthly":
            return MONTHLY_FIELDS.get(field)
        case _:
            raise ValueError(f"unknown endpoint: {endpoint!r}")
