"""The slot of a date in a daily climatology."""

from __future__ import annotations

import xarray as xr
from diagnostix.climate import calendar_day


#: The days of the calendars that have no 29 February in some years only.
DAYS_OF_YEAR: dict[str, int] = {"noleap": 365, "365_day": 365, "360_day": 360}


def day_of_year(times: xr.DataArray) -> xr.DataArray:
    """The day of each date on a leap-year calendar, named ``dayofyear``.

    29 February is 60 and 1 March is 61 in every year. A calendar whose years
    all have the same days keeps its own day of the year.
    """
    if times.dt.calendar in DAYS_OF_YEAR:
        return times.dt.dayofyear
    return calendar_day(times.dt.month, times.dt.day).rename("dayofyear")
