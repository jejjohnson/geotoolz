"""Temporal window type."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pandas as pd


@dataclass(frozen=True)
class TimeRange:
    """Inclusive time window, optionally with a target sampling frequency.

    ``freq`` is a pandas offset alias (``"1D"``, ``"6H"`` ...) used only
    by serializers that need to materialize the range (e.g. the CDS
    ``year/month/day`` form). Adapters accepting continuous ranges
    (CMEMS) ignore ``freq``.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    freq: str | None = None

    def __post_init__(self) -> None:
        # UTC-awareness is a class invariant, not a `parse()` detail:
        # downstream paths (AEMET hourly `tz_convert`, pollution
        # filtering against aware payload timestamps) assume it. Coerce
        # here so direct construction behaves like `parse()`.
        object.__setattr__(self, "start", _as_ts(self.start))
        object.__setattr__(self, "end", _as_ts(self.end))
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")

    # ---- constructors -----------------------------------------------------

    @classmethod
    def parse(
        cls,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
        freq: str | None = None,
    ) -> TimeRange:
        """Parse heterogeneous inputs into a ``TimeRange``.

        Accepts ISO strings, ``datetime``, ``pandas.Timestamp``.
        Strings without timezone are treated as UTC.
        """
        return cls(start=_as_ts(start), end=_as_ts(end), freq=freq)

    # ---- derived ----------------------------------------------------------

    def to_index(self) -> pd.DatetimeIndex:
        """Materialize the range at :attr:`freq` (default daily)."""
        return pd.date_range(self.start, self.end, freq=self.freq or "1D")

    # ---- serializers ------------------------------------------------------

    def as_cmems(self) -> dict[str, str]:
        return {
            "start_datetime": self.start.isoformat(),
            "end_datetime": self.end.isoformat(),
        }

    def as_cds_form(self) -> dict[str, list[str]]:
        """Explode the range into the ``year/month/day`` form CDS expects."""
        idx = self.to_index()
        years = sorted({f"{t.year}" for t in idx})
        months = sorted({f"{t.month:02d}" for t in idx})
        days = sorted({f"{t.day:02d}" for t in idx})
        return {"year": years, "month": months, "day": days}

    def cds_times(self) -> list[str]:
        """Hours-of-day (``"HH:MM"``) the range touches, for the CDS ``time`` key.

        Derived from :meth:`to_index` at :attr:`freq` (default daily, so
        ``["00:00"]``). ERA5-style reanalyses are hourly products and
        require this field alongside ``year/month/day``; a finer ``freq``
        (e.g. ``"6h"``) expands the returned hours accordingly.
        """
        idx = self.to_index()
        return sorted({f"{t.hour:02d}:{t.minute:02d}" for t in idx})

    def as_xarray_sel(self, time: str = "time") -> dict[str, slice]:
        """``ds.sel(**tr.as_xarray_sel())`` selector."""
        return {time: slice(self.start, self.end)}


def _as_ts(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """Coerce to a UTC pandas Timestamp."""
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        raise ValueError(f"could not parse timestamp: {value!r}")
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    return cast(pd.Timestamp, ts)
