"""Shared UTC / RFC 3339 time coercion helpers.

The catalog's stored time-axis contract is UTC: naive timestamps are
assumed to already be UTC, while tz-aware timestamps in other zones are
converted. Inside a catalog, times are held **naive, in UTC** — the form
existing GeoParquet artifacts, `from_stac_items` and the streaming
writer already produce — so naive and aware inputs can be mixed freely.

- `to_naive_utc` / `naive_utc_interval` / `naive_utc_datetimes` /
  `naive_utc_interval_index` produce that stored form; every catalog
  constructor, query path and ingest path goes through them.
- `to_utc_ts` / `to_rfc3339` produce the aware / ``Z``-suffixed form for
  talking to external metadata (STAC, CMR/UMM).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def to_utc_ts(value: Any) -> pd.Timestamp:
    """Coerce any datetime-like to a UTC-aware `pd.Timestamp`.

    Naive inputs are assumed UTC (the catalog's stored time-axis
    contract) and localised; tz-aware inputs in other zones are
    converted.

    Args:
        value: Anything `pd.Timestamp` accepts — ISO string, ``datetime``,
            ``pd.Timestamp``, epoch-like.

    Returns:
        A tz-aware ``pd.Timestamp`` in UTC.
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def to_naive_utc(value: Any) -> pd.Timestamp:
    """Coerce a datetime-like to the catalog's stored form: naive, in UTC.

    Naive inputs are returned unchanged (they are UTC by contract);
    tz-aware inputs are converted to UTC and the zone dropped. ``NaT``
    passes through.

    Args:
        value: Anything `pd.Timestamp` accepts.

    Returns:
        A tz-naive ``pd.Timestamp`` (or ``pd.NaT``).
    """
    ts = pd.Timestamp(value)
    if ts is pd.NaT or ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def naive_utc_interval(interval: pd.Interval) -> pd.Interval:
    """Return ``interval`` with timestamp endpoints in naive UTC.

    Non-timestamp intervals (e.g. numeric) are returned unchanged, as are
    intervals whose endpoints are already naive.
    """
    left, right = interval.left, interval.right
    if not (isinstance(left, pd.Timestamp) and isinstance(right, pd.Timestamp)):
        return interval
    if left.tzinfo is None and right.tzinfo is None:
        return interval
    return pd.Interval(to_naive_utc(left), to_naive_utc(right), closed=interval.closed)


def naive_utc_datetimes(values: pd.Series) -> pd.Series:
    """Return a datetime ``Series`` in naive UTC.

    Naive ``datetime64`` input is returned as-is (no copy). tz-aware
    input is converted. ``object`` input — e.g. a column mixing naive
    and aware timestamps from different sources — is coerced element
    by element.
    """
    dtype = values.dtype
    if getattr(dtype, "tz", None) is not None:
        return values.dt.tz_convert("UTC").dt.tz_localize(None)
    if dtype == object:
        return pd.to_datetime(values.map(to_naive_utc))
    return values


def naive_utc_interval_index(index: pd.IntervalIndex) -> pd.IntervalIndex:
    """Return an ``IntervalIndex`` whose datetime endpoints are naive UTC."""
    left = index.left
    if not (isinstance(left, pd.DatetimeIndex) and left.tz is not None):
        return index
    right = index.right
    assert isinstance(right, pd.DatetimeIndex)
    return pd.IntervalIndex.from_arrays(
        left.tz_convert("UTC").tz_localize(None),
        right.tz_convert("UTC").tz_localize(None),
        closed=index.closed,
        name=index.name,
    )


def to_rfc3339(value: Any) -> str:
    """Serialize a datetime-like as RFC 3339 in UTC with a ``Z`` suffix.

    ``.isoformat()`` on a UTC-aware Timestamp yields ``...+00:00``; the
    STAC-canonical form uses ``Z``, so the offset is normalised.

    Args:
        value: Anything `pd.Timestamp` accepts. Naive inputs are assumed
            UTC (see `to_utc_ts`).

    Returns:
        An RFC 3339 string, e.g. ``"2024-06-15T10:00:00Z"``.
    """
    return to_utc_ts(value).isoformat().replace("+00:00", "Z")
