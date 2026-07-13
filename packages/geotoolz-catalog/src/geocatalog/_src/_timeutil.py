"""Shared UTC / RFC 3339 time coercion helpers.

The catalog's stored time-axis contract is UTC: naive timestamps are
assumed to already be UTC, while tz-aware timestamps in other zones are
converted. These two helpers centralise that contract for every module
that talks to external metadata (STAC, CMR/UMM) so the coercion rule is
spelled exactly once.
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
