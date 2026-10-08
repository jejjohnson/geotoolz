"""`geocatalog.utils` — URI, retry and time helpers the catalog uses.

- `parse_uri` / `ParsedURI` — split ``s3://`` / ``gs://`` / ``az://`` /
  ``https://`` / local paths into scheme, bucket and key.
- `retry_transient_io` — retry a callable on transient network errors.
- UTC time: `to_utc_ts`, `to_naive_utc`, `to_rfc3339`, and
  `is_time_invariant` with its `TIME_INVARIANT_START` /
  `TIME_INVARIANT_END` sentinels.
"""

from __future__ import annotations

from geocatalog._src._timeutil import (
    TIME_INVARIANT_END,
    TIME_INVARIANT_START,
    is_time_invariant,
    to_naive_utc,
    to_rfc3339,
    to_utc_ts,
)
from geocatalog._src.retry import retry_transient_io
from geocatalog._src.uri import (
    ParsedURI,
    parse_uri,
)


__all__ = [
    "TIME_INVARIANT_END",
    "TIME_INVARIANT_START",
    "ParsedURI",
    "is_time_invariant",
    "parse_uri",
    "retry_transient_io",
    "to_naive_utc",
    "to_rfc3339",
    "to_utc_ts",
]
