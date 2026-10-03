"""`geocatalog.io` — path / URI parsing, retries and UTC time helpers.

The helpers the catalog itself uses to reach files and normalise
times, at a public path:

* `parse_uri` / `ParsedURI` — one reading of local paths and cloud URIs
  (scheme, local vs remote, name / suffix, Zarr detection, the local
  path of ``file://`` URIs, the GDAL ``/vsi*/`` path).
* `retry_transient_io` — the retry policy the loaders, adapters and
  staging share (transient remote errors only, exponential backoff).
* `to_utc_ts` / `to_naive_utc` / `to_rfc3339` — the catalog's time
  contract (times stored as naive UTC), plus `is_time_invariant` and the
  `TIME_INVARIANT_START` / `TIME_INVARIANT_END` sentinel interval.

Every name is also at the top level.
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
from geocatalog._src.uri import ParsedURI, parse_uri


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
