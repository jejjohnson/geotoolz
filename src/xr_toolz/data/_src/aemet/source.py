"""AEMET OpenData adapter.

The AEMET API is unusual in two ways:

1. **Two-hop fetch**. Endpoints return a JSON envelope with short-lived
   signed URLs (``datos`` and ``metadatos``). The actual payload must
   be fetched from ``datos`` in a second request; the ``api_key`` is
   only needed on the first hop.
2. **Conservative rate limit**. The response header
   ``Remaining-request-count`` ticks down toward zero; hitting it
   returns HTTP 429. We watch the header and back off before it fires.

Concurrency uses a thread pool (``max_workers``) because AEMET returns
~150 KB blobs per station × chunk and the bottleneck is the network,
not the parser. Async would be marginally more efficient but the sync
wrapper composes with the rest of :mod:`xr_toolz.data`.
"""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from xr_toolz.data._src.aemet.geo import canonical_province, community_for
from xr_toolz.data._src.aemet.parsers import (
    format_aemet_datetime,
    parse_aemet_datetime,
    parse_dms,
    parse_spanish_float,
)
from xr_toolz.data._src.aemet.schema import (
    DAILY_FIELDS,
    DAILY_PASSTHROUGH_FIELDS,
    HOURLY_FIELDS,
    MONTHLY_FIELDS,
)
from xr_toolz.data._src.base import DatasetInfo, DataSource
from xr_toolz.data._src.credentials import AEMETCredentials, load_aemet
from xr_toolz.types import (
    BBox,
    DepthRange,
    PressureLevels,
    Station,
    StationCollection,
    TimeRange,
    Variable,
)


AEMET_BASE_URL = "https://opendata.aemet.es/opendata/api"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_WORKERS = 4
RATE_LIMIT_HEADER = "Remaining-request-count"
RATE_LIMIT_FLOOR = 5  # pause when fewer remaining requests than this
DAILY_WINDOW_DAYS = 180  # AEMET's undocumented per-request cap
# AEMET's monthly endpoint caps each request at 36 months, despite the
# endpoint accepting ``anioini``/``aniofin`` that look year-flexible.
# Requests with wider spans return a 404 body ("El rango de las
# fechas no puede ser superior a 36 meses").
MONTHLY_WINDOW_YEARS = 3
POLLUTION_WINDOW_DAYS = 90


class AemetError(RuntimeError):
    """Any AEMET-level failure (auth, rate limit, bad envelope, 5xx)."""


class AemetAuthError(AemetError):
    """Missing or rejected API key."""


class AemetRateLimitError(AemetError):
    """Quota exhausted even after backoff."""


class AemetNoDataError(AemetError):
    """Envelope returned no ``datos`` URL (station has no data, or bad range)."""


@dataclass(frozen=True)
class _Envelope:
    """Parsed first-hop response from the AEMET API."""

    datos: str
    metadatos: str | None
    status: int
    description: str
    remaining: int | None


class AemetSource(DataSource):
    """Adapter for AEMET OpenData observation endpoints.

    Args:
        credentials: Explicit credentials. When ``None``, resolved from
            the environment / ``.env`` / ``~/.aemet`` via
            :func:`~xr_toolz.data.load_aemet`.
        client: Optional pre-built HTTP client (must expose a
            ``get(url, timeout=...)`` method returning an object with
            ``.status_code``, ``.headers``, ``.json()``, ``.raise_for_status()``
            — compatible with ``httpx.Client`` and test doubles).
        base_url: Override the API root (useful for staging / mocks).
        timeout_s: Per-request timeout.
        max_retries: Retries for 429 / transient network errors.
        max_workers: Thread pool size for parallel station fetches.
        min_interval_s: Minimum seconds between consecutive HTTP calls,
            enforced globally across the worker pool. AEMET's rate
            limit is ``~150`` requests per rolling minute — ``0.5``
            (120 req/min) leaves comfortable headroom and avoids the
            burst-detection revocation we saw at ``max_workers=8``.
            Set to ``0.0`` to disable pacing.
    """

    source_id = "aemet"

    def __init__(
        self,
        credentials: AEMETCredentials | None = None,
        client: Any | None = None,
        base_url: str = AEMET_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_workers: int = DEFAULT_MAX_WORKERS,
        min_interval_s: float = 0.0,
    ) -> None:
        self.credentials = credentials or load_aemet()
        self._client = client
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.min_interval_s = max(0.0, min_interval_s)
        # Global pacing lock — a monotonic timestamp of the last
        # outbound HTTP call, shared across all worker threads.
        self._pace_lock = threading.Lock()
        self._pace_last = 0.0
        # Global 429 backoff — when any worker hits a rate limit, all
        # workers (including that one on its retry) wait until this
        # deadline. Without this shared pause, other workers keep the
        # bucket topped up while the 429'd worker backs off, so the
        # minute window never actually clears.
        self._rate_limited_until = 0.0
        # Scale factor on the 429 global-pause schedule. Production
        # runs want full minute-scale pauses to let AEMET's minute
        # bucket drain; tests set this to 0 to avoid actually sleeping.
        self.rate_limit_pause_scale = 1.0

    # ---- client handling --------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                "AemetSource requires optional AEMET dependencies. "
                "Install with: pip install 'xr_toolz[aemet]'"
            ) from exc
        self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def _rate_limit(self) -> None:
        """Block until it's safe to make another outbound call.

        Enforces two gates, both shared across worker threads:

        1. ``min_interval_s`` between consecutive calls (per-request
           pacing). Acts as a token bucket: one request every
           ``min_interval_s`` seconds.
        2. ``_rate_limited_until`` — a global pause set whenever any
           worker observes a 429. While it's in the future, every
           worker blocks here, which actually lets AEMET's minute
           window drain instead of keeping it topped up from the
           other workers.
        """
        with self._pace_lock:
            now = time.monotonic()
            # Gate 2: global 429 pause takes precedence.
            wait_rl = self._rate_limited_until - now
            if wait_rl > 0:
                time.sleep(wait_rl)
                now = time.monotonic()
            # Gate 1: per-request pacing.
            if self.min_interval_s > 0:
                wait_pace = self._pace_last + self.min_interval_s - now
                if wait_pace > 0:
                    time.sleep(wait_pace)
                    now = time.monotonic()
            self._pace_last = now

    def _trip_rate_limit(self, seconds: float) -> None:
        """Register a global pause of ``seconds`` from now, shared across workers."""
        with self._pace_lock:
            deadline = time.monotonic() + seconds
            if deadline > self._rate_limited_until:
                self._rate_limited_until = deadline

    def _require_key(self) -> str:
        if self.credentials is None:
            raise AemetAuthError(
                "No AEMET API key found. Set AEMET_API_KEY in your .env or "
                "environment (see xr_toolz.data.load_aemet)."
            )
        return self.credentials.api_key

    # ---- DataSource API --------------------------------------------------

    def list_datasets(self) -> list[DatasetInfo]:
        from xr_toolz.data._src.aemet.catalog import AEMET_DATASETS

        return list(AEMET_DATASETS.values())

    def describe(self, dataset_id: str) -> DatasetInfo:
        from xr_toolz.data._src.aemet.catalog import AEMET_DATASETS

        if dataset_id in AEMET_DATASETS:
            return AEMET_DATASETS[dataset_id]
        return DatasetInfo(
            dataset_id=dataset_id,
            source=self.source_id,
            title=dataset_id,
        )

    def download(
        self,
        dataset_id: str,
        output: Path,
        *,
        variables: list[str | Variable] | None = None,
        bbox: BBox | None = None,
        time: TimeRange | None = None,
        depth: DepthRange | None = None,
        levels: PressureLevels | None = None,
        **extras: Any,
    ) -> Path:
        """Download ``dataset_id`` to a NetCDF file at ``output``."""
        ds = self.open(
            dataset_id,
            variables=variables,
            bbox=bbox,
            time=time,
            depth=depth,
            levels=levels,
            **extras,
        )
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output)
        return output

    def open(
        self,
        dataset_id: str,
        *,
        variables: list[str | Variable] | None = None,
        bbox: BBox | None = None,
        time: TimeRange | None = None,
        depth: DepthRange | None = None,
        levels: PressureLevels | None = None,
        stations: Iterable[str] | StationCollection | None = None,
        **extras: Any,
    ) -> xr.Dataset:
        """Open ``dataset_id`` as an ``(station, time)`` xarray Dataset.

        ``dataset_id`` is one of the preset IDs from
        :mod:`~xr_toolz.data._src.aemet.catalog` (``aemet_daily``,
        ``aemet_hourly``, ``aemet_monthly``, ``aemet_pollution``,
        ``aemet_stations``, ``aemet_normals``, ``aemet_extremes``).
        """
        info = self.describe(dataset_id)
        kind = info.extras.get("aemet_kind", dataset_id)
        known = {
            "aemet_stations",
            "aemet_daily",
            "aemet_hourly",
            "aemet_monthly",
            "aemet_normals",
            "aemet_extremes",
            "aemet_pollution",
        }
        if kind not in known:
            raise ValueError(f"unknown AEMET dataset: {dataset_id}")

        # The inventory preset doesn't need station IDs — it *is* the
        # station list. All other presets require at least one station
        # ID or a BBox to derive them from.
        if kind == "aemet_stations":
            return _stations_to_dataset(self.list_stations(bbox=bbox))

        station_ids = self._resolve_station_ids(stations, bbox)

        if kind == "aemet_daily":
            return self.get_daily(station_ids, time=time, variables=variables)
        if kind == "aemet_hourly":
            return self.get_hourly(station_ids, variables=variables)
        if kind == "aemet_monthly":
            return self.get_monthly(station_ids, time=time, variables=variables)
        if kind == "aemet_normals":
            return self.get_normals(station_ids)
        if kind == "aemet_extremes":
            parameter = str(extras.get("parameter", "T"))
            return self.get_extremes(station_ids, parameter=parameter)
        # kind == "aemet_pollution"
        return self.get_pollution(station_ids, time=time)

    # ---- inventory --------------------------------------------------------

    def list_stations(self, bbox: BBox | None = None) -> StationCollection:
        """Return the full AEMET station inventory, optionally filtered."""
        payload = self._get_json(
            "/valores/climatologicos/inventarioestaciones/todasestaciones"
        )
        stations = tuple(_parse_station(row) for row in payload)
        collection = StationCollection(stations)
        if bbox is not None:
            collection = collection.within(bbox)
        return collection

    # ---- daily climatology ------------------------------------------------

    def get_daily(
        self,
        stations: Iterable[str],
        *,
        time: TimeRange | None = None,
        variables: list[str | Variable] | None = None,
    ) -> xr.Dataset:
        """Fetch daily climatology for ``stations`` in ``time``.

        Automatically chunks each station's window into 180-day slices
        (AEMET's undocumented per-request cap) and fetches in parallel.
        """
        station_ids = tuple(stations)
        if not station_ids:
            raise ValueError("at least one station id required")
        if time is None:
            raise ValueError("daily endpoint requires a TimeRange")

        start_dt = time.start.to_pydatetime()
        end_dt = time.end.to_pydatetime()

        frames = self._fetch_station_windows(
            station_ids,
            start_dt,
            end_dt,
            chunk_days=DAILY_WINDOW_DAYS,
            url_fn=_daily_url,
            parse_fn=_parse_daily_rows,
        )
        full_index = pd.date_range(start_dt.date(), end_dt.date(), freq="1D", tz=UTC)
        ds = _frames_to_stations_dataset(
            frames,
            station_ids=station_ids,
            time_index=full_index,
            field_map=DAILY_FIELDS,
            passthrough=DAILY_PASSTHROUGH_FIELDS,
            endpoint_name="daily",
        )
        if variables:
            ds = _subset_variables(ds, variables)
        return ds

    # ---- hourly / conventional -------------------------------------------

    def get_hourly(
        self,
        stations: Iterable[str],
        *,
        variables: list[str | Variable] | None = None,
    ) -> xr.Dataset:
        """Fetch the latest ~24 h of hourly observations for ``stations``.

        The AEMET hourly endpoint is rolling — it returns whatever window
        of recent observations the station has published (typically the
        last 24 hours). There is no historical-window query.
        """
        station_ids = tuple(stations)
        if not station_ids:
            raise ValueError("at least one station id required")

        def fetch(sid: str) -> list[dict[str, Any]]:
            return self._get_json(f"/observacion/convencional/datos/estacion/{sid}")

        rows_by_station: dict[str, list[dict[str, Any]]] = {}
        times_seen: set[datetime] = set()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(fetch, sid): sid for sid in station_ids}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    rows = fut.result() or []
                except AemetNoDataError:
                    rows = []
                except (AemetAuthError, AemetRateLimitError):
                    # Auth / quota failures are not per-chunk
                    # problems — they fail the whole scrape, so we
                    # let them propagate instead of silently
                    # skipping every remaining station.
                    raise
                except AemetError as exc:
                    warnings.warn(
                        f"AEMET hourly failed for station {sid}: {exc}",
                        stacklevel=2,
                    )
                    rows = []
                rows_by_station[sid] = rows
                for row in rows:
                    if "fint" in row:
                        times_seen.add(parse_aemet_datetime(row["fint"]))

        if not times_seen:
            raise AemetError("AEMET returned no hourly observations")
        time_index = pd.DatetimeIndex(sorted(times_seen), tz=UTC)

        frames: dict[str, pd.DataFrame] = {}
        for sid, rows in rows_by_station.items():
            if not rows:
                continue
            frames[sid] = _hourly_frame(rows)

        ds = _frames_to_stations_dataset(
            frames,
            station_ids=station_ids,
            time_index=time_index,
            field_map=HOURLY_FIELDS,
            passthrough=frozenset(),
            endpoint_name="hourly",
        )
        if variables:
            ds = _subset_variables(ds, variables)
        return ds

    # ---- monthly / annual -------------------------------------------------

    def get_monthly(
        self,
        stations: Iterable[str],
        *,
        time: TimeRange | None = None,
        variables: list[str | Variable] | None = None,
    ) -> xr.Dataset:
        """Fetch monthly aggregates for ``stations`` over ``time``."""
        station_ids = tuple(stations)
        if not station_ids:
            raise ValueError("at least one station id required")
        if time is None:
            raise ValueError("monthly endpoint requires a TimeRange")

        year_start = time.start.year
        year_end = time.end.year

        def fetch(sid: str, y1: int, y2: int) -> list[dict[str, Any]]:
            return self._get_json(
                f"/valores/climatologicos/mensualesanuales/datos/"
                f"anioini/{y1}/aniofin/{y2}/estacion/{sid}"
            )

        frames: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for sid in station_ids:
                for y1, y2 in _chunk_years(year_start, year_end, MONTHLY_WINDOW_YEARS):
                    futures[pool.submit(fetch, sid, y1, y2)] = sid
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    rows = fut.result() or []
                except AemetNoDataError:
                    # Chunk empty for this station (e.g. started after
                    # the request window). Skip — a gap is still a
                    # valid, observable outcome (full time axis + NaN).
                    continue
                except (AemetAuthError, AemetRateLimitError):
                    # Auth / quota failures are not per-chunk
                    # problems — they fail the whole scrape, so we
                    # let them propagate instead of silently
                    # skipping every remaining station.
                    raise
                except AemetError as exc:
                    # Hard error on one chunk shouldn't nuke the
                    # whole scrape; log and continue so the archive
                    # gets whatever data we could reach. Re-running
                    # later resumes from the archive's latest time.
                    warnings.warn(
                        f"AEMET chunk failed for station {sid}: {exc}",
                        stacklevel=2,
                    )
                    continue
                if not rows:
                    continue
                frame = _monthly_frame(rows)
                if sid in frames:
                    frames[sid] = pd.concat([frames[sid], frame])
                else:
                    frames[sid] = frame

        # Fetch chunks cover whole years (AEMET endpoint granularity),
        # but return only the months the caller actually asked for.
        # Without this trim, ``TimeRange.parse("2024-06", "2024-09")``
        # would still return Jan-Dec 2024 and incremental archive
        # syncs would repeatedly rewrite entire years.
        req_start = time.start.tz_convert(UTC) if time.start.tz else time.start
        req_end = time.end.tz_convert(UTC) if time.end.tz else time.end
        month_start = pd.Timestamp(
            year=req_start.year, month=req_start.month, day=1, tz=UTC
        )
        month_end = pd.Timestamp(year=req_end.year, month=req_end.month, day=1, tz=UTC)
        time_index = pd.date_range(month_start, month_end, freq="MS", tz=UTC)

        ds = _frames_to_stations_dataset(
            frames,
            station_ids=station_ids,
            time_index=time_index,
            field_map=MONTHLY_FIELDS,
            passthrough=frozenset(),
            endpoint_name="monthly",
        )
        if variables:
            ds = _subset_variables(ds, variables)
        return ds

    # ---- normals / extremes / pollution ----------------------------------

    def get_normals(self, stations: Iterable[str]) -> xr.Dataset:
        """Fetch 1981–2010 climate normals for ``stations``.

        Emitted as a ``(station, month)`` dataset with month=1..12.
        """
        station_ids = tuple(stations)

        def fetch(sid: str) -> list[dict[str, Any]]:
            return self._get_json(f"/valores/climatologicos/normales/estacion/{sid}")

        rows_per: dict[str, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(fetch, sid): sid for sid in station_ids}
            for fut in as_completed(futures):
                sid = futures[fut]
                # Match the per-chunk tolerance used by daily / monthly
                # / pollution: a single station's missing payload
                # shouldn't abort a many-station batch.
                try:
                    rows_per[sid] = fut.result() or []
                except AemetNoDataError:
                    rows_per[sid] = []
                except (AemetAuthError, AemetRateLimitError):
                    raise
                except AemetError as exc:
                    warnings.warn(
                        f"AEMET normals failed for station {sid}: {exc}",
                        stacklevel=2,
                    )
                    rows_per[sid] = []
        return _normals_to_dataset(rows_per, station_ids)

    def get_extremes(
        self, stations: Iterable[str], *, parameter: str = "T"
    ) -> xr.Dataset:
        """Fetch record extremes (P=precip, T=temp, V=wind) per station.

        Extremes are a ragged, per-field set of records. We emit them as
        a minimal ``(station,)`` dataset with the raw dictionaries kept
        in ``attrs['records']`` rather than guessing a schema that may
        vary per station.
        """
        station_ids = tuple(stations)
        if parameter not in {"P", "T", "V"}:
            raise ValueError(f"parameter must be one of P/T/V, got {parameter!r}")

        def fetch(sid: str) -> list[dict[str, Any]]:
            return self._get_json(
                f"/valores/climatologicos/valoresextremos/"
                f"parametro/{parameter}/estacion/{sid}"
            )

        rows_per: dict[str, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(fetch, sid): sid for sid in station_ids}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    rows_per[sid] = fut.result() or []
                except AemetNoDataError:
                    rows_per[sid] = []
                except (AemetAuthError, AemetRateLimitError):
                    raise
                except AemetError as exc:
                    warnings.warn(
                        f"AEMET extremes failed for station {sid}: {exc}",
                        stacklevel=2,
                    )
                    rows_per[sid] = []
        return _extremes_to_dataset(rows_per, station_ids, parameter)

    def get_pollution(
        self,
        stations: Iterable[str],
        *,
        time: TimeRange | None = None,
    ) -> xr.Dataset:
        """Fetch EMEP background-pollution time series for ``stations``.

        AEMET returns irregular hourly/daily mixed samples — we pass
        them through as ``(station, time)`` keeping whatever time axis
        the payload reports.
        """
        station_ids = tuple(stations)

        def fetch(sid: str) -> list[dict[str, Any]]:
            return self._get_json(f"/red/especial/contaminacionfondo/estacion/{sid}")

        frames: dict[str, pd.DataFrame] = {}
        all_times: set[datetime] = set()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(fetch, sid): sid for sid in station_ids}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    rows = fut.result() or []
                except AemetNoDataError:
                    continue
                except (AemetAuthError, AemetRateLimitError):
                    # Auth / quota failures are not per-chunk
                    # problems — they fail the whole scrape, so we
                    # let them propagate instead of silently
                    # skipping every remaining station.
                    raise
                except AemetError as exc:
                    warnings.warn(
                        f"AEMET pollution failed for station {sid}: {exc}",
                        stacklevel=2,
                    )
                    continue
                if not rows:
                    continue
                frame = _pollution_frame(rows)
                frames[sid] = frame
                for ts in pd.DatetimeIndex(frame.index):
                    all_times.add(ts.to_pydatetime())

        if time is not None and all_times:
            all_times = {
                t
                for t in all_times
                if time.start.to_pydatetime() <= t <= time.end.to_pydatetime()
            }
        time_index = pd.DatetimeIndex(sorted(all_times), tz=UTC)
        return _pollution_to_dataset(frames, station_ids, time_index)

    # ---- internals: station-windowed fetcher -----------------------------

    def _fetch_station_windows(
        self,
        station_ids: tuple[str, ...],
        start: datetime,
        end: datetime,
        *,
        chunk_days: int,
        url_fn: Any,
        parse_fn: Any,
    ) -> dict[str, pd.DataFrame]:
        """Fan out (station, chunk) fetches and stitch per station.

        ``url_fn(sid, chunk_start, chunk_end) -> str`` produces the
        endpoint path. ``parse_fn(list[dict]) -> pd.DataFrame`` turns
        a payload into a per-chunk frame.
        """
        tasks: list[tuple[str, datetime, datetime]] = []
        for sid in station_ids:
            tasks.extend(_chunk_days_range(sid, start, end, chunk_days))
        per_chunk: dict[str, list[pd.DataFrame]] = {sid: [] for sid in station_ids}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._get_json, url_fn(sid, s, e)): (sid, s, e)
                for sid, s, e in tasks
            }
            for fut in as_completed(futures):
                sid, _, _ = futures[fut]
                try:
                    rows = fut.result() or []
                except AemetNoDataError:
                    continue
                except (AemetAuthError, AemetRateLimitError):
                    # Auth / quota failures are not per-chunk
                    # problems — they fail the whole scrape, so we
                    # let them propagate instead of silently
                    # skipping every remaining station.
                    raise
                except AemetError as exc:
                    warnings.warn(
                        f"AEMET chunk failed for station {sid}: {exc}",
                        stacklevel=2,
                    )
                    continue
                if rows:
                    per_chunk[sid].append(parse_fn(rows))
        frames: dict[str, pd.DataFrame] = {}
        for sid, chunks in per_chunk.items():
            if not chunks:
                continue
            frames[sid] = pd.concat(chunks).sort_index()
            frames[sid] = frames[sid][~frames[sid].index.duplicated(keep="last")]
        return frames

    def _resolve_station_ids(
        self,
        stations: Iterable[str] | StationCollection | None,
        bbox: BBox | None,
    ) -> tuple[str, ...]:
        if stations is None:
            if bbox is None:
                raise ValueError(
                    "AEMET requests need either ``stations`` (explicit IDs / "
                    "collection) or a ``bbox`` to narrow the station list"
                )
            return self.list_stations(bbox=bbox).ids()
        if isinstance(stations, StationCollection):
            return stations.ids()
        return tuple(stations)

    # ---- internals: HTTP + rate limit -----------------------------------

    def _get_json(self, path: str, *, use_key: bool = True) -> Any:
        """First-hop envelope + second-hop payload, returning parsed JSON."""
        envelope = self._fetch_envelope(path, use_key=use_key)
        return self._fetch_data(envelope.datos)

    def _fetch_envelope(self, path: str, *, use_key: bool) -> _Envelope:
        url = self.base_url + (path if path.startswith("/") else f"/{path}")
        client = self._get_client()
        headers: dict[str, str] = {}
        if use_key:
            headers["api_key"] = self._require_key()

        for attempt in range(self.max_retries + 1):
            self._rate_limit()
            try:
                resp = client.get(url, headers=headers, timeout=self.timeout_s)
            except Exception as exc:
                if _is_transient_transport_error(exc):
                    if attempt == self.max_retries:
                        raise AemetError(f"AEMET network error: {exc}") from exc
                    time.sleep(_backoff_seconds(attempt))
                    continue
                raise
            status = resp.status_code
            if status == 401:
                raise AemetAuthError("AEMET rejected API key (401)")
            if status == 429:
                # Trip a shared pause so *every* worker blocks and the
                # minute window drains instead of being kept hot by
                # concurrent callers. Grow the pause with each attempt.
                self._trip_rate_limit(
                    _rate_limit_pause(attempt) * self.rate_limit_pause_scale
                )
                if attempt == self.max_retries:
                    remaining = _parse_remaining(resp.headers)
                    raise AemetRateLimitError(
                        f"AEMET rate limit, retries exhausted "
                        f"(remaining={remaining}, body={resp.text[:200]!r})"
                    )
                continue
            if status >= 500:
                if attempt == self.max_retries:
                    raise AemetError(f"AEMET server error: {status}")
                time.sleep(_backoff_seconds(attempt))
                continue
            if status >= 400:
                raise AemetError(f"AEMET request failed: {status} {resp.text[:200]}")
            body = resp.json()
            remaining = _parse_remaining(resp.headers)
            # AEMET returns HTTP 200 with ``estado=404`` (and no
            # ``datos`` field) for soft errors like "station has no
            # data in range" or "window exceeds cap". Surface those as
            # a typed error so chunkers can skip/retry cleanly.
            if "datos" not in body:
                estado = body.get("estado", status)
                desc = body.get("descripcion", "(no description)")
                raise AemetNoDataError(
                    f"AEMET envelope missing datos (estado={estado}): {desc}"
                )
            env = _Envelope(
                datos=body["datos"],
                metadatos=body.get("metadatos"),
                status=body.get("estado", status),
                description=body.get("descripcion", ""),
                remaining=remaining,
            )
            if remaining is not None and remaining <= RATE_LIMIT_FLOOR:
                time.sleep(1.0)
            return env
        raise AemetError("unreachable retry loop")

    def _fetch_data(self, data_url: str) -> Any:
        client = self._get_client()
        for attempt in range(self.max_retries + 1):
            self._rate_limit()
            try:
                resp = client.get(data_url, timeout=self.timeout_s)
            except Exception as exc:
                if _is_transient_transport_error(exc):
                    if attempt == self.max_retries:
                        raise AemetError(
                            f"AEMET data-hop network error: {exc}"
                        ) from exc
                    time.sleep(_backoff_seconds(attempt))
                    continue
                raise
            status = resp.status_code
            if status == 429:
                # Data-hop 429s also trip the shared pause so other
                # workers don't keep the bucket hot.
                self._trip_rate_limit(
                    _rate_limit_pause(attempt) * self.rate_limit_pause_scale
                )
                if attempt == self.max_retries:
                    raise AemetRateLimitError(
                        f"AEMET data-hop rate limit, retries exhausted "
                        f"(body={resp.text[:200]!r})"
                    )
                continue
            if status >= 500 and attempt < self.max_retries:
                # AEMET's Tomcat front-end returns occasional 5xx on
                # the signed data URL; treat it as transient.
                time.sleep(_backoff_seconds(attempt))
                continue
            if status >= 400:
                raise AemetError(f"AEMET data-hop failed: {status} {resp.text[:200]}")
            return _decode_aemet_json(resp.content)
        raise AemetError("unreachable data-hop retry loop")


# ---- module-level helpers ------------------------------------------------


def _daily_url(sid: str, start: datetime, end: datetime) -> str:
    s = format_aemet_datetime(start)
    e = format_aemet_datetime(end)
    return (
        f"/valores/climatologicos/diarios/datos/"
        f"fechaini/{s}/fechafin/{e}/estacion/{sid}"
    )


def _chunk_days_range(
    sid: str, start: datetime, end: datetime, chunk_days: int
) -> list[tuple[str, datetime, datetime]]:
    """Split ``[start, end]`` into chunks of at most ``chunk_days`` each.

    Uses ``cur <= end`` so single-day windows (``start == end`` — e.g.
    ``TimeRange.parse("2024-01-01", "2024-01-01")``) still emit exactly
    one ``(start, end)`` chunk and trigger a real AEMET call. ``<``
    would silently return no chunks, producing a NaN-only dataset.
    """
    out: list[tuple[str, datetime, datetime]] = []
    cur = start
    delta = timedelta(days=chunk_days)
    while cur <= end:
        nxt = min(cur + delta, end)
        out.append((sid, cur, nxt))
        cur = nxt + timedelta(seconds=1)
    return out


def _chunk_years(y1: int, y2: int, chunk_years: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur = y1
    while cur <= y2:
        nxt = min(cur + chunk_years - 1, y2)
        out.append((cur, nxt))
        cur = nxt + 1
    return out


def _backoff_seconds(attempt: int) -> float:
    return min(2.0**attempt, 30.0)


def _rate_limit_pause(attempt: int) -> float:
    """Global pause duration on 429, growing with each retry.

    AEMET's per-minute rate bucket drains on its own, but only if we
    actually stop hitting it. The first pause is already a full minute
    so the sliding-minute window fully clears; later attempts extend
    up to five minutes in case the window is wider than advertised.
    """
    return min(60.0 * (attempt + 1), 300.0)


def _is_transient_transport_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a network-layer error we should retry.

    Matches by class name so we don't import ``httpx`` at module import
    time. Covers timeouts, connection resets, DNS failures, and
    ``OSError`` (which httpx re-raises for some transport-level
    issues).
    """
    cls = type(exc).__name__
    return isinstance(exc, (TimeoutError, ConnectionError, OSError)) or cls in {
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadError",
        "ConnectError",
        "WriteError",
        "RemoteProtocolError",
    }


def _decode_aemet_json(content: bytes) -> Any:
    """Parse AEMET's JSON payload, handling their non-UTF-8 encoding.

    AEMET often serves JSON as Latin-1 / Windows-1252 (Spanish accents
    like ``Ó`` appear as raw ``0xD3`` bytes) while declaring
    ``application/json`` without a charset. We try UTF-8 first, then
    fall back to Latin-1 which is byte-compatible and can't fail.
    """
    import json

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    return json.loads(text)


def _time_to_numpy(index: pd.DatetimeIndex) -> np.ndarray:
    """Return a ``datetime64[ns]`` array from a (possibly tz-aware) index.

    xarray's NetCDF backends refuse tz-aware object arrays; timestamps
    are canonically stored as naive UTC datetime64 in CF conventions.
    """
    if index.tz is not None:
        index = index.tz_convert(UTC).tz_localize(None)
    return index.to_numpy(dtype="datetime64[ns]")


def _parse_remaining(headers: Mapping[str, str] | None) -> int | None:
    if headers is None:
        return None
    raw = headers.get(RATE_LIMIT_HEADER) or headers.get(RATE_LIMIT_HEADER.lower())
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _parse_station(row: Mapping[str, Any]) -> Station:
    lat_raw = row.get("latitud", "")
    lon_raw = row.get("longitud", "")
    lat = parse_dms(lat_raw) if isinstance(lat_raw, str) and lat_raw else 0.0
    lon = parse_dms(lon_raw) if isinstance(lon_raw, str) and lon_raw else 0.0
    altitude_raw = row.get("altitud")
    altitude = parse_spanish_float(altitude_raw) if altitude_raw is not None else None
    raw_province = row.get("provincia")
    # AEMET's inventory ships a raw ``provincia`` string and nothing
    # about the autonomous community. ``geo`` maps both to Spanish
    # standard names so filters don't have to know AEMET's quirks.
    province = canonical_province(raw_province) or (
        str(raw_province) if raw_province else None
    )
    return Station(
        id=str(row.get("indicativo", "")),
        name=str(row.get("nombre", "")),
        lon=lon,
        lat=lat,
        altitude=altitude,
        wmo_id=str(row["indsinop"]) if row.get("indsinop") else None,
        source="aemet",
        city=row.get("localidad") or row.get("ciudad"),
        province=province,
        community=community_for(raw_province),
        attrs={"raw": dict(row)},
    )


def _parse_daily_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {}
        for field_name in DAILY_FIELDS:
            rec[field_name] = parse_spanish_float(row.get(field_name))
        for field_name in DAILY_PASSTHROUGH_FIELDS:
            rec[field_name] = row.get(field_name)
        rec["time"] = pd.Timestamp(row["fecha"], tz=UTC)
        records.append(rec)
    frame = pd.DataFrame.from_records(records).set_index("time").sort_index()
    return frame


def _hourly_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {"time": parse_aemet_datetime(row["fint"])}
        for field_name in HOURLY_FIELDS:
            rec[field_name] = parse_spanish_float(row.get(field_name))
        records.append(rec)
    frame = pd.DataFrame.from_records(records)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.set_index("time").sort_index()


def _monthly_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        fecha = str(row.get("fecha", ""))
        if len(fecha) < 6 or "-" not in fecha:
            continue
        try:
            year_str, month_str = fecha.split("-")
            year, month = int(year_str), int(month_str)
            if not 1 <= month <= 12:
                continue
        except ValueError:
            continue
        rec: dict[str, Any] = {
            "time": pd.Timestamp(year=year, month=month, day=1, tz=UTC)
        }
        for field_name in MONTHLY_FIELDS:
            rec[field_name] = parse_spanish_float(row.get(field_name))
        records.append(rec)
    if not records:
        return pd.DataFrame(columns=[*MONTHLY_FIELDS])
    frame = pd.DataFrame.from_records(records).set_index("time").sort_index()
    return frame


def _pollution_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        ts = row.get("fecha") or row.get("fint")
        if ts is None:
            continue
        rec: dict[str, Any] = {"time": parse_aemet_datetime(str(ts))}
        for key, value in row.items():
            if key in {"fecha", "fint", "indicativo", "idema"}:
                continue
            rec[key] = parse_spanish_float(value) if isinstance(value, str) else value
        records.append(rec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).set_index("time").sort_index()


def _frames_to_stations_dataset(
    frames: Mapping[str, pd.DataFrame],
    *,
    station_ids: tuple[str, ...],
    time_index: pd.DatetimeIndex,
    field_map: Mapping[str, Variable],
    passthrough: frozenset[str],
    endpoint_name: str,
) -> xr.Dataset:
    """Stitch per-station frames onto a common time grid as (station, time)."""
    n_station = len(station_ids)
    n_time = len(time_index)
    arrays: dict[str, np.ndarray] = {}
    for field_name in field_map:
        arrays[field_name] = np.full((n_station, n_time), np.nan, dtype=np.float64)
    string_arrays: dict[str, np.ndarray] = {}
    for field_name in passthrough:
        string_arrays[field_name] = np.full((n_station, n_time), "", dtype=object)

    for i, sid in enumerate(station_ids):
        frame = frames.get(sid)
        if frame is None or frame.empty:
            continue
        aligned = frame.reindex(time_index)
        for field_name, buf in arrays.items():
            if field_name in aligned.columns:
                buf[i] = aligned[field_name].to_numpy(dtype=np.float64, na_value=np.nan)
        for field_name, buf in string_arrays.items():
            if field_name in aligned.columns:
                buf[i] = aligned[field_name].fillna("").to_numpy(dtype=object)

    data_vars: dict[str, tuple[tuple[str, str], np.ndarray, dict[str, str]]] = {}
    for field_name, buf in arrays.items():
        variable = field_map[field_name]
        data_vars[variable.name] = (("station", "time"), buf, variable.cf_attrs())
    for field_name, buf in string_arrays.items():
        data_vars[field_name] = (("station", "time"), buf, {})

    ds = xr.Dataset(
        {
            name: xr.DataArray(data=data, dims=dims, attrs=attrs)
            for name, (dims, data, attrs) in data_vars.items()
        },
        coords={
            "station": ("station", list(station_ids)),
            "time": ("time", _time_to_numpy(time_index)),
        },
    )
    ds.attrs["source"] = "aemet"
    ds.attrs["featureType"] = "timeSeries"
    ds.attrs["endpoint"] = endpoint_name
    return ds


def _stations_to_dataset(collection: StationCollection) -> xr.Dataset:
    """Render an inventory snapshot as a station-only ``xr.Dataset``."""
    if not collection:
        return xr.Dataset(coords={"station": ("station", np.array([], dtype=object))})
    lon = np.array([s.lon for s in collection], dtype=np.float64)
    lat = np.array([s.lat for s in collection], dtype=np.float64)
    altitude = np.array(
        [s.altitude if s.altitude is not None else np.nan for s in collection],
        dtype=np.float64,
    )
    return xr.Dataset(
        {
            "lon": (("station",), lon, {"units": "degrees_east"}),
            "lat": (("station",), lat, {"units": "degrees_north"}),
            "altitude": (("station",), altitude, {"units": "m"}),
            "name": (
                ("station",),
                np.array([s.name for s in collection], dtype=object),
            ),
            "province": (
                ("station",),
                np.array([s.province or "" for s in collection], dtype=object),
            ),
            "community": (
                ("station",),
                np.array([s.community or "" for s in collection], dtype=object),
            ),
        },
        coords={
            "station": (
                ("station",),
                np.array([s.id for s in collection], dtype=object),
            ),
        },
        attrs={"source": "aemet", "featureType": "timeSeries"},
    )


def _normals_to_dataset(
    rows_per: Mapping[str, list[dict[str, Any]]],
    station_ids: tuple[str, ...],
) -> xr.Dataset:
    month_index = np.arange(1, 13, dtype=np.int64)
    n_station = len(station_ids)
    fields: dict[str, np.ndarray] = {
        name: np.full((n_station, 12), np.nan, dtype=np.float64)
        for name in ("tm_mes", "ta_min", "ta_max", "p_mes", "inso")
    }
    for i, sid in enumerate(station_ids):
        for row in rows_per.get(sid, []):
            mes_raw = row.get("mes")
            try:
                mes = int(mes_raw) if mes_raw is not None else 0
            except (TypeError, ValueError):
                continue
            if not 1 <= mes <= 12:
                continue
            for field_name in fields:
                # ``parse_spanish_float`` returns ``None`` for missing
                # sentinels and an explicit ``float`` otherwise — zero
                # included. The previous ``or np.nan`` silently wiped
                # legitimate zeros (e.g. zero-precipitation months),
                # so check for ``None`` explicitly.
                parsed = parse_spanish_float(row.get(field_name))
                fields[field_name][i, mes - 1] = np.nan if parsed is None else parsed
    return xr.Dataset(
        {name: (("station", "month"), arr) for name, arr in fields.items()},
        coords={
            "station": (("station",), list(station_ids)),
            "month": (("month",), month_index),
        },
        attrs={"source": "aemet", "endpoint": "normals"},
    )


def _extremes_to_dataset(
    rows_per: Mapping[str, list[dict[str, Any]]],
    station_ids: tuple[str, ...],
    parameter: str,
) -> xr.Dataset:
    records = [list(rows_per.get(sid, [])) for sid in station_ids]
    return xr.Dataset(
        {
            "records": (
                ("station",),
                np.array([str(r) for r in records], dtype=object),
            )
        },
        coords={"station": (("station",), list(station_ids))},
        attrs={
            "source": "aemet",
            "endpoint": "extremes",
            "parameter": parameter,
        },
    )


def _pollution_to_dataset(
    frames: Mapping[str, pd.DataFrame],
    station_ids: tuple[str, ...],
    time_index: pd.DatetimeIndex,
) -> xr.Dataset:
    all_cols: set[str] = set()
    for frame in frames.values():
        all_cols.update(frame.columns)
    n_station = len(station_ids)
    n_time = len(time_index)
    data = {
        col: np.full((n_station, n_time), np.nan, dtype=np.float64) for col in all_cols
    }
    for i, sid in enumerate(station_ids):
        frame = frames.get(sid)
        if frame is None or frame.empty:
            continue
        aligned = frame.reindex(time_index)
        for col in all_cols:
            if col in aligned.columns:
                data[col][i] = aligned[col].to_numpy(dtype=np.float64, na_value=np.nan)
    return xr.Dataset(
        {col: (("station", "time"), arr) for col, arr in data.items()},
        coords={
            "station": (("station",), list(station_ids)),
            "time": (("time",), _time_to_numpy(time_index)),
        },
        attrs={"source": "aemet", "endpoint": "pollution", "featureType": "timeSeries"},
    )


def _subset_variables(ds: xr.Dataset, variables: list[str | Variable]) -> xr.Dataset:
    wanted: set[str] = set()
    for v in variables:
        if isinstance(v, Variable):
            wanted.add(v.name)
        else:
            wanted.add(v)
    keep = [name for name in ds.data_vars if name in wanted]
    if not keep:
        return ds
    subset = ds[keep]
    # ``ds[list]`` is typed as ``DataArray | Dataset``; indexing with a
    # list always yields a Dataset, so narrow the return type for ty.
    assert isinstance(subset, xr.Dataset)
    return subset
