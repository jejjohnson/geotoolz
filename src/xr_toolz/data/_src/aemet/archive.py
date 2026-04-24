"""Local mirror of AEMET observations with incremental sync.

Scraping all stations periodically produces thousands of requests.
:class:`AemetArchive` owns a small local store and exposes:

- :meth:`sync` — backfill everything once; re-runs resume from the
  archive's latest time per station.
- :meth:`load` — read the archive back as a :class:`geopandas.GeoDataFrame`.
- :meth:`coverage` — per-station first / last / gap statistics.

Storage format
--------------

Station observations are **unstructured** — one record per
``(station, time)``. They are stored as **GeoParquet** in long format
with a ``geometry`` column (``Point(lon, lat)``), so:

- reading is a one-liner for anyone with geopandas or pyarrow,
- spatial filters (``.cx[lon_min:lon_max, lat_min:lat_max]``) work out
  of the box,
- the file is chunked, typed and compressed without xarray encoding
  gymnastics.

One file per preset (``aemet_daily.parquet``, ``aemet_monthly.parquet``
...) plus the shared ``stations.parquet`` inventory snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from xr_toolz.data._src.aemet.source import AemetSource
from xr_toolz.types import StationCollection, TimeRange


@dataclass(frozen=True)
class ArchiveCoverage:
    """Per-station span of what the archive already holds for a preset."""

    preset: str
    station_id: str
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    n_timesteps: int
    gap_fraction: float


class AemetArchive:
    """Append-only local mirror of AEMET station observations.

    Everything (inventory + per-preset time series) lives under
    ``root`` as GeoParquet. Dense fields like future gridded AEMET
    products are out of scope for this class.

    Args:
        root: Directory that will hold one file per preset.
        source: :class:`AemetSource` used for fetches.
    """

    def __init__(self, root: Path, source: AemetSource) -> None:
        self.root = Path(root)
        self.source = source
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- inventory -------------------------------------------------------

    def sync_stations(self) -> StationCollection:
        """Refresh the cached station inventory and return it."""
        collection = self.source.list_stations()
        self._write_stations(collection)
        return collection

    def load_stations(self) -> StationCollection:
        """Return the inventory as a :class:`StationCollection`.

        Refreshes from AEMET if no cached copy exists.
        """
        path = self.stations_path
        if not path.is_file():
            return self.sync_stations()
        import geopandas as gpd

        gdf = gpd.read_parquet(path)
        return _stations_from_geodataframe(gdf)

    def load_stations_geodataframe(self):
        """Return the inventory as a :class:`geopandas.GeoDataFrame`.

        Geometry is EPSG:4326 points. Refreshes from AEMET if no
        cached copy exists.
        """
        import geopandas as gpd

        if not self.stations_path.is_file():
            self.sync_stations()
        return gpd.read_parquet(self.stations_path)

    def _write_stations(self, collection: StationCollection) -> None:
        import geopandas as gpd
        from shapely.geometry import Point

        records = [
            {
                "id": s.id,
                "name": s.name,
                "lon": s.lon,
                "lat": s.lat,
                "altitude": s.altitude,
                "wmo_id": s.wmo_id,
                "source": s.source,
                "city": s.city,
                "province": s.province,
                "community": s.community,
                "timezone": s.timezone,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "active": s.active,
                "geometry": Point(s.lon, s.lat),
            }
            for s in collection
        ]
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        gdf.to_parquet(self.stations_path)

    # ---- sync ------------------------------------------------------------

    def sync(
        self,
        preset: str,
        *,
        stations: StationCollection | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> xr.Dataset:
        """Backfill or update a preset. Returns the freshly-fetched slice.

        ``sync`` returns the in-memory ``xr.Dataset`` that was just
        fetched (useful for immediate inspection). The on-disk archive
        is written as a long-format GeoParquet file; call :meth:`load`
        to read it back as a :class:`geopandas.GeoDataFrame`.

        Behaviour:

        - ``since`` defaults to one day after the archive's **global**
          latest stored time (the newest row across all stations), or
          the preset-specific floor (e.g. 1920 for daily / monthly)
          when the archive is empty. Per-station resume — so stations
          missing recent records get backfilled even when others are
          up-to-date — is not implemented; use an explicit ``since``
          if you need that behaviour.
        - ``until`` defaults to ``now`` in UTC.
        - Re-running with the same window overwrites the overlapping
          slice; it is idempotent, not additive.
        """
        if preset not in _FETCH_TABLE:
            raise ValueError(
                f"unknown preset {preset!r}; expected one of {sorted(_FETCH_TABLE)}"
            )
        station_ids = self._station_ids(stations)
        start = self._resolve_start(preset, since)
        end = self._resolve_end(until)

        # Three cases for inverted windows (``start > end``):
        #
        # 1. Hourly preset — always fetch (endpoint is rolling 24h,
        #    ignores TimeRange entirely; stale data would otherwise
        #    accumulate until the next calendar day).
        # 2. Non-hourly + auto-resumed start (``since is None``) —
        #    ``_resolve_start`` set ``start = last + 1 day`` which
        #    overshoots ``end=now`` on routine same-day re-runs.
        #    Correct behaviour is a silent no-op.
        # 3. Non-hourly + caller-specified ``since`` that inverts —
        #    almost certainly a typo. Let ``TimeRange.parse`` raise
        #    so the user sees the error.
        inverted = pd.Timestamp(start, tz="UTC") > pd.Timestamp(end, tz="UTC")
        is_hourly = preset in _TIME_INSENSITIVE_PRESETS
        # Auto-resumed (since=None) + inverted + non-hourly → no-op.
        # Caller-specified inversion on a non-hourly preset → fall
        # through and let TimeRange.parse raise a clear error.
        if inverted and not is_hourly and since is None:
            return xr.Dataset()
        tr = None if (inverted and is_hourly) else TimeRange.parse(start, end)
        fresh = _FETCH_TABLE[preset](self.source, station_ids, tr)
        self._append(preset, fresh)
        return fresh

    def load(self, preset: str):
        """Open the archive for ``preset`` as a :class:`geopandas.GeoDataFrame`.

        The GeoDataFrame is long-format: one row per
        ``(station, time)`` sample, with the ``geometry`` column
        carrying each station's EPSG:4326 point.
        """
        import geopandas as gpd

        path = self._preset_path(preset)
        if not path.exists():
            raise FileNotFoundError(
                f"archive for {preset!r} does not exist at {path}; "
                "run .sync(preset) first."
            )
        return gpd.read_parquet(path)

    def load_dataset(self, preset: str) -> xr.Dataset:
        """Open the archive for ``preset`` as an ``(station, time)`` xarray dataset.

        Convenience for workflows that prefer the cube view over the
        long-format GeoDataFrame.
        """
        gdf = self.load(preset)
        return _geodataframe_to_dataset(gdf)

    def coverage(self, preset: str) -> list[ArchiveCoverage]:
        """Return per-station first / last / gap statistics for ``preset``."""
        try:
            gdf = self.load(preset)
        except FileNotFoundError:
            return []

        # Choose a representative variable for gap counting; fall back
        # to the first value column if none of the usual suspects are
        # present.
        value_cols = _value_columns(gdf)
        pick = next(
            (
                v
                for v in ("air_temperature_daily_mean", "air_temperature")
                if v in value_cols
            ),
            value_cols[0] if value_cols else None,
        )
        all_times = np.sort(pd.to_datetime(gdf["time"].unique()))
        n_time = len(all_times)

        out: list[ArchiveCoverage] = []
        grouped = gdf.groupby("station_id", sort=True)
        for sid, sub in grouped:
            if pick is None or n_time == 0:
                out.append(
                    ArchiveCoverage(
                        preset=preset,
                        station_id=str(sid),
                        first=None,
                        last=None,
                        n_timesteps=0,
                        gap_fraction=1.0,
                    )
                )
                continue
            valid = sub[sub[pick].notna()]
            n_valid = len(valid)
            if n_valid == 0:
                out.append(
                    ArchiveCoverage(
                        preset=preset,
                        station_id=str(sid),
                        first=None,
                        last=None,
                        n_timesteps=0,
                        gap_fraction=1.0,
                    )
                )
                continue
            times = pd.to_datetime(valid["time"])
            # ``Series.min`` / ``max`` are typed as ``Timestamp | NaTType``;
            # we've already filtered to non-null rows so NaT isn't possible,
            # but ty doesn't know that.
            first_ts = cast("pd.Timestamp | None", pd.Timestamp(times.min()))
            last_ts = cast("pd.Timestamp | None", pd.Timestamp(times.max()))
            out.append(
                ArchiveCoverage(
                    preset=preset,
                    station_id=str(sid),
                    first=first_ts,
                    last=last_ts,
                    n_timesteps=n_valid,
                    gap_fraction=1.0 - n_valid / n_time,
                )
            )
        return out

    # ---- paths -----------------------------------------------------------

    @property
    def stations_path(self) -> Path:
        return self.root / "stations.parquet"

    def _preset_path(self, preset: str) -> Path:
        return self.root / f"{preset}.parquet"

    # ---- internals -------------------------------------------------------

    def _station_ids(self, stations: StationCollection | None) -> tuple[str, ...]:
        collection = stations if stations is not None else self.load_stations()
        return collection.ids()

    def _resolve_start(self, preset: str, since: str | None) -> str:
        if since is not None:
            return since
        try:
            gdf = self.load(preset)
            last = pd.to_datetime(gdf["time"]).max()
            return (last + pd.Timedelta(days=1)).isoformat()
        except FileNotFoundError:
            return _DEFAULT_START[preset]

    def _resolve_end(self, until: str | None) -> str:
        if until is not None:
            return until
        return datetime.now(UTC).isoformat()

    def _append(self, preset: str, fresh: xr.Dataset) -> None:
        """Merge ``fresh`` into the archive, overwriting overlapping rows."""
        fresh_gdf = _dataset_to_geodataframe(fresh, self._station_lookup())
        path = self._preset_path(preset)
        if not path.exists():
            fresh_gdf.to_parquet(path)
            return
        import geopandas as gpd

        existing = gpd.read_parquet(path)
        merged = _merge_long(existing, fresh_gdf)
        merged.to_parquet(path)

    def _station_lookup(self) -> dict[str, tuple[float, float]]:
        """Map station_id → (lon, lat) for geometry enrichment."""
        stations = self.load_stations()
        return {s.id: (s.lon, s.lat) for s in stations}


# ---- dataset <-> geodataframe conversion --------------------------------


def _dataset_to_geodataframe(
    ds: xr.Dataset, station_lookup: dict[str, tuple[float, float]]
):
    """Flatten an ``(station, time)`` dataset into a long GeoDataFrame.

    Non-finite (``NaN``-only) rows are retained so the gap structure
    of the original dataset is preserved on disk — callers can always
    ``.dropna(subset=...)`` downstream.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    df = ds.to_dataframe().reset_index()
    # `station` axis becomes the `station_id` column for clarity
    # downstream (the index name also lands here).
    if "station" in df.columns:
        df = df.rename(columns={"station": "station_id"})
    df["station_id"] = df["station_id"].astype(str)
    df["time"] = pd.to_datetime(df["time"])

    # Add geometry from the inventory; stations absent from the
    # lookup fall back to (nan, nan) so the row survives.
    lons = df["station_id"].map(
        lambda sid: station_lookup.get(sid, (np.nan, np.nan))[0]
    )
    lats = df["station_id"].map(
        lambda sid: station_lookup.get(sid, (np.nan, np.nan))[1]
    )
    df["lon"] = lons.astype(float)
    df["lat"] = lats.astype(float)
    geometry = [
        Point(x, y) if pd.notna(x) and pd.notna(y) else None
        for x, y in zip(df["lon"], df["lat"], strict=True)
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    # Put the identifying columns first — matters for readability
    # when someone opens the parquet in DuckDB/pandas without knowing
    # the schema upfront.
    lead = ["station_id", "time", "lon", "lat"]
    rest = [c for c in gdf.columns if c not in (*lead, "geometry")]
    return gdf[[*lead, *rest, "geometry"]]


def _geodataframe_to_dataset(gdf) -> xr.Dataset:
    """Reconstruct a ``(station, time)`` xarray Dataset from long GeoParquet."""
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    value_cols = _value_columns(gdf)
    # Ensure time is a pandas Timestamp index compatible with xarray.
    df["time"] = pd.to_datetime(df["time"])
    pivoted = df.pivot_table(
        index="station_id",
        columns="time",
        values=value_cols,
        aggfunc="first",
    )
    # pivoted columns are (variable, time); unstack gives a cube.
    station_ids = pivoted.index.astype(str).tolist()
    times = sorted({t for _, t in pivoted.columns})
    data: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for var in value_cols:
        sub = pivoted.get(var)
        if sub is None:
            continue
        sub = sub.reindex(columns=times)
        # Preserve the column's original dtype. AEMET's daily preset
        # writes non-numeric passthrough fields (hour-of-extreme
        # strings like ``horatmin``, ``horaracha``); casting those to
        # float64 would either silently lose them or raise.
        if pd.api.types.is_numeric_dtype(df[var]):
            arr = sub.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            arr = sub.to_numpy(dtype=object)
        data[var] = (("station", "time"), arr)
    return xr.Dataset(
        {name: xr.DataArray(arr, dims=dims) for name, (dims, arr) in data.items()},
        coords={
            "station": ("station", station_ids),
            "time": ("time", np.array(times, dtype="datetime64[ns]")),
        },
        attrs={"source": "aemet", "featureType": "timeSeries"},
    )


def _merge_long(existing, fresh):
    """Union existing + fresh GeoParquet rows; fresh wins on (station, time)."""
    import geopandas as gpd

    key_cols = ["station_id", "time"]
    # Drop existing rows that have a fresh replacement.
    key_fresh = pd.MultiIndex.from_frame(fresh[key_cols])
    key_existing = pd.MultiIndex.from_frame(existing[key_cols])
    keep_mask = ~key_existing.isin(key_fresh)
    survivors = existing[keep_mask]
    merged = pd.concat([survivors, fresh], ignore_index=True)
    merged = merged.sort_values(key_cols, kind="stable").reset_index(drop=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


def _value_columns(gdf) -> list[str]:
    """Non-key / non-metadata columns that carry observation values."""
    meta = {"station_id", "time", "lon", "lat", "geometry"}
    return [c for c in gdf.columns if c not in meta]


# ---- inventory <-> geodataframe -----------------------------------------


def _stations_from_geodataframe(gdf) -> StationCollection:
    from xr_toolz.types import Station

    def _opt(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return value

    stations = tuple(
        Station(
            id=str(row["id"]),
            name=str(row["name"]),
            lon=float(row["lon"]),
            lat=float(row["lat"]),
            altitude=_opt(row.get("altitude")),
            wmo_id=_opt(row.get("wmo_id")),
            source=_opt(row.get("source")),
            city=_opt(row.get("city")),
            province=_opt(row.get("province")),
            community=_opt(row.get("community")),
            timezone=_opt(row.get("timezone")),
            start_time=_opt(row.get("start_time")),
            end_time=_opt(row.get("end_time")),
            active=_opt(row.get("active")),
        )
        for _, row in gdf.iterrows()
    )
    return StationCollection(stations)


# ---- preset fetch dispatch ----------------------------------------------


def _fetch_daily(source: AemetSource, sids, tr: TimeRange | None) -> xr.Dataset:
    # The ``start > end`` short-circuit in ``sync`` prevents non-hourly
    # presets from reaching here with ``tr=None``; assert for clarity.
    assert tr is not None
    return source.get_daily(sids, time=tr)


def _fetch_hourly(source: AemetSource, sids, tr: TimeRange | None) -> xr.Dataset:
    # TimeRange is ignored — hourly endpoint is rolling.
    del tr
    return source.get_hourly(sids)


def _fetch_monthly(source: AemetSource, sids, tr: TimeRange | None) -> xr.Dataset:
    assert tr is not None
    return source.get_monthly(sids, time=tr)


def _fetch_pollution(source: AemetSource, sids, tr: TimeRange | None) -> xr.Dataset:
    assert tr is not None
    return source.get_pollution(sids, time=tr)


_FETCH_TABLE = {
    "aemet_daily": _fetch_daily,
    "aemet_hourly": _fetch_hourly,
    "aemet_monthly": _fetch_monthly,
    "aemet_pollution": _fetch_pollution,
}


# Presets whose fetchers ignore the requested ``TimeRange`` — always
# call them, even when the same-day auto-resume makes ``start > end``.
_TIME_INSENSITIVE_PRESETS: frozenset[str] = frozenset({"aemet_hourly"})


_DEFAULT_START = {
    "aemet_daily": "1920-01-01",
    "aemet_hourly": (datetime.now(UTC) - pd.Timedelta(days=2)).isoformat(),
    "aemet_monthly": "1920-01-01",
    "aemet_pollution": (datetime.now(UTC) - pd.Timedelta(days=365 * 5)).isoformat(),
}
