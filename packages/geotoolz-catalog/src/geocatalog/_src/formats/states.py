"""Model states: build a state catalog, and load it back without mixing runs.

A model state is a field a model produced: an analysis, a forecast step,
one ensemble member. Each file is one row, like any xarray file, plus the
reserved state columns (`STATE_COLUMNS`) read from its metadata through a
name map (`CF_NAMES`, `CFGRIB_NAMES`). The loader stacks one file per
(run, member) into an ``(init_time, member, lead_time, y, x)`` cube, so
two runs valid at the same time stay two states (`load_xarray` would keep
one of them).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd

from geocatalog._src._extras import missing_extra
from geocatalog._src._schema import (
    STATE_KINDS,
    StorageEngine,
    coerce_state_columns,
    state_parents,
    state_text,
)
from geocatalog._src.backends.memory import InMemoryGeoCatalog, _reproject_bounds
from geocatalog._src.base import CatalogRow, GeoCatalog
from geocatalog._src.formats.xarray_backend import (  # shared xarray plumbing
    _dataset_row,
    _open_for_row,
    _read_clipped,
    _xy_dims,
)
from geocatalog._src.states import CF_NAMES, StateNames, require_state_columns
from geocatalog._src.utils.timeutil import to_naive_utc


if TYPE_CHECKING:
    import xarray as xr

    from geocatalog._src.backends.duckdb_backend import DuckDBGeoCatalog
    from geocatalog._src.geoslice import GeoSlice

try:
    import xarray as xr
except ImportError:  # pragma: no cover - exercised via the [xarray-raster] extra
    xr = None  # type: ignore[assignment]


_FIELDS = ("valid_time", "run_time", "member", "kind", "producer", "parents")


# ---------------------------------------------------------------------------
# Reading the fields
# ---------------------------------------------------------------------------


def _sources(names: StateNames, field: str) -> tuple[Any, ...]:
    source = names.get(field)
    if source is None:
        return ()
    return source if isinstance(source, tuple) else (source,)


def _lookup(ds: xr.Dataset, filepath: str, names: StateNames, field: str) -> Any:
    """The first source of ``field`` found in ``ds`` (or by a callable), else None."""
    for source in _sources(names, field):
        if callable(source):
            return source(filepath)
        if source in ds.variables:
            values = ds[source].values
            if values.size != 1:
                raise ValueError(
                    f"build_state_catalog: {filepath}: {field} source {source!r} "
                    f"has {values.size} values; a state file holds one run and "
                    "one member (index each separately, or map the field to an "
                    "attribute or a callable)."
                )
            # A numpy scalar: datetime64 stays a time (``.item()`` would
            # turn nanoseconds into an int).
            return values.reshape(())[()]
        if source in ds.attrs:
            return ds.attrs[source]
    return None


def _valid_time_var(ds: xr.Dataset, names: StateNames) -> str:
    """The first ``valid_time`` source that is a coordinate of ``ds``."""
    candidates = [s for s in _sources(names, "valid_time") if isinstance(s, str)]
    for name in candidates:
        if name in ds.coords:
            return name
    # No time axis: the row gets the placeholder interval, as in
    # `build_xarray_catalog`.
    return candidates[0] if candidates else "time"


def _run_time(value: Any, filepath: str) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    try:
        ts = to_naive_utc(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"build_state_catalog: {filepath}: run_time {value!r} is not a time"
        ) from exc
    return None if ts is pd.NaT else ts


def _kind(value: Any, filepath: str) -> str | None:
    text = state_text(value)
    if text is not None and text not in STATE_KINDS:
        raise ValueError(
            f"build_state_catalog: {filepath}: kind {text!r} is not one of "
            f"{list(STATE_KINDS)}."
        )
    return text


def _state_row(
    filepath: str | Path,
    *,
    names: StateNames,
    data_vars: Sequence[str] | None,
    target_crs: Any | None,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One state catalog row: the xarray row plus the five state columns."""
    if xr is None:
        raise missing_extra("`build_state_catalog`", "xarray-raster")
    path = str(filepath)
    with _open_for_row(filepath, storage_options) as ds:
        row = _dataset_row(
            ds,
            filepath,
            data_vars=data_vars,
            time_var=_valid_time_var(ds, names),
            target_crs=target_crs,
        )
        row["kind"] = _kind(_lookup(ds, path, names, "kind"), path)
        row["run_time"] = _run_time(_lookup(ds, path, names, "run_time"), path)
        row["member"] = state_text(_lookup(ds, path, names, "member"))
        row["producer"] = state_text(_lookup(ds, path, names, "producer"))
        parents = _lookup(ds, path, names, "parents")
        row["parents"] = state_parents(
            np.atleast_1d(parents).tolist()
            if isinstance(parents, np.ndarray)
            else parents
        )
    return row


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


def build_state_catalog(
    filepaths: Sequence[str | Path],
    *,
    crs: Any | None = None,
    names: StateNames = CF_NAMES,
    data_vars: Sequence[str] | None = None,
    engine: StorageEngine = "memory",
    out_path: str | Path | None = None,
    write_bbox: bool = True,
    sort_by: tuple[str, ...] | None = ("start_time", "geometry_hilbert"),
    batch_size: int = 10_000,
    n_workers: int = 1,
    storage_options: dict[str, Any] | None = None,
) -> InMemoryGeoCatalog | DuckDBGeoCatalog:
    """Index model-state files: one row per file, with the state columns filled.

    Each file is indexed like `build_xarray_catalog` does (footprint from
    the coordinates, valid-time interval from the time axis), and the
    reserved state columns are read from its metadata through ``names``:
    ``kind``, ``run_time``, ``member``, ``producer`` and ``parents``. A
    field that is not found is null. Lead time is not stored: a row's
    leads are its valid times minus ``run_time``.

    Args:
        filepaths: NetCDF / Zarr files, one run and member each.
        crs: CRS of the files' coordinates, when they don't carry one
            (coordinates are not reprojected).
        names: Where each field lives: a mapping from ``valid_time``,
            ``run_time``, ``member``, ``kind``, ``producer`` and
            ``parents`` to a name (a coordinate or variable, else a
            global attribute), a callable on the file path, or a tuple of
            them tried in order: `CF_NAMES` (the default) for CF files,
            `CFGRIB_NAMES` for cfgrib-decoded GRIB, or your own map
            (``{**CF_NAMES, "kind": lambda p: "forecast"}``).
        data_vars: Data variables recorded per row (as in
            `build_xarray_catalog`).
        engine: ``"memory"`` for an `InMemoryGeoCatalog`, ``"duckdb"``
            to stream the rows to GeoParquet at ``out_path`` and open
            them lazily (``[duckdb]`` extra; needs ``crs``).
        out_path: Destination GeoParquet path for ``engine="duckdb"``.
        write_bbox: Emit the GeoParquet 1.1 ``bbox`` covering struct
            (``engine="duckdb"`` only).
        sort_by: Sort keys for the DuckDB rewrite (``engine="duckdb"``
            only); ``None`` keeps the input order.
        batch_size: Rows per Arrow record batch (``engine="duckdb"``).
        n_workers: Process-pool size for per-file extraction
            (``engine="duckdb"``; ``names`` must then be picklable, so no
            lambdas).
        storage_options: obstore options for remote files (`geocloud.fs`).

    Returns:
        A catalog of kind ``"xarray"`` with the five state columns.

    Raises:
        ImportError: If the ``[xarray-raster]`` extra is missing.
        ValueError: If no file was given, the CRS is unknown, a file's
            ``kind`` is not ``analysis`` / ``forecast`` / ``reanalysis``,
            or a run-time / member source holds more than one value.

    Example:
        >>> from geocatalog.build import build_state_catalog
        >>> from geocatalog.states import CF_NAMES
        >>> catalog = build_state_catalog(paths, crs="EPSG:4326", names=CF_NAMES)
        >>> catalog.gdf[["kind", "run_time", "member"]].iloc[0].tolist()
        ['forecast', Timestamp('2026-01-01 00:00:00'), '0']
    """
    if xr is None:
        raise missing_extra("`build_state_catalog`", "xarray-raster")
    if engine not in ("memory", "duckdb"):
        raise ValueError(
            f"build_state_catalog: engine must be 'memory' or 'duckdb'; got {engine!r}"
        )
    unknown = set(names) - set(_FIELDS)
    if unknown:
        raise ValueError(
            f"build_state_catalog: unknown field(s) {sorted(unknown)} in names; "
            f"expected some of {list(_FIELDS)}."
        )
    extract: Callable[[str | Path], dict[str, Any]] = functools.partial(
        _state_row,
        names=names,
        data_vars=tuple(data_vars) if data_vars is not None else None,
        target_crs=crs,
        storage_options=storage_options,
    )
    if engine == "duckdb":
        from geocatalog._src.storage.streaming import stream_build_duckdb

        if out_path is None:
            raise ValueError("build_state_catalog(engine='duckdb') requires out_path")
        if crs is None:
            raise ValueError(
                "build_state_catalog(engine='duckdb') requires crs (coordinate "
                "bounds are not reprojected)."
            )
        return stream_build_duckdb(
            filepaths,
            extract,
            out_path=out_path,
            crs=crs,
            kind="xarray",
            write_bbox=write_bbox,
            sort_by=sort_by,
            batch_size=batch_size,
            n_workers=n_workers,
        )

    rows = [extract(fp) for fp in filepaths]
    if not rows:
        raise ValueError("build_state_catalog: no files given")
    crs_value = crs if crs is not None else rows[0]["crs"]
    if crs_value is None:
        raise ValueError(
            "build_state_catalog: cannot determine the catalog CRS; pass crs=..."
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs_value)
    return InMemoryGeoCatalog(coerce_state_columns(gdf), kind="xarray")


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

_INIT, _MEMBER, _LEAD = "init_time", "member", "lead_time"


def _to_leads(piece: xr.Dataset, time_var: str, run_time: pd.Timestamp) -> xr.Dataset:
    """Relabel a file's valid-time axis as lead time, ``τ = v - r``."""
    if time_var not in piece.coords:
        raise ValueError(
            f"load_states: the file has no {time_var!r} coordinate, so its "
            "lead times are unknown."
        )
    coord = piece[time_var]
    run = np.datetime64(run_time, "ns")
    if coord.ndim == 0:
        lead = np.atleast_1d(coord.values.astype("datetime64[ns]") - run)
        piece = piece.drop_vars(time_var).expand_dims({_LEAD: lead})
    elif coord.ndim == 1:
        (dim,) = coord.dims
        lead = coord.values.astype("datetime64[ns]") - run
        piece = piece.assign_coords({_LEAD: (dim, lead)}).swap_dims({dim: _LEAD})
    else:
        raise ValueError(f"load_states: {time_var!r} must be 0-d or 1-d")
    # Keep the grid and the lead; drop scalar and auxiliary coordinates
    # (cfgrib's `time`, `step`, `number`, …), which differ per file.
    return piece.reset_coords(drop=True)


_INTEGER = re.compile(r"[+-]?\d+")


def _concat(pieces: list[xr.Dataset], dim: pd.Index) -> xr.Dataset:
    """Stack ``pieces`` along a new ``dim``, outer-joining the other axes."""
    return xr.concat(
        pieces,
        dim=dim,
        join="outer",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="drop",
    )


def _member_order(labels: Iterable[str]) -> list[str]:
    """Members in natural order: by number when every label is an integer.

    CF realizations are numbers written as text, so text order would put
    ``"10"`` before ``"2"``; CMIP variant labels (``r1i1p1f1``) and
    mixed labels keep text order.
    """
    unique = sorted(set(labels))
    if unique and all(_INTEGER.fullmatch(label) for label in unique):
        return sorted(unique, key=int)
    return unique


def _stack(rows: list[CatalogRow], read: Callable[[CatalogRow], xr.Dataset]) -> Any:
    """One cube from rows of the same grid: one (init_time, member) cell per file.

    Members are concatenated within each run, then runs along
    ``init_time``; outer joins put NaN where a run lacks a member or a
    lead. Nothing is reindexed to the whole cube per file, and values are
    never touched, so the cube equals the files exactly.
    """
    seen: dict[tuple[pd.Timestamp, str], str] = {}
    by_run: dict[pd.Timestamp, dict[str, xr.Dataset]] = {}
    grid: tuple[str, str, np.ndarray, np.ndarray] | None = None
    for row in rows:
        state = row.state
        assert state is not None  # `require_state_columns` checked it
        if state.run_time is None:
            raise ValueError(
                f"load_states: {row.filepath} has no run_time, so it is not part "
                "of a run; load analyses and reanalyses with load_xarray."
            )
        member = state.member if state.member is not None else ""
        key = (state.run_time, member)
        if key in seen:
            raise ValueError(
                f"load_states: run {state.run_time} member {member!r} appears "
                f"twice ({seen[key]} and {row.filepath}); a run and member is "
                "one file."
            )
        seen[key] = row.filepath
        time_var = str(row.extras.get("time_var") or "time")
        piece = _to_leads(read(row), time_var, state.run_time)
        if piece.sizes[_LEAD] == 0:
            # The row's interval overlaps the slice, but no time step of
            # the file falls in it: this run has no state in the slice.
            continue
        x_name, y_name = _xy_dims(piece)
        here = (x_name, y_name, piece[x_name].values, piece[y_name].values)
        if grid is None:
            grid = here
        elif not (
            here[:2] == grid[:2]
            and np.array_equal(here[2], grid[2])
            and np.array_equal(here[3], grid[3])
        ):
            raise ValueError(
                f"load_states: {row.filepath} is on a different grid from "
                f"{rows[0].filepath}; pass group_by= (e.g. 'producer') to load "
                "each grid as its own DataTree leaf."
            )
        by_run.setdefault(state.run_time, {})[member] = piece
    if not by_run:
        raise ValueError(
            f"load_states: the slice overlaps {len(rows)} file(s) but contains "
            "none of their time steps; widen the slice's interval to include "
            "a valid time."
        )
    members = _member_order(m for run in by_run.values() for m in run)
    run_times = sorted(by_run)
    runs: list[xr.Dataset] = []
    for run_time in run_times:
        cells = by_run.pop(run_time)
        here = [m for m in members if m in cells]
        run = _concat([cells.pop(m) for m in here], pd.Index(here, name=_MEMBER))
        if here != members:
            # Every run on the same member axis, in natural order, so the
            # run-level join never re-sorts members as text.
            run = run.reindex({_MEMBER: members})
        runs.append(run)
    inits = pd.Index(pd.DatetimeIndex(run_times).as_unit("ns"), name=_INIT)
    cube = _concat(runs, inits)
    cube = cube.sortby(_LEAD).transpose(_INIT, _MEMBER, _LEAD, ...)
    return cube.assign_coords(valid_time=cube[_INIT] + cube[_LEAD])


def load_states(
    catalog: GeoCatalog,
    slice_: GeoSlice,
    *,
    data_vars: Sequence[str] | None = None,
    group_by: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> xr.Dataset | xr.DataTree:
    """Load forecast states as an ``(init_time, member, lead_time, y, x)`` cube.

    Every row overlapping ``slice_`` is one run and member. Each file is
    clipped to the slice (bounds and valid-time interval), its valid
    times are relabelled as leads, ``τ = v - r``, and the files are
    stacked by run time and member. Two runs valid at the same time stay
    two states, unlike `load_xarray`, which keeps one valid time. A
    member or lead missing from a run comes back as NaN, not as a shorter
    cube, and ``valid_time = init_time + lead_time`` is a 2-D coordinate.
    Members are in natural order: by number when every label is an
    integer (``"2"`` before ``"10"``), else as text (``r1i1p1f1``). A run
    whose file overlaps the slice without a time step inside it has no
    state there and is left out.

    Args:
        catalog: A state catalog (`build_state_catalog`), on either
            backend; filter it first with the typed queries
            (`forecasts`, `ensemble_at`, …).
        slice_: The region and valid-time interval to load. Bounds may be
            in another CRS; they are reprojected to the catalog's.
        data_vars: Data variables to keep. ``None`` keeps them all.
        group_by: A row column (``"producer"``) to split the rows by.
            ``None`` returns one cube and requires one grid; a column
            returns an `xarray.DataTree` with one leaf per value, each
            leaf a cube on its own grid.
        storage_options: obstore options for remote files (`geocloud.fs`).

    Returns:
        An `xarray.Dataset` with dims ``(init_time, member, lead_time,
        y, x)`` (the files' own spatial names), or an `xarray.DataTree`
        of them when ``group_by`` is given. A deterministic run (null
        member) is the member ``""``.

    Raises:
        ImportError: If the ``[xarray-raster]`` extra is missing.
        TypeError: If the catalog is not of kind ``"xarray"``.
        ValueError: If the catalog has no state columns, no row matches,
            the slice holds none of the matching files' time steps, a row
            has no run time, a run and member appears twice, or the rows
            are on different grids without ``group_by``.

    Example:
        >>> from geocatalog.states import forecasts, load_states
        >>> cube = load_states(forecasts(catalog), aoi)
        >>> cube["xch4"].dims, cube["xch4"].shape
        (('init_time', 'member', 'lead_time', 'lat', 'lon'), (3, 4, 2, 4, 5))
    """
    if xr is None:
        raise missing_extra("`load_states`", "xarray-raster")
    if catalog.kind != "xarray":
        raise TypeError(
            f"load_states requires an xarray-kind catalog; got {catalog.kind!r}"
        )
    require_state_columns(catalog, "load_states")
    filtered = catalog.query(slice_)
    rows = list(filtered.iter_rows())
    if not rows:
        raise ValueError("load_states: no catalog rows match the slice")
    bounds = _reproject_bounds(slice_.bounds, slice_.crs, filtered.crs)
    interval = (slice_.interval.left, slice_.interval.right)

    def read(row: CatalogRow) -> xr.Dataset:
        return _read_clipped(
            row.filepath,
            bounds=bounds,
            interval=interval,
            time_var=str(row.extras.get("time_var") or "time"),
            data_vars=data_vars,
            storage_options=storage_options,
        )

    # Catalog order is arbitrary (a SQL relation has none): sort first.
    rows.sort(key=lambda r: r.filepath)
    if group_by is None:
        return _stack(rows, read)
    if group_by not in rows[0].extras:
        raise ValueError(f"load_states: group_by column {group_by!r} not in catalog")
    groups: dict[str, list[CatalogRow]] = {}
    for row in rows:
        value = state_text(row.extras.get(group_by))
        if value is None or "/" in value:
            raise ValueError(
                f"load_states: {row.filepath} has {group_by}={value!r}; a group "
                "needs a non-null name without '/'."
            )
        groups.setdefault(value, []).append(row)
    return xr.DataTree.from_dict(
        {f"/{name}": _stack(group, read) for name, group in sorted(groups.items())}
    )


__all__ = ["build_state_catalog", "load_states"]
