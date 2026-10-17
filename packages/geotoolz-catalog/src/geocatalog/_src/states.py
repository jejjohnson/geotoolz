"""Model-state name maps and typed queries over the reserved state columns.

Everything here works on any catalog that carries `STATE_COLUMNS` (one
built by `build_state_catalog`, or a migrated v0 catalog whose extras
used the names), on both backends: an `InMemoryGeoCatalog` is filtered
with a boolean mask, a `DuckDBGeoCatalog` with a SQL predicate whose
literals are typed here (never a user string), so the filter stays lazy.

The queries take `pd.Timestamp` / `pd.Timedelta` arguments rather than
query strings: on pandas 3, ``where("run_time == '2026-01-02'")`` on a
datetime column silently matches nothing.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from geocatalog._src._schema import STATE_COLUMNS
from geocatalog._src.backends.memory import InMemoryGeoCatalog
from geocatalog._src.base import CatalogRow, GeoCatalog
from geocatalog._src.utils.timeutil import to_naive_utc


if TYPE_CHECKING:
    from geocatalog._src.backends.duckdb_backend import DuckDBGeoCatalog


NameSource = str | Callable[[str], Any]
"""Where one state field is read: a name, or a callable on the file path.

A name is looked up first as a coordinate or variable of the dataset (a
scalar, or the time axis for ``valid_time``), then as a global attribute.
A callable receives the file path and returns the value (for archives
that encode the run in the file name).
"""

StateNames = Mapping[str, NameSource | tuple[NameSource, ...]]
"""Field → source(s), tried in order; the first one found wins.

Fields: ``valid_time`` (the time axis the row interval and the leads come
from), ``run_time``, ``member``, ``kind``, ``producer``, ``parents``. A
field left out of the map is null on every row.
"""

CF_NAMES: StateNames = MappingProxyType(
    {
        "valid_time": "time",
        "run_time": "forecast_reference_time",
        "member": "realization",
        "kind": "kind",
        "producer": "producer",
        "parents": "parents",
    }
)
"""Where CF files keep each state field.

The run time is the ``forecast_reference_time`` coordinate or attribute,
the member the ``realization`` one, and the valid time the ``time`` axis.
``kind``, ``producer`` and ``parents`` have no CF name: they are global
attributes of those names, written by the script that saves the states.
"""

CFGRIB_NAMES: StateNames = MappingProxyType(
    {
        "valid_time": "valid_time",
        "run_time": "time",
        "member": "number",
        "kind": "kind",
        "producer": "producer",
        "parents": "parents",
    }
)
"""Where cfgrib-decoded GRIB keeps each state field.

cfgrib names the run time ``time`` (a scalar coordinate), the lead
``step``, the member ``number`` and the valid time ``valid_time`` (a
coordinate along ``step``).
"""


# ---------------------------------------------------------------------------
# Typed arguments
# ---------------------------------------------------------------------------


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    """A naive-UTC timestamp from a datetime-like; strings are refused."""
    if not isinstance(value, (dt.datetime, np.datetime64)):
        raise TypeError(
            f"{name} must be a pd.Timestamp (or datetime / numpy.datetime64); "
            f"got {type(value).__name__}. Parse strings with pd.Timestamp(...)."
        )
    ts = to_naive_utc(value)
    if ts is pd.NaT:
        raise ValueError(f"{name} must not be NaT")
    return ts


def _timedelta(value: Any, name: str) -> pd.Timedelta:
    """A `pd.Timedelta` from a timedelta-like; strings are refused."""
    if not isinstance(value, (dt.timedelta, np.timedelta64)):
        raise TypeError(
            f"{name} must be a pd.Timedelta (or timedelta / numpy.timedelta64); "
            f"got {type(value).__name__}."
        )
    return pd.Timedelta(value)


def _member(value: Any) -> str:
    """The member column is text; an int would silently match nothing."""
    if not isinstance(value, str):
        raise TypeError(
            f"member must be a string such as '0' or 'r1i1p1f1' (the column "
            f"is text); got {type(value).__name__} {value!r}."
        )
    return value


# ---------------------------------------------------------------------------
# One filter, two backends
# ---------------------------------------------------------------------------

# A predicate is (left, op, right): ``left`` names a column or the lead at
# either end of the row (``"lead_start"`` / ``"lead_end"``); ``right`` is a
# typed value (str, pd.Timestamp, pd.Timedelta).
_Predicate = tuple[str, str, Any]
_OPS = {"==", "<=", ">=", "<", ">"}


def _is_duckdb(catalog: GeoCatalog) -> bool:
    try:
        from geocatalog._src.backends.duckdb_backend import DuckDBGeoCatalog
    except ImportError:  # pragma: no cover - the module imports softly
        return False
    return isinstance(catalog, DuckDBGeoCatalog)


def _columns(catalog: GeoCatalog) -> list[str]:
    if _is_duckdb(catalog):
        return list(catalog.relation.columns)  # type: ignore[attr-defined]
    return list(catalog.gdf.columns)


def require_state_columns(catalog: GeoCatalog, where: str) -> None:
    """Raise `ValueError` unless ``catalog`` carries every state column."""
    missing = [c for c in STATE_COLUMNS if c not in _columns(catalog)]
    if missing:
        raise ValueError(
            f"{where}: the catalog has no state column(s) {missing}; index model "
            "states with geocatalog.build.build_state_catalog."
        )


def _mask(gdf: pd.DataFrame, predicates: list[_Predicate]) -> np.ndarray:
    keep = np.ones(len(gdf), dtype=bool)
    left = pd.Series(gdf.index.left, index=gdf.index)
    right = pd.Series(gdf.index.right, index=gdf.index)
    for name, op, value in predicates:
        if name == "lead_start":
            series = left - gdf["run_time"]
        elif name == "lead_end":
            series = right - gdf["run_time"]
        elif name == "end_time":
            series = right
        elif name == "start_time":
            series = left
        else:
            series = gdf[name]
        test = {
            "==": series == value,
            "<=": series <= value,
            ">=": series >= value,
            "<": series < value,
            ">": series > value,
        }[op]
        # Comparisons with a null are False, as in SQL.
        keep &= test.fillna(False).to_numpy(dtype=bool)
    return keep


def _sql_literal(value: Any, *, time_literal: str = "TIMESTAMP") -> str:
    """A SQL literal for a typed value; never a raw user string in the SQL."""
    if isinstance(value, pd.Timestamp):
        suffix = "+00:00" if time_literal == "TIMESTAMPTZ" else ""
        return f"{time_literal} '{value.isoformat()}{suffix}'"
    if isinstance(value, pd.Timedelta):
        return f"to_microseconds({int(value / pd.Timedelta(microseconds=1))})"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise TypeError(f"no SQL literal for {type(value).__name__}")  # pragma: no cover


def _sql(predicates: list[_Predicate], *, time_literal: str) -> str:
    """The predicates as one SQL ``WHERE`` body.

    ``time_literal`` types timestamps compared with the catalog's own
    ``start_time`` / ``end_time`` (``TIMESTAMPTZ`` when they carry a zone,
    so DuckDB never reads a naive literal in the session's zone);
    ``run_time`` is always a naive ``TIMESTAMP`` (`state_arrow_types`):
    subtracted from a ``TIMESTAMPTZ`` it is read as UTC with
    ``timezone('UTC', run_time)``, never cast through the session zone.
    """
    run_time = (
        "timezone('UTC', run_time)" if time_literal == "TIMESTAMPTZ" else "run_time"
    )
    terms = []
    for name, op, value in predicates:
        if op not in _OPS:  # pragma: no cover - internal
            raise ValueError(op)
        column = {
            "lead_start": f"(start_time - {run_time})",
            "lead_end": f"(end_time - {run_time})",
        }.get(name, f'"{name}"')
        literal = time_literal if name in ("start_time", "end_time") else "TIMESTAMP"
        terms.append(f"{column} {op} {_sql_literal(value, time_literal=literal)}")
    return " AND ".join(terms)


def _filter(catalog: GeoCatalog, predicates: list[_Predicate], where: str) -> Any:
    """Rows of ``catalog`` where every predicate holds, on its own backend."""
    require_state_columns(catalog, where)
    if _is_duckdb(catalog):
        duck: DuckDBGeoCatalog = catalog  # type: ignore[assignment]
        if not predicates:
            return duck.query()
        return duck.sql(_sql(predicates, time_literal=duck._time_literal))
    if not isinstance(catalog, InMemoryGeoCatalog):
        catalog = InMemoryGeoCatalog(catalog.gdf, kind=catalog.kind)
    gdf = catalog.gdf
    return InMemoryGeoCatalog(gdf[_mask(gdf, predicates)], kind=catalog.kind)


# ---------------------------------------------------------------------------
# The queries
# ---------------------------------------------------------------------------


def forecasts(catalog: GeoCatalog) -> GeoCatalog:
    """The forecast rows of a state catalog (``kind == "forecast"``).

    Args:
        catalog: A catalog with the reserved state columns, on either
            backend.

    Returns:
        A catalog of the same backend with the forecast rows.

    Example:
        >>> fc = forecasts(catalog)
        >>> load_states(fc, aoi).sizes   # (init_time, member, lead_time, y, x)
    """
    return _filter(catalog, [("kind", "==", "forecast")], "forecasts")


def ensemble_at(catalog: GeoCatalog, *, run_time: pd.Timestamp) -> GeoCatalog:
    """Every member of one forecast run.

    Args:
        catalog: A state catalog.
        run_time: The run's initialisation time, a `pd.Timestamp` (a
            string raises `TypeError`). Matched exactly.

    Returns:
        A catalog of the forecast rows with that ``run_time``, one per
        member file.

    Raises:
        TypeError: If ``run_time`` is not datetime-like.

    Example:
        >>> members = ensemble_at(catalog, run_time=pd.Timestamp("2026-01-01"))
    """
    ts = _timestamp(run_time, "run_time")
    return _filter(
        catalog,
        [("kind", "==", "forecast"), ("run_time", "==", ts)],
        "ensemble_at",
    )


def forecast_chain(
    catalog: GeoCatalog, *, run_time: pd.Timestamp, member: str
) -> GeoCatalog:
    """One member of one forecast run: the files of a single trajectory.

    Args:
        catalog: A state catalog.
        run_time: The run's initialisation time, a `pd.Timestamp`.
        member: The member label, as text (``"0"``, ``"r1i1p1f1"``).

    Returns:
        A catalog of the forecast rows of that run and member.

    Raises:
        TypeError: If ``run_time`` is not datetime-like or ``member`` is
            not a string (the column is text, so ``0`` would match
            nothing).

    Example:
        >>> chain = forecast_chain(catalog, run_time=pd.Timestamp("2026-01-02"),
        ...                        member="0")
    """
    ts = _timestamp(run_time, "run_time")
    label = _member(member)
    return _filter(
        catalog,
        [("kind", "==", "forecast"), ("run_time", "==", ts), ("member", "==", label)],
        "forecast_chain",
    )


def lead_window(
    catalog: GeoCatalog, start: pd.Timedelta, end: pd.Timedelta
) -> GeoCatalog:
    """Forecast rows whose leads overlap ``[start, end]``.

    A row covers valid times ``[v_min, v_max]``, so its leads are
    ``[v_min - run_time, v_max - run_time]``; lead time is derived, not
    stored.

    Args:
        catalog: A state catalog.
        start: First lead of the window, a `pd.Timedelta`.
        end: Last lead of the window (inclusive).

    Returns:
        A catalog of the forecast rows with at least one lead in the
        window.

    Raises:
        TypeError: If a bound is not timedelta-like.
        ValueError: If ``start > end``.

    Example:
        >>> day_two = lead_window(catalog, pd.Timedelta("2D"), pd.Timedelta("2D"))
    """
    lo, hi = _timedelta(start, "start"), _timedelta(end, "end")
    if lo > hi:
        raise ValueError(f"lead_window: start {lo} is after end {hi}")
    return _filter(
        catalog,
        [("kind", "==", "forecast"), ("lead_end", ">=", lo), ("lead_start", "<=", hi)],
        "lead_window",
    )


def at_time(catalog: GeoCatalog, time: pd.Timestamp) -> GeoCatalog:
    """States of every kind valid at ``time``: rows whose interval contains it.

    Args:
        catalog: A state catalog.
        time: The valid time, a `pd.Timestamp`.

    Returns:
        A catalog of the rows whose valid-time interval contains ``time``
        (analyses, and every run and member that reaches it).

    Example:
        >>> jan3 = at_time(catalog, pd.Timestamp("2026-01-03"))
    """
    ts = _timestamp(time, "time")
    return _filter(
        catalog,
        [("start_time", "<=", ts), ("end_time", ">=", ts)],
        "at_time",
    )


def latest_analysis(catalog: GeoCatalog, *, before: pd.Timestamp) -> CatalogRow | None:
    """The analysis with the latest valid time at or before ``before``.

    Filters on the reserved ``kind`` column, not on a zero lead: a lead-0
    forecast row is valid at its run time too. Analyses valid at the same
    time are ordered by run time, then by file path (the last wins), so
    every backend returns the same row.

    Args:
        catalog: A state catalog.
        before: The cut-off valid time, a `pd.Timestamp` (inclusive).

    Returns:
        The `CatalogRow` of that analysis, or ``None`` when there is none.

    Example:
        >>> row = latest_analysis(catalog, before=pd.Timestamp("2026-01-03 06:00"))
        >>> row.filepath
        'ana_20260103.nc'
    """
    ts = _timestamp(before, "before")
    rows = _filter(
        catalog,
        [("kind", "==", "analysis"), ("end_time", "<=", ts)],
        "latest_analysis",
    )
    best: CatalogRow | None = None
    for row in rows.iter_rows():
        if best is None or _recency(row) > _recency(best):
            best = row
    return best


def _recency(row: CatalogRow) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Sort key of `latest_analysis`: valid time, then run time, then path.

    The same on every backend, whatever order the rows arrive in (a SQL
    relation has none); a missing run time sorts first.
    """
    state = row.state
    run = state.run_time if state is not None else None
    return (row.interval.right, pd.Timestamp.min if run is None else run, row.filepath)


__all__ = [
    "CFGRIB_NAMES",
    "CF_NAMES",
    "NameSource",
    "StateNames",
    "at_time",
    "ensemble_at",
    "forecast_chain",
    "forecasts",
    "latest_analysis",
    "lead_window",
    "require_state_columns",
]
