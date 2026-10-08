"""`geocatalog` console command.

Thin CLI over the library API — every subcommand maps to a single
public function and adds nothing more than argument parsing, exit-code
mapping, and human / JSON-friendly output. Business logic stays in
the library.

Exit codes:

* 0 — success.
* 1 — user error (bad args, missing extra, no files match glob,
       invalid bounds / time range / CRS, partial ``--start`` / ``--end``,
       a ``build vector`` input that is not a vector dataset).
* 2 — catalog error (corrupt artifact, schema mismatch).
* 3 — I/O error (path not readable / writable, parent dir missing).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from cyclopts import App, Parameter

from geocatalog import __version__
from geocatalog._src._schema import StorageEngine


# Sub-app + root. Cyclopts lets us register sub-apps via
# `app.command(sub_app)`; `name=` controls the verb the user types.
# `version=` is explicit: cyclopts would otherwise look up a distribution
# named after the module (`geocatalog`), which does not exist — the
# distribution is `geotoolz-catalog`.
app = App(
    name="geocatalog",
    help="Spatiotemporal catalog over geospatial files.",
    version=__version__,
)
build_app = App(
    name="build", help="Build a catalog from raster / xarray / vector files."
)
app.command(build_app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_glob(pattern: str) -> list[Path]:
    """Local-disk glob expansion with ``**`` support.

    Remote URIs (``s3://``, ``gs://``, ``https://``) raise — a future
    PR can plug in fsspec; for now the CLI is local-only, matching
    the library's loader surface.
    """
    if "://" in pattern:
        raise ValueError(
            f"Remote URIs are not supported by the CLI yet: "
            f"{pattern!r}. Expand the URI list yourself and pass concrete paths."
        )
    matches = sorted(glob.glob(pattern, recursive=True))
    return [Path(m) for m in matches]


def _parse_bounds(s: str) -> tuple[float, float, float, float]:
    """``"xmin,ymin,xmax,ymax"`` → tuple of four floats."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"--bounds must be 'xmin,ymin,xmax,ymax' (4 floats); got {s!r}"
        )
    try:
        xmin, ymin, xmax, ymax = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"--bounds values must be numeric; got {s!r}") from exc
    return (xmin, ymin, xmax, ymax)


def _crs_error(value: str | None, flag: str) -> str | None:
    """A one-line message when ``value`` is not a CRS pyproj understands."""
    if value is None:
        return None
    import pyproj

    try:
        pyproj.CRS.from_user_input(value)
    except pyproj.exceptions.CRSError as exc:
        # pyproj may echo a multi-line input (WKT); keep the message on
        # one line.
        return f"invalid {flag} {value!r}: {' '.join(str(exc).split())}"
    return None


def _unreadable_input_errors() -> tuple[type[Exception], ...]:
    """Reader errors for an input that exists but is not a supported file.

    pyogrio / fiona raise `RuntimeError`-family errors for "not a vector
    file"; they are user errors (exit 1), not crashes.
    """
    errors: list[type[Exception]] = []
    try:
        from pyogrio.errors import DataSourceError

        errors.append(DataSourceError)
    except ImportError:  # pragma: no cover - pyogrio ships with geopandas
        pass
    try:
        from fiona.errors import DriverError

        errors.append(DriverError)
    except ImportError:
        pass
    return tuple(errors)


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    """Print ``payload`` as JSON or pretty key/value lines."""
    if as_json:
        print(json.dumps(payload, default=str, indent=2))
        return
    width = max((len(str(k)) for k in payload), default=0)
    for key, value in payload.items():
        print(f"{key.ljust(width)}  {value}")


def _write_catalog(cat: Any, out: Path) -> int | None:
    """Persist `cat` to `out`, returning a CLI exit code on failure.

    Returns ``None`` on success, ``3`` on filesystem failure. Pulled
    into a helper so each `build` subcommand maps OSError to exit 3
    consistently.
    """
    from geocatalog.storage import to_geoparquet

    try:
        to_geoparquet(cat, out)
    except OSError as exc:
        print(f"could not write {out}: {exc}", file=sys.stderr)
        return 3
    return None


def _emit_build_result(out: Path, n_rows: int, *, json_output: bool) -> None:
    """Standardised success line for the build subcommands."""
    if json_output:
        print(json.dumps({"out": str(out), "rows": n_rows}))
    else:
        print(f"wrote {out} ({n_rows} rows)")


def _parse_partition_by(value: str | None) -> tuple[str, ...] | None:
    """Parse ``"year,month"`` into a tuple of Hive partition columns."""
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError("--partition-by must name at least one column")
    return parts


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _retired_flag[T](
    old_flag: str, new_flag: str, old: T | None, new: T | None
) -> T | None:
    """Resolve a renamed flag (#246): the new one, or the old one with a notice.

    The old spelling still works for one minor release; using it prints
    a deprecation notice on stderr. Passing both raises `ValueError`.
    """
    if old is None:
        return new
    if new is not None:
        raise ValueError(f"pass {new_flag} or the deprecated {old_flag}, not both")
    print(
        f"warning: {old_flag} is deprecated, use {new_flag}; the old flag will "
        "be removed in the next minor release.",
        file=sys.stderr,
    )
    return old


def _run_build(
    verb: str,
    build: Callable[[list[Path]], Any],
    *,
    input_glob: str,
    out: Path,
    crs: str | None,
    json_output: bool,
    write: bool = True,
    input_errors: tuple[type[Exception], ...] = (),
) -> int:
    """The body every ``build`` subcommand shares, with its exit codes.

    Validates ``--crs``, expands the glob, runs ``build`` and
    writes the catalog to ``out`` (unless ``write=False``: the builder
    already did). ``input_errors`` are reader errors that mean "not a
    file of this kind" — user errors (exit 1), not crashes.
    """
    if (err := _crs_error(crs, "--crs")) is not None:
        print(err, file=sys.stderr)
        return 1
    try:
        paths = _expand_glob(input_glob)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not paths:
        print(f"no files matched {input_glob!r}", file=sys.stderr)
        return 1
    try:
        cat = build(paths)
    except ImportError as exc:
        print(f"build {verb} needs an extra: {exc}", file=sys.stderr)
        return 1
    except (ValueError, TypeError, *input_errors) as exc:
        print(f"build {verb} failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"build {verb} I/O error: {exc}", file=sys.stderr)
        return 3
    if write:
        code = _write_catalog(cat, out)
        if code is not None:
            return code
    _emit_build_result(out, len(cat), json_output=json_output)
    return 0


@build_app.command
def raster(
    *,
    input_glob: Annotated[
        str, Parameter(help="Glob over raster files. Use ** for recursion.")
    ],
    out: Annotated[Path, Parameter(help="Destination GeoParquet path.")],
    regex: Annotated[
        str | None,
        Parameter(
            help=(
                "Filename regex with `(?P<date>...)` or `(?P<start>...)+(?P<stop>...)`."
            )
        ),
    ] = None,
    date_format: Annotated[
        str, Parameter(help="strptime fmt for regex date groups.")
    ] = "%Y%m%d",
    crs: Annotated[
        str | None,
        Parameter(help="Catalog CRS. None latches onto the first file's native CRS."),
    ] = None,
    engine: Annotated[
        StorageEngine | None,
        Parameter(
            help="`memory` (default) builds in RAM; `duckdb` streams to GeoParquet."
        ),
    ] = None,
    json_output: Annotated[
        bool, Parameter(name=["--json"], help="Emit machine-readable JSON.")
    ] = False,
    target_crs: Annotated[
        str | None, Parameter(name="--target-crs", show=False)
    ] = None,  # deprecated spelling of --crs
    backend: Annotated[
        StorageEngine | None, Parameter(name="--backend", show=False)
    ] = None,  # deprecated spelling of --engine
) -> int:
    """Build a raster catalog from a glob of GeoTIFFs."""
    try:
        crs = _retired_flag("--target-crs", "--crs", target_crs, crs)
        engine = _retired_flag("--backend", "--engine", backend, engine) or "memory"
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    def build(paths: list[Path]) -> Any:
        from geocatalog.build import build_raster_catalog

        return build_raster_catalog(
            paths,
            filename_regex=regex,
            date_format=date_format,
            crs=crs,
            engine=engine,
            out_path=out if engine == "duckdb" else None,
        )

    return _run_build(
        "raster",
        build,
        input_glob=input_glob,
        out=out,
        crs=crs,
        json_output=json_output,
        write=engine == "memory",  # the duckdb engine streams straight to `out`
    )


@build_app.command
def xarray(
    *,
    input_glob: Annotated[str, Parameter(help="Glob over NetCDF / Zarr / HDF stores.")],
    out: Annotated[Path, Parameter(help="Destination GeoParquet path.")],
    time_var: Annotated[
        str, Parameter(help="Coordinate name for the time axis.")
    ] = "time",
    crs: Annotated[
        str | None,
        Parameter(help="CRS to tag the catalog with (not used to reproject)."),
    ] = None,
    json_output: Annotated[
        bool, Parameter(name=["--json"], help="Emit machine-readable JSON.")
    ] = False,
    target_crs: Annotated[
        str | None, Parameter(name="--target-crs", show=False)
    ] = None,  # deprecated spelling of --crs
) -> int:
    """Build an xarray-shaped catalog. Requires the `[xarray-raster]` extra."""
    try:
        crs = _retired_flag("--target-crs", "--crs", target_crs, crs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    def build(paths: list[Path]) -> Any:
        from geocatalog.build import build_xarray_catalog

        return build_xarray_catalog(paths, time_var=time_var, crs=crs)

    return _run_build(
        "xarray",
        build,
        input_glob=input_glob,
        out=out,
        crs=crs,
        json_output=json_output,
    )


@build_app.command
def vector(
    *,
    input_glob: Annotated[str, Parameter(help="Glob over vector files.")],
    out: Annotated[Path, Parameter(help="Destination GeoParquet path.")],
    layer: Annotated[
        str | None, Parameter(help="Layer name/index for multi-layer files.")
    ] = None,
    regex: Annotated[
        str | None, Parameter(help="Filename regex for time parsing.")
    ] = None,
    date_format: Annotated[
        str, Parameter(help="strptime fmt for regex date groups.")
    ] = "%Y%m%d",
    crs: Annotated[str | None, Parameter(help="Catalog CRS.")] = None,
    json_output: Annotated[
        bool, Parameter(name=["--json"], help="Emit machine-readable JSON.")
    ] = False,
    target_crs: Annotated[
        str | None, Parameter(name="--target-crs", show=False)
    ] = None,  # deprecated spelling of --crs
) -> int:
    """Build a vector catalog (Shapefile / GeoPackage / GeoJSON)."""
    try:
        crs = _retired_flag("--target-crs", "--crs", target_crs, crs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    def build(paths: list[Path]) -> Any:
        from geocatalog.build import build_vector_catalog

        return build_vector_catalog(
            paths,
            filename_regex=regex,
            date_format=date_format,
            crs=crs,
            layer=layer,
        )

    return _run_build(
        "vector",
        build,
        input_glob=input_glob,
        out=out,
        crs=crs,
        json_output=json_output,
        input_errors=_unreadable_input_errors(),
    )


# ---------------------------------------------------------------------------
# query / stats / info
# ---------------------------------------------------------------------------


def _open_catalog(source: Path) -> Any:
    """Open a catalog artifact, mapping errors to CLI exit codes.

    Uses ``engine='auto'`` so the DuckDB backend (lazy, scales) is
    preferred when the ``[duckdb]`` extra is installed — `stats` /
    `query` are correspondingly cheap on large artifacts. The
    in-memory backend is the fallback.

    Exit codes:

    * 3 — file missing or not readable (checked before opening so we
      catch the DuckDB / pyarrow surface error consistently across
      backends).
    * 2 — file is present and readable but the open failed (corrupt
      parquet, unknown column layout, schema-version mismatch).

    Returns the opened catalog directly; the caller is expected to
    propagate any raised SystemExit through.
    """
    from geocatalog import open_catalog

    if not source.exists():
        print(f"catalog not found: {source}", file=sys.stderr)
        raise SystemExit(3)
    # Filesystem-level readability check before handing off. DuckDB
    # surfaces permission errors as `duckdb.IOException` (not
    # `OSError`), and geopandas' pyarrow path mostly raises `OSError`
    # but not always — pre-checking access keeps the exit-code mapping
    # well-defined regardless of which backend `engine='auto'` lands on.
    if source.is_file() and not os.access(source, os.R_OK):
        print(f"could not read {source}: permission denied", file=sys.stderr)
        raise SystemExit(3)
    try:
        return open_catalog(source)
    except OSError as exc:
        print(f"could not read {source}: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
    except (ValueError, KeyError) as exc:
        print(f"corrupt or unrecognised catalog ({source}): {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        # DuckDB raises its own DatabaseError subclasses (IOException,
        # InvalidInputException, …) that don't inherit from OSError or
        # ValueError. Inspect the message to land on the right exit
        # code — pre-check above handles the common "permission denied"
        # path, so anything reaching here is treated as a catalog
        # error rather than I/O.
        print(f"failed to open catalog ({source}): {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _catalog_crs(cat: Any) -> str:
    """Backend-agnostic CRS readout (`InMemoryGeoCatalog` and `DuckDBGeoCatalog`).

    `DuckDBGeoCatalog` exposes ``.crs`` directly; `InMemoryGeoCatalog`
    routes through ``cat.gdf.crs``. Both backends populate
    ``cat.get_config()["crs"]``, so we read from there to avoid
    materialising the relation just to print metadata.
    """
    config = cat.get_config()
    crs = config.get("crs")
    return "" if crs is None else str(crs)


@app.command
def query(
    source: Annotated[Path, Parameter(help="GeoParquet catalog to query.")],
    *,
    bounds: Annotated[
        str | None, Parameter(help='"xmin,ymin,xmax,ymax" in --crs units.')
    ] = None,
    crs: Annotated[str, Parameter(help="CRS of --bounds.")] = "EPSG:4326",
    start: Annotated[str | None, Parameter(help="Start of time window (ISO).")] = None,
    end: Annotated[str | None, Parameter(help="End of time window (ISO).")] = None,
    json_output: Annotated[
        bool, Parameter(name=["--json"], help="Emit machine-readable JSON.")
    ] = False,
    bbox: Annotated[
        str | None, Parameter(name="--bbox", show=False)
    ] = None,  # deprecated spelling of --bounds
) -> int:
    """Filter ``source`` by bounds + time and print the matching row count.

    ``--start`` and ``--end`` are paired — pass either both or neither.
    `_coerce_interval` requires two timestamp-likes, so passing one
    half of the window would otherwise crash; we fail fast with exit 1
    instead.
    """
    try:
        bounds = _retired_flag("--bbox", "--bounds", bbox, bounds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if (start is None) != (end is None):
        print(
            "--start and --end must be passed together (or both omitted)",
            file=sys.stderr,
        )
        return 1
    try:
        box = _parse_bounds(bounds) if bounds else None
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if (err := _crs_error(crs, "--crs")) is not None:
        print(err, file=sys.stderr)
        return 1
    time: tuple[Any, Any] | None = None
    if start is not None and end is not None:
        try:
            time = (pd.Timestamp(start), pd.Timestamp(end))
        except (ValueError, TypeError) as exc:
            print(f"invalid --start / --end timestamp: {exc}", file=sys.stderr)
            return 1
    cat = _open_catalog(source)
    try:
        result = cat.query(bounds=box, crs=crs, time=time)
    except (ValueError, TypeError) as exc:
        print(f"query failed: {exc}", file=sys.stderr)
        return 1
    _emit(
        {
            "source": str(source),
            "rows": len(result),
            "bounds": box,
            "bbox": box,  # deprecated key, kept for one release
            "time": [str(start), str(end)] if time else None,
        },
        as_json=json_output,
    )
    return 0


@app.command
def stats(
    source: Annotated[Path, Parameter(help="GeoParquet catalog to summarise.")],
    *,
    json_output: Annotated[
        bool, Parameter(name=["--json"], help="Emit machine-readable JSON.")
    ] = False,
) -> int:
    """Print rows / bounds / temporal extent / kind / CRS for ``source``.

    Uses the Protocol-level ``total_bounds`` / ``temporal_extent`` /
    ``kind`` properties + ``get_config()`` for CRS — none of these
    materialise the relation through pandas on the DuckDB backend.
    """
    cat = _open_catalog(source)
    extent = cat.temporal_extent
    _emit(
        {
            "rows": len(cat),
            "bounds": list(cat.total_bounds),
            "temporal_start": None if extent is None else extent.left,
            "temporal_end": None if extent is None else extent.right,
            "kind": cat.kind,
            "backend": cat.kind,  # deprecated key, kept for one release
            "crs": _catalog_crs(cat),
        },
        as_json=json_output,
    )
    return 0


@app.command
def migrate(
    source: Annotated[Path, Parameter(help="GeoParquet catalog to migrate in-place.")],
    *,
    to_version: Annotated[
        int | None,
        Parameter(help="Target schema version. Defaults to the reader's current."),
    ] = None,
) -> int:
    """Rewrite ``source`` at the requested schema version.

    Mirrors `_open_catalog`'s exit-code mapping:

    * 3 — source missing or unreadable.
    * 2 — corrupt artifact / schema mismatch (`CatalogSchemaError`).
    """
    import os

    from geocatalog._src.base import CatalogSchemaError
    from geocatalog.storage import SCHEMA_VERSION_CURRENT, migrate_geoparquet

    if not source.exists():
        print(f"catalog not found: {source}", file=sys.stderr)
        return 3
    if source.is_file() and not os.access(source, os.R_OK):
        print(f"could not read {source}: permission denied", file=sys.stderr)
        return 3
    target = SCHEMA_VERSION_CURRENT if to_version is None else to_version
    try:
        v_before = migrate_geoparquet(source, to_version=target)
    except CatalogSchemaError as exc:
        print(f"migrate failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"migrate I/O error: {exc}", file=sys.stderr)
        return 3
    except (ValueError, KeyError) as exc:
        print(f"migrate failed: {exc}", file=sys.stderr)
        return 2
    if v_before == target:
        print(f"{source} already at v{target}")
    else:
        print(f"wrote {source} (v{v_before} -> v{target})")
    return 0


@app.command
def convert(
    source: Annotated[Path, Parameter(help="Input GeoParquet catalog.")],
    *,
    out: Annotated[
        Path | None,
        Parameter(help="Destination GeoParquet file or directory."),
    ] = None,
    partition_by: Annotated[
        str | None,
        Parameter(help='Comma-separated Hive partition columns, e.g. "year,month".'),
    ] = None,
    json_output: Annotated[
        bool, Parameter(name=["--json"], help="Emit machine-readable JSON.")
    ] = False,
) -> int:
    """Convert a catalog artifact, optionally to Hive-partitioned layout."""
    from geocatalog.storage import to_geoparquet

    try:
        partitions = _parse_partition_by(partition_by)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    cat = _open_catalog(source)
    materialized = cat.materialize() if hasattr(cat, "materialize") else cat
    destination = out if out is not None else source.with_suffix("")
    # `to_geoparquet(..., partition_by=...)` is destructive — it wipes
    # `destination` before writing. When the source has no extension
    # (already a partitioned dir) the default `source.with_suffix("")`
    # resolves to the source itself, which would read-then-delete the
    # input. Require an explicit `--out` for that case rather than
    # silently destroying the user's archive.
    if destination.resolve() == source.resolve():
        print(
            f"convert: refusing to overwrite source in place; pass --out to "
            f"specify a destination different from {source}",
            file=sys.stderr,
        )
        return 1
    try:
        to_geoparquet(materialized, destination, partition_by=partitions)
    except OSError as exc:
        print(f"convert I/O error: {exc}", file=sys.stderr)
        return 3
    except (ValueError, TypeError, KeyError) as exc:
        print(f"convert failed: {exc}", file=sys.stderr)
        return 1
    _emit(
        {"source": str(source), "out": str(destination), "rows": len(cat)},
        as_json=json_output,
    )
    return 0


@app.command
def info(
    source: Annotated[Path, Parameter(help="GeoParquet catalog to inspect.")],
    *,
    row: Annotated[int, Parameter(help="Row index to inspect.")] = 0,
    json_output: Annotated[
        bool, Parameter(name=["--json"], help="Emit machine-readable JSON.")
    ] = False,
) -> int:
    """Show one row of ``source`` in detail.

    Reads via ``cat.gdf`` — the DuckDB backend will materialise the
    relation through pandas here. Acceptable because the user asked
    for one specific row; if it ever matters in practice we can
    swap to a SQL ``LIMIT 1 OFFSET N`` shortcut on `DuckDBGeoCatalog`.
    """
    cat = _open_catalog(source)
    if row < 0 or row >= len(cat):
        print(f"row {row} out of range [0, {len(cat)})", file=sys.stderr)
        return 1
    series = cat.gdf.iloc[row]
    payload: dict[str, object] = {col: series[col] for col in series.index}
    interval = cat.gdf.index[row]
    if hasattr(interval, "left"):
        payload["start_time"] = interval.left
        payload["end_time"] = interval.right
    _emit(payload, as_json=json_output)
    return 0
