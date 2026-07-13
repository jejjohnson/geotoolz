"""Streaming GeoParquet writer + parallel row extraction for catalog builders.

The `backend="duckdb"` branch of the per-format builders
(`build_raster_catalog` / `build_vector_catalog` / `build_xarray_catalog`)
funnels through this module rather than materialising a `gpd.GeoDataFrame`
in RAM. The flow is:

1. **Extract** per-file metadata rows with `_iter_rows_parallel`, optionally
   distributing extraction across `n_workers` processes (a single writer
   thread consumes the pool's output — see `geoduckdb.md` §4.6).
2. **Stream-write** rows to a temp GeoParquet via `StreamingParquetWriter`,
   batching into Arrow record batches of `batch_size` rows (peak RAM is
   `O(batch_size * row_size)`, not `O(n_rows)`).
3. **Sort-rewrite** the temp file via DuckDB when `sort_by` is non-None,
   expanding the literal token ``"geometry_hilbert"`` to
   ``ST_Hilbert(ST_Centroid(geometry))`` and streaming the sorted relation
   back through `pyarrow.parquet.ParquetWriter` (the design defers
   DuckDB's native GeoParquet writer — see §sharp-edges line 974).
4. **Open** the final artifact as a `DuckDBGeoCatalog`.

The streaming writer produces a fully GeoParquet 1.1-compliant file (PROJJSON
CRS metadata + per-row ``bbox`` covering struct), so it round-trips through
`geopandas.read_parquet` and `DuckDBGeoCatalog.open` without ceremony.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import multiprocessing
import os
import shutil
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import shapely.geometry.base
import shapely.wkb


if TYPE_CHECKING:
    from geocatalog._src.duckdb_backend import DuckDBGeoCatalog


from loguru import logger as log

from geocatalog._src.base import INTERNAL_COLUMNS, RESERVED_COLUMNS
from geocatalog._src.parquet import SCHEMA_VERSION_CURRENT as _SCHEMA_VERSION


_BACKEND_T = Literal["raster", "xarray", "vector"]
_GEOPARQUET_VERSION = "1.1.0"


# Default cap on concurrently-open shard writers in
# `_write_partitioned_rows`. One file descriptor per open writer; with
# high-cardinality `partition_by` (say, ``("year","month","day")`` across
# a decade of daily data), the unbounded version would burn through the
# default ``ulimit -n`` of 1024 deterministically. 64 keeps headroom for
# the rest of the process (DuckDB pages, log files, etc.) while still
# amortising the per-shard open overhead.
_DEFAULT_MAX_OPEN_WRITERS: int = 64


# ---------------------------------------------------------------------------
# Parallel row extraction
# ---------------------------------------------------------------------------


def _iter_rows_parallel(
    filepaths: Sequence[str | Path] | Iterator[str | Path],
    extract_fn: Callable[[str | Path], dict[str, Any] | None],
    *,
    n_workers: int = 1,
    ordered: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield per-file row dicts, optionally extracted across a process pool.

    Args:
        filepaths: Files to extract from. May be a sequence or any
            iterable — the parallel path streams the input rather than
            materialising it, so 10⁶-file iterables don't blow up the
            coordinator process.
        extract_fn: Module-level (picklable) callable that takes a single
            path and returns a row dict, or ``None`` to skip the file.
        n_workers: Pool size. ``1`` runs everything in the calling process
            with no `multiprocessing` overhead. ``>1`` spawns a process
            pool with the ``"spawn"`` start method (rasterio + ``fork``
            deadlocks on macOS — spawn is the portable default).
        ordered: With ``n_workers>1``, yield rows in input order instead
            of completion order. Keeps the pending-future buffer bounded
            by ``n_workers``. A slow input EARLIER in the queue stalls
            every subsequent yield AND can temporarily reduce parallelism
            (workers may sit idle while the coordinator waits for the
            next-in-line future before refilling). The cost is most
            visible on skewed workloads where the first few inputs take
            much longer than the rest. For workloads with significant
            variance, prefer ``ordered=False`` and sort post-hoc if you
            need a stable byte layout.

    Yields:
        Row dicts. ``None`` returns from ``extract_fn`` are filtered out
        (filenames that didn't match the regex etc.). With ``n_workers=1``
        the order is the input order. With ``n_workers>1`` the default
        order is completion order; pass ``ordered=True`` for input order
        without requiring a downstream ``sort_by`` rewrite.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1; got {n_workers}")
    if n_workers == 1:
        for fp in filepaths:
            row = extract_fn(fp)
            if row is not None:
                yield row
        return

    ctx = multiprocessing.get_context("spawn")
    # Keep at most `window` futures pending at once. Bounded submission
    # avoids the O(n_files) coordinator-memory blowup of
    # `ProcessPoolExecutor.map(fn, list(filepaths))`, which materialises
    # the full input list and queues every future up front.
    # Ordered mode waits for the next-in-line future, so keep only one
    # worker-width of results buffered behind any slow file.
    window = n_workers if ordered else max(n_workers * 4, 8)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
    ) as pool:
        iterator = iter(filepaths)
        if ordered:
            indexed = enumerate(iterator)
            pending_by_index: dict[
                int,
                concurrent.futures.Future[dict[str, Any] | None],
            ] = {}

            def submit_next() -> None:
                try:
                    idx, fp = next(indexed)
                except StopIteration:
                    return
                pending_by_index[idx] = pool.submit(extract_fn, fp)

            for _ in range(window):
                submit_next()
            next_yield = 0
            while pending_by_index:
                fut = pending_by_index[next_yield]
                try:
                    row = fut.result()
                finally:
                    del pending_by_index[next_yield]
                    next_yield += 1
                submit_next()
                if row is not None:
                    yield row
            return

        pending: set[concurrent.futures.Future[dict[str, Any] | None]] = set()
        # Prime the window.
        for fp in itertools.islice(iterator, window):
            pending.add(pool.submit(extract_fn, fp))
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                row = fut.result()
                if row is not None:
                    yield row
                # Refill the window one-for-one.
                try:
                    nxt = next(iterator)
                except StopIteration:
                    continue
                pending.add(pool.submit(extract_fn, nxt))


# ---------------------------------------------------------------------------
# Streaming GeoParquet writer
# ---------------------------------------------------------------------------


class StreamingParquetWriter:
    """Append-only GeoParquet 1.1 writer over a row iterator.

    Buffers rows into Arrow record batches of `batch_size`, writes each
    batch via `pyarrow.parquet.ParquetWriter`, and on `close()` injects the
    aggregate GeoParquet ``geo`` key-value metadata (PROJJSON CRS, observed
    geometry types, total bbox, and the bbox-covering struct pointer).
    Geometries are encoded as WKB on the fly; the per-row bbox column
    (`xmin`/`ymin`/`xmax`/`ymax` struct) is computed per batch when
    `write_bbox=True`.

    Args:
        path: Destination GeoParquet file. Parent directory must exist.
        crs: CRS of every input geometry. Written into ``geo.columns.geometry.crs``
            as PROJJSON. Callers must canonicalise *before* writing — this
            class does not reproject.
        backend: Catalog backend tag, copied to the reserved ``_backend``
            column on every row (preserves loader-dispatch on reopen).
        schema_version: Value for the reserved ``_schema_version`` column.
            Bump from 0 on first substantive schema change.
        write_bbox: Emit the GeoParquet 1.1 ``bbox`` covering struct. Default
            True; turn off only if a downstream consumer chokes on 1.1.
        batch_size: Rows per Arrow record batch. Default 10 000 → peak
            memory ≈ 10 MB for ~1 KB-per-row catalogs (design §4.6).

    Usage::

        with StreamingParquetWriter(path, crs=crs, backend="raster") as w:
            for row in row_iter:
                w.write_row(row)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        crs: Any,
        backend: _BACKEND_T,
        schema_version: int = _SCHEMA_VERSION,
        write_bbox: bool = True,
        batch_size: int = 10_000,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        self._path = Path(path)
        self._crs = pyproj.CRS.from_user_input(crs)
        self._backend: _BACKEND_T = backend
        self._schema_version = schema_version
        self._write_bbox = write_bbox
        self._batch_size = batch_size

        self._buffer: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self._geometry_types: set[str] = set()
        self._total_bbox: list[float] | None = None  # [xmin, ymin, xmax, ymax]
        self._closed = False

    def __enter__(self) -> StreamingParquetWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Always close — even on exception — so partial files don't dangle
        # with an open file handle on Windows / network filesystems.
        try:
            self.close()
        except Exception:
            if exc is None:
                raise
            # If we're already unwinding from a write error, prefer the
            # original exception; swallow this one.
            log.exception(
                "StreamingParquetWriter: close failed during exception unwind"
            )

    def write_row(self, row: dict[str, Any]) -> None:
        """Buffer one row; flush when the batch fills."""
        if self._closed:
            raise RuntimeError("StreamingParquetWriter: write_row after close")
        self._buffer.append(row)
        if len(self._buffer) >= self._batch_size:
            self._flush_batch()

    def close(self) -> None:
        """Flush remaining rows and finalise the GeoParquet metadata."""
        if self._closed:
            return
        self._closed = True
        if self._buffer:
            self._flush_batch()
        if self._writer is None:
            # No rows were written. Build an empty file so the artifact path
            # exists with valid metadata — downstream code can still open it.
            schema = self._build_schema_from_first(row=None)
            self._writer = pq.ParquetWriter(self._path, schema=schema)
        self._writer.add_key_value_metadata(
            {"geo": json.dumps(self._build_geo_metadata())}
        )
        self._writer.close()

    # -- internals -----------------------------------------------------------

    def _flush_batch(self) -> None:
        rows = self._buffer
        self._buffer = []
        if self._schema is None:
            self._schema = self._build_schema_from_first(row=rows[0])
            self._writer = pq.ParquetWriter(self._path, schema=self._schema)
        table = self._rows_to_table(rows, schema=self._schema)
        assert self._writer is not None  # for ty
        self._writer.write_table(table)

    def _build_schema_from_first(self, row: dict[str, Any] | None) -> pa.Schema:
        """Infer the Arrow schema from the first row (or build a minimal one).

        The schema is sealed from the first row's extras keyset; subsequent
        rows must share the same keys. The current per-backend extractors
        (`_filepath_to_row`, `_xarray_row`, `_vector_row`) all return a
        uniform shape, so this holds. If a future extractor returns
        per-row optional extras, this branch will need to pre-union keys
        across the first N rows before sealing the schema.
        """
        # Reserved column order (matches what `to_geoparquet` writes for the
        # InMemory backend — keeps reader code uniform).
        fields: list[pa.Field] = []
        sample = dict(row) if row is not None else {}
        # Required columns first.
        fields.append(pa.field("filepath", pa.string()))
        fields.append(pa.field("start_time", pa.timestamp("us")))
        fields.append(pa.field("end_time", pa.timestamp("us")))
        fields.append(pa.field("geometry", pa.binary()))
        if self._write_bbox:
            fields.append(
                pa.field(
                    "bbox",
                    pa.struct(
                        [
                            pa.field("xmin", pa.float64()),
                            pa.field("ymin", pa.float64()),
                            pa.field("xmax", pa.float64()),
                            pa.field("ymax", pa.float64()),
                        ]
                    ),
                )
            )
        # Extras: infer per-column type from the sample row.
        for key, value in sample.items():
            if key in RESERVED_COLUMNS:
                continue
            fields.append(pa.field(key, _infer_arrow_type(value)))
        fields.append(pa.field("_backend", pa.string()))
        fields.append(pa.field("_schema_version", pa.int32()))
        return pa.schema(fields)

    def _rows_to_table(
        self,
        rows: list[dict[str, Any]],
        *,
        schema: pa.Schema,
    ) -> pa.Table:
        """Convert a list of row dicts into an Arrow table matching `schema`."""
        encoded: list[dict[str, Any]] = []
        for row in rows:
            geom = row.get("geometry")
            if not isinstance(geom, shapely.geometry.base.BaseGeometry):
                raise TypeError(
                    f"StreamingParquetWriter: 'geometry' must be a shapely "
                    f"geometry; got {type(geom).__name__}"
                )
            self._geometry_types.add(geom.geom_type)
            xmin, ymin, xmax, ymax = geom.bounds
            if self._total_bbox is None:
                self._total_bbox = [xmin, ymin, xmax, ymax]
            else:
                self._total_bbox[0] = min(self._total_bbox[0], xmin)
                self._total_bbox[1] = min(self._total_bbox[1], ymin)
                self._total_bbox[2] = max(self._total_bbox[2], xmax)
                self._total_bbox[3] = max(self._total_bbox[3], ymax)

            out = {k: v for k, v in row.items() if k != "geometry"}
            out["geometry"] = shapely.wkb.dumps(geom)
            if self._write_bbox:
                out["bbox"] = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            out["_backend"] = self._backend
            out["_schema_version"] = self._schema_version
            encoded.append(out)
        return pa.Table.from_pylist(encoded, schema=schema)

    def _build_geo_metadata(self) -> dict[str, Any]:
        column_meta: dict[str, Any] = {
            "encoding": "WKB",
            "geometry_types": sorted(self._geometry_types) or ["Polygon"],
            "crs": self._crs.to_json_dict(),
        }
        if self._total_bbox is not None:
            column_meta["bbox"] = list(self._total_bbox)
        if self._write_bbox:
            column_meta["covering"] = {
                "bbox": {
                    "xmin": ["bbox", "xmin"],
                    "ymin": ["bbox", "ymin"],
                    "xmax": ["bbox", "xmax"],
                    "ymax": ["bbox", "ymax"],
                }
            }
        return {
            "version": _GEOPARQUET_VERSION,
            "primary_column": "geometry",
            "columns": {"geometry": column_meta},
        }


def _infer_arrow_type(value: Any) -> pa.DataType:
    """Conservative type inference for extras columns.

    Only the types our three builders actually produce; anything weirder
    falls through to a string column so the writer doesn't crash on a
    badly-typed extras value.
    """
    if value is None:
        return pa.string()
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    if isinstance(value, str):
        return pa.string()
    if isinstance(value, list):
        if not value:
            return pa.list_(pa.string())
        return pa.list_(_infer_arrow_type(value[0]))
    return pa.string()


# ---------------------------------------------------------------------------
# DuckDB-driven sort-rewrite pass
# ---------------------------------------------------------------------------


def sort_geoparquet(
    src: str | Path,
    dst: str | Path,
    *,
    sort_by: Sequence[str],
    crs: Any,
    backend: _BACKEND_T,
    schema_version: int = _SCHEMA_VERSION,
    write_bbox: bool = True,
    batch_size: int = 10_000,
) -> None:
    """Rewrite a streamed GeoParquet file in `sort_by` order via DuckDB.

    Opens ``src`` as a DuckDB relation, sorts by the given keys (the
    literal token ``"geometry_hilbert"`` expands to
    ``ST_Hilbert(ST_Centroid(geometry))``), and streams the sorted
    relation back through `pyarrow.parquet.ParquetWriter` so the final
    file carries the same GeoParquet 1.1 ``geo`` metadata + bbox covering
    struct as the streamed input.

    Args:
        src: Path to the unsorted GeoParquet (written by
            `StreamingParquetWriter`).
        dst: Path to write the sorted result. Overwrites if it exists.
        sort_by: Sort keys. Plain column names pass through; the literal
            token ``"geometry_hilbert"`` expands to
            ``ST_Hilbert(ST_Centroid(geometry))``.
        crs: CRS to encode in the destination's GeoParquet metadata.
        backend: Backend tag for the destination's metadata.
        schema_version: Schema version for the destination.
        write_bbox: Emit the GeoParquet 1.1 covering bbox struct.
        batch_size: Pyarrow batch size for the streamed rewrite.

    Raises:
        ImportError: ``[duckdb]`` extra missing.
    """
    from geocatalog._src.duckdb_backend import _require_duckdb

    duckdb_mod = _require_duckdb()
    src_str = str(src)
    dst_path = Path(dst)

    # Build the sort SQL. Underlying assumption: `start_time` / `geometry`
    # exist as columns (every streamed builder produces them).
    order_exprs: list[str] = []
    needs_hilbert = False
    for key in sort_by:
        if key == "geometry_hilbert":
            order_exprs.append("ST_Hilbert(ST_Centroid(geometry))")
            needs_hilbert = True
        else:
            order_exprs.append(_quote_ident(key))
    if not order_exprs:
        raise ValueError("sort_by must contain at least one key")

    con = duckdb_mod.connect(":memory:")
    try:
        con.install_extension("spatial")
        con.load_extension("spatial")
        order_sql = ", ".join(order_exprs)
        sql = f"""
            SELECT *
            FROM read_parquet($src)
            ORDER BY {order_sql}
        """
        relation = con.sql(sql, params={"src": src_str})

        # Stream the sorted relation back through pyarrow. We re-encode via
        # `StreamingParquetWriter` so the destination carries the canonical
        # GeoParquet 1.1 metadata + bbox struct, regardless of what DuckDB's
        # arrow output looks like.
        # `to_arrow_reader` since duckdb 1.1; `fetch_arrow_reader` deprecated.
        reader = relation.to_arrow_reader(batch_size=batch_size)
        with StreamingParquetWriter(
            dst_path,
            crs=crs,
            backend=backend,
            schema_version=schema_version,
            write_bbox=write_bbox,
            batch_size=batch_size,
        ) as writer:
            for batch in reader:
                _write_arrow_batch(writer, batch)
    finally:
        con.close()
    if needs_hilbert:
        log.debug("sort_geoparquet: rewrote {} with Hilbert sort -> {}", src, dst_path)


def _quote_ident(name: str) -> str:
    """Quote a DuckDB identifier with safe escaping."""
    return '"' + name.replace('"', '""') + '"'


def _write_arrow_batch(writer: StreamingParquetWriter, batch: pa.RecordBatch) -> None:
    """Feed a DuckDB-emitted Arrow batch back into `StreamingParquetWriter`.

    DuckDB hands back the geometry column as WKB binary plus a ``bbox``
    struct column; we decode just the geometry to shapely so the writer
    can re-aggregate bbox/types from the canonical path, then drop the
    DuckDB-provided ``bbox`` / ``_backend`` / ``_schema_version`` columns
    (the writer re-emits its own).
    """
    table = pa.Table.from_batches([batch])
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    # The writer re-emits its own `bbox` covering struct on top of the
    # internal schema columns, so all three are dropped from the input.
    drop = {"bbox"} | INTERNAL_COLUMNS
    geom_wkb = cols.pop("geometry")
    for i in range(table.num_rows):
        row = {name: vals[i] for name, vals in cols.items() if name not in drop}
        wkb = geom_wkb[i]
        if isinstance(wkb, (bytes, bytearray, memoryview)):
            row["geometry"] = shapely.wkb.loads(bytes(wkb))
        else:
            raise TypeError(
                f"sort rewrite: unexpected geometry type {type(wkb).__name__}; "
                "expected bytes-like WKB"
            )
        writer.write_row(row)


# ---------------------------------------------------------------------------
# End-to-end streaming-build orchestrator
# ---------------------------------------------------------------------------


def stream_build_duckdb(
    filepaths: Sequence[str | Path],
    extract_fn: Callable[[str | Path], dict[str, Any] | None],
    *,
    out_path: str | Path,
    crs: Any,
    backend: _BACKEND_T,
    write_bbox: bool = True,
    sort_by: tuple[str, ...] | None = ("start_time", "geometry_hilbert"),
    partition_by: Sequence[str] | None = None,
    batch_size: int = 10_000,
    n_workers: int = 1,
    ordered: bool = False,
    max_open_writers: int = _DEFAULT_MAX_OPEN_WRITERS,
) -> DuckDBGeoCatalog:
    """Stream-build a `DuckDBGeoCatalog` artifact from per-file extraction.

    Drives the three-step pipeline (extract → stream-write → optionally
    sort-rewrite) and returns the catalog opened on the final path. Callers
    are responsible for canonicalising geometries to ``crs`` *inside*
    ``extract_fn`` — this orchestrator does no reprojection.

    Args:
        filepaths: Files to index.
        extract_fn: Picklable per-file extractor; returns a row dict or
            ``None`` to skip.
        out_path: Final GeoParquet destination.
        crs: CRS to record in the artifact (must match the geometries
            ``extract_fn`` produces).
        backend: Backend tag.
        write_bbox: Emit GeoParquet 1.1 ``bbox`` covering struct.
        sort_by: Sort keys for the post-write rewrite. ``None`` skips the
            rewrite and leaves rows in extraction order.
        partition_by: Optional Hive partition columns for directory output.
            Built-in ``"year"``, ``"month"``, and ``"day"`` values are
            derived from ``start_time``; other names are read from each row.
            Partitioned output skips the global sort rewrite because the
            directory layout provides the coarse pruning axis.
        batch_size: Streaming batch size.
        n_workers: Process-pool size for extraction.
        ordered: With ``n_workers>1``, preserve input row order instead of
            completion order. Useful for reproducible artifacts when
            ``sort_by=None``. A slow input earlier in the queue stalls
            every subsequent yield and can temporarily reduce parallelism
            (workers may sit idle waiting on the next-in-line future).
            Prefer ``ordered=False`` for skewed workloads and sort
            post-hoc if you need a stable byte layout.
        max_open_writers: Cap on concurrently open shard writers when
            ``partition_by`` is set. Ignored otherwise. See
            `write_partitioned_rows` for the LRU eviction semantics.

    Returns:
        A `DuckDBGeoCatalog` opened on ``out_path``.

    Raises:
        ImportError: ``[duckdb]`` extra missing — surfaced before any
            writes so a missing extra leaves the filesystem untouched.
        ValueError: No files yielded a row. ``out_path`` is left
            untouched (any pre-existing artifact is preserved).
    """
    # Fail fast before touching the filesystem: the final
    # `DuckDBGeoCatalog.open` (and the optional sort-rewrite) both need
    # the extra, so dying here saves us from writing a temp file we
    # can't open anyway.
    from geocatalog._src.duckdb_backend import (
        DuckDBGeoCatalog,
        _require_duckdb,
    )

    _require_duckdb()

    out_path = Path(out_path)
    if out_path.parent != Path() and not out_path.parent.exists():
        raise FileNotFoundError(
            f"stream_build_duckdb: parent dir does not exist: {out_path.parent}"
        )

    if partition_by is not None:
        rows = _iter_rows_parallel(filepaths, extract_fn, n_workers=n_workers)
        rows_written = _write_partitioned_rows(
            rows,
            out_path=out_path,
            crs=crs,
            backend=backend,
            partition_by=partition_by,
            write_bbox=write_bbox,
            batch_size=batch_size,
            replace=True,
            max_open_writers=max_open_writers,
        )
        if rows_written == 0:
            raise ValueError(
                "stream_build_duckdb: no files yielded a row (every file "
                "skipped or unmatched). Existing artifact at out_path "
                "(if any) was not modified."
            )
        return DuckDBGeoCatalog.open(out_path, backend=backend, crs=crs)

    # Always stream into a sibling temp file, regardless of `sort_by`.
    # Only move/rename to `out_path` after we've confirmed at least one
    # row was written and (when sorting) the rewrite succeeded. This is
    # what keeps a failed build (no matching files, sort-rewrite error,
    # interrupted process) from clobbering a pre-existing artifact at
    # `out_path`.
    target_dir = out_path.parent if str(out_path.parent) else Path()
    fd, tmp_str = tempfile.mkstemp(
        prefix=out_path.stem + ".staging.",
        suffix=".parquet",
        dir=str(target_dir) if str(target_dir) else None,
    )
    os.close(fd)
    staged = Path(tmp_str)

    sort_tmp: Path | None = None
    rows_written = 0
    try:
        with StreamingParquetWriter(
            staged,
            crs=crs,
            backend=backend,
            write_bbox=write_bbox,
            batch_size=batch_size,
        ) as writer:
            for row in _iter_rows_parallel(
                filepaths,
                extract_fn,
                n_workers=n_workers,
                ordered=ordered,
            ):
                writer.write_row(row)
                rows_written += 1

        if rows_written == 0:
            # Nothing was written that's worth keeping. Don't touch out_path.
            staged.unlink(missing_ok=True)
            raise ValueError(
                "stream_build_duckdb: no files yielded a row (every file "
                "skipped or unmatched). Existing artifact at out_path "
                "(if any) was not modified."
            )

        if sort_by is None:
            # Atomic replace — `os.replace` is rename(2) on POSIX, which
            # is atomic within a single filesystem.
            os.replace(staged, out_path)
        else:
            # Sort-rewrite reads `staged` and writes to a second temp,
            # which is then atomically moved into place.
            fd2, sort_tmp_str = tempfile.mkstemp(
                prefix=out_path.stem + ".sorted.",
                suffix=".parquet",
                dir=str(target_dir) if str(target_dir) else None,
            )
            os.close(fd2)
            sort_tmp = Path(sort_tmp_str)
            sort_geoparquet(
                staged,
                sort_tmp,
                sort_by=sort_by,
                crs=crs,
                backend=backend,
                write_bbox=write_bbox,
                batch_size=batch_size,
            )
            os.replace(sort_tmp, out_path)
            sort_tmp = None  # ownership transferred to out_path
            staged.unlink(missing_ok=True)
    except BaseException:
        # Surface the original exception; clean up only the temp files we
        # own, never `out_path` (the user's existing artifact, if any).
        staged.unlink(missing_ok=True)
        if sort_tmp is not None:
            sort_tmp.unlink(missing_ok=True)
        raise

    return DuckDBGeoCatalog.open(out_path, backend=backend, crs=crs)


def append_files(
    archive: str | Path,
    filepaths: Sequence[str | Path],
    extract_fn: Callable[[str | Path], dict[str, Any] | None],
    *,
    crs: Any,
    backend: _BACKEND_T,
    partition_by: Sequence[str],
    write_bbox: bool = True,
    batch_size: int = 10_000,
    n_workers: int = 1,
    max_open_writers: int = _DEFAULT_MAX_OPEN_WRITERS,
) -> DuckDBGeoCatalog:
    """Append new files to a Hive-partitioned GeoParquet archive.

    Only the new rows are extracted and written; existing shards are left
    untouched, so append work is ``O(N_new)`` plus the number of new
    partitions touched. The archive is created if it does not exist.

    Before any rows are written, the caller-supplied ``partition_by`` is
    validated against the archive's existing layout (the directory tree
    of ``key=value`` dirs). A mismatch raises `ValueError` rather than
    silently producing a mixed-layout archive that downstream readers
    can't reconstruct cleanly. A fresh archive (no existing shards)
    accepts the caller's ``partition_by`` as the source of truth.

    Args:
        archive: Hive-partitioned GeoParquet directory.
        filepaths: New source files to index.
        extract_fn: Picklable per-file extractor, same contract as
            `stream_build_duckdb`.
        crs: CRS to encode in the new shard metadata.
        backend: Backend tag for loader dispatch.
        partition_by: Hive partition columns. ``"year"``, ``"month"``,
            and ``"day"`` are derived from each row's ``start_time``.
            Must match the existing archive's layout, if any.
        write_bbox: Emit the GeoParquet 1.1 ``bbox`` covering struct.
        batch_size: Rows per Arrow record batch.
        n_workers: Process-pool size for extraction.
        max_open_writers: Cap on concurrently open shard writers; see
            `write_partitioned_rows`.

    Returns:
        A `DuckDBGeoCatalog` opened on the updated archive.

    Raises:
        ValueError: ``partition_by`` differs from the archive's existing
            layout, or no input files yielded a row.
    """
    from geocatalog._src.duckdb_backend import DuckDBGeoCatalog, _require_duckdb

    _require_duckdb()
    archive = Path(archive)
    if archive.exists() and not archive.is_dir():
        raise ValueError(f"append_files requires a partitioned directory: {archive}")
    if archive.parent != Path() and not archive.parent.exists():
        raise FileNotFoundError(
            f"append_files: parent dir does not exist: {archive.parent}"
        )
    requested = tuple(partition_by)
    existing = _detect_partition_layout(archive)
    if existing is not None and existing != requested:
        raise ValueError(
            f"append_files: partition_by={requested} does not match the "
            f"archive's existing layout {existing} at {archive}. Mixed "
            "Hive layouts produce shards that downstream readers cannot "
            "join cleanly — rebuild the archive at the new layout, or "
            "pass partition_by matching the existing one."
        )
    rows = _iter_rows_parallel(filepaths, extract_fn, n_workers=n_workers)
    rows_written = _write_partitioned_rows(
        rows,
        out_path=archive,
        crs=crs,
        backend=backend,
        partition_by=requested,
        write_bbox=write_bbox,
        batch_size=batch_size,
        replace=False,
        max_open_writers=max_open_writers,
    )
    if rows_written == 0:
        raise ValueError("append_files: no files yielded a row")
    return DuckDBGeoCatalog.open(archive, backend=backend, crs=crs)


def _detect_partition_layout(archive: Path) -> tuple[str, ...] | None:
    """Return the Hive partition column order under ``archive``, or None.

    Walks the directory tree under ``archive`` looking for the first
    ``.parquet`` shard, then walks back up reading the ``key=`` prefix
    from each ancestor directory until it reaches ``archive``. The
    returned tuple is the partition column order at write time.

    Returns ``None`` if ``archive`` does not exist, contains no parquet
    shards (fresh init), or contains parquet directly without any
    ``key=value`` ancestors (a non-partitioned archive — treated as
    "no layout to compare against" so the caller can still init).
    """
    if not archive.exists() or not archive.is_dir():
        return None
    shard = next(archive.rglob("*.parquet"), None)
    if shard is None:
        return None
    keys: list[str] = []
    current = shard.parent
    while current != archive and current.parent != current:
        name = current.name
        if "=" not in name:
            break
        keys.append(name.split("=", 1)[0])
        current = current.parent
    if not keys:
        return None
    keys.reverse()
    return tuple(keys)


def write_partitioned_rows(
    rows: Iterator[dict[str, Any]],
    *,
    out_path: str | Path,
    crs: Any,
    backend: _BACKEND_T,
    partition_by: Sequence[str],
    schema_version: int = _SCHEMA_VERSION,
    write_bbox: bool = True,
    batch_size: int = 10_000,
    replace: bool = True,
    max_open_writers: int = _DEFAULT_MAX_OPEN_WRITERS,
) -> int:
    """Write row dicts to a Hive-partitioned GeoParquet directory.

    This is the lower-level writer used by `to_geoparquet(...,
    partition_by=...)` and append workflows. Rows are streamed into one
    shard per touched partition; ``replace=False`` moves only those new
    shards into an existing archive.

    To bound file-descriptor usage with high-cardinality
    ``partition_by``, the writer keeps at most ``max_open_writers``
    `StreamingParquetWriter` instances open at any time and evicts the
    least-recently-used one when the cap is hit. If a previously-evicted
    partition receives more rows, a *new* shard file is opened for it
    (with a fresh shard ID) so we never truncate a finalised shard;
    Hive readers union all shards under a partition directory, so this
    is observationally identical to a single-shard partition.

    Args:
        rows: Iterator of catalog row dictionaries with shapely geometry.
        out_path: Destination partitioned directory.
        crs: CRS to encode in each shard's GeoParquet metadata.
        backend: Backend tag for loader dispatch.
        partition_by: Hive partition columns. ``"year"``, ``"month"``,
            and ``"day"`` are derived from each row's ``start_time``.
        schema_version: Reserved catalog schema version written per row.
        write_bbox: Emit the GeoParquet 1.1 ``bbox`` covering struct.
        batch_size: Rows per Arrow record batch.
        replace: Replace the whole output directory when True; append only
            the new shards when False.
        max_open_writers: Hard cap on concurrently open shard writers.
            Default 64 — enough headroom under a typical 1024 fd
            ulimit while still amortising open overhead. Set higher on
            generous-ulimit machines for fewer shard files per
            partition.

    Returns:
        Number of rows written.
    """
    return _write_partitioned_rows(
        rows,
        out_path=out_path,
        crs=crs,
        backend=backend,
        partition_by=partition_by,
        schema_version=schema_version,
        write_bbox=write_bbox,
        batch_size=batch_size,
        replace=replace,
        max_open_writers=max_open_writers,
    )


def _write_partitioned_rows(
    rows: Iterator[dict[str, Any]],
    *,
    out_path: str | Path,
    crs: Any,
    backend: _BACKEND_T,
    partition_by: Sequence[str],
    schema_version: int = _SCHEMA_VERSION,
    write_bbox: bool = True,
    batch_size: int = 10_000,
    replace: bool,
    max_open_writers: int = _DEFAULT_MAX_OPEN_WRITERS,
) -> int:
    partitions = tuple(partition_by)
    if not partitions:
        raise ValueError("partition_by must contain at least one column")
    if max_open_writers < 1:
        raise ValueError(f"max_open_writers must be >= 1; got {max_open_writers}")
    partition_set = set(partitions)

    out_path = Path(out_path)
    target_dir = out_path.parent
    staging = Path(
        tempfile.mkdtemp(
            prefix=out_path.name + ".partitioned.",
            dir=str(target_dir),
        )
    )
    # Open writers tracked LRU-style: insertion order = oldest-first, and
    # `move_to_end` on access promotes the touched key to most-recent.
    open_writers: OrderedDict[tuple[str, ...], StreamingParquetWriter] = OrderedDict()
    rows_written = 0
    write_session_id = uuid.uuid4().hex
    partition_counter = itertools.count()

    def _new_shard(values: tuple[str, ...]) -> StreamingParquetWriter:
        partition_dir = staging.joinpath(
            *(
                f"{name}={_format_partition_value(value)}"
                for name, value in zip(partitions, values, strict=True)
            )
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        partition_shard_id = f"{write_session_id}-{next(partition_counter):08d}"
        shard = partition_dir / f"part-{partition_shard_id}.parquet"
        return StreamingParquetWriter(
            shard,
            crs=crs,
            backend=backend,
            schema_version=schema_version,
            write_bbox=write_bbox,
            batch_size=batch_size,
        )

    try:
        for row in rows:
            values = tuple(_partition_value(row, name) for name in partitions)
            writer = open_writers.get(values)
            if writer is None:
                # Evict the LRU writer if we're at the cap. A re-opened
                # partition gets a *fresh* shard ID so we never write
                # to a finalised file.
                while len(open_writers) >= max_open_writers:
                    _, evicted = open_writers.popitem(last=False)
                    evicted.close()
                writer = _new_shard(values)
                open_writers[values] = writer
            else:
                open_writers.move_to_end(values)
            writer.write_row({k: v for k, v in row.items() if k not in partition_set})
            rows_written += 1
    except BaseException:
        for writer in open_writers.values():
            writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for writer in open_writers.values():
        writer.close()

    if rows_written == 0:
        shutil.rmtree(staging, ignore_errors=True)
        return 0

    if replace:
        if out_path.exists():
            if not out_path.is_dir():
                out_path.unlink()
            else:
                shutil.rmtree(out_path)
        os.replace(staging, out_path)
        return rows_written

    out_path.mkdir(parents=True, exist_ok=True)
    for shard in staging.rglob("*.parquet"):
        rel = shard.relative_to(staging)
        dest = out_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(shard, dest)
    shutil.rmtree(staging, ignore_errors=True)
    return rows_written


def _partition_value(row: dict[str, Any], name: str) -> Any:
    if name in row and row[name] is not None:
        return row[name]
    if name in {"year", "month", "day"}:
        if "start_time" not in row or row["start_time"] is None:
            raise ValueError(
                f"partition column {name!r} requires a 'start_time' field on "
                "every row (derived via pd.Timestamp). Set start_time in the "
                "extractor, or remove year/month/day from partition_by."
            )
        ts = pd.Timestamp(row["start_time"])
        if pd.isna(ts):
            # Silently producing "year=nan/month=nan/" shards corrupts the
            # archive layout — downstream readers can't tell those rows
            # apart from a string-valued "nan" partition.
            raise ValueError(
                f"partition column {name!r} cannot be derived from a NaT "
                "start_time. Filter out time-less rows upstream or supply "
                "explicit partition values on those rows."
            )
        return getattr(ts, name)
    raise KeyError(f"partition column {name!r} not present in row")


def _format_partition_value(value: Any) -> str:
    text = str(value)
    return text.replace("/", "_").replace("\\", "_").replace("=", "_")
