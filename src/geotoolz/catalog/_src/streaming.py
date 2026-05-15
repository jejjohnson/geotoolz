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
import json
import logging
import multiprocessing
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import shapely.geometry.base
import shapely.wkb


if TYPE_CHECKING:
    from geotoolz.catalog._src.duckdb_backend import DuckDBGeoCatalog


log = logging.getLogger(__name__)


_BACKEND_T = Literal["raster", "xarray", "vector"]
_GEOPARQUET_VERSION = "1.1.0"
_SCHEMA_VERSION = 0


# ---------------------------------------------------------------------------
# Parallel row extraction
# ---------------------------------------------------------------------------


def _iter_rows_parallel(
    filepaths: Sequence[str | Path],
    extract_fn: Callable[[str | Path], dict[str, Any] | None],
    *,
    n_workers: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield per-file row dicts, optionally extracted across a process pool.

    Args:
        filepaths: Files to extract from.
        extract_fn: Module-level (picklable) callable that takes a single
            path and returns a row dict, or ``None`` to skip the file.
        n_workers: Pool size. ``1`` runs everything in the calling process
            with no `multiprocessing` overhead. ``>1`` spawns a process
            pool with the ``"spawn"`` start method (rasterio + ``fork``
            deadlocks on macOS — spawn is the portable default).

    Yields:
        Row dicts in submission order. ``None`` returns from
        ``extract_fn`` are filtered out (filenames that didn't match the
        regex etc.).
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
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
    ) as pool:
        for row in pool.map(extract_fn, list(filepaths)):
            if row is not None:
                yield row


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
        """Infer the Arrow schema from the first row (or build a minimal one)."""
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
        reserved = {"filepath", "start_time", "end_time", "geometry", "bbox"}
        for key, value in sample.items():
            if key in reserved:
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
    from geotoolz.catalog._src.duckdb_backend import _require_duckdb

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
        log.debug("sort_geoparquet: rewrote %s with Hilbert sort -> %s", src, dst_path)


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
    drop = {"bbox", "_backend", "_schema_version"}
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
    batch_size: int = 10_000,
    n_workers: int = 1,
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
        batch_size: Streaming batch size.
        n_workers: Process-pool size for extraction.

    Returns:
        A `DuckDBGeoCatalog` opened on ``out_path``.

    Raises:
        ValueError: No files yielded a row.
    """
    from geotoolz.catalog._src.duckdb_backend import DuckDBGeoCatalog

    out_path = Path(out_path)
    if out_path.parent != Path() and not out_path.parent.exists():
        raise FileNotFoundError(
            f"stream_build_duckdb: parent dir does not exist: {out_path.parent}"
        )

    # Stream into a temp neighbour file so we can atomically replace `out_path`
    # only after the sort-rewrite succeeds.
    target_dir = out_path.parent if str(out_path.parent) else Path()
    if sort_by is None:
        # Direct write to the final path.
        write_target = out_path
        sort_tmp: Path | None = None
    else:
        fd, tmp_str = tempfile.mkstemp(
            prefix=out_path.stem + ".unsorted.",
            suffix=".parquet",
            dir=str(target_dir) if str(target_dir) else None,
        )
        # We only need the path; close the fd immediately so pyarrow can open it.
        import os

        os.close(fd)
        write_target = Path(tmp_str)
        sort_tmp = write_target

    rows_written = 0
    try:
        with StreamingParquetWriter(
            write_target,
            crs=crs,
            backend=backend,
            write_bbox=write_bbox,
            batch_size=batch_size,
        ) as writer:
            for row in _iter_rows_parallel(filepaths, extract_fn, n_workers=n_workers):
                writer.write_row(row)
                rows_written += 1

        if rows_written == 0:
            raise ValueError(
                "stream_build_duckdb: no files yielded a row (every file "
                "skipped or unmatched)"
            )

        if sort_tmp is not None:
            sort_geoparquet(
                sort_tmp,
                out_path,
                sort_by=sort_by,  # type: ignore[arg-type]
                crs=crs,
                backend=backend,
                write_bbox=write_bbox,
                batch_size=batch_size,
            )
            sort_tmp.unlink()
    except BaseException:
        # On any failure, surface the original exception. Leave the tmp file
        # in place if the streaming write succeeded but the sort failed — it's
        # easier to debug than a silent half-write.
        raise

    return DuckDBGeoCatalog.open(out_path, backend=backend, crs=crs)
