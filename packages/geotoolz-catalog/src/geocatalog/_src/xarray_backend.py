"""Xarray catalog builder + loader — extras-gated via `[xarray-raster]`.

The xarray backend lets the catalog index NetCDF / Zarr / HDF stores
the same way `build_raster_catalog` indexes GeoTIFFs. Each row's
footprint is derived from the dataset's coordinate min/max; the time
axis is parsed from a ``time`` coordinate (configurable).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import geopandas as gpd
import pandas as pd
import shapely.geometry


if TYPE_CHECKING:
    import xarray as xr

    from geocatalog._src.duckdb_backend import DuckDBGeoCatalog
    from geocatalog._src.geoslice import GeoSlice


from geocatalog._src.io import _close_resolved_uri, _resolve_uri
from geocatalog._src.memory import InMemoryGeoCatalog


# Only `xarray` is genuinely optional — geopandas + shapely are base deps.
# Importing this module without the [xarray-raster] extra is allowed (it's
# how `geocatalog.__getattr__` raises the friendly ImportError); the
# `build_xarray_catalog` / `load_xarray` functions assert presence at call
# time.
try:
    import xarray as xr
except ImportError:  # pragma: no cover - exercised via the [xarray-raster] extra
    xr = None  # type: ignore[assignment]


def _xy_dims(ds: xr.Dataset) -> tuple[str, str]:
    """Resolve the (x, y) coordinate names for a dataset.

    Returns ``(x_name, y_name)`` after checking the common conventions:
    ``(x, y)``, ``(lon, lat)``, ``(longitude, latitude)``.
    """
    candidates = (("x", "y"), ("lon", "lat"), ("longitude", "latitude"))
    for x_name, y_name in candidates:
        if x_name in ds.coords and y_name in ds.coords:
            return x_name, y_name
    raise ValueError(
        "Could not resolve (x, y) coordinate names in dataset; "
        f"have coords {list(ds.coords)}."
    )


def _xarray_engine(filepath: str | Path) -> str | None:
    """Pick the xarray engine for ``filepath``.

    Directories and ``.zarr`` paths use the zarr engine; everything else
    falls through to xarray's default (netcdf4 / h5netcdf). Centralised
    so the build + load paths can't disagree.

    For ``str`` inputs, the scheme is inspected so we don't run ``is_dir()``
    against a remote URI. Empty schemes (and single-character schemes, which
    is how ``urlsplit`` parses Windows drive letters like ``C:/...``) are
    treated as local paths and routed through the ``Path`` branch — this
    keeps local Zarr directories without a ``.zarr`` suffix working when
    the caller passes them as strings.
    """
    if isinstance(filepath, Path):
        if filepath.suffix == ".zarr" or filepath.is_dir():
            return "zarr"
        return None
    scheme = urlsplit(filepath).scheme
    if not scheme or len(scheme) == 1:
        # Local path expressed as str (incl. Windows ``C:/...``).
        return _xarray_engine(Path(filepath))
    if Path(urlsplit(filepath).path).suffix == ".zarr":
        return "zarr"
    return None


def _xarray_row(
    filepath: str | Path,
    *,
    data_vars: Sequence[str] | None,
    time_var: str,
    target_crs: Any | None,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if xr is None:
        raise ImportError(
            "build_xarray_catalog requires xarray; install via "
            "`pip install 'geocatalog[xarray-raster]'`."
        )
    engine = _xarray_engine(filepath)
    resolved = _resolve_uri(filepath, storage_options=storage_options)
    try:
        with xr.open_dataset(resolved, engine=engine) as ds:
            x_name, y_name = _xy_dims(ds)
            xmin, xmax = float(ds[x_name].min()), float(ds[x_name].max())
            ymin, ymax = float(ds[y_name].min()), float(ds[y_name].max())
            polygon = shapely.geometry.box(xmin, ymin, xmax, ymax)

            if time_var in ds.coords:
                times = pd.to_datetime(ds[time_var].values)
                start = pd.Timestamp(times.min())
                end = pd.Timestamp(times.max())
                n_timesteps = int(ds[time_var].size)
            else:
                start = pd.Timestamp("1900-01-01")
                end = pd.Timestamp("2100-01-01")
                n_timesteps = 0

            # Resolve CRS: rio accessor if rioxarray is loaded, otherwise the
            # caller's target_crs as a fallback.
            crs_value = None
            try:
                crs_value = ds.rio.crs  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                crs_value = None
            if crs_value is None:
                crs_value = target_crs
    finally:
        _close_resolved_uri(resolved)

    return {
        "filepath": str(filepath),
        "geometry": polygon,
        "start_time": start,
        "end_time": end,
        "n_timesteps": n_timesteps,
        "time_var": time_var,
        "data_vars": list(data_vars) if data_vars is not None else None,
        "crs": str(crs_value) if crs_value is not None else None,
    }


def build_xarray_catalog(
    filepaths: Sequence[str | Path],
    *,
    target_crs: Any | None = None,
    data_vars: Sequence[str] | None = None,
    time_var: str = "time",
    backend: Literal["memory", "duckdb"] = "memory",
    out_path: str | Path | None = None,
    write_bbox: bool = True,
    sort_by: tuple[str, ...] | None = ("start_time", "geometry_hilbert"),
    partition_by: tuple[str, ...] | None = None,
    batch_size: int = 10_000,
    n_workers: int = 1,
    ordered: bool = False,
    storage_options: dict[str, Any] | None = None,
) -> InMemoryGeoCatalog | DuckDBGeoCatalog:
    """Build an xarray-shaped catalog — in-memory (default) or streamed.

    For each input file, opens it with ``xr.open_dataset``, derives the
    spatial footprint from the min/max of its (x, y) / (lon, lat) /
    (longitude, latitude) coordinates, and reads the time interval from
    the ``time_var`` coordinate. If `rioxarray` is loaded and the
    dataset carries a CRS through its ``rio`` accessor, that wins;
    otherwise the catalog falls back to ``target_crs``.

    Backends mirror `build_raster_catalog`. The streaming branch
    (`backend="duckdb"`) requires the ``[duckdb]`` extra and an
    *explicit* ``target_crs`` — unlike the raster builder, xarray
    coordinate bounds are *not* reprojected, so the CRS metadata must
    match the dataset's native coords. Passing ``target_crs=None`` in
    the duckdb branch raises `ValueError`.

    Args:
        filepaths: Files to index. ``.zarr`` paths (and directories)
            are opened with the zarr engine; everything else falls
            back to netcdf4 / h5netcdf.
        target_crs: CRS to tag the catalog with when files don't carry
            their own. Coordinate bounds are *not* reprojected — there's
            no `WarpedVRT` analogue for xarray — so this should match
            the files' native CRS.
        data_vars: Subset of data variables recorded per row. Loaders
            consult this to pick which arrays to materialise. ``None``
            leaves it open-ended; downstream `load_xarray` can still
            override.
        time_var: Coordinate name for the time axis. Default ``"time"``.
            Files where this coordinate is missing get the sentinel
            interval ``[1900-01-01, 2100-01-01]``.
        backend: ``"memory"`` for the existing in-RAM path,
            ``"duckdb"`` for the streamed GeoParquet path.
        out_path: Destination GeoParquet path. Required when
            ``backend="duckdb"``.
        write_bbox: Emit the GeoParquet 1.1 covering ``bbox`` struct.
            Only consulted when ``backend="duckdb"``.
        sort_by: Sort keys for the post-write DuckDB rewrite; literal
            ``"geometry_hilbert"`` expands to
            ``ST_Hilbert(ST_Centroid(geometry))``. ``None`` skips the
            rewrite. Only consulted when ``backend="duckdb"``.
        partition_by: Optional Hive partition columns for directory
            output when ``backend="duckdb"``. ``"year"``, ``"month"``,
            and ``"day"`` are derived from ``start_time``.
        batch_size: Rows per Arrow record batch. Default 10 000.
        n_workers: Process-pool size for per-file extraction.
        ordered: With ``backend="duckdb"`` and ``n_workers>1``, preserve
            input row order instead of completion order. Useful for
            reproducible artifacts when ``sort_by=None``. A slow input
            earlier in the queue stalls every subsequent yield and can
            temporarily reduce parallelism (workers may sit idle waiting
            on the next-in-line future). Prefer ``ordered=False`` for
            skewed workloads and sort post-hoc if you need a stable byte
            layout.

    Returns:
        `InMemoryGeoCatalog` for ``backend="memory"``, otherwise a
        `DuckDBGeoCatalog`.

    Raises:
        ImportError: If the ``[xarray-raster]`` extra is not installed.
        ValueError: If no files yielded a row or ``out_path`` missing
            in the duckdb branch.
    """
    if backend not in ("memory", "duckdb"):
        raise ValueError(
            f"build_xarray_catalog: backend must be 'memory' or 'duckdb'; "
            f"got {backend!r}"
        )
    if backend == "duckdb":
        if out_path is None:
            raise ValueError("build_xarray_catalog(backend='duckdb') requires out_path")
        return _build_xarray_catalog_duckdb(
            filepaths,
            target_crs=target_crs,
            data_vars=data_vars,
            time_var=time_var,
            out_path=out_path,
            write_bbox=write_bbox,
            sort_by=sort_by,
            partition_by=partition_by,
            batch_size=batch_size,
            n_workers=n_workers,
            ordered=ordered,
            storage_options=storage_options,
        )

    rows: list[dict[str, Any]] = [
        _xarray_row(
            fp,
            data_vars=data_vars,
            time_var=time_var,
            target_crs=target_crs,
            storage_options=storage_options,
        )
        for fp in filepaths
    ]
    if not rows:
        raise ValueError("build_xarray_catalog: no files yielded a row")
    crs_value = target_crs if target_crs is not None else rows[0]["crs"]
    if crs_value is None:
        raise ValueError(
            "build_xarray_catalog: cannot determine catalog CRS — pass "
            "`target_crs=...` explicitly, or load `rioxarray` so the "
            "dataset's `.rio.crs` accessor reports a CRS."
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs_value)
    return InMemoryGeoCatalog(gdf, backend="xarray")


def _build_xarray_catalog_duckdb(
    filepaths: Sequence[str | Path],
    *,
    target_crs: Any | None,
    data_vars: Sequence[str] | None,
    time_var: str,
    out_path: str | Path,
    write_bbox: bool,
    sort_by: tuple[str, ...] | None,
    partition_by: tuple[str, ...] | None,
    batch_size: int,
    n_workers: int,
    ordered: bool,
    storage_options: dict[str, Any] | None,
) -> DuckDBGeoCatalog:
    """Streaming-write branch for `build_xarray_catalog`.

    Unlike the raster branch (which has `WarpedVRT` and can canonicalise
    to EPSG:4326), xarray's `_xarray_row` reads raw coordinate min/max
    and does *not* reproject. Silently defaulting to EPSG:4326 here
    would mislabel artifacts whose source coords are in UTM/Web-Mercator/
    etc., so the duckdb branch instead requires an explicit
    ``target_crs`` matching the data's native CRS.
    """
    from geocatalog._src.streaming import stream_build_duckdb

    if target_crs is None:
        raise ValueError(
            "build_xarray_catalog(backend='duckdb') requires target_crs. "
            "Unlike the raster builder, the xarray branch does not "
            "reproject coordinate bounds, so the CRS metadata must match "
            "the dataset's native coordinate system. Pass target_crs "
            "explicitly (e.g. 'EPSG:4326' for lon/lat, or your data's "
            "actual projected CRS for UTM/Web-Mercator/etc.)."
        )
    extract_fn = functools.partial(
        _xarray_row,
        data_vars=tuple(data_vars) if data_vars is not None else None,
        time_var=time_var,
        target_crs=target_crs,
        storage_options=storage_options,
    )
    return stream_build_duckdb(
        filepaths,
        extract_fn,
        out_path=out_path,
        crs=target_crs,
        backend="xarray",
        write_bbox=write_bbox,
        sort_by=sort_by,
        partition_by=partition_by,
        batch_size=batch_size,
        n_workers=n_workers,
        ordered=ordered,
    )


def load_xarray(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice,
    *,
    data_vars: Sequence[str] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Load + concat the catalog rows matching ``slice_`` into an ``xr.Dataset``.

    For each matching file, opens it inside a context manager, clips to
    ``slice_.bounds`` along the spatial coords *and* to
    ``slice_.interval`` along the time coord, ``.load()``s the clipped
    piece so the data persists after the file handle closes, and
    concatenates the pieces along the time coordinate. Single-file
    results skip the concat.

    The time-clip is what makes this loader honest for files that span
    many years — without it, a query for one month against a multi-year
    NetCDF would return every timestep in the file.

    Args:
        catalog: An xarray-backend catalog.
        slice_: Window to read. Bounds may be in a different CRS than
            the catalog; the loader reprojects internally on the query
            but the *coordinate selection* still uses the catalog CRS,
            so cross-CRS slicing only works if the catalog and slice
            CRSs agree.
        data_vars: Subset of data variables to keep per file. ``None``
            preserves the dataset's full variable set.

    Returns:
        An ``xr.Dataset`` concatenated along the time coordinate, or a
        single-file Dataset if only one row matched.

    Raises:
        ImportError: If xarray is not installed (``[xarray-raster]``
            extra missing).
        TypeError: If the catalog's backend tag is not ``"xarray"``.
        ValueError: If no catalog rows match the slice.
    """
    if xr is None:
        raise ImportError(
            "load_xarray requires xarray; install via "
            "`pip install 'geocatalog[xarray-raster]'`."
        )
    if catalog.backend != "xarray":
        raise TypeError(
            f"load_xarray requires an xarray-backend catalog; got {catalog.backend!r}"
        )
    filtered = catalog.query(slice_)
    if len(filtered) == 0:
        raise ValueError("load_xarray: no catalog rows match the slice")

    xmin, ymin, xmax, ymax = slice_.bounds
    t_start, t_end = slice_.interval.left, slice_.interval.right
    pieces: list[xr.Dataset] = []
    for fp, row_time_var in zip(
        filtered.gdf["filepath"], filtered.gdf["time_var"], strict=False
    ):
        engine = _xarray_engine(fp)
        resolved = _resolve_uri(fp, storage_options=storage_options)
        try:
            with xr.open_dataset(resolved, engine=engine) as ds:
                x_name, y_name = _xy_dims(ds)
                # `.sel(slice)` requires monotonic coords; fall back to `.where`.
                try:
                    piece = ds.sel(
                        {x_name: slice(xmin, xmax), y_name: slice(ymin, ymax)}
                    )
                except KeyError:
                    mask = (
                        (ds[x_name] >= xmin)
                        & (ds[x_name] <= xmax)
                        & (ds[y_name] >= ymin)
                        & (ds[y_name] <= ymax)
                    )
                    piece = ds.where(mask, drop=True)
                # Time-clip to the slice interval so multi-year files don't
                # smuggle in out-of-range timesteps. Files without the time
                # coord (n_timesteps == 0) pass through.
                if row_time_var in piece.coords:
                    try:
                        piece = piece.sel({row_time_var: slice(t_start, t_end)})
                    except KeyError:
                        t_mask = (piece[row_time_var] >= t_start) & (
                            piece[row_time_var] <= t_end
                        )
                        piece = piece.where(t_mask, drop=True)
                if data_vars is not None:
                    piece = piece[list(data_vars)]
                # Materialise so the array survives the `with` block close.
                pieces.append(piece.load())
        finally:
            _close_resolved_uri(resolved)
    if len(pieces) == 1:
        return pieces[0]
    time_var = filtered.gdf["time_var"].iloc[0]
    if time_var in pieces[0].coords:
        return xr.concat(pieces, dim=time_var)
    return xr.merge(pieces)
