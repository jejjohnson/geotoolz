"""`field_for()` — bridge a staged catalog to a geopatcher `Field`.

A staged catalog (the output of `stage()`) has its ``filepath`` /
``assets`` columns rewritten to local paths. ``field_for`` mosaics the
rows a `GeoSlice` selects onto the slice grid with `load_raster` and
wraps the resulting `GeoTensor` in a `geopatcher.RasterField`, so a
`SpatialPatcher` can chip it and reassemble its output:

    staged = stage(bundle.catalog)
    field = field_for(staged, slice_, asset="red")
    patches = list(patcher.split(field))

The field is in the slice CRS (the catalog CRS when no slice is given),
whatever CRS the individual files are in. ``materialize=False`` skips
the mosaic and returns one lazy field per row, in each file's own CRS.

`geopatcher` is a soft dependency — imports happen inside
`field_for` so a base install is unaffected. Opt in with
``pip install 'geotoolz-catalog[patch]'``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

from geocatalog._src._extras import missing_extra
from geocatalog._src.uri import parse_uri


if TYPE_CHECKING:
    from geopatcher import RasterField

    from geocatalog._src.geoslice import GeoSlice
    from geocatalog._src.memory import InMemoryGeoCatalog


_GEOREADER_HINT = (
    "field_for() requires georeader, a base dependency of geotoolz-catalog. "
    "Reinstall it with `pip install georeader-spaceml`."
)


def _is_local_path(path: str) -> bool:
    """True if ``path`` is a local filesystem path, not a remote URI.

    A remote URI here means staging didn't actually fetch the bytes —
    typically a `stage(on_error="skip")` row whose original URI was
    preserved. Windows drive paths and ``file://`` URIs are local.
    """
    return parse_uri(path).is_local


@overload
def field_for(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice | None = ...,
    *,
    asset: str | None = ...,
    band_indexes: Sequence[int] | None = ...,
    resampling: Any | None = ...,
    merge_method: str = ...,
    nodata: float | None = ...,
    materialize: Literal[True] = ...,
    mode: str = ...,
) -> RasterField: ...


@overload
def field_for(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice | None = ...,
    *,
    asset: str | None = ...,
    band_indexes: Sequence[int] | None = ...,
    resampling: Any | None = ...,
    merge_method: str = ...,
    nodata: float | None = ...,
    materialize: Literal[False],
    mode: str = ...,
) -> list[RasterField]: ...


def field_for(
    catalog: InMemoryGeoCatalog,
    slice_: GeoSlice | None = None,
    *,
    asset: str | None = None,
    band_indexes: Sequence[int] | None = None,
    resampling: Any | None = None,
    merge_method: str = "last",
    nodata: float | None = None,
    materialize: bool = True,
    mode: str = "raster",
) -> RasterField | list[RasterField]:
    """Build a geopatcher `RasterField` over a staged catalog.

    Args:
        catalog: A raster catalog whose rows reference local files —
            typically the output of `stage()`.
        slice_: The grid to build the field on: its bounds, CRS and
            resolution. ``None`` covers the whole catalog in the
            catalog CRS, on the first row's file grid (resolution and
            pixel alignment; that file must then be in the catalog CRS),
            with the bounds grown outward to whole pixels. With
            ``materialize=False`` it selects every row and needs no grid.
        asset: Asset key to read from each row's ``assets`` map.
            ``None`` reads the row's ``filepath`` — the right default
            for `build_raster_catalog` catalogs, which carry no map.
        band_indexes: 1-indexed bands to keep (see `load_raster`).
        resampling: A ``rasterio.enums.Resampling`` for files whose
            grid differs from the slice (default bilinear; use
            ``Resampling.nearest`` for categorical rasters).
        merge_method: How overlapping rows combine (see `load_raster`).
        nodata: Nodata override for the mosaic (see `load_raster`).
        materialize: ``True`` (default) reads and mosaics the selected
            rows into one `GeoTensor` on the slice grid and wraps it —
            a field `SpatialPatcher.split` / ``merge`` can round-trip.
            ``False`` reads nothing and returns one lazy field per
            selected row (catalog order), each over a
            `georeader.RasterioReader` in its file's own CRS and grid;
            each ``select`` reads its window on demand, restricted to
            ``band_indexes``. There is no mosaic and no reprojection on
            this path (``resampling``, ``merge_method`` and ``nodata``
            do not apply).
        mode: Field flavour; only ``"raster"`` is supported.

    Returns:
        One `RasterField` (``materialize=True``) or a list of lazy ones.

    Raises:
        ImportError: If geopatcher (``[patch]`` extra) or georeader is
            missing — each with its own install hint.
        TypeError: If ``slice_`` is a string (``asset`` is keyword-only).
        ValueError: If ``mode`` is unsupported, the backend is not
            raster, the catalog (or the slice's selection) is empty, or
            ``slice_`` is ``None`` and the grid cannot be inferred.
        KeyError: If ``asset`` is missing from a row's asset map, or a
            resolved path is a non-local URI — which happens when
            ``stage(on_error="skip")`` kept an unstaged URI. The message
            names the offending rows so the caller can re-stage or drop
            them.
    """
    try:
        from geopatcher import RasterField
    except ImportError as exc:
        raise missing_extra(
            "`field_for()`", "patch", packages="geotoolz-patcher"
        ) from exc
    try:
        from georeader.rasterio_reader import RasterioReader
    except ImportError as exc:
        raise ImportError(_GEOREADER_HINT) from exc

    if isinstance(slice_, str):
        raise TypeError(
            f"field_for(catalog, {slice_!r}): the second argument is a GeoSlice; "
            f"pass the asset by keyword — field_for(catalog, asset={slice_!r})."
        )
    if mode != "raster":
        raise ValueError(f"field_for(mode={mode!r}): only 'raster' is supported today.")
    # `backend` is set by every catalog constructor we ship; tolerate
    # third-party catalogs that omit it by skipping the check rather
    # than crashing with AttributeError.
    backend = getattr(catalog, "backend", None)
    if backend is not None and backend != "raster":
        raise ValueError(
            f"field_for(mode='raster') requires a raster-backed catalog; "
            f"got backend={backend!r}. Pass a catalog produced by "
            "`build_raster_catalog` / `stage()` over raster sources."
        )
    if len(catalog) == 0:
        raise ValueError("field_for: catalog is empty; nothing to wrap.")

    if slice_ is None:
        selected = _with_asset_paths(catalog, asset=asset)
        if materialize:  # the lazy path needs no grid
            slice_ = _whole_catalog_slice(selected)
    else:
        # Only the rows the slice selects need to carry `asset`.
        selected = catalog.query(slice_)
        if len(selected) == 0:
            raise ValueError(f"field_for: no catalog row intersects {slice_!r}.")
        selected = _with_asset_paths(selected, asset=asset)
    paths = [str(p) for p in selected.gdf["filepath"]]
    _reject_unstaged_uris(paths, asset=asset)

    if not materialize:
        indexes = list(band_indexes) if band_indexes is not None else None
        return [RasterField(RasterioReader(p, indexes=indexes)) for p in paths]
    assert slice_ is not None

    from geocatalog._src.raster import load_raster

    tensor = load_raster(
        selected,
        slice_,
        band_indexes=band_indexes,
        resampling=resampling,
        merge_method=merge_method,  # type: ignore[arg-type]
        nodata=nodata,
    )
    return RasterField(tensor)


def _whole_catalog_slice(catalog: InMemoryGeoCatalog) -> GeoSlice:
    """A `GeoSlice` over every row, at the first file's native resolution."""
    import pyproj
    import rasterio

    from geocatalog._src.geoslice import GeoSlice

    first = str(catalog.gdf["filepath"].iloc[0])
    _reject_unstaged_uris([first], asset=None)
    with rasterio.open(first) as src:
        file_crs = pyproj.CRS.from_user_input(src.crs) if src.crs else None
        res = (abs(src.res[0]), abs(src.res[1]))
        origin = (src.transform.c, src.transform.f)
    if file_crs is None or not file_crs.equals(catalog.crs):
        raise ValueError(
            f"field_for: cannot infer a grid — {first!r} is in "
            f"{file_crs.to_string() if file_crs else 'no CRS'}, not the catalog "
            f"CRS {catalog.crs.to_string()}. Pass slice_= with the resolution "
            "to build the field on."
        )
    interval = catalog.temporal_extent
    assert interval is not None  # the catalog is not empty
    return GeoSlice(
        bounds=_snap_outward(catalog.total_bounds, res, origin),
        interval=interval,
        resolution=res,
        crs=catalog.crs,
    )


def _snap_outward(
    bounds: Any, res: tuple[float, float], origin: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Grow ``bounds`` to whole pixels of the grid through ``origin``.

    `rasterio.merge` rounds the output size, so bounds that are not a
    whole number of pixels would lose the last partial pixel; snapping
    outward to the first file's grid covers every row and keeps that
    file's pixels aligned.
    """
    xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    rx, ry = res
    x0, y0 = origin
    eps = 1e-9  # absorb float noise in bounds that already sit on the grid
    return (
        x0 + math.floor((xmin - x0) / rx + eps) * rx,
        y0 - math.ceil((y0 - ymin) / ry - eps) * ry,
        x0 + math.ceil((xmax - x0) / rx - eps) * rx,
        y0 - math.floor((y0 - ymax) / ry + eps) * ry,
    )


def _with_asset_paths(
    catalog: InMemoryGeoCatalog, *, asset: str | None
) -> InMemoryGeoCatalog:
    """``catalog`` with ``filepath`` pointing at ``asset`` on every row."""
    from geocatalog._src.memory import InMemoryGeoCatalog

    if asset is None:
        if "filepath" not in catalog.gdf.columns:
            raise KeyError(
                "field_for(asset=None) needs a 'filepath' column; "
                f"catalog columns: {list(catalog.gdf.columns)}"
            )
        return catalog
    gdf = catalog.gdf.copy()
    gdf["filepath"] = _asset_paths(catalog, asset=asset)
    return InMemoryGeoCatalog(gdf, backend=catalog.backend)


def _reject_unstaged_uris(paths: list[str], *, asset: str | None) -> None:
    """Surface unstaged remote URIs as a `KeyError` with row context.

    `stage(on_error="skip")` preserves the original URI on a row whose
    fetch failed; reading it would silently go to the network (or
    fail deep inside rasterio). Fail loudly so the user can either
    retry staging or drop the row.
    """
    bad = [(i, p) for i, p in enumerate(paths) if not _is_local_path(p)]
    if not bad:
        return
    asset_clause = f"asset {asset!r}" if asset is not None else "filepath"
    sample = ", ".join(f"row {i}: {u!r}" for i, u in bad[:3])
    suffix = f" (and {len(bad) - 3} more)" if len(bad) > 3 else ""
    raise KeyError(
        f"field_for: {asset_clause} resolved to non-local URIs on "
        f"{len(bad)} row(s); re-run stage() (or drop the rows). "
        f"Examples: {sample}{suffix}."
    )


def _asset_paths(catalog: InMemoryGeoCatalog, *, asset: str) -> list[str]:
    """``assets[asset]`` for every row, raising `KeyError` on the first gap."""
    gdf = catalog.gdf
    if "assets" not in gdf.columns:
        raise KeyError(
            f"field_for(asset={asset!r}) needs an 'assets' column on "
            "the catalog; did you forget to stage() first, or pass "
            "`asset=None` to use `filepath`?"
        )

    out: list[str] = []
    for row_idx, blob in enumerate(gdf["assets"].tolist()):
        if isinstance(blob, dict):
            decoded: Any = blob
        elif isinstance(blob, str) and blob:
            try:
                decoded = json.loads(blob)
            except json.JSONDecodeError as exc:
                raise KeyError(
                    f"field_for: row {row_idx} asset map is not valid JSON "
                    f"({exc}); can't resolve asset {asset!r}."
                ) from exc
        else:
            raise KeyError(
                f"field_for: row {row_idx} has no asset map; "
                f"can't resolve asset {asset!r}."
            )
        if not isinstance(decoded, dict) or asset not in decoded:
            available = sorted(decoded) if isinstance(decoded, dict) else []
            raise KeyError(
                f"field_for: row {row_idx} has no asset {asset!r}; "
                f"available: {available}"
            )
        out.append(str(decoded[asset]))
    return out


__all__ = ["field_for"]
