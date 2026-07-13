"""`field_for()` — bridge a staged catalog to geopatcher `Field`s.

A staged catalog (the output of `stage()`) has its ``filepath`` /
``assets`` columns rewritten to local paths. The downstream pattern
in the design doc (§7) is to hand that catalog to ``geopatcher`` so
a `SpatialPatcher` can read patches:

    cat = stage(bundle.catalog)
    fields = field_for(cat, "red")          # one RasterField per row
    patcher.split(fields[0])

This helper saves the user from writing the per-row
``RasterioReader(...) → RasterField(...)`` shim by hand. It lives
under ``geocatalog._src.staging`` because the staged-asset column
rewrite is the precondition that makes path-based Field construction
meaningful.

`geopatcher` is a soft dependency — imports happen inside
`field_for` so a base `pip install geocatalog` is unaffected.
Users opt in via ``pip install 'geocatalog[patch]'`` (or by
installing geopatcher directly).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from urllib.parse import urlparse


if TYPE_CHECKING:
    from geopatcher import Field

    from geocatalog._src.base import GeoCatalog


_GEOPATCHER_HINT = (
    "field_for() requires geopatcher. Install with "
    "`pip install 'geocatalog[patch]'` or `pip install geopatcher`."
)


# Schemes that resolve to a local file we can hand to RasterioReader
# without a network round-trip. Anything else (https://, s3://, gs://, …)
# means staging didn't actually fetch the bytes — typically a
# `stage(on_error="skip")` row whose original URI was preserved.
_LOCAL_URI_SCHEMES = frozenset({"", "file"})


def _is_local_path(path: str) -> bool:
    """True if ``path`` is a local filesystem path, not a remote URI.

    `urlparse("C:/data/tile.tif").scheme` returns ``'c'`` on every
    platform, so a naive scheme-membership check would reject valid
    Windows local paths. Real URI schemes are always 2+ characters
    (``http``, ``s3``, ``gs``, …), so a single-character scheme is
    treated as a drive letter rather than a remote scheme.
    """
    scheme = urlparse(path).scheme
    if scheme in _LOCAL_URI_SCHEMES:
        return True
    return len(scheme) == 1 and scheme.isalpha()


def field_for(
    catalog: GeoCatalog,
    asset: str | None = None,
    *,
    mode: str = "raster",
) -> list[Field]:
    """Build one geopatcher `Field` per row of a staged catalog.

    Args:
        catalog: A catalog whose rows reference local files. Typically
            the output of `stage()` — its ``filepath`` column (and
            ``assets`` JSON map, when present) point at cached copies
            already on disk. The catalog's ``backend`` must be
            ``"raster"`` when ``mode="raster"`` (the only mode today);
            other backends raise.
        asset: Which asset key to read for each row. ``None`` falls
            back to the row's ``filepath`` column — the right default
            for catalogs built by `build_raster_catalog` (which don't
            carry an asset map). When a string is passed, the per-row
            ``assets`` JSON dict is consulted; rows that do not carry
            that key raise `KeyError`.
        mode: Field flavor. ``"raster"`` (the only value supported
            today) wraps each path in a `RasterioReader` and then a
            `RasterField`. Reserved for future expansion to vector /
            xarray fields.

    Returns:
        A list of geopatcher `Field` instances in catalog row order.
        Single-row catalogs return a single-element list; the caller
        unpacks. A future multi-raster `Field` constructor in
        geopatcher would let this return one composite Field; the
        list-of-Fields shape is the truthful bridge until then.

    Raises:
        ImportError: If geopatcher is not installed.
        ValueError: If ``mode`` is not a supported flavor, the catalog
            is empty, or the catalog's ``backend`` does not match
            ``mode`` (e.g. ``backend="vector"`` with ``mode="raster"``).
        KeyError: If ``asset`` is a string and any row's asset map
            does not contain that key, or if any resolved path is a
            non-local URI — which happens when ``stage(on_error="skip")``
            preserved an unstaged URI for a row whose fetch failed.
            The error message points at the offending row + URI so the
            caller can retry staging or drop the row.
    """
    try:
        from geopatcher import RasterField
        from georeader.rasterio_reader import RasterioReader
    except ImportError as exc:  # pragma: no cover - exercised via patched sys.modules
        raise ImportError(_GEOPATCHER_HINT) from exc

    if mode != "raster":
        raise ValueError(f"field_for(mode={mode!r}): only 'raster' is supported today.")
    # `backend` is set by every catalog constructor we ship; tolerate
    # third-party catalogs that omit it by skipping the check rather
    # than crashing with AttributeError.
    backend = getattr(catalog, "backend", None)
    if backend is not None and mode == "raster" and backend != "raster":
        raise ValueError(
            f"field_for(mode='raster') requires a raster-backed catalog; "
            f"got backend={backend!r}. Pass a catalog produced by "
            "`build_raster_catalog` / `stage()` over raster sources."
        )
    if len(catalog) == 0:
        raise ValueError("field_for: catalog is empty; nothing to wrap.")

    paths = _resolve_paths(catalog, asset=asset)
    _reject_unstaged_uris(paths, asset=asset)
    return [RasterField(RasterioReader(p)) for p in paths]


def _reject_unstaged_uris(paths: list[str], *, asset: str | None) -> None:
    """Surface unstaged remote URIs as a `KeyError` with row context.

    `stage(on_error="skip")` preserves the original URI on a row whose
    fetch failed, which silently turns into a `RasterField` pointing
    at a remote object if `field_for` doesn't catch it. We want a loud
    failure so the user can either retry staging or drop the row.
    """
    bad: list[tuple[int, str]] = []
    for row_idx, p in enumerate(paths):
        if not _is_local_path(p):
            bad.append((row_idx, p))
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


def _resolve_paths(catalog: GeoCatalog, *, asset: str | None) -> list[str]:
    """Pull one local path per row, either ``filepath`` or assets[asset].

    When ``asset is None`` the function returns the ``filepath`` column
    verbatim; when ``asset`` is a string it decodes each row's JSON
    asset map and pulls the matching key, raising `KeyError` on the
    first row that is missing it.
    """
    gdf = catalog.gdf
    if asset is None:
        if "filepath" not in gdf.columns:
            raise KeyError(
                "field_for(asset=None) needs a 'filepath' column; "
                f"catalog columns: {list(gdf.columns)}"
            )
        return [str(p) for p in gdf["filepath"].tolist()]

    if "assets" not in gdf.columns:
        raise KeyError(
            f"field_for(asset={asset!r}) needs an 'assets' column on "
            "the catalog; did you forget to stage() first, or pass "
            "`asset=None` to use `filepath`?"
        )

    out: list[str] = []
    for row_idx, blob in enumerate(gdf["assets"].tolist()):
        if not isinstance(blob, str) or not blob:
            raise KeyError(
                f"field_for: row {row_idx} has no asset map; "
                f"can't resolve asset {asset!r}."
            )
        try:
            decoded = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise KeyError(
                f"field_for: row {row_idx} asset map is not valid JSON "
                f"({exc}); can't resolve asset {asset!r}."
            ) from exc
        if not isinstance(decoded, dict) or asset not in decoded:
            available = sorted(decoded) if isinstance(decoded, dict) else []
            raise KeyError(
                f"field_for: row {row_idx} has no asset {asset!r}; "
                f"available: {available}"
            )
        out.append(str(decoded[asset]))
    return out


__all__ = ["field_for"]
