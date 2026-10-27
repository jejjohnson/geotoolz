"""Fixed-size batches of patches as one `GeoArray` with a ``patch`` dim (ADR-013).

`batch_split` groups a patcher's patches into `PatchBatch` tuples: the
patches as one `GeoArray` (``(patch, *dims)`` data, ``(patch, 6)``
transforms) and a ``(patch,)`` validity mask for the padding of the last
batch. The batch is NumPy; the caller moves it to JAX once, with
``batch.patches.to_jax()``. `unbatch` turns a batch (or a model's output
for it) back into `Patch` objects that ``patcher.merge`` places, from the
transform rows.

`BatchedPatch` is the 0.12 batch (stacked payloads next to per-patch
lists). It stays one release, for patches that are not raster carriers,
through the deprecated `BatchedPatch.split`.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from geoarray import DIMS, GeoArray, GeoArrayError, isel

from geopatcher._src.carrier import raster_grid
from geopatcher._src.patch import Patch


__all__ = ["BatchedPatch", "PatchBatch", "batch_split", "unbatch"]

PATCH = "patch"
"""The batch dim of a `PatchBatch`: one place on the ground per item."""


class PatchBatch(NamedTuple):
    """A batch of patches: one carrier and a mask of its real entries.

    A `NamedTuple`, so JAX treats it as a pytree: jitting a step on a
    whole batch compiles once per patch shape, whatever the patches'
    positions or the mask.

    Attributes:
        patches: The patches as one `GeoArray` with dims
            ``("patch", *patch_dims)``, a ``(patch, 6)`` transform (one row
            per patch) and the field's CRS, fill, coordinates and attrs.
            NumPy until the caller calls ``patches.to_jax()``.
        valid: ``(patch,)`` bool, ``True`` for real patches. Padding rows
            of a final batch copy its last real patch (data and transform
            row), so they are finite and look like real data.

    Example:
        >>> for patches, valid in batch_split(patcher, field, batch_size=64):
        ...     patches = patches.to_jax()  # (64, 4, 256, 256), once
        ...     scores = geoarray.map_over(patches, model, "patch")
        ...     loss = masked_mean(scores.data, valid)
    """

    patches: GeoArray
    valid: np.ndarray


def batch_split(
    patcher: Any,
    field: Any,
    *,
    batch_size: int,
    pad_last: bool = True,
) -> Iterator[PatchBatch]:
    """Yield the patches of ``patcher.split(field)`` as fixed-size batches.

    Groups consecutive patches into batches of exactly ``batch_size``; only
    the final batch can be short, and ``pad_last`` decides what happens to
    it. Every patch must be a `GeoArray` (any raster field adapter) or a
    ``DataArray`` that reads as a grid (`GeoArray.from_xarray`), of one
    shape.

    Args:
        patcher: Any patcher exposing ``split(field) -> Iterator[Patch]``
            (e.g. `SpatialPatcher`).
        field: The field adapter to split; passed through to
            ``patcher.split``.
        batch_size: Number of patches per batch. Must be positive.
        pad_last: If ``True`` (default), a short final batch is padded to
            ``batch_size`` with copies of its last real patch and
            ``valid == False``, so every batch has one shape (no
            recompilation on the last batch). If ``False``, the final batch
            is yielded short, all valid.

    Yields:
        One `PatchBatch` per ``batch_size`` patches, plus a final short
        batch (padded or not, per ``pad_last``) when the patch count is not
        a multiple of ``batch_size``. The batch is NumPy whatever is
        installed.

    Raises:
        ValueError: ``batch_size`` is not positive; the patches of one
            batch differ in shape (``boundary="shrink"`` edge chips) or in
            CRS, fill, coordinates or attrs (a float field's ``None`` fill
            and the NaN its ``boundary="pad"`` edge chips declare count as
            one: the batch's fill is NaN).
        TypeError: A patch is not a raster carrier (a plain array, a
            ``DataArray`` with no grid): use the deprecated
            `BatchedPatch.split` for those, until geotoolz-patcher 0.14.

    Example:
        >>> batches = list(batch_split(patcher, field, batch_size=16))
        >>> batches[0].patches.dims
        ('patch', 'band', 'y', 'x')
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    group: list[Patch] = []
    for patch in patcher.split(field):
        group.append(patch)
        if len(group) == batch_size:
            yield _patch_batch(group, batch_size, pad=False)
            group = []
    if group:
        yield _patch_batch(group, batch_size, pad=pad_last)


def unbatch(
    batch: PatchBatch | BatchedPatch,
    data: Any | None = None,
    *,
    patcher: Any | None = None,
    domain: Any | None = None,
) -> list[Patch]:
    """Turn a batch, or a model's output for it, back into `Patch` objects.

    Padding entries (``valid == False``) are dropped, so a padded final
    batch gives fewer patches than its length. For a `PatchBatch`, each
    patch is placed from its transform row against ``domain``'s grid: its
    anchor and indices are what ``patcher.geometry`` gives that pixel
    offset, and its weights are ``patcher.window``'s, so
    ``patcher.merge(unbatch(...), domain)`` reassembles the field.

    Args:
        batch: A `PatchBatch` from `batch_split` (or, until 0.14, a
            deprecated `BatchedPatch`, which carries its own anchors).
        data: Optional replacement payload, typically the model output for
            the batch: a `GeoArray` with the ``patch`` dim (from
            `geoarray.map_over`) or an array whose rows line up with the
            batch. Padding rows are ignored.
        patcher: The patcher that split the field; needed for a
            `PatchBatch` (its geometry must be
            `spatial.geometry.Rectangular`).
        domain: The field's domain (``field.domain``, or its bare
            `geoarray.Grid`); needed for a `PatchBatch`.

    Returns:
        One `Patch` per valid entry, in batch order, with NumPy data. A row
        that keeps the patch's spatial shape comes back as a `GeoArray` on
        the patch's grid (a per-pixel output with a new band count drops
        the stale per-band attrs); any other row stays a bare array.

    Raises:
        TypeError: A `PatchBatch` without ``patcher`` and ``domain``.
        ValueError: The patcher's geometry is not
            `spatial.geometry.Rectangular`, the patches were clipped by
            ``boundary="shrink"``, ``domain`` is not a raster domain, or a
            transform row is not on the domain's pixel grid (another pixel
            size or rotation, naming the patch, or an origin between pixels).

    Example:
        >>> place = dict(patcher=patcher, domain=field.domain)
        >>> outs = [
        ...     p
        ...     for b in batch_split(patcher, field, batch_size=16)
        ...     for p in unbatch(b, model(b.patches), **place)
        ... ]
        >>> merged = patcher.merge(outs, field.domain)
    """
    if isinstance(batch, BatchedPatch):
        return _unbatch_legacy(batch, data)
    if patcher is None or domain is None:
        raise TypeError(
            "unbatch(PatchBatch) places patches from their transform rows: "
            "pass patcher= and domain= (the field's domain)"
        )
    return _unbatch(batch, data, patcher, domain)


def _patch_batch(patches: list[Patch], batch_size: int, *, pad: bool) -> PatchBatch:
    _check_shapes(patches)
    items = [_as_geoarray(p.data) for p in patches]
    first = items[0]
    fill = _batch_fill(items)
    for item in items[1:]:
        if not _same_static(item, first):
            raise ValueError(
                "batch_split needs patches with one CRS, dims, coordinates "
                f"and attrs; got {_describe(first)} and {_describe(item)}"
            )
    data = [np.asarray(item.data) for item in items]
    transforms = [np.asarray(item.transform, dtype=np.float64) for item in items]
    valid = [True] * len(items)
    if pad:
        # copies of the last real patch: finite, and in distribution (ADR-013)
        for _ in range(batch_size - len(items)):
            data.append(data[-1])
            transforms.append(transforms[-1])
            valid.append(False)
    stacked = GeoArray(
        np.stack(data),
        np.stack(transforms),
        first.crs,
        fill_value=fill,
        attrs=first.attrs,
        dims=(PATCH, *first.dims),
        coords=first.coords,
    )
    return PatchBatch(stacked, np.asarray(valid, dtype=bool))


def _check_shapes(patches: list[Patch]) -> None:
    shapes = [np.shape(p.data) for p in patches]
    if len(set(shapes)) > 1:
        detail = ", ".join(
            f"{p.anchor!r}: {shape}" for p, shape in zip(patches, shapes, strict=True)
        )
        raise ValueError(
            "batch_split needs equal-shaped patches to stack, got "
            f"{detail}. Edge chips shrink under boundary='shrink'; use "
            "boundary='pad', 'reflect' or 'drop' for fixed-size chips."
        )


def _as_geoarray(data: Any) -> GeoArray:
    """A patch as a raster carrier, or a `TypeError` naming the old path."""
    if isinstance(data, GeoArray) and not data.batch_shape:
        return data
    if _is_dataarray(data):
        try:
            carrier = GeoArray.from_xarray(data)
        except (ValueError, GeoArrayError) as exc:
            raise _not_a_raster(data) from exc
        if isinstance(carrier, GeoArray):
            return carrier
    raise _not_a_raster(data)


def _not_a_raster(data: Any) -> TypeError:
    return TypeError(
        "batch_split batches raster patches (a GeoArray, or a DataArray on an "
        f"affine grid), got {type(data).__name__}. For other patches use the "
        "deprecated geopatcher.run.BatchedPatch.split (removed in "
        "geotoolz-patcher 0.14)."
    )


def _batch_fill(items: list[GeoArray]) -> Any:
    """The fill every patch of a batch shares.

    ``boundary="pad"`` declares NaN on the edge chips of a float field with
    no fill, while interior chips keep ``None``: one field, so the batch
    takes NaN. Any other difference is a real mismatch.
    """
    fills = [item.fill_value for item in items]
    first = fills[0]
    if all(_same_fill(f, first) for f in fills):
        return first
    floating = np.issubdtype(items[0].dtype, np.floating)
    if floating and all(f is None or _is_nan(f) for f in fills):
        return np.nan
    raise ValueError(
        f"batch_split needs patches with one fill, got {sorted(set(map(repr, fills)))}"
    )


def _same_fill(a: Any, b: Any) -> bool:
    return (_is_nan(a) and _is_nan(b)) or (
        a is None if b is None else (a is not None and bool(a == b))
    )


def _same_static(a: GeoArray, b: GeoArray) -> bool:
    return (
        a.dims == b.dims
        and a.coords == b.coords
        and a.crs == b.crs
        and a.attrs == b.attrs
    )


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def _describe(g: GeoArray) -> str:
    return f"(crs={g.crs.to_string()}, fill={g.fill_value!r}, dims={g.dims})"


def _unbatch(
    batch: PatchBatch, data: Any | None, patcher: Any, domain: Any
) -> list[Patch]:
    from geopatcher._src.spatial.geometry import Rectangular
    from geopatcher._src.spatial.patcher import _safe_base_weights

    geometry = patcher.geometry
    if not isinstance(geometry, Rectangular):
        raise ValueError(
            "unbatch places a PatchBatch with a Rectangular geometry, got "
            f"{type(geometry).__name__}"
        )
    size = tuple(int(n) for n in geometry.size[-2:])
    if tuple(batch.patches.shape[-2:]) != size:
        # a chip clipped by boundary="shrink": its anchor and window weights
        # cannot be recovered from the transform row alone
        raise ValueError(
            f"unbatch places full {size} patches, got {batch.patches.shape[-2:]} "
            "ones: patches clipped by boundary='shrink' cannot be placed from "
            "their transforms; use boundary='pad', 'reflect' or 'drop'"
        )
    grid = raster_grid(domain)
    if grid is None:
        raise ValueError(
            "unbatch places a PatchBatch on a raster domain (a geoarray.Grid or an "
            f"object with a .grid), got {type(domain).__name__}"
        )
    _check_resolution(batch.patches, grid)
    to_pixel = ~grid.affine
    weights = _safe_base_weights(patcher.window, geometry)
    patches = []
    for i in np.flatnonzero(np.asarray(batch.valid, dtype=bool)):
        item = isel(batch.patches, **{PATCH: int(i)}).to_numpy()
        anchor = _pixel_offset(to_pixel, item)
        if data is None:
            out: Any = item
        elif isinstance(data, GeoArray) and PATCH in data.dims:
            out = isel(data, **{PATCH: int(i)}).to_numpy()
        else:
            out = _rebuild(item, data[i])
        patches.append(
            Patch(
                data=out,
                anchor=anchor,
                indices=geometry.neighborhood(domain, anchor),
                weights=weights,
            )
        )
    return patches


def _check_resolution(patches: GeoArray, grid: Any) -> None:
    """Every transform row has the target grid's pixel size and rotation."""
    rows = np.asarray(patches.transform, dtype=np.float64)[:, [0, 1, 3, 4]]
    target = np.asarray(grid.transform, dtype=np.float64)[[0, 1, 3, 4]]
    scale = float(np.abs(target).max())
    bad = ~np.isclose(rows, target, rtol=1e-9, atol=1e-12 * scale).all(axis=1)
    if bad.any():
        p = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"patch {p} of the batch has pixel size / rotation (a, b, d, e) = "
            f"{tuple(rows[p].tolist())}, but the target grid has "
            f"{tuple(target.tolist())}: patches are placed on the grid they "
            "were split from, never resampled"
        )


def expand_batches(patches: Any, patcher: Any, domain: Any) -> Iterable[Any]:
    """``patches`` with every batch unpacked into `Patch` objects, lazily.

    A batch is a `PatchBatch` or a `GeoArray` whose batch dim is ``patch``
    (every row real); `SpatialPatcher.merge` and its variants accept either
    in place of patches, one or a stream of them, mixed with `Patch`
    objects. A sequence with no batch in it is returned as it is, so the
    merge hooks still see its length.
    """
    if _is_batch(patches):
        patches = [patches]
    elif isinstance(patches, Sequence) and not any(_is_batch(p) for p in patches):
        return patches
    return _expanded(patches, patcher, domain)


def aexpand_batches(
    patches: AsyncIterable[Any] | Iterable[Any], patcher: Any, domain: Any
) -> AsyncIterable[Any] | Iterable[Any]:
    """`expand_batches` for a sync or an async stream (`SpatialPatcher.amerge`)."""
    if not isinstance(patches, AsyncIterable):
        return expand_batches(patches, patcher, domain)
    return _aexpanded(patches, patcher, domain)


def _expanded(patches: Iterable[Any], patcher: Any, domain: Any) -> Iterator[Any]:
    # closing this generator (a merge that stops early) closes the producer
    iterator = iter(patches)
    try:
        for item in iterator:
            if _is_batch(item):
                batch = _as_patch_batch(item)
                yield from unbatch(batch, patcher=patcher, domain=domain)
            else:
                yield item
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


async def _aexpanded(
    patches: AsyncIterable[Any], patcher: Any, domain: Any
) -> AsyncIterator[Any]:
    # amerge closes its input when the aggregation fails: forward that to
    # the producer, so its cleanup runs before amerge returns
    iterator = aiter(patches)
    try:
        async for item in iterator:
            if _is_batch(item):
                batch = _as_patch_batch(item)
                for patch in unbatch(batch, patcher=patcher, domain=domain):
                    yield patch
            else:
                yield item
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


def _is_batch(item: Any) -> bool:
    if isinstance(item, PatchBatch):
        return True
    return isinstance(item, GeoArray) and item.batch_dims == (PATCH,)


def _as_patch_batch(item: Any) -> PatchBatch:
    if isinstance(item, PatchBatch):
        return item
    return PatchBatch(item, np.ones(item.batch_shape[0], dtype=bool))


def _pixel_offset(to_pixel: Any, item: GeoArray) -> tuple[int, int]:
    """``item``'s top-left pixel on the domain grid, as ``(row, col)``."""
    t = item.transform
    col, row = to_pixel * (float(t[2]), float(t[5]))
    r, c = round(row), round(col)
    if abs(row - r) > 1e-6 or abs(col - c) > 1e-6:
        raise ValueError(
            f"a patch's origin falls at pixel ({row}, {col}) of the domain, "
            "not on its grid: unbatch needs the domain of the field that was split"
        )
    return int(r), int(c)


def _rebuild(template: Any, row: Any) -> Any:
    """Wrap ``row`` in ``template``'s carrier when the spatial shape matches."""
    if template is None:
        return row
    values = np.asarray(row)
    if _is_dataarray(template):
        if values.shape != template.shape:
            return row
        return template.copy(deep=False, data=values)
    if values.shape[-2:] != template.shape[-2:] or not 2 <= values.ndim <= 4:
        return row
    dims = template.dims if values.ndim == template.ndim else DIMS[-values.ndim :]
    return template.with_data(values, dims=dims)


def _is_dataarray(data: Any) -> bool:
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - xarray is an optional extra
        return False
    return isinstance(data, xr.DataArray)


# --- the 0.12 batch, deprecated (ADR-013) ---------------------------------


@dataclass(eq=False)
class BatchedPatch:
    """Deprecated: a leading-axis batch of patches with per-patch lists.

    Replaced by `PatchBatch` (ADR-013) and removed in geotoolz-patcher
    0.14. `BatchedPatch.split` still batches patches that are not raster
    carriers (plain arrays, ``DataArray`` with no grid); `batch_split`
    raises for those.

    Attributes:
        data: ``(batch, *patch_shape)`` NumPy array of the stacked payloads.
        anchors: Per-patch anchors; ``None`` for padding entries.
        valid: ``(batch,)`` bool; ``False`` for zero-padding.
        indices: Per-patch indices (as on `Patch.indices`).
        weights: Per-patch window weights (as on `Patch.weights`).
        carriers: Per-patch pixel-free carrier templates, used by `unbatch`
            to rebuild ``DataArray`` / `GeoArray` chips.

    It is a JAX pytree once JAX is imported (``data`` and ``valid`` are the
    leaves; the lists are static and compare by identity).

    Example:
        >>> batches = list(BatchedPatch.split(patcher, field, batch_size=8))
    """

    data: Any
    anchors: list[Any]
    valid: Any
    indices: list[Any]
    weights: list[Any]
    carriers: list[Any] = field(default_factory=list)

    @classmethod
    def split(
        cls,
        patcher: Any,
        field: Any,
        *,
        batch_size: int,
        pad_last: bool = True,
    ) -> Iterator[BatchedPatch]:
        """Deprecated: the 0.12 `batch_split`, NumPy only.

        Yields `BatchedPatch` objects whose short final batch is
        zero-padded (``pad_last``). Removed in geotoolz-patcher 0.14.

        Raises:
            ValueError: ``batch_size`` is not positive, or the patches of
                one batch differ in shape.

        Example:
            >>> next(BatchedPatch.split(patcher, field, batch_size=4)).data.shape
            (4, 8, 8)
        """
        warnings.warn(
            "BatchedPatch.split is deprecated and is removed in geotoolz-patcher "
            "0.14; batch_split yields a PatchBatch (one GeoArray with a 'patch' "
            "dim) for raster patches",
            DeprecationWarning,
            stacklevel=2,
        )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        _register_when_jax_loads()
        return _legacy_split(patcher, field, batch_size, pad_last)


def _legacy_split(
    patcher: Any, field: Any, batch_size: int, pad_last: bool
) -> Iterator[BatchedPatch]:
    group: list[Patch] = []
    for patch in patcher.split(field):
        group.append(patch)
        if len(group) == batch_size:
            yield _legacy_batch(group, batch_size, pad=False)
            group = []
    if group:
        yield _legacy_batch(group, batch_size, pad=pad_last)


def _legacy_batch(patches: list[Patch], batch_size: int, *, pad: bool) -> BatchedPatch:
    _check_shapes(patches)
    arrays = [np.asarray(p.data) for p in patches]
    n_pad = batch_size - len(patches) if pad else 0
    arrays += [np.zeros_like(arrays[0])] * n_pad
    return BatchedPatch(
        data=np.stack(arrays, axis=0),
        anchors=[p.anchor for p in patches] + [None] * n_pad,
        valid=np.asarray([True] * len(patches) + [False] * n_pad, dtype=bool),
        indices=[p.indices for p in patches] + [None] * n_pad,
        weights=[p.weights for p in patches] + [None] * n_pad,
        carriers=[_template(p.data) for p in patches] + [None] * n_pad,
    )


def _unbatch_legacy(batch: BatchedPatch, data: Any | None) -> list[Patch]:
    arrays = batch.data if data is None else data
    valid = np.asarray(batch.valid, dtype=bool)
    carriers = batch.carriers or [None] * len(valid)
    return [
        Patch(
            data=_rebuild(carriers[i], arrays[i]),
            anchor=batch.anchors[i],
            indices=batch.indices[i],
            weights=batch.weights[i],
        )
        for i in np.flatnonzero(valid)
    ]


def _template(data: Any) -> Any:
    """A pixel-free copy of ``data``'s carrier metadata (``None`` for arrays)."""
    if isinstance(data, GeoArray):
        return GeoArray(
            np.broadcast_to(np.zeros((), dtype=data.dtype), data.shape),
            np.asarray(data.transform, dtype=np.float64),
            data.crs,
            fill_value=data.fill_value,
            attrs=data.attrs,
            dims=data.dims,
            coords=data.coords,
        )
    if _is_dataarray(data):
        placeholder = np.broadcast_to(np.zeros((), dtype=data.dtype), data.shape)
        return data.copy(deep=False, data=placeholder)
    return None


def _flatten(batch: BatchedPatch) -> tuple[tuple[Any, Any], _StaticMeta]:
    meta = _StaticMeta((batch.anchors, batch.indices, batch.weights, batch.carriers))
    return (batch.data, batch.valid), meta


def _unflatten(meta: _StaticMeta, children: tuple[Any, Any]) -> BatchedPatch:
    anchors, indices, weights, carriers = meta.value
    data, valid = children
    return BatchedPatch(data, anchors, valid, indices, weights, carriers)


class _StaticMeta:
    """Pytree aux data compared by identity.

    The per-patch metadata holds numpy weight arrays, whose ``==`` is
    elementwise — unusable for JAX's treedef comparison. Identity is the
    only safe equality: a structural "always equal" would let a cached jit
    trace hand back a stale batch's anchors.
    """

    __slots__ = ("value",)

    def __init__(self, value: tuple[Any, ...]) -> None:
        self.value = value


_PYTREE_HOOKED = False


def _register_when_jax_loads() -> None:
    """Register `BatchedPatch` with JAX now, or when JAX is imported."""
    global _PYTREE_HOOKED
    if _PYTREE_HOOKED:
        return
    _PYTREE_HOOKED = True
    # pipekit core comes with geoarray; importing geopatcher never imports JAX
    from pipekit import on_import

    on_import("jax", _register_pytree)


def _register_pytree(jax: Any) -> None:
    jax.tree_util.register_pytree_node(BatchedPatch, _flatten, _unflatten)
