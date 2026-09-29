"""Carrier rewrap discipline shared by every operator family.

`georeader.GeoTensor` (>=2.0) is an ``np.ndarray`` subclass, so the
Tier-A primitives accept it transparently via ``np.asarray``. The one
carrier-aware step left in each ``_apply`` is the *rewrap*: a GeoTensor
input should come back as a GeoTensor (georeferencing and metadata
propagated from the input), while a plain ndarray input should come back
as a plain ndarray. :func:`wrap_like` centralises that duck-typed
dispatch so operators support both carriers with a single call, and owns
the metadata rules every output follows:

* ``attrs`` is always a fresh (shallow-copied) dict, never the input's.
* Per-band metadata (:data:`~geotoolz._src.bands.PER_BAND_KEYS`) is
  dropped when the band axis changes size, unless new ``band_names`` are
  supplied, in which case they are written under
  :data:`~geotoolz._src.bands.CANONICAL_BAND_KEY`.
"""

from __future__ import annotations

import enum
import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from geotoolz._src.bands import (
    BAND_NAME_KEYS,
    CANONICAL_BAND_KEY,
    PER_BAND_KEYS,
    band_count,
)


if TYPE_CHECKING:
    from affine import Affine
    from georeader.geotensor import GeoTensor

__all__ = ["INHERIT", "Inherit", "adopt_attrs", "rewrap_attrs", "wrap_like"]


class Inherit(enum.Enum):
    """Sentinel type for "take this value from the reference carrier"."""

    INHERIT = "inherit"

    def __repr__(self) -> str:
        return "INHERIT"


#: Default for :func:`wrap_like`'s ``fill_value_default``: inherit
#: ``ref.fill_value_default``. Any other value -- including ``None``,
#: ``NaN``, ``False`` and ``0`` -- is used verbatim.
INHERIT: Final = Inherit.INHERIT

FillValue = numbers.Number | None | Literal[Inherit.INHERIT]


def rewrap_attrs(
    ref_attrs: Mapping[str, Any] | None,
    *,
    in_bands: int,
    out_bands: int,
    band_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Derive an output ``attrs`` dict from an input's ``attrs``.

    Always returns a new dict (a shallow copy), so mutating the output's
    attrs never leaks back into the input.

    Args:
        ref_attrs: The input carrier's ``attrs`` (``None`` means empty).
        in_bands: Band count of the input carrier.
        out_bands: Band count of the output carrier.
        band_names: New band names. When given, they are written under
            :data:`~geotoolz._src.bands.CANONICAL_BAND_KEY` and the other
            band-name aliases (``descriptions``, ``bands``,
            ``band_descriptions``) are dropped so no lookup can find a
            disagreeing name list. Spectral per-band keys
            (``wavelengths`` / ``wavelengths_nm``) are kept only when the
            band count is unchanged.

    Returns:
        The new attrs. When ``band_names`` is ``None`` and the band count
        changed, every key in :data:`~geotoolz._src.bands.PER_BAND_KEYS`
        is dropped.

    Raises:
        ValueError: If ``len(band_names) != out_bands``.
    """
    attrs = dict(ref_attrs or {})
    changed = in_bands != out_bands
    if band_names is not None:
        names = [str(name) for name in band_names]
        if len(names) != out_bands:
            raise ValueError(
                f"band_names has {len(names)} entries but the output has "
                f"{out_bands} band(s)"
            )
        for key in PER_BAND_KEYS if changed else BAND_NAME_KEYS:
            attrs.pop(key, None)
        attrs[CANONICAL_BAND_KEY] = names
    elif changed:
        for key in PER_BAND_KEYS:
            attrs.pop(key, None)
    return attrs


def adopt_attrs(
    ref: GeoTensor | np.ndarray,
    result: Any,
    *,
    attrs: Mapping[str, Any] | None = None,
    band_names: Sequence[str] | None = None,
) -> Any:
    """Give a GeoTensor built by a georeader call fresh attrs derived from ``ref``.

    georeader methods that move a raster onto a new grid
    (``read_from_window``, ``pad``, ``isel``, ``resize``, ``read_reproject``,
    ...) either alias the input's ``attrs`` dict or drop it. This keeps
    ``result``'s grid (``transform``, ``crs``, ``fill_value_default``) and
    pixel values (a view, not a copy) and replaces its attrs following the
    same rules as :func:`wrap_like`. A new GeoTensor is always returned,
    so ``result`` -- which georeader sometimes returns as ``ref`` itself --
    is never mutated.

    Args:
        ref: The operator's input carrier.
        result: The georeader output. Anything without a ``transform``
            (a plain array) is returned unchanged.
        attrs: Explicit attrs, as in :func:`wrap_like`. ``None`` derives
            them from ``ref.attrs``.
        band_names: New band names, as in :func:`wrap_like`.

    Returns:
        A GeoTensor on ``result``'s grid with fresh attrs, or ``result``
        unchanged when it is not GeoTensor-like.
    """
    if getattr(result, "transform", None) is None:
        return result

    from georeader.geotensor import GeoTensor

    values = np.asarray(result)
    out_bands = band_count(values.shape)
    new_attrs = rewrap_attrs(
        getattr(ref, "attrs", None) if attrs is None else attrs,
        in_bands=band_count(tuple(np.shape(ref))) if attrs is None else out_bands,
        out_bands=out_bands,
        band_names=band_names,
    )
    return GeoTensor(
        values,
        transform=result.transform,
        crs=result.crs,
        fill_value_default=result.fill_value_default,
        attrs=new_attrs,
    )


def wrap_like(
    ref: GeoTensor | np.ndarray,
    out: np.ndarray,
    *,
    fill_value_default: FillValue = INHERIT,
    attrs: Mapping[str, Any] | None = None,
    transform: Affine | None = None,
    band_names: Sequence[str] | None = None,
) -> GeoTensor | np.ndarray:
    """Rewrap a computed array to match the carrier of ``ref``.

    Args:
        ref: The operator's input carrier -- a ``GeoTensor`` or any plain
            array-like. Dispatch is duck-typed on the presence of
            ``array_as_geotensor`` rather than an isinstance check, so
            GeoTensor-compatible carriers from other libraries also work.
        out: The result array produced by a Tier-A primitive. Unless
            ``transform`` is given, its trailing two (spatial) dims must
            match ``ref``'s when ``ref`` is a GeoTensor.
        fill_value_default: Fill value for the returned GeoTensor. The
            default, :data:`INHERIT`, propagates ``ref.fill_value_default``;
            any other value is used verbatim, so ``None`` means "no fill
            value" and ``NaN`` / ``False`` / ``0`` are explicit fills.
            Ignored for plain-array carriers.
        attrs: Attributes for the output. ``None`` (default) derives them
            from ``ref.attrs`` via :func:`rewrap_attrs`: a shallow copy with
            per-band keys dropped when the band count changes. An explicit
            mapping is shallow-copied and used as-is (the caller owns its
            band keys); ``band_names`` is still written into it.
        transform: Affine transform for the output. ``None`` (default)
            keeps ``ref.transform`` and requires ``out`` to share ``ref``'s
            spatial shape; pass one for outputs on a new grid (crops,
            resampling) -- the spatial-shape check is then skipped.
        band_names: Names of the output bands, written under
            ``attrs["band_names"]``. Must have one entry per output band
            (``out.shape[-3]`` for 3-D / 4-D outputs, 1 for 2-D).

    Returns:
        A new GeoTensor with ``ref``'s CRS when ``ref`` is GeoTensor-like,
        otherwise ``out`` as a plain ``np.ndarray``.

    Raises:
        ValueError: If ``transform`` is ``None`` and the spatial shape of
            ``out`` differs from ``ref``'s, or if ``band_names`` has the
            wrong length.
    """
    if getattr(ref, "array_as_geotensor", None) is None:
        return np.asarray(out)

    from georeader.geotensor import GeoTensor

    values = np.asarray(out)
    ref_shape = tuple(np.shape(ref))
    if transform is None:
        if values.shape[-2:] != ref_shape[-2:]:
            raise ValueError(
                "Operation altered spatial dimensions! "
                f"{ref_shape[-2:]} -> {values.shape[-2:]}; pass transform= "
                "for outputs on a new grid."
            )
        transform = ref.transform
    out_bands = band_count(values.shape)
    if attrs is None:
        new_attrs = rewrap_attrs(
            getattr(ref, "attrs", None),
            in_bands=band_count(ref_shape),
            out_bands=out_bands,
            band_names=band_names,
        )
    else:
        new_attrs = rewrap_attrs(
            attrs, in_bands=out_bands, out_bands=out_bands, band_names=band_names
        )
    if fill_value_default is INHERIT:
        fill_value_default = ref.fill_value_default
    return GeoTensor(
        values,
        transform=transform,
        crs=ref.crs,
        fill_value_default=fill_value_default,
        attrs=new_attrs,
    )
