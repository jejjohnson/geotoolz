"""Carrier rewrap discipline shared by every operator family.

The Tier-A primitives take plain arrays: ``np.asarray(g)`` is a
`geoarray.GeoArray`'s data (a host copy for JAX data). The one
carrier-aware step left in each ``_apply`` is the *rewrap*: a GeoArray
input comes back as a GeoArray (georeferencing and metadata propagated
from the input), while a plain ndarray input comes back as a plain
ndarray. :func:`wrap_like` centralises that dispatch so operators support
both carriers with a single call, refuses a georeader ``GeoTensor``
(naming ``from_georeader``), and owns the metadata rules every output
follows:

* ``attrs`` is derived from the input's (immutable `FrozenAttrs`).
* Per-band metadata (:data:`~geoops._src.bands.PER_BAND_KEYS`) is
  dropped when the band axis changes size, unless new ``band_names`` are
  supplied, in which case they are written under
  :data:`~geoops._src.bands.CANONICAL_BAND_KEY`.
"""

from __future__ import annotations

import enum
import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from geoops._src.bands import (
    BAND_NAME_KEYS,
    CANONICAL_BAND_KEY,
    PER_BAND_KEYS,
    band_count,
)


from geoarray import FrozenAttrs, GeoArray

from geoops._src.engine import from_engine
from geoops._src.geo import reject_geotensor


if TYPE_CHECKING:
    from affine import Affine

__all__ = ["INHERIT", "Inherit", "adopt_attrs", "rewrap_attrs", "wrap_like"]


class Inherit(enum.Enum):
    """Sentinel type for "take this value from the reference carrier"."""

    INHERIT = "inherit"

    def __repr__(self) -> str:
        return "INHERIT"


#: Default for :func:`wrap_like`'s ``fill_value``: inherit
#: ``ref.fill_value``. Any other value -- including ``None``,
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

    Returns a new dict, which the caller freezes into the output's attrs.

    Args:
        ref_attrs: The input carrier's ``attrs`` (``None`` means empty).
        in_bands: Band count of the input carrier.
        out_bands: Band count of the output carrier.
        band_names: New band names. When given, they are written under
            :data:`~geoops._src.bands.CANONICAL_BAND_KEY` and the other
            band-name aliases (``descriptions``, ``bands``,
            ``band_descriptions``) are dropped so no lookup can find a
            disagreeing name list. Spectral per-band keys
            (``wavelengths`` / ``wavelengths_nm``) are kept only when the
            band count is unchanged.

    Returns:
        The new attrs. When ``band_names`` is ``None`` and the band count
        changed, every key in :data:`~geoops._src.bands.PER_BAND_KEYS`
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
    ref: GeoArray | np.ndarray,
    result: Any,
    *,
    attrs: Mapping[str, Any] | None = None,
    band_names: Sequence[str] | None = None,
) -> Any:
    """Give a raster on a new grid the attrs derived from ``ref``.

    For results that come back from a grid-changing step (a georeader
    engine call such as ``read_reproject``, a geoarray ``read_window`` /
    ``pad``, ...): keeps ``result``'s grid (``transform``, ``crs``,
    ``fill_value``) and pixel values (no copy) and derives its attrs from
    ``ref`` following the same rules as :func:`wrap_like`. A georeader
    ``GeoTensor`` result is converted with ``from_georeader`` first.

    Args:
        ref: The operator's input carrier.
        result: The new raster. Anything that is neither a `GeoArray` nor
            a ``GeoTensor`` (a plain array) is returned unchanged.
        attrs: Explicit attrs, as in :func:`wrap_like`. ``None`` derives
            them from ``ref.attrs``.
        band_names: New band names, as in :func:`wrap_like`.

    Returns:
        A GeoArray on ``result``'s grid with attrs derived from ``ref``, or
        ``result`` unchanged when it is not a raster carrier.
    """
    result = from_engine(result)
    if not isinstance(result, GeoArray):
        return result
    out_bands = band_count(tuple(result.shape))
    new_attrs = rewrap_attrs(
        getattr(ref, "attrs", None) if attrs is None else attrs,
        in_bands=band_count(tuple(np.shape(ref))) if attrs is None else out_bands,
        out_bands=out_bands,
        band_names=band_names,
    )
    return result.replace(attrs=FrozenAttrs(new_attrs))


def wrap_like(
    ref: GeoArray | np.ndarray,
    out: np.ndarray,
    *,
    fill_value: FillValue = INHERIT,
    attrs: Mapping[str, Any] | None = None,
    transform: Affine | Any | None = None,
    band_names: Sequence[str] | None = None,
) -> GeoArray | np.ndarray:
    """Rewrap a computed array to match the carrier of ``ref``.

    Args:
        ref: The operator's input carrier -- a `geoarray.GeoArray` or any
            plain array-like. A georeader ``GeoTensor`` raises.
        out: The result array produced by a Tier-A primitive. Unless
            ``transform`` is given, its trailing two (spatial) dims must
            match ``ref``'s when ``ref`` is a GeoArray.
        fill_value: Fill value for the returned GeoArray. The default,
            :data:`INHERIT`, propagates ``ref.fill_value``; any other value
            is used verbatim, so ``None`` means "no fill value" and
            ``NaN`` / ``False`` / ``0`` are explicit fills. Ignored for
            plain-array carriers.
        attrs: Attributes for the output. ``None`` (default) derives them
            from ``ref.attrs`` via :func:`rewrap_attrs`: per-band keys
            dropped when the band count changes. An explicit mapping is
            used as-is (the caller owns its band keys); ``band_names`` is
            still written into it.
        transform: Transform for the output (an ``Affine``, six numbers or
            a ``(6,)`` array). ``None`` (default) keeps ``ref.transform``
            and requires ``out`` to share ``ref``'s spatial shape; pass one
            for outputs on a new grid (crops, resampling) -- the
            spatial-shape check is then skipped.
        band_names: Names of the output bands, written under
            ``attrs["band_names"]``. Must have one entry per output band
            (``out.shape[-3]`` for 3-D / 4-D outputs, 1 for 2-D).

    Returns:
        A new GeoArray with ``ref``'s CRS when ``ref`` is a GeoArray,
        otherwise ``out`` as a plain ``np.ndarray``.

    Raises:
        TypeError: ``ref`` is a georeader ``GeoTensor``.
        ValueError: If ``transform`` is ``None`` and the spatial shape of
            ``out`` differs from ``ref``'s, or if ``band_names`` has the
            wrong length.
    """
    reject_geotensor(ref)
    if not isinstance(ref, GeoArray):
        return np.asarray(out)

    values = np.asarray(out)
    ref_shape = tuple(ref.shape)
    if transform is None:
        if values.shape[-2:] != ref_shape[-2:]:
            raise ValueError(
                "Operation altered spatial dimensions! "
                f"{ref_shape[-2:]} -> {values.shape[-2:]}; pass transform= "
                "for outputs on a new grid."
            )
        transform = np.asarray(ref.transform)
    out_bands = band_count(values.shape)
    if attrs is None:
        new_attrs = rewrap_attrs(
            ref.attrs,
            in_bands=band_count(ref_shape),
            out_bands=out_bands,
            band_names=band_names,
        )
    else:
        new_attrs = rewrap_attrs(
            attrs, in_bands=out_bands, out_bands=out_bands, band_names=band_names
        )
    if fill_value is INHERIT:
        fill_value = ref.fill_value
    return GeoArray(
        values,
        transform,
        ref.crs,
        fill_value=fill_value,
        attrs=FrozenAttrs(new_attrs),
    )
