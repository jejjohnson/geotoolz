"""Tier-B Operators — band-space spectral operations.

Each Operator wraps a Tier-A primitive in `array.py` and re-attaches
geospatial metadata via :func:`geotoolz._src.wrap.wrap_like`: a
GeoTensor input comes back as a GeoTensor (``transform``, ``crs``,
``fill_value_default``, and a copy of ``attrs`` propagated) while a
plain ``np.ndarray`` input comes back as a plain ndarray. When the band
axis is *altered* (selected, reordered, binned, stacked, etc.) the
per-band ``attrs`` entries (``band_names``, ``descriptions``,
``wavelengths``, ...) are subset, concatenated, rewritten or dropped in
lockstep (see :mod:`geotoolz._src.bands`) so downstream operators see
the correct labels (plain-array carriers have no attrs, so this
bookkeeping is skipped for them).

Nodata: pixels that are non-finite or equal the input's
``fill_value_default`` in any band (see :mod:`geotoolz._src.valid`) hold
the output's fill value. Band subsets and stacks keep the input's fill;
value-preserving transforms (SRF convolution, binning, smoothing) keep
it too, switching to ``NaN`` when integer input is promoted to float
(:func:`geotoolz._src.valid.carried_fill`); derived products
(:class:`BandMath`, :class:`NormalizedDifference`, :class:`BandRatio`,
:class:`ContinuumRemoval`) hold ``NaN`` and declare
``fill_value_default=NaN``.

Band names are resolved from an explicit ``band_names=`` constructor
argument when present; otherwise operators look for
``gt.attrs["band_names"]``. Wavelength-dependent operators follow the
same convention with explicit ``source_wavelengths=`` / ``wavelengths=``
arguments first, then ``gt.attrs["wavelengths"]``. On plain arrays the
attrs fallbacks are unavailable, so name/wavelength resolution raises a
clear ``ValueError`` unless the values are given explicitly.

The band axis is ``-3`` by default (georeader's ``(C, H, W)`` /
``(T, C, H, W)`` layout). On a ``(T, C, H, W)`` time stack every operator
works per frame: band-collapsing products (:class:`BandMath`,
:class:`NormalizedDifference`, :class:`BandRatio`) come back as
``(T, 1, H, W)`` and SRF convolutions convolve each frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from georeader.reflectance import srf, transform_to_srf
from pipekit import Operator

from geotoolz._src.bands import concat_band_attrs, strip_band_attrs, take_band_attrs
from geotoolz._src.config import jsonable
from geotoolz._src.shape import keep_band_axis, over_frames
from geotoolz._src.valid import carried_fill, wrap_filled
from geotoolz._src.wrap import wrap_like
from geotoolz.spectral._src.array import (
    band_ratio,
    continuum_removal,
    evaluate_band_math,
    normalized_difference,
    select_bands,
    spectral_binning,
    spectral_smoothing,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


BandKey = int | str


def _attrs(gt: GeoTensor | np.ndarray) -> dict[str, Any]:
    """Return a shallow copy of ``gt.attrs`` (or ``{}`` if missing)."""
    attrs = getattr(gt, "attrs", None)
    return {} if attrs is None else dict(attrs)


def _band_names(
    gt: GeoTensor | np.ndarray, band_names: list[str] | None
) -> list[str] | None:
    """Resolve band names from an explicit override or carrier attrs."""
    if band_names is not None:
        return list(band_names)
    names = _attrs(gt).get("band_names")
    if names is None:
        return None
    return [str(name) for name in names]


def _default_band_names(n_bands: int) -> list[str]:
    return [f"B{idx}" for idx in range(n_bands)]


def _resolve_band(key: BandKey, names: list[str] | None) -> int:
    if isinstance(key, int):
        return key
    if names is None:
        raise ValueError(
            f"Cannot resolve band name {key!r}: band_names must be provided "
            "or present in gt.attrs['band_names']"
        )
    try:
        return names.index(key)
    except ValueError as exc:
        raise ValueError(f"Band name {key!r} is not present in band_names") from exc


def _resolve_bands(keys: list[BandKey], names: list[str] | None) -> list[int]:
    indexes = []
    for idx, key in enumerate(keys):
        try:
            indexes.append(_resolve_band(key, names))
        except ValueError as exc:
            raise ValueError(f"Failed to resolve band at index {idx}: {key!r}") from exc
    return indexes


def _with_band_attrs(
    gt: GeoTensor | np.ndarray,
    values: np.ndarray,
    *,
    attrs: dict[str, Any] | None = None,
    band_names: list[str] | None = None,
    wavelengths: np.ndarray | None = None,
) -> GeoTensor | np.ndarray:
    """:func:`wrap_filled` that also writes ``attrs["wavelengths"]``.

    ``values`` carries ``gt``'s values, so nodata pixels get the carried
    fill (:func:`~geotoolz._src.valid.carried_fill`). ``attrs`` and
    ``band_names`` go straight to :func:`wrap_like` (which copies attrs
    and drops stale per-band keys); ``wavelengths``, when given, is then
    stored as a JSON-friendly float list. Plain ndarray carriers come back
    as plain arrays with no metadata to update.
    """
    out = wrap_filled(
        gt,
        values,
        fill_value_default=carried_fill(gt, np.asarray(values).dtype),
        attrs=attrs,
        band_names=band_names,
    )
    if wavelengths is not None and hasattr(out, "attrs"):
        out.attrs["wavelengths"] = jsonable(np.asarray(wavelengths, dtype=float))
    return out


def _wavelengths(
    gt: GeoTensor | np.ndarray, wavelengths: np.ndarray | list[float] | None
) -> np.ndarray:
    if wavelengths is None:
        wavelengths = _attrs(gt).get("wavelengths")
    if wavelengths is None:
        raise ValueError(
            "wavelengths must be provided or available as gt.attrs['wavelengths']"
        )
    return np.asarray(wavelengths, dtype=float)


class SelectBands(Operator):
    """Select bands by integer index or band name along the band axis.

    Resolves string keys against ``gt.attrs["band_names"]`` (or an
    explicit ``band_names=`` override at the call site). Selected
    ``band_names`` and ``wavelengths`` attrs travel with the output.
    Plain ``np.ndarray`` input is supported for integer indexes (a
    plain array comes back); string names require carrier attrs.

    Args:
        indexes: Bands to keep, in output order. Items are either
            integer positions along ``axis`` or string names looked up
            in ``gt.attrs["band_names"]``.
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> # Sentinel-2 RGB stacked as (B2, B3, B4, B8) -> keep (B4, B3, B2).
        >>> rgb = spectral.SelectBands(indexes=["B4", "B3", "B2"])
        >>> out = rgb(reflectance_geotensor)
    """

    def __init__(self, *, indexes: list[BandKey], axis: int = -3) -> None:
        self.indexes = indexes
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        names = _band_names(gt, None)
        indexes = _resolve_bands(self.indexes, names)
        attrs = _attrs(gt)
        band_axis_len = arr.shape[self.axis]
        wavelengths = attrs.get("wavelengths")
        if wavelengths is not None:
            wavelengths_arr = np.asarray(wavelengths, dtype=float)
            if wavelengths_arr.size != band_axis_len:
                raise ValueError(
                    "gt.attrs['wavelengths'] length "
                    f"({wavelengths_arr.size}) does not match the band axis "
                    f"length ({band_axis_len}) at axis {self.axis}"
                )
        out = select_bands(arr, indexes, axis=self.axis)
        return wrap_like(
            gt, out, attrs=take_band_attrs(attrs, indexes, n_bands=band_axis_len)
        )

    def get_config(self) -> dict[str, Any]:
        return {"indexes": jsonable(list(self.indexes)), "axis": self.axis}


class ReorderBands(SelectBands):
    """Reorder bands by integer index or band name.

    Same semantics as :class:`SelectBands` but emphasises that the
    output keeps every input band — just in a new order. Length of
    ``order`` should equal the number of bands on the input.

    Args:
        order: New band ordering. Items are integer positions or names.
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> # Reorder a BGRN stack to RGBN.
        >>> reorder = spectral.ReorderBands(order=[2, 1, 0, 3])
        >>> rgbn = reorder(bgrn_geotensor)
    """

    def __init__(self, *, order: list[BandKey], axis: int = -3) -> None:
        super().__init__(indexes=order, axis=axis)
        self.order = order

    def get_config(self) -> dict[str, Any]:
        return {"order": jsonable(list(self.order)), "axis": self.axis}


class StackBands(Operator):
    """Concatenate GeoTensors along the band axis.

    All inputs must share spatial shape, transform, and CRS. Each per-band
    attrs key (``band_names``, ``descriptions``, ``wavelengths``, ...)
    is concatenated in input order when every input carries it with one
    entry per band; otherwise that key is dropped on the output.
    Plain ``np.ndarray`` inputs are supported (the transform/CRS check
    is skipped and a plain array is returned); mixing GeoTensors and
    plain arrays raises because their georeferencing cannot agree.

    Args:
        axis: Position of the band axis. Default ``-3``. 2-D inputs are
            promoted to 3-D by inserting a unit dimension at ``axis``.

    Examples:
        >>> from geotoolz import spectral
        >>> stack = spectral.StackBands()
        >>> stacked = stack([gt_b4, gt_b8])  # (2, H, W)
    """

    def __init__(self, *, axis: int = -3) -> None:
        self.axis = axis

    def _apply(self, tensors: list[GeoTensor | np.ndarray]) -> GeoTensor | np.ndarray:
        if isinstance(tensors, np.ndarray):
            raise TypeError(
                "StackBands takes a sequence of carriers to concatenate along the "
                f"band axis; got a single {tensors.ndim}-D array"
            )
        if not tensors:
            raise ValueError("StackBands requires at least one GeoTensor")
        first = tensors[0]
        first_transform = getattr(first, "transform", None)
        first_crs = getattr(first, "crs", None)
        arrays = []
        for idx, gt in enumerate(tensors):
            if gt.shape[-2:] != first.shape[-2:]:
                raise ValueError(
                    "All GeoTensors must share spatial shape; "
                    f"GeoTensor at index {idx} has shape {gt.shape[-2:]}, "
                    f"expected {first.shape[-2:]}"
                )
            if (
                getattr(gt, "transform", None) != first_transform
                or getattr(gt, "crs", None) != first_crs
            ):
                raise ValueError("All GeoTensors must share transform and CRS")
            arr = np.asarray(gt)
            expanded = np.expand_dims(arr, axis=self.axis) if arr.ndim == 2 else arr
            arrays.append(expanded)
        out = np.concatenate(arrays, axis=self.axis)
        attrs = strip_band_attrs(_attrs(first))
        attrs.update(
            concat_band_attrs(
                [_attrs(gt) for gt in tensors],
                [arr.shape[self.axis] for arr in arrays],
            )
        )
        return wrap_like(first, out, attrs=attrs)


class SplitBands(Operator):
    """Split a multiband GeoTensor into one single-band GeoTensor per band.

    Plain ``np.ndarray`` input is supported and yields a list of plain
    single-band arrays.

    Args:
        names: Optional override for band names. If omitted, names are
            taken from ``gt.attrs["band_names"]``.
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> bands = spectral.SplitBands()(rgb_geotensor)  # list of (1, H, W)
        >>> red, green, blue = bands
    """

    def __init__(self, *, names: list[str] | None = None, axis: int = -3) -> None:
        self.names = names
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> list[GeoTensor | np.ndarray]:
        arr = np.asarray(gt)
        axis = self.axis + arr.ndim if self.axis < 0 else self.axis
        if not 0 <= axis < arr.ndim:
            raise ValueError(
                f"SplitBands axis {self.axis} is out of range for a "
                f"{arr.ndim}-D tensor (valid axes: "
                f"{-arr.ndim} to {arr.ndim - 1})"
            )
        n_bands = arr.shape[axis]
        source_names = _band_names(gt, self.names)
        if source_names is not None and len(source_names) != n_bands:
            raise ValueError("names length must match the number of bands")
        attrs = _attrs(gt)
        outputs = []
        for idx in range(n_bands):
            out = np.take(arr, [idx], axis=axis)
            outputs.append(
                wrap_like(
                    gt,
                    out,
                    attrs=take_band_attrs(attrs, [idx], n_bands=n_bands),
                    band_names=None if self.names is None else [self.names[idx]],
                )
            )
        return outputs


class BandMath(Operator):
    """Evaluate a restricted arithmetic expression over named bands.

    The expression is parsed with Python's ``ast`` module and only
    permits constants, unary +/-, binary ``+ - * / **``, and the
    whitelisted functions ``abs, sqrt, log, log10, exp, where, minimum,
    maximum, clip``. Band variables resolve to slices along the band
    axis named by ``band_names`` (default: ``gt.attrs["band_names"]``
    or synthetic ``B0, B1, ...`` labels). Plain ``np.ndarray`` input is
    supported (synthetic ``B0, B1, ...`` names apply unless
    ``band_names`` is given) and returns a plain array.

    Args:
        expression: Arithmetic expression over band names, e.g.
            ``"(B8 - B4) / (B8 + B4 + 1e-6)"``.
        band_names: Override for the names used in ``expression``.
            Default ``None`` (read from ``gt.attrs["band_names"]``).
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> # NDVI by name (assumes gt.attrs["band_names"] contains B4/B8).
        >>> ndvi = spectral.BandMath(expression="(B8 - B4) / (B8 + B4 + 1e-6)")
        >>> ndvi_map = ndvi(reflectance_geotensor)
    """

    def __init__(
        self,
        *,
        expression: str,
        band_names: list[str] | None = None,
        axis: int = -3,
    ) -> None:
        self.expression = expression
        self.band_names = band_names
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        names = _band_names(gt, self.band_names) or _default_band_names(
            arr.shape[self.axis]
        )
        variables = {
            name: np.take(arr, idx, axis=self.axis) for idx, name in enumerate(names)
        }
        out = keep_band_axis(
            np.asarray(evaluate_band_math(self.expression, variables)), gt
        )
        # A float result is a new quantity (NaN nodata); an integer or
        # boolean one (``b0 + b1`` on DN, ``b0 > b1``) keeps a fill its
        # dtype can hold.
        if np.issubdtype(out.dtype, np.inexact):
            fill = np.nan
        else:
            fill = carried_fill(gt, out.dtype)
        return wrap_filled(
            gt, out, fill_value_default=fill, attrs=strip_band_attrs(_attrs(gt))
        )


class NormalizedDifference(Operator):
    r"""Generic normalized-difference index by band index or name.

    .. math::

        \mathrm{ND}(a, b) = \frac{a - b}{a + b + \varepsilon}

    Unlike :class:`geotoolz.indices.NormalizedDifference` (integer
    indices only), this variant also accepts band-name strings resolved
    against ``gt.attrs["band_names"]``. Plain ``np.ndarray`` input is
    supported for integer keys and returns a plain array.

    Args:
        a: Index or name of the "high" band (numerator-positive term).
        b: Index or name of the "low" band.
        eps: Denominator stabiliser. Default ``1e-6``.
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> ndvi = spectral.NormalizedDifference(a="B8", b="B4")
        >>> ndvi_map = ndvi(reflectance_geotensor)
    """

    def __init__(
        self, *, a: BandKey, b: BandKey, eps: float = 1e-6, axis: int = -3
    ) -> None:
        self.a = a
        self.b = b
        self.eps = eps
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        names = _band_names(gt, None)
        out = normalized_difference(
            np.asarray(gt),
            _resolve_band(self.a, names),
            _resolve_band(self.b, names),
            axis=self.axis,
            eps=self.eps,
        )
        return wrap_filled(
            gt,
            keep_band_axis(out, gt),
            fill_value_default=np.nan,
            attrs=strip_band_attrs(_attrs(gt)),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "a": jsonable(self.a),
            "b": jsonable(self.b),
            "eps": self.eps,
            "axis": self.axis,
        }


class BandRatio(Operator):
    r"""Simple two-band ratio ``numerator / (denominator + eps)``.

    Plain ``np.ndarray`` input is supported for integer keys and
    returns a plain array; string names require carrier attrs.

    Args:
        numerator: Index or name of the numerator band.
        denominator: Index or name of the denominator band.
        eps: Denominator stabiliser. Default ``1e-6``.
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> # Simple Ratio Vegetation Index (NIR / Red).
        >>> srvi = spectral.BandRatio(numerator="B8", denominator="B4")
        >>> ratio = srvi(reflectance_geotensor)
    """

    def __init__(
        self,
        *,
        numerator: BandKey,
        denominator: BandKey,
        eps: float = 1e-6,
        axis: int = -3,
    ) -> None:
        self.numerator = numerator
        self.denominator = denominator
        self.eps = eps
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        names = _band_names(gt, None)
        out = band_ratio(
            np.asarray(gt),
            _resolve_band(self.numerator, names),
            _resolve_band(self.denominator, names),
            axis=self.axis,
            eps=self.eps,
        )
        return wrap_filled(
            gt,
            keep_band_axis(out, gt),
            fill_value_default=np.nan,
            attrs=strip_band_attrs(_attrs(gt)),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "numerator": jsonable(self.numerator),
            "denominator": jsonable(self.denominator),
            "eps": self.eps,
            "axis": self.axis,
        }


class ApplySRF(Operator):
    """Convolve band-first hyperspectral data through Gaussian SRFs.

    Constructs Gaussian spectral response functions from
    ``target_center_wavelengths`` and ``target_fwhm`` (via
    :func:`georeader.reflectance.srf`) and integrates the hyperspectral
    cube to the target multispectral bands via
    :func:`georeader.reflectance.transform_to_srf`. Plain
    ``np.ndarray`` input ``(B, H, W)`` is supported and returns a plain
    float32 array (no band-metadata bookkeeping).

    Args:
        target_center_wavelengths: Center wavelengths of the synthetic
            target bands, in nanometres. Shape ``(K,)``.
        target_fwhm: Full-width-half-maximum of each target band.
            Same units and shape as ``target_center_wavelengths``.
        source_wavelengths: Wavelengths of the hyperspectral source
            bands. Shape ``(B,)``.
        band_names: Optional names for the target bands. Default
            ``["B0", "B1", ...]``.

    Examples:
        >>> import geotoolz as gz
        >>> # Convolve EMIT hyperspectral to Sentinel-2 bands.
        >>> srf = gz.spectral.ApplySRF(
        ...     target_center_wavelengths=S2_CENTERS,
        ...     target_fwhm=S2_FWHM,
        ...     source_wavelengths=emit_wavelengths,
        ...     band_names=["B2", "B3", "B4", "B8"],
        ... )
        >>> s2_like = srf(emit_geotensor)
    """

    def __init__(
        self,
        *,
        target_center_wavelengths: np.ndarray | list[float],
        target_fwhm: np.ndarray | list[float],
        source_wavelengths: np.ndarray | list[float],
        band_names: list[str] | None = None,
    ) -> None:
        self.target_center_wavelengths = target_center_wavelengths
        self.target_fwhm = target_fwhm
        self.source_wavelengths = source_wavelengths
        self.band_names = band_names

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        source_wavelengths = np.asarray(self.source_wavelengths, dtype=float)
        target_center_wavelengths = np.asarray(
            self.target_center_wavelengths, dtype=float
        )
        responses = srf(target_center_wavelengths, self.target_fwhm, source_wavelengths)
        names = self.band_names or _default_band_names(target_center_wavelengths.size)
        srf_df = pd.DataFrame(responses, index=source_wavelengths, columns=names)
        out = transform_to_srf(
            gt,
            srf_df,
            source_wavelengths.tolist(),
            fill_value_default=getattr(gt, "fill_value_default", 0.0),
        )
        if not hasattr(out, "attrs"):  # plain-array carrier: nothing to rewrite
            return out
        return _with_band_attrs(
            gt,
            np.asarray(out),
            attrs=strip_band_attrs(_attrs(gt)),
            band_names=list(names),
            wavelengths=target_center_wavelengths,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "target_center_wavelengths": jsonable(
                np.asarray(self.target_center_wavelengths, dtype=float)
            ),
            "target_fwhm": jsonable(np.asarray(self.target_fwhm, dtype=float)),
            "source_wavelengths": jsonable(
                np.asarray(self.source_wavelengths, dtype=float)
            ),
            "band_names": self.band_names,
        }


class GaussianSRF(Operator):
    """Convolve to synthetic Gaussian SRFs using source wavelengths from attrs.

    Thin wrapper around :class:`ApplySRF` that reads the source
    hyperspectral wavelengths from ``gt.attrs["wavelengths"]`` when
    ``source_wavelengths`` is omitted. Plain ``np.ndarray`` input is
    supported when ``source_wavelengths`` is given explicitly (there
    are no carrier attrs to fall back on) and returns a plain array.

    Args:
        target_center_wavelengths: Center wavelengths of the target
            bands, in nanometres.
        target_fwhm: FWHM of each target band.
        source_wavelengths: Source hyperspectral wavelengths. Default
            ``None`` (read from ``gt.attrs["wavelengths"]``).
        band_names: Optional names for the target bands.

    Examples:
        >>> import geotoolz as gz
        >>> # Source wavelengths come from gt.attrs["wavelengths"].
        >>> conv = gz.spectral.GaussianSRF(
        ...     target_center_wavelengths=[490.0, 665.0, 842.0],
        ...     target_fwhm=[65.0, 30.0, 115.0],
        ...     band_names=["blue", "red", "nir"],
        ... )
        >>> out = conv(hyperspectral_geotensor)
    """

    def __init__(
        self,
        *,
        target_center_wavelengths: np.ndarray | list[float],
        target_fwhm: np.ndarray | list[float],
        source_wavelengths: np.ndarray | list[float] | None = None,
        band_names: list[str] | None = None,
    ) -> None:
        self.target_center_wavelengths = target_center_wavelengths
        self.target_fwhm = target_fwhm
        self.source_wavelengths = source_wavelengths
        self.band_names = band_names

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return ApplySRF(
            target_center_wavelengths=self.target_center_wavelengths,
            target_fwhm=self.target_fwhm,
            source_wavelengths=_wavelengths(gt, self.source_wavelengths),
            band_names=self.band_names,
        )(gt)

    def get_config(self) -> dict[str, Any]:
        return {
            "target_center_wavelengths": jsonable(
                np.asarray(self.target_center_wavelengths, dtype=float)
            ),
            "target_fwhm": jsonable(np.asarray(self.target_fwhm, dtype=float)),
            "source_wavelengths": (
                None
                if self.source_wavelengths is None
                else jsonable(np.asarray(self.source_wavelengths, dtype=float))
            ),
            "band_names": self.band_names,
        }


class ContinuumRemoval(Operator):
    """Hull-quotient continuum removal along the band axis.

    For each spectrum (each spatial pixel along the band axis), divides
    by an envelope:

    * ``method="convex_hull"`` — the upper convex hull of the spectrum
      vs wavelength. Output is the spectrum / hull, in ``(0, 1]``, with
      absorption features pulled to <1 and the hull itself at 1.
    * ``method="linear"`` — a straight line between the first and last
      band. Cheaper but only valid when no spectral curvature lives
      outside the absorption band of interest.

    Plain ``np.ndarray`` input is supported when ``wavelengths`` is
    given explicitly and returns a plain array.

    Args:
        method: ``"convex_hull"`` or ``"linear"``. Default
            ``"convex_hull"``.
        wavelengths: Source wavelengths in strictly increasing order.
            Default ``None`` (read from ``gt.attrs["wavelengths"]``).
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> # Continuum-remove a SWIR-2 mineral absorption window.
        >>> cr = spectral.ContinuumRemoval(method="convex_hull")
        >>> band_depth = cr(swir2_geotensor)  # (B, H, W), values in (0, 1]
    """

    def __init__(
        self,
        *,
        method: str = "convex_hull",
        wavelengths: np.ndarray | list[float] | None = None,
        axis: int = -3,
    ) -> None:
        self.method = method
        self.wavelengths = wavelengths
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = continuum_removal(
            np.asarray(gt),
            _wavelengths(gt, self.wavelengths),
            axis=self.axis,
            method=self.method,
        )
        return wrap_filled(gt, out, fill_value_default=np.nan)

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "wavelengths": None
            if self.wavelengths is None
            else jsonable(np.asarray(self.wavelengths, dtype=float)),
            "axis": self.axis,
        }


class SpectralBinning(Operator):
    """Aggregate source bands into wavelength-centered bins.

    For each ``target_wavelength`` :math:`\\lambda_c` with bin width
    :math:`w`, aggregates source bands with
    :math:`|\\lambda_s - \\lambda_c| \\le w/2`. Aggregation modes:

    * ``"mean"`` — uniform average.
    * ``"median"`` — robust to outliers in narrow bins.
    * ``"weighted_mean"`` — Gaussian weights with
      :math:`\\sigma = w / (2 \\sqrt{2 \\ln 2})` (FWHM equals ``width``).

    Plain ``np.ndarray`` input is supported when ``source_wavelengths``
    is given explicitly and returns a plain array.

    Args:
        target_wavelengths: Center wavelengths of the output bins.
        width: Scalar or per-bin widths (same units as wavelengths).
        method: ``"mean"``, ``"median"``, or ``"weighted_mean"``.
            Default ``"mean"``.
        source_wavelengths: Source band wavelengths. Default ``None``
            (read from ``gt.attrs["wavelengths"]``).
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> # Coarsen hyperspectral cube to broad-band averages.
        >>> binner = spectral.SpectralBinning(
        ...     target_wavelengths=[490.0, 560.0, 665.0],
        ...     width=30.0,
        ...     method="weighted_mean",
        ... )
        >>> coarse = binner(hyperspectral_geotensor)
    """

    def __init__(
        self,
        *,
        target_wavelengths: np.ndarray | list[float],
        width: float | np.ndarray | list[float],
        method: str = "mean",
        source_wavelengths: np.ndarray | list[float] | None = None,
        axis: int = -3,
    ) -> None:
        self.target_wavelengths = target_wavelengths
        self.width = width
        self.method = method
        self.source_wavelengths = source_wavelengths
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        target_wavelengths = np.asarray(self.target_wavelengths, dtype=float)
        out = spectral_binning(
            np.asarray(gt),
            _wavelengths(gt, self.source_wavelengths),
            target_wavelengths,
            self.width,
            axis=self.axis,
            method=self.method,
        )
        # Band axis is reshaped; band_names from the source no longer
        # apply, but new wavelengths do.
        return _with_band_attrs(
            gt,
            out,
            attrs=strip_band_attrs(_attrs(gt)),
            wavelengths=target_wavelengths,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "target_wavelengths": jsonable(
                np.asarray(self.target_wavelengths, dtype=float)
            ),
            "width": jsonable(self.width),
            "method": self.method,
            "source_wavelengths": (
                None
                if self.source_wavelengths is None
                else jsonable(np.asarray(self.source_wavelengths, dtype=float))
            ),
            "axis": self.axis,
        }


class SpectralSmoothing(Operator):
    """Smooth spectra along the band axis.

    Pure per-spectrum math — accepts a GeoTensor or a plain
    ``np.ndarray`` and returns the same carrier kind.

    Args:
        method: ``"savgol"`` (Savitzky-Golay), ``"gaussian"``, or
            ``"moving_average"``. Default ``"savgol"``.
        window: Filter window length in bands. Must be odd for
            ``"savgol"``. Default ``7``.
        polyorder: Polynomial order for Savitzky-Golay (ignored by the
            other methods). Default ``2``.
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> # De-noise hyperspectral spectra before continuum removal.
        >>> smooth = spectral.SpectralSmoothing(
        ...     method="savgol", window=9, polyorder=2
        ... )
        >>> denoised = smooth(hyperspectral_geotensor)
    """

    def __init__(
        self,
        *,
        method: str = "savgol",
        window: int = 7,
        polyorder: int = 2,
        axis: int = -3,
    ) -> None:
        self.method = method
        self.window = window
        self.polyorder = polyorder
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = spectral_smoothing(
            np.asarray(gt),
            axis=self.axis,
            method=self.method,
            window=self.window,
            polyorder=self.polyorder,
        )
        return wrap_filled(gt, out, fill_value_default=carried_fill(gt, out.dtype))
