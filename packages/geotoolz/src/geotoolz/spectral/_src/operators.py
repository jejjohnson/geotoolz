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
value-preserving transforms (binning, smoothing) keep it too, switching
to ``NaN`` when integer input is promoted to float
(:func:`geotoolz._src.valid.carried_fill`); derived products
(:class:`BandMath`, :class:`BandRatio`, :class:`ContinuumRemoval`) hold
``NaN`` and declare ``fill_value_default=NaN``.

Band names are resolved from an explicit ``band_names=`` constructor
argument when present; otherwise through the package-wide resolver
(:func:`geotoolz._src.bands.resolve_band`), which reads ``gt.attrs``
under ``band_names``, then ``descriptions``, then ``bands``.
Wavelength-dependent operators follow the same convention with explicit
``source_wavelengths=`` / ``wavelengths=`` arguments first, then
``gt.attrs["wavelengths"]`` (:func:`geotoolz._src.bands.resolve_wavelengths`).
On plain arrays the attrs fallbacks are unavailable, so a string band
reference raises ``TypeError`` and a missing wavelength table raises
``ValueError`` unless the values are given explicitly.

The band axis is ``-3`` by default (georeader's ``(C, H, W)`` /
``(T, C, H, W)`` layout). On a ``(T, C, H, W)`` time stack every operator
works per frame: band-collapsing products (:class:`BandMath`,
:class:`BandRatio`) come back as ``(T, 1, H, W)``.

The normalized difference and the Gaussian-SRF convolution live in one
place each: :class:`geotoolz.indices.NormalizedDifference` and
:class:`geotoolz.radiometry.ApplySRF`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pipekit import Operator

from geotoolz._src.bands import (
    BandRef,
    band_names,
    concat_band_attrs,
    resolve_band,
    resolve_bands,
    resolve_wavelengths,
    strip_band_attrs,
    take_band_attrs,
)
from geotoolz._src.config import jsonable
from geotoolz._src.geo import grid_matches
from geotoolz._src.shape import keep_band_axis
from geotoolz._src.valid import carried_fill, wrap_filled
from geotoolz._src.wrap import wrap_like
from geotoolz.spectral._src.array import (
    band_ratio,
    continuum_removal,
    evaluate_band_math,
    select_bands,
    spectral_binning,
    spectral_smoothing,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


def _attrs(gt: GeoTensor | np.ndarray) -> dict[str, Any]:
    """Return a shallow copy of ``gt.attrs`` (or ``{}`` if missing)."""
    attrs = getattr(gt, "attrs", None)
    return {} if attrs is None else dict(attrs)


def _names_or_attrs(
    gt: GeoTensor | np.ndarray, override: list[str] | None
) -> list[str | None] | None:
    """An explicit ``names`` override, else the carrier's band names."""
    return list(override) if override is not None else band_names(gt)


def _default_band_names(n_bands: int) -> list[str]:
    return [f"B{idx}" for idx in range(n_bands)]


class SelectBands(Operator):
    """Select bands by integer index or band name along the band axis.

    Resolves string keys with the package-wide band resolver
    (``gt.attrs`` ``band_names``, then ``descriptions``, then
    ``bands``). Selected
    ``band_names`` and ``wavelengths`` attrs travel with the output.
    Plain ``np.ndarray`` input is supported for integer positions (a
    plain array comes back); string names require carrier attrs.

    Args:
        bands: Bands to keep, in output order. Items are either
            integer positions along ``axis`` or band names resolved
            against ``gt.attrs``.
        axis: Position of the band axis. Default ``-3``.

    Examples:
        >>> from geotoolz import spectral
        >>> # Sentinel-2 RGB stacked as (B2, B3, B4, B8) -> keep (B4, B3, B2).
        >>> rgb = spectral.SelectBands(bands=["B4", "B3", "B2"])
        >>> out = rgb(reflectance_geotensor)
    """

    def __init__(self, *, bands: list[BandRef], axis: int = -3) -> None:
        self.bands = bands
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        indexes = resolve_bands(gt, self.bands)
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
        return {"bands": jsonable(list(self.bands)), "axis": self.axis}


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

    def __init__(self, *, order: list[BandRef], axis: int = -3) -> None:
        super().__init__(bands=order, axis=axis)
        self.order = order

    def get_config(self) -> dict[str, Any]:
        return {"order": jsonable(list(self.order)), "axis": self.axis}


def _same_fill(a: Any, b: Any) -> bool:
    """Fill-value equality with ``NaN`` equal to ``NaN`` (``None`` only to ``None``)."""
    if a is None or b is None:
        return a is b
    both_nan = bool(np.isnan(a)) and bool(np.isnan(b))
    return both_nan or bool(a == b)


class StackBands(Operator):
    """Concatenate carriers along the band axis.

    The checks of :meth:`georeader.GeoTensor.concatenate`, extended to
    inputs with different band counts: every input must sit on the first
    input's pixel grid (:func:`geotoolz._src.geo.grid_matches` -- equal
    spatial shape, CRS and transform) and declare the same
    ``fill_value_default`` (``NaN`` matches ``NaN``), since the stack can
    declare only one fill. Each per-band attrs key (``band_names``,
    ``descriptions``, ``wavelengths``, ...) is concatenated in input order
    when every input carries it with one entry per band; otherwise that
    key is dropped on the output. Other attrs come from the first input.
    Plain ``np.ndarray`` inputs are supported (only the spatial shape is
    checked and a plain array is returned); mixing GeoTensors and plain
    arrays raises because their georeferencing cannot agree.

    Args:
        axis: Position of the band axis. Default ``-3``. 2-D inputs are
            promoted to 3-D by inserting a unit dimension at ``axis``.

    Raises:
        TypeError: If given a single array instead of a sequence.
        ValueError: If the sequence is empty, mixes GeoTensors and plain
            arrays, or the inputs differ in grid or fill value.

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
        georeferenced = getattr(first, "transform", None) is not None
        first_fill = getattr(first, "fill_value_default", None)
        for idx, gt in enumerate(tensors[1:], start=1):
            if (getattr(gt, "transform", None) is not None) != georeferenced:
                raise ValueError(
                    "StackBands cannot mix GeoTensors and plain arrays; input "
                    f"{idx} differs from input 0"
                )
            if not grid_matches(first, gt):
                raise ValueError(
                    "All inputs must share spatial shape, transform and CRS; "
                    f"input {idx} (shape {np.shape(gt)}) is on a different grid "
                    f"than input 0 (shape {np.shape(first)})"
                )
            fill = getattr(gt, "fill_value_default", None)
            if not _same_fill(fill, first_fill):
                raise ValueError(
                    "All inputs must share fill_value_default; input "
                    f"{idx} has {fill!r}, input 0 has {first_fill!r}"
                )
        arrays = [
            np.expand_dims(arr, axis=self.axis) if arr.ndim == 2 else arr
            for arr in (np.asarray(gt) for gt in tensors)
        ]
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
            read from ``gt.attrs`` (``band_names``, then ``descriptions``,
            then ``bands``).
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
        source_names = _names_or_attrs(gt, self.names)
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
    axis named by ``band_names`` (default: the carrier's band names from
    ``gt.attrs`` — ``band_names``, then ``descriptions``, then ``bands`` —
    or synthetic ``B0, B1, ...`` labels). Plain ``np.ndarray`` input is
    supported (synthetic ``B0, B1, ...`` names apply unless
    ``band_names`` is given) and returns a plain array.

    Args:
        expression: Arithmetic expression over band names, e.g.
            ``"(B8 - B4) / (B8 + B4 + 1e-6)"``.
        band_names: Override for the names used in ``expression``.
            Default ``None`` (read from the carrier's ``attrs``).
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
        names = _names_or_attrs(gt, self.band_names) or _default_band_names(
            arr.shape[self.axis]
        )
        variables = {
            name: np.take(arr, idx, axis=self.axis)
            for idx, name in enumerate(names)
            if name is not None
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


class BandRatio(Operator):
    r"""Simple two-band ratio ``numerator / (denominator + eps)``.

    Plain ``np.ndarray`` input is supported for integer keys and
    returns a plain array; string names require carrier attrs.

    Args:
        numerator: Index or name of the numerator band.
        denominator: Index or name of the denominator band.
        eps: Denominator stabiliser. Default ``1e-10`` (the package-wide
            value, as in :func:`geotoolz.indices.iron_oxide`).
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
        numerator: BandRef,
        denominator: BandRef,
        eps: float = 1e-10,
        axis: int = -3,
    ) -> None:
        self.numerator = numerator
        self.denominator = denominator
        self.eps = eps
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = band_ratio(
            np.asarray(gt),
            resolve_band(gt, self.numerator),
            resolve_band(gt, self.denominator),
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
            resolve_wavelengths(gt, self.wavelengths),
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
            resolve_wavelengths(gt, self.source_wavelengths, name="source_wavelengths"),
            target_wavelengths,
            self.width,
            axis=self.axis,
            method=self.method,
        )
        # Band axis is reshaped; band_names from the source no longer
        # apply, but new wavelengths do.
        attrs = strip_band_attrs(_attrs(gt))
        attrs["wavelengths"] = jsonable(target_wavelengths)
        return wrap_filled(
            gt,
            out,
            fill_value_default=carried_fill(gt, out.dtype),
            attrs=attrs,
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
