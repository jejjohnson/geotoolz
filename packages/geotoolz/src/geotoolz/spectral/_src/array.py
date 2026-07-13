"""Tier-A primitives for band-space spectral operations."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

import numpy as np
from jaxtyping import Float, Num, Shaped
from scipy import ndimage, signal

# Re-use the canonical normalized-difference primitive instead of
# duplicating the maths here. Same algebra as NDVI / NDWI / NDBI / NBR.
from geotoolz.indices._src.array import normalized_difference as normalized_difference


def select_bands(
    arr: Shaped[np.ndarray, "c h w"], indexes: list[int], *, axis: int = 0
) -> Shaped[np.ndarray, "k h w"]:
    """Select bands by integer index along the configured band axis.

    Thin wrapper over :func:`numpy.take`. Shapes are shown for the
    default channel-first ``(C, H, W)`` layout, but any number of
    dimensions is supported — only the axis at position ``axis`` is
    touched.

    Args:
        arr: Input array with the band axis at position ``axis``.
        indexes: Integer band positions to keep, in output order.
            Repeats are allowed (the band is duplicated in the output).
        axis: Position of the band axis. Default ``0`` (band-first
            convention; matches ``rasterio.read()`` output).

    Returns:
        Array with the band axis replaced by the ``len(indexes)``
        selected bands; all other axes are unchanged.
    """
    return np.take(arr, indexes, axis=axis)


def reorder_bands(
    arr: Shaped[np.ndarray, "c h w"], order: list[int], *, axis: int = 0
) -> Shaped[np.ndarray, "c h w"]:
    """Reorder bands by integer index along the configured band axis.

    Alias of :func:`select_bands` that emphasises intent: ``order``
    should be a permutation of ``range(arr.shape[axis])`` so every input
    band appears exactly once in the output (this is not enforced).

    Args:
        arr: Input array with the band axis at position ``axis``.
        order: New band ordering as integer positions.
        axis: Position of the band axis. Default ``0``.

    Returns:
        Array with the same shape as ``arr`` and the band axis permuted
        according to ``order``.
    """
    return select_bands(arr, order, axis=axis)


def band_ratio(
    arr: Num[np.ndarray, "c h w"],
    numerator_idx: int,
    denominator_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-6,
) -> Float[np.ndarray, "h w"]:
    """Compute ``numerator / (denominator + eps)`` with the band axis collapsed.

    Args:
        arr: Input array with the band axis at position ``axis``. Any
            other dimensions are preserved untouched.
        numerator_idx: Integer position of the numerator band.
        denominator_idx: Integer position of the denominator band.
        axis: Position of the band axis. Default ``0``.
        eps: Small constant added to the denominator to shadow division
            by zero on no-data / saturated pixels. Default ``1e-6``.
            Pass ``0.0`` to see ``inf``/``nan`` on zero pixels instead.

    Returns:
        Ratio array with the band axis collapsed.
    """
    numerator = np.take(arr, numerator_idx, axis=axis)
    denominator = np.take(arr, denominator_idx, axis=axis)
    return numerator / (denominator + eps)


_ALLOWED_FUNCS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "where": np.where,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}


def evaluate_band_math(
    expression: str, variables: Mapping[str, Num[np.ndarray, "h w"]]
) -> Num[np.ndarray, "h w"]:
    """Evaluate a restricted arithmetic expression over named band arrays.

    The expression is parsed with Python's :mod:`ast` module — never
    ``eval`` — and only a small arithmetic grammar is permitted:

    * numeric constants,
    * unary ``+`` / ``-``,
    * binary ``+ - * / **``,
    * calls to the whitelisted numpy functions ``abs``, ``sqrt``,
      ``log``, ``log10``, ``exp``, ``where``, ``minimum``, ``maximum``,
      and ``clip``.

    Bare names resolve against ``variables``; anything else (comparison
    operators, attribute access, subscripts, lambdas, ...) raises.

    Args:
        expression: Arithmetic expression over the keys of
            ``variables``, e.g. ``"(B8 - B4) / (B8 + B4 + 1e-6)"``.
        variables: Mapping from band name to band array. All arrays
            should broadcast against each other (typically identical
            ``(H, W)`` slices of one cube).

    Returns:
        The evaluated result as an ndarray (broadcast shape of the
        participating bands).

    Raises:
        ValueError: If the expression references an unknown band name,
            uses a non-numeric constant, or contains any construct
            outside the grammar above.
    """
    tree = ast.parse(expression, mode="eval")
    return np.asarray(_eval_node(tree.body, variables))


def _eval_node(node: ast.AST, variables: Mapping[str, np.ndarray]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("BandMath constants must be numeric")
    if isinstance(node, ast.Name):
        try:
            return variables[node.id]
        except KeyError as exc:
            raise ValueError(f"Unknown band name in expression: {node.id!r}") from exc
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError(
            f"Unsupported unary operator in BandMath: {type(node.op).__name__}"
        )
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        try:
            func = _ALLOWED_FUNCS[node.func.id]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported function in BandMath: {node.func.id!r}"
            ) from exc
        args = [_eval_node(arg, variables) for arg in node.args]
        kwargs = {
            kw.arg: _eval_node(kw.value, variables)
            for kw in node.keywords
            if kw.arg is not None
        }
        return func(*args, **kwargs)
    raise ValueError(f"Unsupported BandMath expression element: {type(node).__name__}")


def _validate_strictly_increasing(wavelengths: np.ndarray, *, context: str) -> None:
    non_increasing = np.flatnonzero(np.diff(wavelengths) <= 0)
    if non_increasing.size:
        idx = int(non_increasing[0])
        raise ValueError(
            f"{context} wavelengths must be strictly increasing; "
            f"found non-increasing pair at indices {idx} and {idx + 1}: "
            f"{wavelengths[idx]} >= {wavelengths[idx + 1]}"
        )


def continuum_removal(
    arr: Float[np.ndarray, "c h w"],
    wavelengths: Float[np.ndarray, " c"],
    *,
    axis: int = 0,
    method: str = "convex_hull",
) -> Float[np.ndarray, "c h w"]:
    """Apply hull-quotient continuum removal along the band axis.

    Each spectrum (the 1-D slice along ``axis`` at every remaining
    position) is divided by a continuum envelope so absorption features
    stand out as dips below 1:

    * ``method="convex_hull"`` — the envelope is the upper convex hull
      of the spectrum vs wavelength. Output values lie in ``(0, 1]``,
      with hull touch-points exactly at 1.
    * ``method="linear"`` — the envelope is the straight line between
      the first and last band. Cheaper, but only appropriate when no
      spectral curvature lies outside the absorption feature of
      interest.

    Where the continuum is exactly zero the quotient is defined as 1
    (flat), avoiding division-by-zero artefacts on empty pixels.

    Args:
        arr: Spectral cube with the band axis at position ``axis``. Any
            number of trailing/leading non-band dimensions is supported.
        wavelengths: Band-center wavelengths, strictly increasing, with
            length equal to ``arr.shape[axis]``.
        axis: Position of the band axis. Default ``0``.
        method: Continuum model, ``"convex_hull"`` (default) or
            ``"linear"``.

    Returns:
        Float array of the same shape as ``arr`` holding the
        continuum-removed (hull-quotient) spectra.

    Raises:
        ValueError: If ``wavelengths`` length does not match the band
            axis, is not strictly increasing, or ``method`` is unknown.
    """
    arr_axis0 = np.moveaxis(np.asarray(arr, dtype=float), axis, 0)
    if arr_axis0.shape[0] != wavelengths.size:
        raise ValueError("wavelengths length must match the band axis")
    _validate_strictly_increasing(wavelengths, context="continuum_removal")

    if method == "linear":
        continuum = _linear_continuum(arr_axis0, wavelengths)
    elif method == "convex_hull":
        continuum = _convex_hull_continuum(arr_axis0, wavelengths)
    else:
        raise ValueError("method must be 'convex_hull' or 'linear'")

    out = np.divide(
        arr_axis0, continuum, out=np.ones_like(arr_axis0), where=continuum != 0
    )
    return np.moveaxis(out, 0, axis)


def _linear_continuum(arr_axis0: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    x0 = wavelengths[0]
    x1 = wavelengths[-1]
    y0 = arr_axis0[0]
    y1 = arr_axis0[-1]
    if x1 == x0:
        return np.broadcast_to(y0, arr_axis0.shape)
    weights = ((wavelengths - x0) / (x1 - x0)).reshape(
        (-1,) + (1,) * (arr_axis0.ndim - 1)
    )
    return y0 + weights * (y1 - y0)


def _convex_hull_continuum(
    arr_axis0: np.ndarray, wavelengths: np.ndarray
) -> np.ndarray:
    flat = arr_axis0.reshape(arr_axis0.shape[0], -1)
    continuum = np.empty_like(flat, dtype=float)
    for idx in range(flat.shape[1]):
        continuum[:, idx] = _upper_hull_line(wavelengths, flat[:, idx])
    return continuum.reshape(arr_axis0.shape)


def _upper_hull_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    hull: list[int] = []
    for idx in range(x.size):
        hull.append(idx)
        while len(hull) >= 3:
            i, j, k = hull[-3:]
            slope_ij = (y[j] - y[i]) / (x[j] - x[i])
            slope_jk = (y[k] - y[j]) / (x[k] - x[j])
            if slope_ij <= slope_jk:
                hull.pop(-2)
            else:
                break
    return np.interp(x, x[hull], y[hull])


def spectral_binning(
    arr: Num[np.ndarray, "c h w"],
    source_wavelengths: Float[np.ndarray, " c"],
    target_wavelengths: Float[np.ndarray, " k"],
    width: float | Float[np.ndarray, " k"],
    *,
    axis: int = 0,
    method: str = "mean",
) -> Float[np.ndarray, "k h w"]:
    """Aggregate source bands into wavelength-centered bins.

    For each target bin center :math:`\\lambda_c` with width :math:`w`,
    the source bands satisfying
    :math:`|\\lambda_s - \\lambda_c| \\le w/2` are aggregated into one
    output band. Aggregation modes:

    * ``method="mean"`` — uniform average of the in-bin bands.
    * ``method="median"`` — per-pixel median; robust to outlier bands
      in narrow bins.
    * ``method="weighted_mean"`` — Gaussian weights centered on
      :math:`\\lambda_c` with :math:`\\sigma = w / (2\\sqrt{2\\ln 2})`,
      i.e. the bin width is interpreted as the FWHM of a synthetic
      Gaussian response; weights are normalised to sum to one.

    Args:
        arr: Spectral cube with the band axis at position ``axis``. Any
            number of non-band dimensions is supported.
        source_wavelengths: Wavelengths of the source bands, strictly
            increasing, with length equal to ``arr.shape[axis]``.
        target_wavelengths: Center wavelengths of the output bins,
            shape ``(K,)``.
        width: Bin width(s) in the same units as the wavelengths — a
            scalar applied to every bin, or a per-bin array broadcast
            against ``target_wavelengths``. Must be strictly positive.
        axis: Position of the band axis. Default ``0``.
        method: ``"mean"`` (default), ``"median"``, or
            ``"weighted_mean"``.

    Returns:
        Float array with the band axis replaced by the ``K`` binned
        bands; all other axes are unchanged.

    Raises:
        ValueError: If ``source_wavelengths`` is not strictly
            increasing, any width is non-positive, a bin captures no
            source band, or ``method`` is unknown.
    """
    arr_axis0 = np.moveaxis(np.asarray(arr), axis, 0)
    _validate_strictly_increasing(source_wavelengths, context="spectral_binning")
    widths = np.broadcast_to(np.asarray(width, dtype=float), target_wavelengths.shape)
    if np.any(widths <= 0):
        bad = int(np.flatnonzero(widths <= 0)[0])
        raise ValueError(
            "spectral_binning widths must be strictly positive; "
            f"got width={float(widths[bad])} at index {bad}"
        )
    out = np.empty((target_wavelengths.size, *arr_axis0.shape[1:]), dtype=float)

    for idx, (center, bin_width) in enumerate(
        zip(target_wavelengths, widths, strict=True)
    ):
        half_width = bin_width / 2.0
        mask = np.abs(source_wavelengths - center) <= half_width
        if not np.any(mask):
            raise ValueError(
                f"No source wavelengths found within +/-{half_width} "
                f"of target bin center {center}; source range: "
                f"[{source_wavelengths.min()}, {source_wavelengths.max()}]"
            )
        values = arr_axis0[mask]
        if method == "mean":
            out[idx] = np.mean(values, axis=0)
        elif method == "median":
            out[idx] = np.median(values, axis=0)
        elif method == "weighted_mean":
            sigma = bin_width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            weights = np.exp(
                -((source_wavelengths[mask] - center) ** 2) / (2.0 * sigma**2)
            )
            weights = weights / weights.sum()
            out[idx] = np.sum(
                weights.reshape((-1,) + (1,) * (values.ndim - 1)) * values, axis=0
            )
        else:
            raise ValueError("method must be 'mean', 'median', or 'weighted_mean'")
    return np.moveaxis(out, 0, axis)


def spectral_smoothing(
    arr: Float[np.ndarray, "c h w"],
    *,
    axis: int = 0,
    method: str = "savgol",
    window: int = 7,
    polyorder: int = 2,
) -> Float[np.ndarray, "c h w"]:
    """Smooth spectra along the band axis.

    Filters each spectrum (1-D slice along ``axis``) independently:

    * ``method="savgol"`` — Savitzky-Golay least-squares polynomial
      fit of order ``polyorder`` over an odd ``window`` of bands.
      Preserves peak shape better than a plain average.
    * ``method="gaussian"`` — Gaussian filter with
      ``sigma = window / 2`` bands (``polyorder`` ignored).
    * ``method="moving_average"`` — uniform boxcar of ``window`` bands
      via ``np.convolve(..., mode="same")`` (``polyorder`` ignored);
      spectrum edges taper because the kernel overhangs zeros.

    Args:
        arr: Spectral cube with the band axis at position ``axis``. Any
            number of non-band dimensions is supported.
        axis: Position of the band axis. Default ``0``.
        method: ``"savgol"`` (default), ``"gaussian"``, or
            ``"moving_average"``.
        window: Filter window length in bands. Must be odd for
            ``"savgol"``. Default ``7``.
        polyorder: Polynomial order of the Savitzky-Golay fit; must be
            less than ``window``. Default ``2``.

    Returns:
        Float array of the same shape as ``arr`` holding the smoothed
        spectra.

    Raises:
        ValueError: If ``method`` is unknown, or ``method="savgol"``
            with an even ``window``.
    """
    if method == "savgol":
        if window % 2 == 0:
            raise ValueError(f"savgol window must be odd, got {window}")
        return signal.savgol_filter(
            arr, window_length=window, polyorder=polyorder, axis=axis
        )
    if method == "gaussian":
        return ndimage.gaussian_filter1d(arr, sigma=window / 2.0, axis=axis)
    if method == "moving_average":
        kernel = np.ones(window, dtype=float) / float(window)
        return np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), axis, arr
        )
    raise ValueError("method must be 'savgol', 'gaussian', or 'moving_average'")
