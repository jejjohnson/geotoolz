"""Coordinate sample-spacing helper shared across spectral / wavelet code.

Centralises the ``_coord_spacing`` logic that was previously copied into
the Fourier (:mod:`xrtoolz.transforms._src.fourier`) and wavelet
(:mod:`xrtoolz.geo._src.wavelet_utils`,
:mod:`xrtoolz.geo._src.wavelet1d`) modules. The three copies differed
only in how strict they were about missing, singleton, and non-uniform
coordinates; those differences are now expressed as flags.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def coord_spacing(
    da: xr.DataArray,
    dim: str,
    *,
    require_coord: bool = True,
    require_min_samples: bool = True,
    require_uniform: bool = True,
    fallback: float = 1.0,
) -> float:
    """Absolute sample spacing of the coordinate for ``dim``.

    ``datetime64`` coordinates are measured in seconds.

    Args:
        da: Field carrying the coordinate.
        dim: Dimension whose coordinate spacing is measured.
        require_coord: When ``dim`` has no coordinate variable, raise
            (``True``) or return ``fallback`` (``False``). xarray permits
            dims without an explicit coordinate.
        require_min_samples: When the coordinate has fewer than two
            samples, raise (``True``) or return ``fallback`` (``False``).
        require_uniform: When the spacing is non-uniform, raise (``True``)
            or return the median ``|Δ|`` (``False``).
        fallback: Spacing returned for the degenerate cases above whose
            ``require_*`` flag is ``False``.

    Returns:
        The positive sample spacing.

    Raises:
        ValueError: per the ``require_*`` flags — missing coordinate,
            fewer than two samples, or non-uniform spacing.
    """
    if dim not in da.coords:
        if require_coord:
            raise ValueError(
                f"dim {dim!r} has no coordinate variable; cannot measure its "
                "sample spacing."
            )
        return fallback

    values = np.asarray(da[dim].values)
    if values.size < 2:
        if require_min_samples:
            raise ValueError(f"coord {dim!r} must contain at least two samples.")
        return fallback

    if np.issubdtype(values.dtype, np.datetime64):
        numeric = values.astype("datetime64[ns]").astype("int64").astype(float) / 1e9
    else:
        numeric = values.astype(float)

    diffs = np.diff(numeric)
    if require_uniform:
        if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"coord {dim!r} is not uniformly spaced; resample onto a regular "
                "grid first."
            )
        return float(abs(diffs[0]))
    return float(np.median(np.abs(diffs)))
