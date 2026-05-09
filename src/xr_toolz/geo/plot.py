"""Plotting helpers for wavelet spectra."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_resolved_scale_map(
    rs_map: xr.DataArray,
    *,
    ax: Axes | None = None,
    cmap: str = "viridis",
    levels: Sequence[float] | int | None = None,
) -> Axes:
    """Plot a 2-D resolved-scale map.

    Args:
        rs_map: Two-dimensional resolved-scale field, typically in kilometres.
        ax: Existing axes to draw on. A new axes is created when omitted.
        cmap: Matplotlib colormap name.
        levels: Optional contour levels. When omitted, an ``imshow`` raster is
            drawn; otherwise filled contours are drawn.

    Returns:
        The axes object for further customization.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    data = np.asarray(rs_map.values, dtype=float)
    extent = _extent(rs_map)
    if levels is None:
        image = ax.imshow(data, origin="lower", cmap=cmap, extent=extent, aspect="auto")
        ax.figure.colorbar(image, ax=ax)
    else:
        y = np.asarray(rs_map[rs_map.dims[0]].values, dtype=float)
        x = np.asarray(rs_map[rs_map.dims[1]].values, dtype=float)
        ax.contourf(x, y, data, cmap=cmap, levels=levels)
    ax.set_title(rs_map.name or "Resolved scale")
    return ax


def plot_wavelet_spectrum_1d(
    spectrum: xr.DataArray,
    *,
    ax: Axes | None = None,
    ref_slopes: Sequence[str | float] = ("-3", "-5/3"),
) -> Axes:
    """Plot a one-dimensional wavelet spectrum with reference slopes.

    Args:
        spectrum: One-dimensional spectrum whose first dimension is scale.
        ax: Existing axes to draw on. A new axes is created when omitted.
        ref_slopes: Reference spectral slopes, such as ``"-3"`` or ``"-5/3"``,
            plotted as dashed comparison lines through the first finite value.

    Returns:
        The axes object for further customization.
    """
    import matplotlib.pyplot as plt

    if spectrum.ndim != 1:
        raise ValueError(
            f"plot_wavelet_spectrum_1d expects a 1-D spectrum; got dims "
            f"{spectrum.dims}. Reduce the extra dims first (e.g. .mean('angle'))."
        )
    if ax is None:
        _, ax = plt.subplots()
    scale_dim = spectrum.dims[0]
    x = np.asarray(spectrum[scale_dim].values, dtype=float)
    y = np.asarray(spectrum.values, dtype=float)
    ax.loglog(x, y, label=spectrum.name or "spectrum")
    finite = np.isfinite(y) & (y > 0)
    if finite.any():
        x0 = float(x[finite][0])
        y0 = float(y[finite][0])
        for slope in ref_slopes:
            exponent = _parse_slope(slope)
            ax.loglog(x, y0 * (x / x0) ** exponent, linestyle="--", label=slope)
    ax.set_xlabel(scale_dim)
    ax.set_ylabel("power")
    ax.legend()
    return ax


def plot_wavelet_anisotropy(
    spectrum: xr.DataArray,
    *,
    ax: Axes | None = None,
    log: bool = True,
) -> Axes:
    """Plot a polar angle-scale heatmap from a directional spectrum.

    Args:
        spectrum: Directional spectrum containing an ``"angle"`` dimension and
            one non-angle scale dimension.
        ax: Existing polar axes to draw on. A new polar axes is created when
            omitted.
        log: If ``True``, plot ``log10`` power values.

    Returns:
        The polar axes object for further customization.
    """
    import matplotlib.pyplot as plt

    if "angle" not in spectrum.dims:
        raise ValueError("spectrum must include an 'angle' dimension")
    if spectrum.ndim != 2:
        raise ValueError(
            f"plot_wavelet_anisotropy expects exactly 2 dims (angle + one "
            f"scale axis); got {spectrum.dims}. Average / reduce extra dims first."
        )
    scale_dim = next(dim for dim in spectrum.dims if dim != "angle")
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    values = np.asarray(spectrum.transpose("angle", scale_dim).values, dtype=float)
    if log:
        values = np.log10(np.maximum(values, np.finfo(float).tiny))
    theta = np.asarray(spectrum["angle"].values, dtype=float)
    radius = np.asarray(spectrum[scale_dim].values, dtype=float)
    ax.grid(False)
    ax.pcolormesh(theta, radius, values.T, shading="auto")
    ax.set_ylabel(scale_dim)
    return ax


def _parse_slope(slope: str | float) -> float:
    """Convert slope labels such as ``"-5/3"`` to float exponents."""
    if isinstance(slope, str) and "/" in slope:
        num, den = slope.split("/", maxsplit=1)
        return float(num) / float(den)
    return float(slope)


def _extent(da: xr.DataArray) -> tuple[float, float, float, float] | None:
    """Return ``imshow`` extent from 2-D coordinate centers."""
    if da.ndim != 2:
        return None
    y = np.asarray(da[da.dims[0]].values, dtype=float)
    x = np.asarray(da[da.dims[1]].values, dtype=float)
    if x.size < 2 or y.size < 2:
        return None
    dx = float(np.diff(x).mean())
    dy = float(np.diff(y).mean())
    return (
        float(x[0] - 0.5 * dx),
        float(x[-1] + 0.5 * dx),
        float(y[0] - 0.5 * dy),
        float(y[-1] + 0.5 * dy),
    )
