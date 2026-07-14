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


def plot_scalogram(
    power: xr.DataArray,
    *,
    coi: xr.DataArray | None = None,
    signif_mask: xr.DataArray | None = None,
    ax: Axes | None = None,
    log_period: bool = True,
    cmap: str = "viridis",
) -> Axes:
    """Plot a time-period wavelet scalogram.

    Args:
        power: Two-dimensional wavelet power with one scale dimension and one
            time dimension. If present, the ``period`` coordinate is used for
            the y-axis.
        coi: Optional cone-of-influence period series to overlay.
        signif_mask: Optional boolean mask to contour significant regions.
            The plot draws a contour at ``0.5``, i.e. the boundary between
            insignificant (``False``) and significant (``True``) samples.
        ax: Existing axes to draw on. A new axes is created when omitted.
        log_period: If ``True``, draw the period axis on a logarithmic scale.
        cmap: Matplotlib colormap name for the power heatmap.

    Returns:
        The axes object for further customization.
    """
    import matplotlib.pyplot as plt

    if power.ndim != 2:
        raise ValueError(f"plot_scalogram expects a 2-D array; got dims {power.dims}.")
    if ax is None:
        _, ax = plt.subplots()
    scale_dim = _infer_scale_dim(power)
    time_dim = str(next(dim for dim in power.dims if dim != scale_dim))
    ycoord = "period" if "period" in power.coords else scale_dim
    values = np.asarray(power.transpose(scale_dim, time_dim).values, dtype=float)
    x = np.asarray(power[time_dim].values)
    y = np.asarray(power[ycoord].values, dtype=float)
    mesh = ax.pcolormesh(x, y, values, shading="auto", cmap=cmap)
    ax.figure.colorbar(mesh, ax=ax)
    if coi is not None:
        # Plot COI against the scalogram's own time axis. Taking x from
        # coi[coi.dims[0]] silently misaligns when coi was produced on a
        # different grid.
        if coi.ndim != 1:
            raise ValueError(
                f"coi must be 1-D over the time axis; got dims {coi.dims}."
            )
        if coi.size != x.size:
            raise ValueError(
                f"coi length ({coi.size}) does not match scalogram time axis "
                f"({x.size}); reindex coi onto power[{time_dim!r}] first."
            )
        ax.plot(x, np.asarray(coi.values), color="white")
    if signif_mask is not None:
        mask = signif_mask.transpose(scale_dim, time_dim)
        ax.contour(
            x,
            y,
            np.asarray(mask.values, dtype=float),
            levels=[0.5],
            colors="black",
            linewidths=0.8,
        )
    if log_period:
        ax.set_yscale("log")
    ax.set_xlabel(time_dim)
    ax.set_ylabel(ycoord)
    ax.set_title(str(power.name) if power.name is not None else "Wavelet scalogram")
    return ax


def plot_global_wavelet_spectrum(
    power: xr.DataArray,
    *,
    signif: xr.DataArray | None = None,
    ax: Axes | None = None,
    time_dim: str = "time",
) -> Axes:
    """Plot global wavelet spectrum as time-averaged power vs. period.

    Args:
        power: Wavelet power with a scale dimension and exactly one
            additional time-like dimension to average over (named
            ``time_dim``). For pixelwise inputs with extra outer dims,
            reduce the spatial/ensemble axes first (e.g.
            ``power.mean(["lat", "lon"])``).
        signif: Optional threshold or reference spectrum with the same
            scale/period coordinate as ``power``. If it includes the
            ``time_dim`` axis it is time-averaged before being drawn as a
            dashed line.
        ax: Existing axes to draw on. A new axes is created when omitted.
        time_dim: Name of the time-like dimension to reduce.

    Returns:
        The axes object for further customization.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    scale_dim = _infer_scale_dim(power)
    extra = tuple(d for d in power.dims if d not in {scale_dim, time_dim})
    if extra:
        raise ValueError(
            f"plot_global_wavelet_spectrum expects only "
            f"({scale_dim!r}, {time_dim!r}) (or just {scale_dim!r}); got extra "
            f"dims {extra!r}. Reduce them first, e.g. .mean({list(extra)!r})."
        )
    spectrum = power.mean(time_dim, skipna=True) if time_dim in power.dims else power
    if spectrum.ndim != 1:
        raise ValueError("plot_global_wavelet_spectrum expects one scale dimension.")
    xcoord = "period" if "period" in spectrum.coords else scale_dim
    ax.plot(
        np.asarray(spectrum[xcoord].values, dtype=float), np.asarray(spectrum.values)
    )
    if signif is not None:
        sig = signif.mean(time_dim, skipna=True) if time_dim in signif.dims else signif
        ax.plot(
            np.asarray(sig[xcoord].values, dtype=float), np.asarray(sig.values), "--"
        )
    ax.set_xscale("log")
    ax.set_xlabel(xcoord)
    ax.set_ylabel("power")
    return ax


def plot_dominant_period_map(
    pmap: xr.DataArray,
    *,
    ax: Axes | None = None,
    cmap: str = "cividis",
    levels: Sequence[float] | int | None = None,
) -> Axes:
    """Plot a 2-D map of dominant Fourier period.

    Args:
        pmap: Two-dimensional dominant-period field, typically produced by
            :func:`xrtoolz.geo.dominant_period_map` after averaging rectified
            wavelet power over time and selecting the peak period.
        ax: Existing axes to draw on. A new axes is created when omitted.
        cmap: Matplotlib colormap name.
        levels: Optional contour levels. When omitted, an ``imshow`` raster is
            drawn; otherwise filled contours are drawn.

    Returns:
        The axes object for further customization.
    """
    if pmap.ndim != 2:
        raise ValueError(
            f"plot_dominant_period_map expects a 2-D field; got dims {pmap.dims}."
        )
    out = plot_resolved_scale_map(pmap, ax=ax, cmap=cmap, levels=levels)
    out.set_title(str(pmap.name) if pmap.name is not None else "Dominant period")
    out.set_ylabel(str(pmap.dims[0]))
    out.set_xlabel(str(pmap.dims[1]))
    return out


def _parse_slope(slope: str | float) -> float:
    """Convert slope labels such as ``"-5/3"`` to float exponents."""
    if isinstance(slope, str) and "/" in slope:
        num, den = slope.split("/", maxsplit=1)
        return float(num) / float(den)
    return float(slope)


def _infer_scale_dim(da: xr.DataArray) -> str:
    """Return the scale dimension, preferring the canonical ``scale`` name."""
    return "scale" if "scale" in da.dims else str(da.dims[0])


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
