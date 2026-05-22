"""V1.5 PSD plot panels — 1-D isotropic and 2-D space-time PSDs.

Mirrors the ``PlotPSDIsotropic`` / ``PlotPSDSpaceTime`` classes from
the oceanbench NeurIPS notebook utilities, refactored as
:class:`_ValidationPanel` subclasses so they slot into ``Sequential``
and ``Graph`` pipelines.

Inputs are :class:`xr.DataArray` outputs of
:func:`xrtoolz.transforms.power_spectrum` (``isotropic=True`` or
``False``) and :func:`xrtoolz.metrics.psd_score`.
"""

from __future__ import annotations

from typing import Any

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

from xrtoolz.metrics import find_intercept_1D
from xrtoolz.viz.validation._src.base import _NullContext, _ValidationPanel


class _PSDPanelBase(_ValidationPanel):
    """PSD panels use constrained layout + per-axes title to avoid
    collisions between the optional top wavelength axis and a
    figure-level suptitle."""

    def _make_fig_axes(self) -> tuple[mpl_figure.Figure, Any]:
        nrows, ncols = self._default_axes_layout
        fig, axes = plt.subplots(
            nrows, ncols, figsize=self.figsize, constrained_layout=True
        )
        return fig, axes

    def _apply(self, *args: Any, **kwargs: Any) -> mpl_figure.Figure:
        ctx = (
            plt.style.context(self.style) if self.style is not None else _NullContext()
        )
        with ctx:
            fig, axes = self._make_fig_axes()
            self._build(fig, axes, *args, **kwargs)
            title = self.title if self.title is not None else self._default_title()
            if title:
                # Put the title on the axes (or top axes) so it doesn't
                # collide with a top secondary x-axis label.
                ax = axes if not isinstance(axes, np.ndarray) else axes.flat[0]
                ax.set_title(title, pad=24)
        self._maybe_save(fig)
        self._maybe_show(fig)
        return fig


def _coerce_da(obj: xr.DataArray | xr.Dataset, var: str | None = None) -> xr.DataArray:
    if isinstance(obj, xr.Dataset):
        name = var or next(iter(obj.data_vars))
        return obj[name]
    return obj


def _format_log_plain(axis: Any) -> None:
    """Replace scientific-notation log labels (``10⁻¹``) with plain
    decimals (``0.1``) on a log-scaled axis. Minor ticks unlabeled."""
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    sf = ScalarFormatter()
    sf.set_scientific(False)
    axis.set_major_formatter(sf)
    axis.set_minor_formatter(NullFormatter())


_DEFAULT_WAVELENGTH_TICKS: tuple[float, ...] = (
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
)
_DEFAULT_PERIOD_TICKS: tuple[float, ...] = (1, 2, 3, 5, 10, 20, 30, 60, 90, 180, 365)


def _wavelength_axis(
    ax: Any,
    *,
    space_scale: float,
    label: str,
    ticks: tuple[float, ...] = _DEFAULT_WAVELENGTH_TICKS,
    location: str = "top",
) -> Any:
    """Add a secondary axis showing ``1/(freq * space_scale)`` with
    plain-integer tick labels at hand-picked round values."""
    from matplotlib.ticker import FixedLocator, NullFormatter

    scale = float(space_scale)
    secax = ax.secondary_xaxis(
        location,
        functions=(
            lambda f: 1.0 / np.where(np.asarray(f) == 0, np.nan, np.asarray(f) * scale),
            lambda w: 1.0 / np.where(np.asarray(w) == 0, np.nan, np.asarray(w) * scale),
        ),
    )
    secax.set_xlabel(label)
    secax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    secax.xaxis.set_major_formatter("{x:.0f}")
    secax.xaxis.set_minor_formatter(NullFormatter())
    return secax


def _period_axis(
    ax: Any,
    *,
    time_scale: float,
    label: str,
    ticks: tuple[float, ...] = _DEFAULT_PERIOD_TICKS,
) -> Any:
    """Twin y-axis showing ``1/(freq * time_scale)`` with plain labels
    at hand-picked round values."""
    from matplotlib.ticker import FixedLocator, NullFormatter

    scale = float(time_scale)
    secay = ax.secondary_yaxis(
        "right",
        functions=(
            lambda f: 1.0 / np.where(np.asarray(f) == 0, np.nan, np.asarray(f) * scale),
            lambda w: 1.0 / np.where(np.asarray(w) == 0, np.nan, np.asarray(w) * scale),
        ),
    )
    secay.set_ylabel(label)
    secay.yaxis.set_major_locator(FixedLocator(list(ticks)))
    secay.yaxis.set_major_formatter("{x:.0f}")
    secay.yaxis.set_minor_formatter(NullFormatter())
    return secay


class PSDIsotropicPanel(_PSDPanelBase):
    """Log-log isotropic PSD vs radial wavenumber.

    Args:
        freq_dim: Radial-frequency dim name (``xrft`` default
            ``"freq_r"``).
        space_scale: Multiplier converting ``freq_dim`` units to inverse
            length used by the wavelength axis (e.g. ``1e-3`` for
            cycles/m → cycles/km). Default ``1.0``.
        wavelength_label: Top-axis label. Default
            ``"Wavelength [units]"``.
        wavenumber_label: Bottom-axis label. Default
            ``"Wavenumber [cycles / unit]"``.
        ylabel: Y-axis label. Default ``"PSD"``.
        show_wavelength: Toggle the wavelength twin axis. Default
            ``True``.
    """

    def __init__(
        self,
        *,
        freq_dim: str = "freq_r",
        space_scale: float = 1.0,
        wavelength_label: str = "Wavelength [units]",
        wavenumber_label: str = "Wavenumber [cycles / unit]",
        ylabel: str = "PSD",
        show_wavelength: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.freq_dim = freq_dim
        self.space_scale = float(space_scale)
        self.wavelength_label = wavelength_label
        self.wavenumber_label = wavenumber_label
        self.ylabel = ylabel
        self.show_wavelength = show_wavelength

    def _default_title(self) -> str:
        return "Isotropic PSD"

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        psd: xr.DataArray | xr.Dataset,
    ) -> None:
        ax = axes
        if isinstance(psd, xr.Dataset):
            for name, da in psd.data_vars.items():
                f = np.asarray(da[self.freq_dim].values)
                ax.plot(f, np.asarray(da.values), label=name)
            ax.legend()
        else:
            f = np.asarray(psd[self.freq_dim].values)
            ax.plot(f, np.asarray(psd.values))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(self.wavenumber_label)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, which="both", alpha=0.3)
        _format_log_plain(ax.xaxis)
        if self.show_wavelength:
            _wavelength_axis(
                ax,
                space_scale=self.space_scale,
                label=self.wavelength_label,
            )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "freq_dim": self.freq_dim,
            "space_scale": self.space_scale,
            "wavelength_label": self.wavelength_label,
            "wavenumber_label": self.wavenumber_label,
            "ylabel": self.ylabel,
            "show_wavelength": self.show_wavelength,
        }


class PSDIsotropicScorePanel(_PSDPanelBase):
    """PSD score vs radial wavenumber with resolved-scale marker.

    Plots the 1-D PSD score (typically from
    :func:`xrtoolz.metrics.psd_score` with ``isotropic=True``) on a
    log-x axis, marks the score threshold with a horizontal dashed
    line, and annotates the resolved wavelength where the score first
    crosses ``threshold`` (computed via
    :func:`xrtoolz.metrics._src.spectral.find_intercept_1D`).

    Args:
        freq_dim: Frequency dim name. Default ``"freq_r"``.
        threshold: Score threshold for the resolved-scale marker.
            Default ``0.5``.
        space_scale: As :class:`PSDIsotropicPanel`.
        wavelength_label: Top-axis label.
        wavenumber_label: Bottom-axis label.
        ylabel: Y-axis label.
        score_var: Variable name when input is a Dataset. Default
            ``"score"``.
        show_wavelength: Toggle the twin axis.
        resolved_units: Unit string used in the resolved-scale legend
            text. ``None`` (default) parses it from the trailing
            ``[unit]`` of ``wavelength_label`` (e.g.
            ``"Wavelength [km]"`` → ``"km"``); empty string suppresses
            the unit.
        clip: When ``True`` (default), display the score clipped to
            ``[0, 1]`` and pin the y-axis to that range — matches the
            oceanbench / Ballarotta 2019 convention where negative
            values just mean "worse than zero predictor". Set to
            ``False`` for diagnostic mode that exposes the signed
            score, with the y-limits chosen from the data unless
            ``ylim`` is given.
        ylim: Optional ``(ymin, ymax)`` override for the y-axis. When
            unset and ``clip=True`` the panel uses ``(0, 1)``; when
            unset and ``clip=False`` matplotlib auto-scales.
    """

    def __init__(
        self,
        *,
        freq_dim: str = "freq_r",
        threshold: float = 0.5,
        space_scale: float = 1.0,
        wavelength_label: str = "Wavelength [units]",
        wavenumber_label: str = "Wavenumber [cycles / unit]",
        ylabel: str = "PSD score",
        score_var: str = "score",
        show_wavelength: bool = True,
        resolved_units: str | None = None,
        clip: bool = True,
        ylim: tuple[float, float] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.freq_dim = freq_dim
        self.threshold = float(threshold)
        self.space_scale = float(space_scale)
        self.wavelength_label = wavelength_label
        self.wavenumber_label = wavenumber_label
        self.ylabel = ylabel
        self.score_var = score_var
        self.show_wavelength = show_wavelength
        # Units string for the resolved-scale legend. If unset, parsed
        # from the trailing ``[unit]`` of ``wavelength_label`` (so
        # ``"Wavelength [km]"`` → ``"km"``); falls back to empty.
        self.resolved_units = resolved_units
        self.clip = bool(clip)
        if ylim is not None:
            ylim_t = tuple(ylim)
            if len(ylim_t) != 2:
                raise ValueError(
                    f"ylim must be a (ymin, ymax) 2-tuple; got length {len(ylim_t)}."
                )
            if ylim_t[0] > ylim_t[1]:
                raise ValueError(f"ylim must satisfy ymin <= ymax; got {ylim_t!r}.")
            self.ylim = ylim_t
        else:
            self.ylim = None

    def _default_title(self) -> str:
        return "Isotropic PSD Score"

    def _resolve_units(self) -> str:
        if self.resolved_units is not None:
            return self.resolved_units
        # Parse trailing "[unit]" from wavelength_label.
        import re

        match = re.search(r"\[([^\]]+)\]", self.wavelength_label)
        if match and match.group(1) != "units":
            return match.group(1)
        return ""

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        score: xr.DataArray | xr.Dataset,
    ) -> None:
        ax = axes
        da = _coerce_da(score, self.score_var)
        f = np.asarray(da[self.freq_dim].values)
        raw = np.asarray(da.values)
        # Clip to [0, 1] for display by default — score < 0 just means
        # the prediction is worse than zero at that scale and the
        # precise value isn't informative (matches oceanbench /
        # Ballarotta 2019 convention). ``clip=False`` exposes the
        # signed score for diagnostic deep-dives.
        vals = np.clip(raw, 0.0, 1.0) if self.clip else raw
        ax.plot(f, vals, color="C0")
        ax.axhline(self.threshold, color="k", linestyle="--", alpha=0.6)
        if not self.clip:
            ax.axhline(0.0, color="k", linestyle="-", alpha=0.3, linewidth=0.6)
        # Resolved scale: wavelength where score crosses threshold.
        nonzero = f > 0
        if nonzero.any():
            s_min = float(np.min(vals[nonzero]))
            s_max = float(np.max(vals[nonzero]))
            crosses = s_min < self.threshold < s_max
            wavelengths = 1.0 / (f[nonzero] * self.space_scale)
            if crosses:
                try:
                    resolved = find_intercept_1D(
                        x=wavelengths,
                        y=vals[nonzero],
                        level=self.threshold,
                    )
                    if np.isfinite(resolved) and resolved > 0:
                        units = self._resolve_units()
                        suffix = f" {units}" if units else ""
                        ax.axvline(
                            1.0 / (resolved * self.space_scale),
                            color="C3",
                            linestyle=":",
                            label=f"Resolved scale ≈ {resolved:.0f}{suffix}",
                        )
                        ax.legend(loc="best")
                except (ValueError, RuntimeError):
                    pass
        ax.set_xscale("log")
        ax.set_xlabel(self.wavenumber_label)
        ax.set_ylabel(self.ylabel)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
        elif self.clip:
            ax.set_ylim(0.0, 1.0)
        # else: matplotlib autoscale on raw values
        ax.grid(True, which="both", alpha=0.3)
        _format_log_plain(ax.xaxis)
        if self.show_wavelength:
            _wavelength_axis(
                ax,
                space_scale=self.space_scale,
                label=self.wavelength_label,
            )

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "freq_dim": self.freq_dim,
            "threshold": self.threshold,
            "space_scale": self.space_scale,
            "wavelength_label": self.wavelength_label,
            "wavenumber_label": self.wavenumber_label,
            "ylabel": self.ylabel,
            "score_var": self.score_var,
            "show_wavelength": self.show_wavelength,
            "resolved_units": self.resolved_units,
            "clip": self.clip,
            "ylim": list(self.ylim) if self.ylim is not None else None,
        }


class PSDSpaceTimePanel(_PSDPanelBase):
    """2-D space-time PSD with log-norm color scale.

    Renders ``|PSD|`` as a ``pcolormesh`` over
    ``(freq_space, freq_time)`` with a :class:`matplotlib.colors.LogNorm`
    colour scale, with optional twin axes converting to wavelength and
    period. Non-positive cells are masked before normalisation so
    ``LogNorm`` can't trip on zeros.

    Args:
        freq_space_dim: Spatial-frequency dim. Default ``"freq_lon"``.
        freq_time_dim: Temporal-frequency dim. Default ``"freq_time"``.
        space_scale: Multiplier converting freq units to inverse
            length used by the wavelength axis. Default ``1.0``.
        time_scale: Multiplier converting freq units to inverse time
            used by the period axis. Default ``1.0``.
        wavelength_label: Top-axis label.
        period_label: Right-axis label.
        wavenumber_label: Bottom-axis label.
        frequency_label: Left-axis label.
        cmap: Colormap. Default ``"RdYlBu_r"``.
        vmin: Optional log-norm lower limit.
        vmax: Optional log-norm upper limit.
        show_dual_axes: Toggle wavelength + period twin axes.
    """

    _default_axes_layout = (1, 1)

    def __init__(
        self,
        *,
        freq_space_dim: str = "freq_lon",
        freq_time_dim: str = "freq_time",
        space_scale: float = 1.0,
        time_scale: float = 1.0,
        wavelength_label: str = "Wavelength [units]",
        period_label: str = "Period [units]",
        wavenumber_label: str = "Wavenumber [cycles / unit]",
        frequency_label: str = "Frequency [cycles / unit]",
        cmap: str = "RdYlBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        show_dual_axes: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.freq_space_dim = freq_space_dim
        self.freq_time_dim = freq_time_dim
        self.space_scale = float(space_scale)
        self.time_scale = float(time_scale)
        self.wavelength_label = wavelength_label
        self.period_label = period_label
        self.wavenumber_label = wavenumber_label
        self.frequency_label = frequency_label
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.show_dual_axes = show_dual_axes

    def _default_title(self) -> str:
        return "Space-Time PSD"

    def _positive_slice(self, da: xr.DataArray) -> xr.DataArray:
        # Keep only positive frequencies on each axis for a log plot.
        sel: dict[str, Any] = {}
        for dim in (self.freq_space_dim, self.freq_time_dim):
            f = da[dim].values
            sel[dim] = f > 0
        return da.isel({d: m for d, m in sel.items()})

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        psd: xr.DataArray | xr.Dataset,
    ) -> None:
        ax = axes
        da = _coerce_da(psd)
        da = da.transpose(self.freq_time_dim, self.freq_space_dim)
        da = self._positive_slice(da)
        fs = np.asarray(da[self.freq_space_dim].values)
        ft = np.asarray(da[self.freq_time_dim].values)
        vals = np.abs(np.asarray(da.values))
        # Mask non-positive cells so LogNorm autoscaling can't see vmin <= 0.
        vals_masked = np.ma.masked_where(~np.isfinite(vals) | (vals <= 0), vals)
        vmin = self.vmin
        if vmin is None and vals_masked.count():
            vmin = float(vals_masked.min())
        norm = colors.LogNorm(vmin=vmin, vmax=self.vmax)
        im = ax.pcolormesh(
            fs, ft, vals_masked, cmap=self.cmap, norm=norm, shading="auto"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(self.wavenumber_label)
        ax.set_ylabel(self.frequency_label)
        fig.colorbar(im, ax=ax, label="PSD")
        _format_log_plain(ax.xaxis)
        _format_log_plain(ax.yaxis)
        if self.show_dual_axes:
            _wavelength_axis(
                ax, space_scale=self.space_scale, label=self.wavelength_label
            )
            _period_axis(ax, time_scale=self.time_scale, label=self.period_label)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "freq_space_dim": self.freq_space_dim,
            "freq_time_dim": self.freq_time_dim,
            "space_scale": self.space_scale,
            "time_scale": self.time_scale,
            "wavelength_label": self.wavelength_label,
            "period_label": self.period_label,
            "wavenumber_label": self.wavenumber_label,
            "frequency_label": self.frequency_label,
            "cmap": self.cmap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "show_dual_axes": self.show_dual_axes,
        }


class PSDSpaceTimeScorePanel(PSDSpaceTimePanel):
    """2-D space-time PSD score with threshold contour.

    Same axes layout as :class:`PSDSpaceTimePanel`, but renders score
    (linear ``[0, 1]``) and overlays a contour line at ``threshold``
    marking the resolved boundary.

    Args:
        threshold: Score threshold contour. Default ``0.5``.
        score_var: Variable name when input is a Dataset. Default
            ``"score"``.
        levels: Filled-contour levels. Default ``np.linspace(0, 1, 11)``
            when ``clip=True``; ignored (matplotlib auto-picks) when
            ``clip=False`` and ``levels`` is unset.
        cmap: Colormap. Default ``"RdYlGn"``.
        clip: When ``True`` (default), display the score clipped to
            ``[0, 1]`` — matches the oceanbench / Ballarotta 2019
            convention. ``False`` exposes signed values for diagnostic
            mode; the colour scale and contour levels then auto-stretch
            to the data range unless ``levels`` is given.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        score_var: str = "score",
        levels: np.ndarray | None = None,
        cmap: str = "RdYlGn",
        clip: bool = True,
        **kw: Any,
    ) -> None:
        # Replace cmap default and forward.
        super().__init__(cmap=cmap, **kw)
        self.threshold = float(threshold)
        self.score_var = score_var
        self.clip = bool(clip)
        # Only pin levels when clipping. With clip=False we let
        # contourf pick a sensible range from the data.
        self._levels_user = None if levels is None else np.asarray(levels)
        self.levels = (
            self._levels_user
            if self._levels_user is not None
            else (np.linspace(0, 1, 11) if self.clip else None)
        )

    def _default_title(self) -> str:
        return "Space-Time PSD Score"

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        psd: xr.DataArray | xr.Dataset,
    ) -> None:
        ax = axes
        da = _coerce_da(psd, self.score_var)
        da = da.transpose(self.freq_time_dim, self.freq_space_dim)
        da = self._positive_slice(da)
        fs = np.asarray(da[self.freq_space_dim].values)
        ft = np.asarray(da[self.freq_time_dim].values)
        # Clip to [0, 1] for display by default (see PSDIsotropicScorePanel).
        raw = np.asarray(da.values)
        vals = np.clip(raw, 0.0, 1.0) if self.clip else raw
        contour_kw: dict[str, Any] = {"cmap": self.cmap}
        if self.levels is not None:
            contour_kw["levels"] = self.levels
        cf = ax.contourf(fs, ft, vals, **contour_kw)
        ax.contour(fs, ft, vals, levels=[self.threshold], colors="k", linewidths=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(self.wavenumber_label)
        ax.set_ylabel(self.frequency_label)
        fig.colorbar(cf, ax=ax, label="PSD score")
        _format_log_plain(ax.xaxis)
        _format_log_plain(ax.yaxis)
        if self.show_dual_axes:
            _wavelength_axis(
                ax, space_scale=self.space_scale, label=self.wavelength_label
            )
            _period_axis(ax, time_scale=self.time_scale, label=self.period_label)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "threshold": self.threshold,
            "score_var": self.score_var,
            "levels": (
                None
                if self._levels_user is None
                else list(map(float, self._levels_user))
            ),
            "clip": self.clip,
        }


__all__ = [
    "PSDIsotropicPanel",
    "PSDIsotropicScorePanel",
    "PSDSpaceTimePanel",
    "PSDSpaceTimeScorePanel",
]
