"""Rotary-spectrum validation panels."""

from __future__ import annotations

from typing import Any

import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz.validation._src.psd import _PSDPanelBase


class RotaryPolarizationPanel(_PSDPanelBase):
    """Heatmap of rotary polarization over wavenumber and latitude.

    Inherits the constrained-layout / per-axes title pattern from
    :class:`_PSDPanelBase` so the optional top wavelength axis doesn't
    collide with the figure title.
    """

    _default_axes_layout = (1, 1)

    def __init__(
        self,
        *,
        var: str = "polarization",
        wavenumber_dim: str = "wavenumber",
        y_dim: str = "lat",
        cmap: str = "RdBu_r",
        vmin: float = -1.0,
        vmax: float = 1.0,
        wavelength_axis: bool = True,
        space_scale: float = 1.0,
        wavelength_label: str = "Wavelength [km]",
        figsize: tuple[float, float] = (6, 8),
        **kw: Any,
    ) -> None:
        super().__init__(figsize=figsize, **kw)
        if vmin >= vmax:
            raise ValueError(f"vmin must be less than vmax; got {vmin=} and {vmax=}.")
        self.var = var
        self.wavenumber_dim = wavenumber_dim
        self.y_dim = y_dim
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.wavelength_axis = bool(wavelength_axis)
        # ``space_scale`` multiplies 1/k to convert into the units used in
        # ``wavelength_label``. xrft outputs cycles/<coord-unit>, so for
        # coords in metres + a "[km]" label this is 1e-3; for coords
        # already in km, leave it at the default 1.0.
        self.space_scale = float(space_scale)
        self.wavelength_label = str(wavelength_label)

    def _default_title(self) -> str:
        return "Rotary Polarization"

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        spectrum: xr.Dataset | xr.DataArray,
    ) -> None:
        da = spectrum[self.var] if isinstance(spectrum, xr.Dataset) else spectrum
        da = da.transpose(self.y_dim, self.wavenumber_dim)
        ax = axes
        mesh = ax.pcolormesh(
            da[self.wavenumber_dim].values,
            da[self.y_dim].values,
            da.values,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            shading="auto",
        )
        fig.colorbar(mesh, ax=ax, label=self.var)
        ax.set_xlabel(self.wavenumber_dim)
        ax.set_ylabel(self.y_dim)
        if self.wavelength_axis:
            scale = self.space_scale
            secax = ax.secondary_xaxis(
                "top",
                functions=(
                    lambda k: _safe_reciprocal(k) * scale,
                    lambda lam: _safe_reciprocal(lam) * scale,
                ),
            )
            secax.set_xlabel(self.wavelength_label)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "var": self.var,
            "wavenumber_dim": self.wavenumber_dim,
            "y_dim": self.y_dim,
            "cmap": self.cmap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "wavelength_axis": self.wavelength_axis,
            "space_scale": self.space_scale,
            "wavelength_label": self.wavelength_label,
        }


def _safe_reciprocal(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.where(arr == 0.0, np.nan, arr)


__all__ = ["RotaryPolarizationPanel"]
