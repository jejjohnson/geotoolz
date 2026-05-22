"""Spatial-snapshot validation panel.

:class:`SpatialMapPanel` renders a single 2-D ``(lat, lon)`` snapshot
with optional cartopy projection, coastlines, and gridlines. The
default colormap auto-resolves from the
:data:`xrtoolz.types.REGISTRY` via
:func:`xrtoolz.viz.cmap_for` so common variables pick the
field-appropriate convention without manual ``cmap=`` kwargs (e.g.
SSH → ``RdBu_r``, SST → ``RdYlBu_r``).
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import matplotlib.figure as mpl_figure
import numpy as np
import xarray as xr

from xrtoolz.viz._src.cmaps import cmap_for
from xrtoolz.viz._src.projections import PRESETS, _resolve_projection
from xrtoolz.viz.validation._src.base import _NullContext, _ValidationPanel


class SpatialMapPanel(_ValidationPanel):
    """Single 2-D ``(lat, lon)`` snapshot with optional cartopy projection.

    Args:
        var: Data-variable name when input is a Dataset. ``None``
            (default) auto-picks the first ``data_var``. Also used as
            the lookup key for :func:`xrtoolz.viz.cmap_for` when
            ``cmap`` is unset.
        time_index: When the input has a ``time`` dim, ``isel(time=…)``
            this index before plotting. Default ``0`` (first snapshot).
            Set to ``None`` to skip selection (useful for inputs that
            were already pre-reduced).
        cmap: Matplotlib colormap. ``None`` (default) auto-resolves
            from the variable registry; falls back to ``"viridis"`` for
            unknown names.
        vmin, vmax: Optional colour-scale limits.
        projection: Preset name (``"global"``, ``"north_atlantic"``,
            ``"gulf_stream"``, ``"kuroshio"``, ``"mediterranean"``), a
            cartopy class name, a cartopy CRS instance, or ``None`` for
            plain matplotlib axes (no cartopy).
        coastlines: Add :meth:`cartopy.mpl.geoaxes.GeoAxes.coastlines`.
            Ignored when ``projection`` is ``None``. Default ``True``.
        gridlines: Add :meth:`cartopy.mpl.geoaxes.GeoAxes.gridlines`
            with labelled lat/lon ticks. Ignored without a projection.
            Default ``True``.
        cbar_label: Colorbar label. Empty string suppresses it. Default
            ``""``.
        lon: Longitude coord name. Default ``"lon"``.
        lat: Latitude coord name. Default ``"lat"``.
    """

    _default_axes_layout = (1, 1)

    def __init__(
        self,
        *,
        var: str | None = None,
        time_index: int | None = 0,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        projection: str | ccrs.Projection | None = None,
        coastlines: bool = True,
        gridlines: bool = True,
        cbar_label: str = "",
        lon: str = "lon",
        lat: str = "lat",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.var = var
        self.time_index = time_index
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.projection = projection
        self.coastlines = bool(coastlines)
        self.gridlines = bool(gridlines)
        self.cbar_label = cbar_label
        self.lon = lon
        self.lat = lat

    def _default_title(self) -> str:
        return self.var or "Snapshot"

    def _make_fig_axes(self) -> tuple[mpl_figure.Figure, Any]:
        # Override so we can route through cartopy when a projection is set.
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=self.figsize)
        crs = _resolve_projection(self.projection)
        if crs is None:
            ax = fig.add_subplot(1, 1, 1)
            return fig, ax
        ax = fig.add_subplot(1, 1, 1, projection=crs)
        if isinstance(self.projection, str) and self.projection in PRESETS:
            extent = PRESETS[self.projection]["extent"]
            if extent is not None:
                # ax is a cartopy GeoAxes when a projection is set.
                ax.set_extent(extent, crs=ccrs.PlateCarree())  # ty: ignore[unresolved-attribute]
        return fig, ax

    def _select_var(self, obj: xr.DataArray | xr.Dataset) -> xr.DataArray:
        if isinstance(obj, xr.Dataset):
            name = self.var or next(iter(obj.data_vars))
            return obj[name]
        return obj

    def _select_time(self, da: xr.DataArray) -> xr.DataArray:
        if self.time_index is None or "time" not in da.dims:
            return da
        return da.isel(time=self.time_index)

    def _build(
        self,
        fig: mpl_figure.Figure,
        axes: Any,
        snapshot: xr.DataArray | xr.Dataset,
    ) -> None:
        ax = axes
        da = self._select_var(snapshot)
        da = self._select_time(da)
        cmap = self.cmap if self.cmap is not None else cmap_for(self.var)
        lon = np.asarray(da[self.lon].values)
        lat = np.asarray(da[self.lat].values)
        vals = np.asarray(da.values)
        pcm_kw: dict[str, Any] = {
            "cmap": cmap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "shading": "auto",
        }
        if self.projection is not None:
            pcm_kw["transform"] = ccrs.PlateCarree()
        im = ax.pcolormesh(lon, lat, vals, **pcm_kw)
        if self.projection is not None:
            if self.coastlines:
                ax.coastlines(resolution="50m", linewidth=0.6)
            if self.gridlines:
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label=self.cbar_label)

    def _apply(self, *args: Any, **kwargs: Any) -> mpl_figure.Figure:
        # Mirrors _ValidationPanel._apply but with the title pinned to
        # the axes (avoids overlapping with cartopy gridline labels).
        import matplotlib.pyplot as plt

        ctx = (
            plt.style.context(self.style) if self.style is not None else _NullContext()
        )
        with ctx:
            fig, axes = self._make_fig_axes()
            self._build(fig, axes, *args, **kwargs)
            title = self.title if self.title is not None else self._default_title()
            if title:
                axes.set_title(title)
            fig.tight_layout()
        self._maybe_save(fig)
        self._maybe_show(fig)
        return fig

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "var": self.var,
            "time_index": self.time_index,
            "cmap": self.cmap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "projection": (
                self.projection
                if isinstance(self.projection, str | type(None))
                else repr(self.projection)
            ),
            "coastlines": self.coastlines,
            "gridlines": self.gridlines,
            "cbar_label": self.cbar_label,
            "lon": self.lon,
            "lat": self.lat,
        }


__all__ = ["SpatialMapPanel"]
