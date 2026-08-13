"""Cartopy projection presets for ocean-paper viz panels.

Standardises basin-specific cartopy boilerplate that every ocean-paper
notebook reproduces (Gulf Stream, North Atlantic, Kuroshio, ...) so a
panel call like ``SpatialMapPanel(projection="gulf_stream")`` applies
the right ``set_extent`` + projection without the user memorising
lat/lon bounds.

Cartopy is a hard dependency of ``xrtoolz``; importing this module
imports cartopy.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt

from xrtoolz.geo.regions import REGIONS


# Each preset: cartopy CRS class name + optional (lon_min, lon_max,
# lat_min, lat_max) extent. Extent is interpreted in PlateCarree.
def _build_presets() -> dict[str, dict[str, Any]]:
    presets: dict[str, dict[str, Any]] = {}
    for region_id, spec in REGIONS.items():
        lon_min, lat_min, lon_max, lat_max = spec.regions.bounds_global
        presets[region_id] = {
            "projection": spec.projection,
            "extent": None
            if region_id == "global"
            else (float(lon_min), float(lon_max), float(lat_min), float(lat_max)),
        }
    return presets


PRESETS: dict[str, dict[str, Any]] = _build_presets()


def _resolve_projection(spec: str | ccrs.Projection | None) -> ccrs.Projection | None:
    """Coerce ``spec`` to a cartopy CRS, looking up presets by name."""
    if spec is None or isinstance(spec, ccrs.Projection):
        return spec
    cls_name = PRESETS[spec]["projection"] if spec in PRESETS else spec
    cls = getattr(ccrs, cls_name, None)
    if cls is None:
        raise ValueError(
            f"Unknown projection {spec!r}. "
            f"Pass a preset name from {sorted(PRESETS)}, "
            "a cartopy class name (e.g. 'PlateCarree'), or a cartopy "
            "CRS instance."
        )
    return cls()


def make_axes(
    projection: str | ccrs.Projection | None = None,
    *,
    fig: mpl_figure.Figure | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> tuple[mpl_figure.Figure, Any]:
    """Create a Figure + Axes pair, applying a preset extent if requested.

    Args:
        projection: Preset name (one of :data:`PRESETS`), a cartopy
            class name (``"PlateCarree"``), an instantiated cartopy
            :class:`~cartopy.crs.Projection`, or ``None`` for plain
            matplotlib axes.
        fig: Optional pre-existing Figure to draw into. When ``None`` a
            new Figure is created with ``figsize``.
        figsize: Figure size when ``fig`` is ``None``.

    Returns:
        ``(Figure, Axes)``. With a preset, the Axes is a
        :class:`cartopy.mpl.geoaxes.GeoAxes` with ``set_extent`` already
        applied. With ``projection=None``, plain matplotlib axes.
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    crs = _resolve_projection(projection)
    if crs is None:
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax

    ax = fig.add_subplot(1, 1, 1, projection=crs)
    if isinstance(projection, str) and projection in PRESETS:
        extent = PRESETS[projection]["extent"]
        if extent is not None:
            # ax is a cartopy GeoAxes when a projection is set.
            ax.set_extent(extent, crs=ccrs.PlateCarree())  # ty: ignore[unresolved-attribute]
    return fig, ax


__all__ = ["PRESETS", "make_axes"]
