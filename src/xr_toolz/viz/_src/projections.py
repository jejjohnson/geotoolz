"""Cartopy projection presets for ocean-paper viz panels.

Standardises basin-specific cartopy boilerplate that every ocean-paper
notebook reproduces (Gulf Stream, North Atlantic, Kuroshio, ...) so a
panel call like ``SpatialMapPanel(projection="gulf_stream")`` applies
the right ``set_extent`` + projection without the user memorising
lat/lon bounds.

Cartopy is a hard dependency of ``xr_toolz``; importing this module
imports cartopy.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt


# Each preset: cartopy CRS class name + optional (lon_min, lon_max,
# lat_min, lat_max) extent. Extent is interpreted in PlateCarree.
PRESETS: dict[str, dict[str, Any]] = {
    "global": {"projection": "Robinson", "extent": None},
    "north_atlantic": {
        "projection": "PlateCarree",
        "extent": (-80, 0, 10, 65),
    },
    "gulf_stream": {
        "projection": "PlateCarree",
        "extent": (-80, -50, 30, 45),
    },
    "kuroshio": {
        "projection": "PlateCarree",
        "extent": (130, 180, 25, 45),
    },
    "mediterranean": {
        "projection": "PlateCarree",
        "extent": (-6, 36, 30, 46),
    },
}


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
