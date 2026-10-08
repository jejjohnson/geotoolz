"""The geostationary fixed grid shared by geostationary imagers.

GOES ABI, MTG FCI, Himawari AHI (and SEVIRI once decoded) all sample a
fixed grid of scan angles. In CF terms the file carries a ``geostationary``
grid mapping (``perspective_point_height``, ``semi_major_axis``,
``semi_minor_axis``, ``longitude_of_projection_origin``,
``sweep_angle_axis``) and 1-D ``x`` / ``y`` pixel-centre scan angles in
radians. Scaled by the perspective-point height those angles are metres
in ``+proj=geos``, evenly spaced, so the grid is a clean affine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from geoproducts._src.hdf import attr, unpacked


__all__ = ["FixedGrid", "geos_crs", "scan_angle_transform"]


def geos_crs(
    *,
    lon_0: float,
    height_m: float,
    semi_major_m: float,
    semi_minor_m: float,
    sweep: str = "x",
) -> CRS:
    """The ``+proj=geos`` CRS of a geostationary view.

    ``sweep`` is always spelled out: GOES ABI sweeps along ``x``, MTG FCI and
    SEVIRI along ``y`` (proj's default).

    Examples:
        >>> crs = geos_crs(lon_0=-75.0, height_m=35786023.0,
        ...                semi_major_m=6378137.0, semi_minor_m=6356752.31414)
        >>> "+sweep=x" in crs.to_wkt()
        True
    """
    return CRS.from_proj4(
        f"+proj=geos +lon_0={float(lon_0)} +h={float(height_m)}"
        f" +a={float(semi_major_m)} +b={float(semi_minor_m)}"
        f" +sweep={sweep} +units=m +no_defs"
    )


def scan_angle_transform(
    x_rad: np.ndarray, y_rad: np.ndarray, height_m: float
) -> Affine:
    """The affine transform of evenly spaced pixel-centre scan angles.

    Args:
        x_rad: Column-centre scan angles (radians), length ``W >= 2``.
        y_rad: Row-centre scan angles (radians), length ``H >= 2``.
        height_m: Perspective-point height (metres).

    Returns:
        Transform mapping pixel corners to ``+proj=geos`` metres.
    """

    def edges(centres: np.ndarray) -> tuple[float, float]:
        half = (centres[-1] - centres[0]) / (len(centres) - 1) / 2.0
        return float((centres[0] - half) * height_m), float(
            (centres[-1] + half) * height_m
        )

    left, right = edges(np.asarray(x_rad, dtype=np.float64))
    top, bottom = edges(np.asarray(y_rad, dtype=np.float64))
    n_rows, n_cols = len(y_rad), len(x_rad)
    return Affine((right - left) / n_cols, 0.0, left, 0.0, (bottom - top) / n_rows, top)


@dataclass(frozen=True)
class FixedGrid:
    """A file's geostationary grid: shape, affine transform and CRS."""

    height: int
    width: int
    transform: Affine
    crs: CRS

    @classmethod
    def from_cf(
        cls, f: Any, grid_mapping: str, *, x: str = "x", y: str = "y"
    ) -> FixedGrid:
        """Read the grid from a CF ``geostationary`` grid mapping and scan angles.

        Args:
            f: Open HDF5 / NetCDF-4 file (or group).
            grid_mapping: Name of the grid-mapping variable
                (``"goes_imager_projection"`` for ABI).
            x: Name of the column scan-angle coordinate.
            y: Name of the row scan-angle coordinate.

        Raises:
            ValueError: The file lacks the grid mapping or the coordinates.
        """
        if grid_mapping not in f or x not in f or y not in f:
            raise ValueError(
                f"not a geostationary fixed-grid file: expected the {x!r} / {y!r} "
                f"scan angles and the {grid_mapping!r} grid mapping."
            )
        proj = f[grid_mapping]
        height_m = float(attr(proj, "perspective_point_height"))
        x_rad, y_rad = unpacked(f, x), unpacked(f, y)
        return cls(
            height=len(y_rad),
            width=len(x_rad),
            transform=scan_angle_transform(x_rad, y_rad, height_m),
            crs=geos_crs(
                lon_0=float(attr(proj, "longitude_of_projection_origin")),
                height_m=height_m,
                semi_major_m=float(attr(proj, "semi_major_axis")),
                semi_minor_m=float(attr(proj, "semi_minor_axis")),
                sweep=str(attr(proj, "sweep_angle_axis", "x")),
            ),
        )
