"""The geostationary fixed grid shared by geostationary imagers.

GOES ABI, MTG FCI, Himawari AHI (and SEVIRI once decoded) all sample a
fixed grid of scan angles. In CF terms the file carries a ``geostationary``
grid mapping (``perspective_point_height``, ``semi_major_axis``,
``semi_minor_axis``, ``longitude_of_projection_origin``,
``sweep_angle_axis``) and 1-D ``x`` / ``y`` pixel-centre scan angles in
radians; binary formats (HSD, HRIT) give the same grid as the CGMS
``CFAC`` / ``LFAC`` / ``COFF`` / ``LOFF`` scaling. Scaled by the
perspective-point height those angles are metres in ``+proj=geos``, evenly
spaced, so the grid is a clean affine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from geoproducts._src import hdf


__all__ = ["FixedGrid", "geos_crs", "on_earth", "scan_angle_transform"]


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


def on_earth(
    x_rad: np.ndarray,
    y_rad: np.ndarray,
    *,
    height_m: float,
    semi_major_m: float,
    semi_minor_m: float,
    sweep: str = "x",
) -> np.ndarray:
    """Whether the line of sight at each scan angle meets the Earth.

    A geostationary imager also samples space around the disk; those
    pixels have no geolocation. The line of sight from the satellite (at
    ``semi_major_m + height_m`` from the Earth's centre, over the equator)
    meets the ellipsoid exactly when its intersection quadratic has a real
    root.

    Args:
        x_rad: Column scan angles (radians), broadcastable with ``y_rad``.
        y_rad: Row scan angles (radians), positive north.
        height_m: Satellite height above the equator (metres).
        semi_major_m: Equatorial radius (metres).
        semi_minor_m: Polar radius (metres).
        sweep: Scan sweep axis, ``"x"`` (GOES ABI) or ``"y"`` (AHI, FCI,
            SEVIRI).

    Returns:
        Boolean array, ``True`` where the pixel views the Earth.

    Examples:
        >>> on_earth(np.array([0.0, 0.16]), np.array([0.0, 0.0]),
        ...          height_m=35786023.0, semi_major_m=6378137.0,
        ...          semi_minor_m=6356752.31414).tolist()
        [True, False]
    """
    x, y = np.broadcast_arrays(np.asarray(x_rad, float), np.asarray(y_rad, float))
    # Unit line of sight (towards the Earth = +d1) in the satellite frame.
    if sweep == "x":
        d1, d2, d3 = np.cos(x) * np.cos(y), np.sin(x), np.cos(x) * np.sin(y)
    else:
        d1, d2, d3 = np.cos(x) * np.cos(y), np.sin(x) * np.cos(y), np.sin(y)
    distance = semi_major_m + height_m
    a2, b2 = semi_major_m**2, semi_minor_m**2
    quad = (d1**2 + d2**2) / a2 + d3**2 / b2
    half_lin = distance * d1 / a2
    const = distance**2 / a2 - 1.0
    return half_lin**2 - quad * const >= 0.0


@dataclass(frozen=True)
class FixedGrid:
    """A file's geostationary grid: shape, affine transform and projection.

    Attributes:
        height: Grid rows.
        width: Grid columns.
        transform: Pixel corners → ``+proj=geos`` metres.
        lon_0: Sub-satellite longitude (degrees east).
        height_m: Satellite height above the equator (metres).
        semi_major_m: Equatorial radius (metres).
        semi_minor_m: Polar radius (metres).
        sweep: Scan sweep axis, ``"x"`` (GOES ABI) or ``"y"``.
    """

    height: int
    width: int
    transform: Affine
    lon_0: float
    height_m: float
    semi_major_m: float
    semi_minor_m: float
    sweep: str

    @property
    def crs(self) -> CRS:
        """The grid's ``+proj=geos`` CRS."""
        return geos_crs(
            lon_0=self.lon_0,
            height_m=self.height_m,
            semi_major_m=self.semi_major_m,
            semi_minor_m=self.semi_minor_m,
            sweep=self.sweep,
        )

    @classmethod
    def from_cgms(
        cls,
        *,
        columns: int,
        lines: int,
        cfac: float,
        lfac: float,
        coff: float,
        loff: float,
        lon_0: float,
        height_m: float,
        semi_major_m: float,
        semi_minor_m: float,
    ) -> FixedGrid:
        """The grid of the CGMS normalised geostationary projection (sweep ``y``).

        Column ``c`` and line ``l`` (1-based) are centred at the scan angles
        ``(c - COFF) · 2¹⁶ / CFAC`` and ``-(l - LOFF) · 2¹⁶ / LFAC`` degrees
        (lines run north to south).

        Args:
            columns: Grid width.
            lines: Grid height.
            cfac: Column scaling factor.
            lfac: Line scaling factor.
            coff: Column offset (1-based).
            loff: Line offset (1-based).
            lon_0: Sub-satellite longitude (degrees east).
            height_m: Satellite height above the equator (metres).
            semi_major_m: Equatorial radius (metres).
            semi_minor_m: Polar radius (metres).

        Examples:
            >>> grid = FixedGrid.from_cgms(
            ...     columns=5500, lines=5500, cfac=20466275, lfac=20466275,
            ...     coff=2750.5, loff=2750.5, lon_0=140.7, height_m=35785863.0,
            ...     semi_major_m=6378137.0, semi_minor_m=6356752.3)
            >>> round(grid.transform.a), round(grid.transform.c / 1000)
            (2000, -5500)
        """
        dx = float(np.deg2rad(2.0**16 / cfac)) * height_m
        dy = float(np.deg2rad(2.0**16 / lfac)) * height_m
        # Pixel centres sit at (c - COFF) / -(l - LOFF) steps; the grid's
        # outer edge is half a pixel further out.
        return cls(
            height=lines,
            width=columns,
            transform=Affine(dx, 0.0, (0.5 - coff) * dx, 0.0, -dy, (loff - 0.5) * dy),
            lon_0=float(lon_0),
            height_m=float(height_m),
            semi_major_m=float(semi_major_m),
            semi_minor_m=float(semi_minor_m),
            sweep="y",
        )

    def on_earth(self, rows: slice, cols: slice) -> np.ndarray:
        """Whether each pixel of ``[rows, cols]`` views the Earth.

        See :func:`on_earth`.

        Args:
            rows: Grid rows (0-based, step 1; may extend past the grid).
            cols: Grid columns (0-based, step 1).

        Returns:
            Boolean ``(len(rows), len(cols))`` array.
        """
        t = self.transform
        x = t.c + t.a * (np.arange(cols.start, cols.stop) + 0.5)
        y = t.f + t.e * (np.arange(rows.start, rows.stop) + 0.5)
        return on_earth(
            x[None, :] / self.height_m,
            y[:, None] / self.height_m,
            height_m=self.height_m,
            semi_major_m=self.semi_major_m,
            semi_minor_m=self.semi_minor_m,
            sweep=self.sweep,
        )

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
        height_m = float(hdf.attr(proj, "perspective_point_height"))
        x_rad, y_rad = hdf.unpacked(f, x), hdf.unpacked(f, y)
        return cls(
            height=len(y_rad),
            width=len(x_rad),
            transform=scan_angle_transform(x_rad, y_rad, height_m),
            lon_0=float(hdf.attr(proj, "longitude_of_projection_origin")),
            height_m=height_m,
            semi_major_m=float(hdf.attr(proj, "semi_major_axis")),
            semi_minor_m=float(hdf.attr(proj, "semi_minor_axis")),
            sweep=str(hdf.attr(proj, "sweep_angle_axis", "x")),
        )
