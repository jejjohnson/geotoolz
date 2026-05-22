"""Coordinate-space range conversions for longitude and latitude."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lon_360_to_180(coord: ArrayLike) -> NDArray:
    """Wrap longitudes from ``[0, 360)`` into ``[-180, 180)``.

    Args:
        coord: Array of longitude values in the ``[0, 360)`` convention.

    Returns:
        Array of longitude values in the ``[-180, 180)`` convention, with
        the same shape as the input.
    """
    return (np.asarray(coord) + 180.0) % 360.0 - 180.0


def lon_180_to_360(coord: ArrayLike) -> NDArray:
    """Wrap longitudes from ``[-180, 180)`` into ``[0, 360)``.

    Args:
        coord: Array of longitude values in the ``[-180, 180)`` convention.

    Returns:
        Array of longitude values in the ``[0, 360)`` convention, with the
        same shape as the input.
    """
    return np.asarray(coord) % 360.0


def lat_180_to_90(coord: ArrayLike) -> NDArray:
    """Wrap latitudes from ``[0, 180)`` into ``[-90, 90)``.

    Useful when a dataset stores latitude as a 0-based index rather than
    the standard geographic convention.

    Args:
        coord: Array of latitude values in the ``[0, 180)`` convention.

    Returns:
        Array of latitude values in the ``[-90, 90)`` convention.
    """
    return (np.asarray(coord) + 90.0) % 180.0 - 90.0


def lat_90_to_180(coord: ArrayLike) -> NDArray:
    """Wrap latitudes from ``[-90, 90)`` into ``[0, 180)``.

    Args:
        coord: Array of latitude values in the ``[-90, 90)`` convention.

    Returns:
        Array of latitude values in the ``[0, 180)`` convention.
    """
    return np.asarray(coord) % 180.0


__all__ = [
    "lat_90_to_180",
    "lat_180_to_90",
    "lon_180_to_360",
    "lon_360_to_180",
]
