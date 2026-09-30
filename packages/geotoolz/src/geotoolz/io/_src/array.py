"""Tier-A primitives for :mod:`geotoolz.io` -- array and metadata decoding.

Format-agnostic helpers the multi-format readers (:class:`ReadHDF`,
:class:`ReadNetCDF`) apply to what h5py / pyhdf / netCDF4 hand back:
band selection with rasterio's 1-based ``indexes`` convention, CF / HDF
fill-value lookup, and GDAL ``GeoTransform`` parsing. No ``GeoTensor``
and no file handles here beyond an array-like dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from affine import Affine

from geotoolz.io._src.errors import GeoToolzIOError


def select_indexes(values: Any, indexes: list[int] | None) -> np.ndarray:
    """Select 1-based band ``indexes`` along the leading axis of ``values``.

    Args:
        values: Array-like ``(C, ...)`` stack, or a 2-D single-layer map.
        indexes: 1-based band numbers (rasterio convention), or ``None``
            for every band. A 2-D map accepts only ``[1]`` (a no-op).

    Returns:
        The selected bands (``values`` itself when ``indexes`` is
        ``None``).

    Raises:
        GeoToolzIOError: If an index is not positive, or ``values`` has
            no band axis and ``indexes`` is not ``[1]``.

    Examples:
        >>> select_indexes(np.arange(12).reshape(3, 2, 2), [3, 1])[:, 0, 0]
        array([8, 0])
    """
    array = np.asanyarray(values)
    if indexes is None:
        return array
    zero_based = [index - 1 for index in indexes]
    if any(index < 0 for index in zero_based):
        raise GeoToolzIOError("indexes are 1-based and must be positive.")
    if array.ndim < 3:
        # Treat a non-band dataset as a single-layer raster: indexes=[1]
        # selects the only layer and is a no-op; anything else is invalid.
        if zero_based != [0]:
            raise GeoToolzIOError("indexes require a dataset with a leading band axis.")
        return array
    return np.take(array, zero_based, axis=0)


def read_indexes(source: Any, indexes: list[int] | None) -> np.ndarray:
    """Read 1-based band ``indexes`` from a lazily sliced dataset.

    Same contract as :func:`select_indexes`, but ``source`` is sliced
    (an h5py-style hyperslab read) instead of materialised first, so
    requesting one band reads only that band off disk. h5py only
    fancy-indexes the leading axis with increasing indices, so the read is
    sorted and then reordered.

    Args:
        source: Dataset supporting ``source[...]`` and
            ``source[list_of_ints]`` (e.g. ``h5py.Dataset``).
        indexes: 1-based band numbers, or ``None`` for every band.

    Returns:
        The selected bands.

    Raises:
        GeoToolzIOError: See :func:`select_indexes`.

    Examples:
        >>> read_indexes(np.arange(12).reshape(3, 2, 2), [3, 1])[:, 0, 0]
        array([8, 0])
    """
    if indexes is None:
        return np.asanyarray(source[...])
    zero_based = [index - 1 for index in indexes]
    if any(index < 0 for index in zero_based):
        raise GeoToolzIOError("indexes are 1-based and must be positive.")
    ndim = getattr(source, "ndim", None)
    if ndim is None:
        ndim = np.asarray(source.shape).size
    if ndim < 3:
        if zero_based != [0]:
            raise GeoToolzIOError("indexes require a dataset with a leading band axis.")
        return np.asanyarray(source[...])
    order = np.argsort(zero_based)
    sorted_indexes = [zero_based[i] for i in order]
    sliced = np.asanyarray(source[sorted_indexes])
    if list(order) == list(range(len(order))):
        return sliced
    inverse = np.argsort(order)
    return sliced[inverse]


def fill_value_from_attrs(attrs: dict[str, Any]) -> Any:
    """Return the scalar fill value declared in ``attrs`` (default ``0``).

    Looks up ``_FillValue``, ``missing_value``, ``fill_value`` and
    ``nodata`` in that order. HDF5 attributes written by netCDF4/xarray
    are ``(1,)``-shaped arrays (lists after
    :func:`~geotoolz._src.config.jsonable`); size-1 values are unwrapped to
    a scalar and multi-element values are skipped, since a ``GeoTensor``
    fill must be a scalar.

    Args:
        attrs: Dataset / variable attributes.

    Returns:
        The fill value, or ``0`` when none is declared.

    Examples:
        >>> fill_value_from_attrs({"_FillValue": [-9999]})
        -9999
        >>> fill_value_from_attrs({})
        0
    """
    for name in ("_FillValue", "missing_value", "fill_value", "nodata"):
        if name not in attrs:
            continue
        value = attrs[name]
        if isinstance(value, list | tuple | np.ndarray):
            flat = np.ravel(np.asarray(value))
            if flat.size != 1:
                continue
            value = flat[0].item()
        return value
    return 0


def affine_from_geotransform(geotransform: Any) -> Affine:
    """Parse a GDAL ``GeoTransform`` attribute into an :class:`~affine.Affine`.

    Args:
        geotransform: Six whitespace-separated numbers
            ``"x0 dx rx y0 ry dy"`` (GDAL order), or ``None``.

    Returns:
        The affine, or the identity when ``geotransform`` is ``None`` or
        does not hold exactly six numbers.

    Examples:
        >>> affine_from_geotransform("100 10 0 200 0 -10")
        Affine(10.0, 0.0, 100.0,
               0.0, -10.0, 200.0)
    """
    if geotransform is None:
        return Affine.identity()
    parts = [float(part) for part in str(geotransform).split()]
    if len(parts) != 6:
        return Affine.identity()
    return Affine.from_gdal(*parts)


__all__ = [
    "affine_from_geotransform",
    "fill_value_from_attrs",
    "read_indexes",
    "select_indexes",
]
