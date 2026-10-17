"""The georeader adapter: `GeoTensor` ↔ `GeoArray`, NumPy only.

georeader is imported here and nowhere else in geoarray (the ``[georeader]``
extra), and only when a function runs. The adapter maps public attributes
one to one; it shares no code with georeader.

| GeoTensor | GeoArray |
| --- | --- |
| ``values`` (``np.ndarray`` view) | ``data`` (the same memory) |
| ``transform`` (``affine.Affine``) | ``transform`` ((6,) float64) |
| ``crs`` | ``crs`` (``pyproj.CRS``) |
| ``fill_value_default`` | ``fill_value`` |
| ``attrs`` (dict) | ``attrs`` (`FrozenAttrs`) |
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
from affine import Affine

from geoarray._src.array import GeoArray
from geoarray._src.attrs import FrozenAttrs
from geoarray._src.grid import crs_string


__all__ = ["from_georeader", "to_georeader"]

_HINT = "the georeader adapter needs georeader: pip install 'geotoolz-array[georeader]'"


def _geotensor_module() -> ModuleType:
    try:
        from georeader import geotensor
    except ImportError as exc:  # pragma: no cover - exercised by the base-install job
        raise ImportError(_HINT) from exc
    return geotensor


def from_georeader(gt: Any) -> GeoArray:
    """A NumPy `GeoArray` from a georeader ``GeoTensor``, sharing its memory.

    Args:
        gt: A ``georeader.geotensor.GeoTensor`` (2-D to 4-D). A lazy
            reader (``GeoData``) must be loaded first: ``gt.load()``.

    Returns:
        The carrier with the same values (no copy), transform, CRS, fill
        value and attrs (frozen: lists become tuples).

    Raises:
        ImportError: georeader is not installed.
        TypeError: ``gt`` is not a ``GeoTensor``, or an attr value cannot
            be frozen (the error names the key).

    Example:
        >>> import numpy as np
        >>> from affine import Affine
        >>> from georeader.geotensor import GeoTensor
        >>> gt = GeoTensor(
        ...     np.zeros((2, 4, 4), np.float32), Affine(20, 0, 600000, 0, -20, 3550000),
        ...     "EPSG:32613", fill_value_default=np.nan,
        ... )
        >>> g = from_georeader(gt)
        >>> g.shape, g.grid.bounds[0], np.shares_memory(g.data, gt.values)
        ((2, 4, 4), 600000.0, True)
    """
    geotensor = _geotensor_module()
    if not isinstance(gt, geotensor.GeoTensor):
        hint = "; load a lazy reader first with .load()" if hasattr(gt, "load") else ""
        raise TypeError(
            f"from_georeader takes a GeoTensor, got {type(gt).__name__}{hint}"
        )
    return GeoArray(
        np.asarray(gt.values),
        gt.transform,
        gt.crs,
        fill_value=gt.fill_value_default,
        attrs=FrozenAttrs(dict(gt.attrs or {})),
    )


def to_georeader(g: GeoArray) -> Any:
    """A georeader ``GeoTensor`` from a NumPy `GeoArray`, sharing its memory.

    Args:
        g: A NumPy carrier, not batched. Move a JAX carrier to the host
            first with `GeoArray.to_numpy`.

    Returns:
        A ``GeoTensor`` with the same values (no copy), an ``affine.Affine``
        transform, the CRS as ``"AUTH:CODE"`` when that is exact (else
        WKT), the fill value (``None`` stays ``None``; georeader's default
        of 0 is never assumed) and the attrs as a plain dict.

    Raises:
        ImportError: georeader is not installed.
        TypeError: ``g`` is not a GeoArray, or holds JAX data.
        ValueError: ``g`` is batched (GeoTensor has one transform).

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> g = GeoArray(np.ones((4, 4)), (20, 0, 600000, 0, -20, 3550000), 32613)
        >>> gt = to_georeader(g)
        >>> gt.crs, gt.transform.c, gt.fill_value_default is None
        ('EPSG:32613', 600000.0, True)
    """
    geotensor = _geotensor_module()
    if not isinstance(g, GeoArray):
        raise TypeError(f"to_georeader takes a GeoArray, got {type(g).__name__}")
    if g.backend != "numpy":
        raise TypeError(
            "GeoTensor holds NumPy data: move the carrier to the host first "
            "with g.to_numpy()"
        )
    if g.batch_shape:
        raise ValueError(
            f"a batched GeoArray (batch {g.batch_shape}) has one transform per "
            "element; convert each element, e.g. [to_georeader(g[i]) for i ...]"
        )
    return geotensor.GeoTensor(
        g.data,
        transform=Affine(*np.asarray(g.transform).tolist()),
        crs=crs_string(g.crs),
        fill_value_default=g.fill_value,
        attrs=g.attrs.to_dict(),
    )
