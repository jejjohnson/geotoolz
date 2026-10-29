"""Refusing a georeader ``GeoTensor``, without importing georeader or geoarray.

The one check behind every public function's "a GeoTensor is refused" rule.
It lives apart from `geoops._src.geo` so dependency-free modules (the Tier-A
``array.py`` kernels, which may not import geoarray, GA-G23) can use it;
`geoops._src.geo` re-exports the same objects.
"""

from __future__ import annotations

from typing import Any


__all__ = ["is_geotensor", "reject_geotensor"]

_FROM_GEOREADER = (
    "convert it at the boundary with "
    "geoarray.interop.georeader.from_georeader(gt) (zero-copy)"
)


def is_geotensor(x: Any) -> bool:
    """Whether ``x`` is a georeader ``GeoTensor`` (checked without importing it)."""
    return any(
        cls.__name__ == "GeoTensor" and cls.__module__.startswith("georeader")
        for cls in type(x).__mro__
    )


def reject_geotensor(x: Any, op_name: str = "geotoolz") -> None:
    """Raise ``TypeError`` for a georeader ``GeoTensor``; pass anything else.

    A ``GeoTensor`` is an ``np.ndarray`` subclass, so ``np.asarray`` would
    take it silently as a plain array and the output would lose its
    georeferencing. geotoolz takes `geoarray.GeoArray` instead.

    Args:
        x: The value to check.
        op_name: Who refuses it, for the message.

    Raises:
        TypeError: ``x`` is a ``GeoTensor``; the message names
            ``from_georeader``.
    """
    if is_geotensor(x):
        raise TypeError(
            f"{op_name} takes a geoarray.GeoArray, not a georeader GeoTensor; "
            f"{_FROM_GEOREADER}"
        )
