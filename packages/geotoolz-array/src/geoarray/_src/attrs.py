"""`FrozenAttrs`: the hashable metadata a `GeoArray` carries.

A GeoArray's ``attrs`` are static pytree data: JAX hashes them into the
cache key of every compiled function. So they must be hashable, and two
equal mappings must hash equal. `FrozenAttrs` freezes values on the way in:

- lists and tuples become tuples (recursively);
- 1-D numeric NumPy arrays (EMIT's 285 wavelengths) become tuples of
  Python numbers, and NumPy scalars become Python scalars;
- mappings (georeader's rasterio ``tags``) become nested `FrozenAttrs`;
- every NaN becomes the one `math.nan` object, so equal attrs compare equal
  even though ``nan != nan``.

Anything else that is not hashable (a 2-D array, a live object) raises.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np


__all__ = ["FrozenAttrs"]

_SCALARS = (str, bytes, bool, int, float, complex, type(None))


def _freeze(value: Any, key: str) -> Any:
    """Return a hashable version of ``value``, or raise `TypeError`."""
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, np.generic):
        return _freeze(value.item(), key)
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.dtype.kind in "biufc":
            return tuple(_freeze(v, key) for v in value.tolist())
        raise TypeError(
            f"attrs[{key!r}]: only 1-D numeric arrays can be frozen "
            f"(got shape {value.shape}, dtype {value.dtype}); store it outside attrs"
        )
    if isinstance(value, FrozenAttrs):
        return value
    if isinstance(value, Mapping):
        return FrozenAttrs(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze(v, key) for v in value)
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            f"attrs[{key!r}] must be hashable (scalars, tuples, 1-D numeric arrays "
            f"or mappings of them); got {type(value).__name__}"
        ) from exc
    return value


def _thaw(value: Any) -> Any:
    """Inverse of `_freeze` for mappings; tuples stay tuples."""
    if isinstance(value, FrozenAttrs):
        return value.to_dict()
    if isinstance(value, tuple):
        return tuple(_thaw(v) for v in value)
    return value


class FrozenAttrs(Mapping[str, Any]):
    """An immutable, hashable mapping of string keys to frozen values.

    Args:
        items: A mapping or iterable of ``(key, value)`` pairs.
        **kwargs: More items, added after ``items``.

    Raises:
        TypeError: A key is not a string, or a value cannot be frozen.

    Example:
        >>> import numpy as np
        >>> wavelengths = np.array([1610.0, 2190.0])
        >>> attrs = FrozenAttrs(band_names=["B11", "B12"], wavelengths=wavelengths)
        >>> attrs["band_names"]
        ('B11', 'B12')
        >>> attrs["wavelengths"]
        (1610.0, 2190.0)
        >>> attrs.set(sensor="S2A").drop("wavelengths")
        FrozenAttrs({'band_names': ('B11', 'B12'), 'sensor': 'S2A'})
    """

    __slots__ = ("_hash", "_items")

    def __init__(
        self,
        items: Mapping[str, Any] | Any = None,
        /,
        **kwargs: Any,
    ) -> None:
        merged = dict(items) if items is not None else {}
        merged.update(kwargs)
        frozen: dict[str, Any] = {}
        for key, value in merged.items():
            if not isinstance(key, str):
                raise TypeError(f"attrs keys must be strings, got {key!r}")
            frozen[key] = _freeze(value, key)
        self._items = frozen
        self._hash = hash(frozenset(frozen.items()))

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenAttrs):
            return self._hash == other._hash and self._items == other._items
        if isinstance(other, Mapping):
            return self == FrozenAttrs(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"FrozenAttrs({self._items!r})"

    def set(self, **items: Any) -> FrozenAttrs:
        """A copy with ``items`` added or replaced.

        Example:
            >>> FrozenAttrs(a=1).set(b=2)
            FrozenAttrs({'a': 1, 'b': 2})
        """
        return FrozenAttrs(self._items, **items)

    def drop(self, *keys: str) -> FrozenAttrs:
        """A copy without ``keys``; missing keys are ignored.

        Example:
            >>> FrozenAttrs(a=1, b=2).drop("b", "c")
            FrozenAttrs({'a': 1})
        """
        return FrozenAttrs({k: v for k, v in self._items.items() if k not in keys})

    def to_dict(self) -> dict[str, Any]:
        """A plain ``dict``; nested `FrozenAttrs` become dicts, tuples stay tuples.

        Example:
            >>> FrozenAttrs(tags={"AREA_OR_POINT": "Area"}).to_dict()
            {'tags': {'AREA_OR_POINT': 'Area'}}
        """
        return {k: _thaw(v) for k, v in self._items.items()}
