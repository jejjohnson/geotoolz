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

from geoarray._src.backend import is_tracer
from geoarray._src.errors import GeoArrayTraceError


__all__ = [
    "PER_BAND_KEYS",
    "FrozenAttrs",
    "concat_bands",
    "drop_band_keys",
    "select_bands",
]

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

    def __reduce__(self) -> tuple[Any, ...]:
        # Rebuild through the constructor: the cached hash of string keys
        # differs between processes (hash randomisation), and every NaN
        # must become the one ``math.nan`` again.
        return (FrozenAttrs, (self.to_dict(),))

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


#: The attrs keys that hold one entry per band. Band selection selects
#: their entries; a change in the band count that no selection explains
#: drops them (so every per-band key always matches the band count).
PER_BAND_KEYS: tuple[str, ...] = (
    "band_names",
    "descriptions",
    "bands",
    "band_descriptions",
    "wavelengths",
    "wavelengths_nm",
)


def select_bands(attrs: FrozenAttrs, index: Any, count: int) -> FrozenAttrs:
    """Apply a band-axis index to every per-band key.

    Args:
        attrs: The carrier's attrs.
        index: What indexed the band axis: an int, a slice, a list or array
            of ints, or a boolean mask. Must be concrete.
        count: The band count before indexing.

    Returns:
        Attrs whose per-band keys hold the selected entries; a per-band
        key whose length does not match ``count`` is dropped.

    Example:
        >>> attrs = FrozenAttrs(band_names=["B04", "B08", "B11"], sensor="S2A")
        >>> select_bands(attrs, [2, 0], 3)
        FrozenAttrs({'band_names': ('B11', 'B04'), 'sensor': 'S2A'})
    """
    if is_tracer(index):
        raise GeoArrayTraceError(
            "a band index must be concrete: the per-band attrs are static, so "
            "the selection must be known when tracing; index g.data instead, "
            "or select the bands outside jit"
        )
    positions = np.atleast_1d(np.arange(count)[index]).tolist()
    selected: dict[str, Any] = {}
    for key, value in attrs.items():
        if key not in PER_BAND_KEYS:
            selected[key] = value
        elif isinstance(value, tuple) and len(value) == count:
            selected[key] = tuple(value[i] for i in positions)
    return FrozenAttrs(selected)


def drop_band_keys(attrs: FrozenAttrs) -> FrozenAttrs:
    """Attrs without any per-band key (see `PER_BAND_KEYS`)."""
    return attrs.drop(*PER_BAND_KEYS)


def concat_bands(parts: list[tuple[FrozenAttrs, int]]) -> FrozenAttrs:
    """Attrs for carriers concatenated along the band axis.

    The first carrier's attrs, with each per-band key concatenated when
    every part has it with one entry per band, and dropped otherwise.

    Args:
        parts: ``(attrs, count)`` for each carrier, in order.
    """
    first = parts[0][0]
    merged: dict[str, Any] = {k: v for k, v in first.items() if k not in PER_BAND_KEYS}
    for key in PER_BAND_KEYS:
        values = [attrs.get(key) for attrs, _ in parts]
        tuples = [v for v in values if isinstance(v, tuple)]
        counts = [n for _, n in parts]
        if len(tuples) == len(parts) and all(
            len(v) == n for v, n in zip(tuples, counts, strict=True)
        ):
            merged[key] = tuple(item for v in tuples for item in v)
    return FrozenAttrs(merged)
