"""Lightweight shape descriptors for operator summary/introspection.

A :class:`Signature` is a value object that captures *what shape an
operator expects or produces* without holding any data. The shape
inference protocol on :class:`xrtoolz.Operator` threads
``Signature`` instances through a pipeline so :meth:`Sequential.summary`
and :meth:`Graph.summary` can render keras-style structural tables.

Signatures are **immutable** — instances are placed in shape-inference
caches that are reused across calls, so in-place mutation would
silently corrupt later runs. The dataclass is ``frozen=True`` and
``dims`` is wrapped in :class:`types.MappingProxyType`, so
``sig.dims["time"] = 5`` raises rather than poisoning the cache.

``dtype`` is canonicalized to ``np.dtype(...).name`` (a string) on
construction so that ``Signature(..., dtype="float32")`` and
``Signature(..., dtype=np.float32)`` compare equal. ``None`` (unknown)
is left as-is, and anything that fails to construct via :func:`np.dtype`
falls back to ``str(dtype)`` for forward compatibility.

Example:
    >>> sig = Signature({"time": 365, "lat": 181, "lon": 360}, dtype="float32")
    >>> sig.format()
    '(time=365, lat=181, lon=360); dtype=float32'
    >>> sig.drop_dims("time").format()
    '(lat=181, lon=360); dtype=float32'
    >>> sig.replace_dims({"lat": None}).format()
    '(time=365, lat=?, lon=360); dtype=float32'
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


def _canonical_dtype(dtype: Any) -> str | None:
    """Normalize a dtype tag so equality is independent of input form.

    ``"float32"``, ``np.float32``, and ``np.dtype("float32")`` all map
    to ``"float32"``. ``None`` round-trips. Anything :func:`np.dtype`
    rejects (custom sentinel strings like ``"category"``) is preserved
    as ``str(dtype)`` so callers using non-numpy tags still round-trip.
    """
    if dtype is None:
        return None
    try:
        return np.dtype(dtype).name
    except TypeError:
        return str(dtype)


@dataclass(frozen=True, eq=False)
class Signature:
    """Shape-and-dtype descriptor propagated without executing data paths.

    Args:
        dims: Mapping from dim name to size. ``None`` marks a dim whose
            size cannot be inferred symbolically (e.g. an irregular
            subset). Wrapped in :class:`MappingProxyType` on
            construction; mutating the proxy raises :class:`TypeError`.
        dtype: Optional dtype tag. Anything :func:`numpy.dtype` accepts
            (string, numpy scalar type, ``np.dtype`` instance) is
            normalized to the canonical numpy name (``"float32"``,
            ``"int64"``, …). ``None`` means "unknown / not tracked".

    Raises:
        TypeError: If ``dims`` keys aren't strings or values aren't
            ``int | None``.

    Example:
        >>> import numpy as np
        >>> from xrtoolz import Signature
        >>> Signature({"time": 12}, dtype="float32") == Signature(
        ...     {"time": 12}, dtype=np.float32,
        ... )
        True
        >>> Signature({"time": 12, "lat": 4}, dtype="float32") == Signature(
        ...     {"lat": 4, "time": 12}, dtype="float32",
        ... )
        True
    """

    dims: Mapping[str, int | None]
    dtype: Any = None

    def __post_init__(self) -> None:
        # Validate + freeze dims into a MappingProxyType so external
        # mutation (sig.dims["time"] = 5) raises instead of corrupting
        # cached signatures held by Graph._compute_signatures.
        normalized: dict[str, int | None] = {}
        for name, size in self.dims.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"Signature dim name must be a str, got "
                    f"{type(name).__name__}: {name!r}"
                )
            if size is not None and not isinstance(size, int | np.integer):
                raise TypeError(
                    f"Signature dim size for {name!r} must be int or None, "
                    f"got {type(size).__name__}: {size!r}"
                )
            normalized[name] = None if size is None else int(size)
        # Frozen dataclass: must use object.__setattr__ to assign.
        object.__setattr__(self, "dims", MappingProxyType(normalized))
        object.__setattr__(self, "dtype", _canonical_dtype(self.dtype))

    def __repr__(self) -> str:
        # Hide the MappingProxyType wrapper so reprs read like a plain
        # dict — matches the constructor signature users actually pass.
        return f"Signature(dims={dict(self.dims)!r}, dtype={self.dtype!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signature):
            return NotImplemented
        # dict(...) compare is order-insensitive — matches Python dict
        # equality semantics. dtype is already canonicalized in __post_init__.
        return dict(self.dims) == dict(other.dims) and self.dtype == other.dtype

    def __hash__(self) -> int:
        # Sorted tuple-of-pairs gives an order-stable hash that aligns
        # with __eq__'s order-insensitive dict comparison.
        return hash((tuple(sorted(self.dims.items())), self.dtype))

    def replace_dims(
        self,
        updates: Mapping[str, int | None],
        *,
        strict: bool = False,
    ) -> Signature:
        """Return a copy with selected dimension sizes replaced.

        Args:
            updates: Mapping of dim name → new size (``None`` means
                "size unknown after this op").
            strict: If True, raise :class:`KeyError` when ``updates``
                refers to a dim not present in ``self.dims``. Default is
                False — unknown keys are silently ignored, which lets
                shape-preserving operators call ``replace_dims`` on a
                superset of their target dims without failing on inputs
                that don't carry the optional dim.

        Example:
            >>> sig = Signature({"time": 12, "lat": 4})
            >>> sig.replace_dims({"time": None}).format()
            '(time=?, lat=4)'
        """
        if strict:
            unknown = set(updates) - set(self.dims)
            if unknown:
                raise KeyError(
                    f"replace_dims(strict=True): {sorted(unknown)!r} not in "
                    f"signature dims {sorted(self.dims)!r}."
                )
        dims = dict(self.dims)
        for name, size in updates.items():
            if name in dims:
                dims[name] = size
        return Signature(dims, dtype=self.dtype)

    def rename_dims(self, mapping: Mapping[str, str]) -> Signature:
        """Return a copy with dimension names renamed by ``mapping``.

        Example:
            >>> sig = Signature({"lon": 360, "lat": 181})
            >>> sig.rename_dims({"lon": "longitude"}).format()
            '(longitude=360, lat=181)'
        """
        return Signature(
            {mapping.get(name, name): size for name, size in self.dims.items()},
            dtype=self.dtype,
        )

    def drop_dims(self, names: str | Sequence[str]) -> Signature:
        """Return a copy without the requested dimensions.

        Example:
            >>> sig = Signature({"time": 12, "lat": 4, "lon": 8})
            >>> sig.drop_dims(("time", "lat")).format()
            '(lon=8)'
        """
        drop = {names} if isinstance(names, str) else set(names)
        return Signature(
            {name: size for name, size in self.dims.items() if name not in drop},
            dtype=self.dtype,
        )

    def format(self) -> str:
        """Render dimensions and dtype as a compact summary string.

        Unknown sizes (``None``) appear as ``?``. Used by
        :meth:`Sequential.summary` and :meth:`Graph.summary`.
        """
        shape = ", ".join(
            f"{name}={size if size is not None else '?'}"
            for name, size in self.dims.items()
        )
        rendered = f"({shape})"
        if self.dtype is not None:
            rendered = f"{rendered}; dtype={self.dtype}"
        return rendered
