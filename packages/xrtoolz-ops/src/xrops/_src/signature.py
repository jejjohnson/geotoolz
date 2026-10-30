"""Dims-by-name helpers over `pipekit.Signature` for xrops's shape inference.

`pipekit.Signature` keeps its dims as ordered ``(name, size)`` pairs; xrops
operators read and rewrite them by name. These helpers hold that idiom in
one place. They are lenient where the operators need it: ``replace_dims``,
``rename_dims`` and ``drop_dims`` ignore names the signature lacks, so a
shape-preserving operator can name an optional dim without failing on
inputs that do not carry it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pipekit import Signature


def sizes(sig: Signature) -> dict[str, int | None]:
    """The dims of ``sig`` as ``{name: size}``, in order."""
    return dict(sig.dims)


def make(dims: Mapping[Any, int | None], dtype: Any = None) -> Signature:
    """A `pipekit.Signature` from ``{name: size}`` (``None`` for unknown).

    ``dtype`` may be anything `numpy.dtype` accepts and is stored as its
    canonical name (``"float32"``); other tags are kept as ``str(dtype)``.
    """
    pairs = tuple(
        (str(name), None if size is None else int(size)) for name, size in dims.items()
    )
    return Signature(dims=pairs, dtype=_dtype(dtype))


def replace_dims(sig: Signature, updates: Mapping[str, int | None]) -> Signature:
    """``sig`` with the sizes of the named dims replaced; other names are ignored."""
    return make({n: updates.get(n, s) for n, s in sig.dims}, sig.dtype)


def rename_dims(sig: Signature, mapping: Mapping[str, str]) -> Signature:
    """``sig`` with dims renamed by ``mapping``; other names stay."""
    return make({mapping.get(n, n): s for n, s in sig.dims}, sig.dtype)


def drop_dims(sig: Signature, names: str | Sequence[str]) -> Signature:
    """``sig`` without the named dims; names it lacks are ignored."""
    drop = {names} if isinstance(names, str) else set(names)
    return make({n: s for n, s in sig.dims if n not in drop}, sig.dtype)


def _dtype(dtype: Any) -> str | None:
    if dtype is None:
        return None
    try:
        return np.dtype(dtype).name
    except TypeError:
        return str(dtype)


__all__ = ["drop_dims", "make", "rename_dims", "replace_dims", "sizes"]
