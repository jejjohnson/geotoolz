"""Deprecated alias — forwards to :mod:`xreinx.dataset`.

Kept so historical ``from xrtoolz.einx.dataset import pack_dataset``
imports keep resolving through 0.x; the package-level
``DeprecationWarning`` fires via :mod:`xrtoolz.einx`.
"""

from __future__ import annotations

from xreinx.dataset import pack_dataset, unpack_dataset


__all__ = ["pack_dataset", "unpack_dataset"]
