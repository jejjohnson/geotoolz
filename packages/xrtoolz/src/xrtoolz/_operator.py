"""Backward-compatibility shim — the xarray-aware ``Operator`` moved to
:mod:`xrcore` (the ``xrtoolz-core`` distribution).

Import :class:`xrcore.Operator` directly in new code; this module exists
so historical ``xrtoolz._operator`` imports keep resolving.
"""

from __future__ import annotations

from xrcore import Operator


__all__ = ["Operator"]
