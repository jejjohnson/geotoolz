"""Backward-compatibility shim — :class:`Signature` moved to
:mod:`xrcore` (the ``xrtoolz-core`` distribution).

Import :class:`xrcore.Signature` directly in new code; this module exists
so historical ``xrtoolz.signature`` imports keep resolving. The name is
also still re-exported at the top level (``xrtoolz.Signature``).
"""

from __future__ import annotations

from xrcore import Signature


__all__ = ["Signature"]
