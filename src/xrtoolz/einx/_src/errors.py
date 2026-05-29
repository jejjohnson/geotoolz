"""Exception hierarchy for :mod:`xrtoolz.einx`.

These are raised by the bridge itself when it catches a problem before
dispatching to einx (e.g. a pattern that names a dim absent from the
input, or shared dims with mismatched coords). Upstream einx / xarray
errors propagate unchanged.
"""

from __future__ import annotations


class EinxBridgeError(Exception):
    """Base for all :mod:`xrtoolz.einx` errors."""


class CoordMismatch(EinxBridgeError):
    """Shared dims have unequal coords and ``align=False``."""


class PatternError(EinxBridgeError, ValueError):
    """Pattern is malformed or references dims not on the inputs."""


__all__ = ["CoordMismatch", "EinxBridgeError", "PatternError"]
