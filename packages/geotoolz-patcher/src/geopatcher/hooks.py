"""Public observability hooks for patcher callbacks.

`PatcherHook` is the callback protocol; `UNKNOWN_TOTAL` is the ``total``
value ``on_split_start`` receives when the patch count is not known up
front (e.g. a product-mode `SpatioTemporalPatcher` or an unsized anchor
stream).
"""

from __future__ import annotations

from geopatcher._src.hooks import UNKNOWN_TOTAL, PatcherHook


__all__ = ["UNKNOWN_TOTAL", "PatcherHook"]
