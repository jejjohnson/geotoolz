"""Backwards-compatible re-export of :mod:`geotoolz._src.bands`.

The band resolver moved to ``geotoolz._src.bands`` so every operator
family shares it; import from there in new code.
"""

from __future__ import annotations

from geotoolz._src.bands import (
    DEFAULT_BAND_KEYS,
    BandRef,
    configured_ref,
    resolve_band,
)


__all__ = ["DEFAULT_BAND_KEYS", "BandRef", "configured_ref", "resolve_band"]
