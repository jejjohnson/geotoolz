"""Compatibility re-export for shared validation helpers."""

from __future__ import annotations

from xrtoolz.utils._src.validation import (
    _is_int_like,
    _require_coords,
    _require_dims,
    _validate_bool_mask,
    _validate_coarsen_factor,
    _validate_idw_args,
    _validate_positive_int,
)


__all__ = [
    "_is_int_like",
    "_require_coords",
    "_require_dims",
    "_validate_bool_mask",
    "_validate_coarsen_factor",
    "_validate_idw_args",
    "_validate_positive_int",
]
