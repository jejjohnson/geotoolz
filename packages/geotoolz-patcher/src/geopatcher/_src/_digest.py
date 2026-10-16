"""Content digests shared by the patch cache and the training datasets."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def array_digest(values: Any) -> str:
    """sha256 of an array's dtype, shape and values.

    Object-dtype arrays (cftime dates, caller-supplied object origins) are
    hashed through ``repr`` of their values, so equal values in different
    processes give the same digest instead of hashing object pointers.

    Examples:
        >>> array_digest([1, 2]) == array_digest(np.array([1, 2]))
        True
        >>> array_digest([1, 2]) == array_digest([1, 3])
        False
    """
    arr = np.asarray(values)
    h = hashlib.sha256(f"{arr.dtype.str}{arr.shape}".encode())
    if arr.dtype.hasobject:
        h.update(repr(arr.tolist()).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()
