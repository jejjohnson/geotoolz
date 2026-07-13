"""Carrier rewrap discipline shared by every operator family.

`georeader.GeoTensor` (>=2.0) is an ``np.ndarray`` subclass, so the
Tier-A primitives accept it transparently via ``np.asarray``. The one
carrier-aware step left in each ``_apply`` is the *rewrap*: a GeoTensor
input should come back as a GeoTensor (metadata propagated through
``array_as_geotensor``), while a plain ndarray input should come back
as a plain ndarray. :func:`wrap_like` centralises that duck-typed
dispatch so operators support both carriers with a single call.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor

__all__ = ["wrap_like"]


def wrap_like(
    ref: GeoTensor | np.ndarray,
    out: np.ndarray,
    *,
    fill_value_default: numbers.Number | None = None,
) -> GeoTensor | np.ndarray:
    """Rewrap a computed array to match the carrier of ``ref``.

    Args:
        ref: The operator's input carrier — a ``GeoTensor`` or any plain
            array-like. Dispatch is duck-typed on the presence of
            ``array_as_geotensor`` rather than an isinstance check, so
            GeoTensor-compatible carriers from other libraries also work.
        out: The result array produced by a Tier-A primitive. Its trailing
            two (spatial) dims must match ``ref``'s when ``ref`` is a
            GeoTensor — ``array_as_geotensor`` enforces this.
        fill_value_default: Fill value for the returned GeoTensor. ``None``
            (default) propagates ``ref.fill_value_default``. Ignored for
            plain-array carriers.

    Returns:
        ``ref.array_as_geotensor(out)`` when ``ref`` is GeoTensor-like,
        otherwise ``out`` as a plain ``np.ndarray``.
    """
    rewrap = getattr(ref, "array_as_geotensor", None)
    if rewrap is not None:
        return rewrap(out, fill_value_default=fill_value_default)
    return np.asarray(out)
