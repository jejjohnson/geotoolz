"""`stack_patches` — uniform-shape `Patch` collection → single ndarray.

The minimum useful primitive for JAX `vmap` / `pmap` and PyTorch
`default_collate` users: take a list of `Patch` / `TemporalPatch` /
`SpatioTemporalPatch` carriers, stack their ``data`` (or any other
ndarray attribute) into a single ``(N, *patch_shape)`` array. Fails
loudly on ragged geometries so the failure mode is visible at the
boundary rather than producing a numpy object array further down the
JIT trace.

See ``docs/notebooks/recipes_jax_vmap.ipynb`` for the canonical use.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def stack_patches(
    patches: Iterable[Any],
    attr: str = "data",
) -> np.ndarray:
    """Stack a `Patch` collection's per-patch ``attr`` into a single ndarray.

    Args:
        patches: Iterable of `Patch` / `TemporalPatch` /
            `SpatioTemporalPatch`. Materialised to a list internally.
        attr: Which patch attribute to stack. Default ``"data"``;
            ``"weights"`` is the other common choice (for COLA
            normalisation, sample weighting, masking).

    Returns:
        ``(N, *element_shape)`` ``np.ndarray`` where ``N == len(patches)``
        and ``element_shape`` is the shape every per-patch ``attr``
        agrees on.

    Raises:
        ValueError: When ``patches`` is empty, when any patch's
            ``attr`` is ``None``, or when the per-patch shapes don't
            agree. The error message names the first mismatching patch
            so ragged geometries (`SpatialPolygonIntersection`,
            `SpatialRadiusGraph`) surface early rather than producing
            numpy object arrays.
    """
    patches_list = list(patches)
    if not patches_list:
        raise ValueError(
            "stack_patches: empty input. Materialise patches first via "
            "list(patcher.split(field)), patcher.patch_at(field, anchor) "
            "(spatial), or patcher.patches_at(series, anchor) (temporal), "
            "and check the field actually has data."
        )

    arrays: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for i, p in enumerate(patches_list):
        value = getattr(p, attr, None)
        if value is None:
            raise ValueError(
                f"stack_patches: patch {i} has {attr}=None — "
                "every patch must carry the requested attribute."
            )
        arr = np.asarray(value)
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise ValueError(
                f"stack_patches: patch {i} has {attr}.shape={arr.shape}, "
                f"expected {expected_shape}. Stacking only works for "
                "uniform-shape geometries (e.g. SpatialRectangular, "
                "SpatialKNNGraph). For ragged geometries "
                "(SpatialRadiusGraph, SpatialPolygonIntersection) keep "
                "the per-patch list and feed your model one patch at a "
                "time, or pad to a common shape first."
            )
        arrays.append(arr)

    return np.stack(arrays, axis=0)


__all__ = ["stack_patches"]
