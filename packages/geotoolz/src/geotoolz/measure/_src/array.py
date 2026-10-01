"""Tier-A primitives for :mod:`geotoolz.measure` -- no ``GeoTensor`` here.

The label / region / skeleton maths is shared with :mod:`geotoolz.plume`
and :mod:`geotoolz.mask`, so it lives in :mod:`geotoolz._src.labels` and
is re-exported here as the measure family's primitive surface. This
module adds the unit conversion used by
:class:`~geotoolz.measure.RegionProps` (``scale_to_crs=True``), which
needs only the affine coefficients, not a carrier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geotoolz._src.labels import (
    DEFAULT_REGIONPROPS,
    Connectivity,
    label_components,
    regionprops_frame,
    skeleton_length,
)


# Region properties that ``regionprops_to_crs_units`` converts from pixel
# units: areas scale by the pixel area, lengths by the pixel size, second
# moments by the squared pixel size. Everything else (positions, angles,
# ratios, the label itself) stays as skimage reports it.
AREA_PROPS = frozenset({"area", "area_bbox", "area_convex", "area_filled"})
LENGTH_PROPS = frozenset(
    {
        "axis_major_length",
        "axis_minor_length",
        "equivalent_diameter_area",
        "feret_diameter_max",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "perimeter_crofton",
    }
)
MOMENT_PROPS = frozenset({"inertia_tensor", "inertia_tensor_eigvals"})


def regionprops_to_crs_units(frame: pd.DataFrame, transform: Any) -> pd.DataFrame:
    """Convert pixel-unit region-property columns to CRS units.

    With ``A = |a·e - b·d|`` the pixel area and ``s`` the pixel side:

        area → area · A,   length → length · s,   moment → moment · s²

    Columns are matched by their base name (``inertia_tensor-0-1`` →
    ``inertia_tensor``); unlisted columns are copied unchanged.

    Args:
        frame: A :func:`regionprops_frame` table in pixel units.
        transform: Affine-like object exposing ``a, b, d, e``.

    Returns:
        A converted copy of ``frame``.

    Raises:
        ValueError: If a length or moment column is present and the
            pixels are not square (anisotropic or sheared), where no
            single pixel length exists.

    Examples:
        >>> from affine import Affine
        >>> frame = pd.DataFrame({"label": [1], "area": [4.0], "perimeter": [8.0]})
        >>> regionprops_to_crs_units(frame, Affine(10, 0, 0, 0, -10, 0))
           label   area  perimeter
        0      1  400.0       80.0
    """
    a, b = float(transform.a), float(transform.b)
    d, e = float(transform.d), float(transform.e)
    pixel_area = abs(a * e - b * d)
    # Transform columns: the CRS step of one pixel column / row.
    col_step, row_step = float(np.hypot(a, d)), float(np.hypot(b, e))
    square = bool(
        np.isclose(col_step, row_step)
        and np.isclose(a * b + d * e, 0.0, atol=1e-12 * max(pixel_area, 1.0))
    )
    out = frame.copy()
    for column in frame.columns:
        base = str(column).split("-")[0]
        if base in AREA_PROPS:
            out[column] = frame[column] * pixel_area
        elif base in LENGTH_PROPS or base in MOMENT_PROPS:
            if not square:
                raise ValueError(
                    f"RegionProps(scale_to_crs=True): {base!r} has no CRS-unit "
                    f"equivalent on non-square pixels (transform a={a}, b={b}, "
                    f"d={d}, e={e}); request area properties only, or resample "
                    "to square pixels first."
                )
            power = 1 if base in LENGTH_PROPS else 2
            out[column] = frame[column] * col_step**power
    return out


__all__ = [
    "AREA_PROPS",
    "DEFAULT_REGIONPROPS",
    "LENGTH_PROPS",
    "MOMENT_PROPS",
    "Connectivity",
    "label_components",
    "regionprops_frame",
    "regionprops_to_crs_units",
    "skeleton_length",
]
