"""Exact-division and grid-alignment helpers.

`divide_evenly` is derived from
``terrax.xreader.stencils._divide_evenly``
(Apache-2.0, © Google LLC; original author Stephan Hoyer).
Modifications © 2026 J. Emmanuel Johnson, licensed MIT under the
geocatalog project's overall MIT licence; the original function is
re-licensed compatibly. See ``NOTICE`` for upstream attribution.

----------------------------------------------------------------------
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
----------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np


if TYPE_CHECKING:
    from geocatalog._src.geoslice import GeoSlice


Align = Literal["off", "warn", "error", "snap"]
"""Construction-time alignment policy for `GeoSlice`.

- ``"off"``: skip the check entirely (today's behaviour).
- ``"warn"``: emit a `GridAlignmentWarning` via the standard library
  ``warnings`` module and continue with the original bounds.
- ``"error"``: raise ``ValueError`` on the first misaligned axis.
- ``"snap"``: round outward while preserving the affine origin
  (``xmin`` and ``ymax`` for north-up rasters).
"""


class GridAlignmentWarning(UserWarning):
    """Emitted by `GeoSlice` for ``align="warn"`` and ``align="snap"``.

    A dedicated subclass so applications can filter alignment notices
    independently of other ``UserWarning``\\ s — e.g. via
    ``warnings.simplefilter("error", GridAlignmentWarning)`` in tests
    or ``"ignore"`` in pipelines that have audited their slice sites.

    Inherits from `UserWarning` so it is visible by default; unlike a
    loguru log record it is not silenced by the library's
    ``logger.disable("geocatalog")`` import-time hygiene.
    """


def _default_tol() -> float:
    """Default tolerance for `divide_evenly`, tied to `PIXEL_PRECISION`.

    Imported lazily to dodge the circular import between this module
    and ``geoslice`` (which depends on `Align` from here).
    """
    from geocatalog._src.geoslice import PIXEL_PRECISION

    return 10**-PIXEL_PRECISION


def divide_evenly(
    length: float,
    step: float,
    *,
    tol: float | None = None,
    label: str = "length",
) -> int:
    """``round(length/step)``, raising if not within ``tol`` of exact.

    The "loud" counterpart to a bare ``round(L / r)`` — useful when a
    subpixel residual would silently shift a grid by a fraction of a
    pixel and surface much later as an off-by-one elsewhere.

    Args:
        length: numerator in CRS units (e.g. ``xmax - xmin``).
        step: denominator in CRS units (e.g. ``x_res``).
        tol: absolute tolerance on the residual ``q*step - length``;
            ``None`` (default) uses ``10 ** -PIXEL_PRECISION``.
        label: name of the dividend, surfaced in the error message.

    Returns:
        Integer pixel count ``q``.

    Raises:
        ValueError: if ``|q*step - length| > tol``.
    """
    if tol is None:
        tol = _default_tol()
    q = int(np.round(length / step))
    residual = q * step - length
    if abs(residual) > tol:
        raise ValueError(
            f"{label}={length!r} is not an integer multiple of "
            f"step={step!r} (nearest is {q} -> residual "
            f"{residual:.6g}). Use align='snap' to round outward, or "
            f"fix the bounds/resolution."
        )
    return q


def is_grid_aligned(
    a: GeoSlice,
    b: GeoSlice,
    *,
    tol: float | None = None,
    explain: bool = False,
) -> bool | dict[str, Any]:
    """Are ``a`` and ``b`` on the same pixel lattice?

    Two slices are co-registered iff their resolutions match (within
    ``tol``) and their per-axis origin offsets are congruent modulo
    resolution.

    CRS mismatch returns ``False`` — the lattice question is undefined
    across CRSs. Reproject one side first via `GeoSlice.to_crs` if
    that's the intent.

    Args:
        a: First slice.
        b: Second slice.
        tol: Absolute tolerance for the residual checks; defaults to
            ``10 ** -PIXEL_PRECISION``.
        explain: If ``True``, return a dict with per-axis residuals
            and resolution-match booleans instead of a plain bool.

    Returns:
        ``bool`` (or a diagnostic dict if ``explain=True``).
    """
    if tol is None:
        tol = _default_tol()

    rx_a, ry_a = a.resolution
    rx_b, ry_b = b.resolution
    # North-up affine origin is (xmin, ymax) — the top-left corner
    # — so the lattice question is about congruence of xmin and ymax
    # modulo resolution, not ymin.
    xmin_a, _, _, ymax_a = a.bounds
    xmin_b, _, _, ymax_b = b.bounds

    crs_match = a.crs == b.crs
    x_res_match = abs(rx_a - rx_b) <= tol
    y_res_match = abs(ry_a - ry_b) <= tol

    if x_res_match:
        x_off = (xmin_a - xmin_b) / rx_a
        x_origin_residual = (x_off - round(x_off)) * rx_a
        x_origin_match = abs(x_origin_residual) <= tol
    else:
        x_origin_residual = float("nan")
        x_origin_match = False

    if y_res_match:
        y_off = (ymax_a - ymax_b) / ry_a
        y_origin_residual = (y_off - round(y_off)) * ry_a
        y_origin_match = abs(y_origin_residual) <= tol
    else:
        y_origin_residual = float("nan")
        y_origin_match = False

    aligned = bool(
        crs_match and x_res_match and y_res_match and x_origin_match and y_origin_match
    )

    if explain:
        return {
            "aligned": aligned,
            "crs_match": crs_match,
            "x_res_match": x_res_match,
            "y_res_match": y_res_match,
            "x_origin_residual": float(x_origin_residual),
            "y_origin_residual": float(y_origin_residual),
        }
    return aligned
