"""Region-integrated face-flux primitive — V4.2.

Returns the area-weighted advective flux ``∑ φ u A_face`` summed over
the cells of a region (or the full grid). This is **not** a strict
boundary-only integral — it sums every face the region covers, so
interior contributions cancel only when the region is a closed
control volume on a divergence-free flow. For a domain-boundary
diagnostic, supply ``region`` as a boundary mask shifted to face
centres.
"""

from __future__ import annotations

import xarray as xr


def boundary_flux(
    ds: xr.Dataset,
    *,
    variable: str | None,
    velocity_vars: dict[str, str],
    face_metrics: xr.Dataset,
    region: xr.DataArray | None = None,
    time_dim: str = "time",
) -> xr.Dataset:
    """Region-integrated advective face flux.

    Computes ``∑ φ u A_face`` over each face direction. By Gauss's
    theorem the same sum equals the closed-boundary flux when ``φ u``
    is divergence-free over the integration domain; for general flows
    the result is the area-weighted total of the face-flux field
    rather than a pure boundary integral. See the module docstring.

    Args:
        ds: Dataset containing ``variable`` (or ``None`` for volume
            flux only) and the velocity components.
        variable: Tracer name or ``None``. If ``None`` the flux is the
            volume flux ``u A_face`` (used by the volume budget).
        velocity_vars: Mapping ``{"u": "u_var", "v": "v_var", "w": "w_var"}``;
            ``"w"`` is optional. Names are looked up in ``ds``.
        face_metrics: Dataset carrying ``area_e`` (east face area, m²),
            ``area_n`` (north face area, m²), and optionally ``area_top``.
        region: Optional boolean mask. ``True`` cells contribute; the
            mask should be shifted to face centres if a boundary-only
            integral is desired.
        time_dim: Name of the time coordinate to preserve as a non-
            spatial dim. Default ``"time"``.

    Returns:
        Dataset with one scalar (or ``time_dim``-indexed) variable per
        active face direction: ``flux_x``, ``flux_y``, optionally
        ``flux_z``.
    """
    out: dict[str, xr.DataArray] = {}
    tracer = ds[variable] if variable is not None else None

    for axis, area_key in (("u", "area_e"), ("v", "area_n"), ("w", "area_top")):
        if axis not in velocity_vars:
            continue
        if area_key not in face_metrics:
            continue
        u = ds[velocity_vars[axis]]
        a = face_metrics[area_key]
        flux = u if tracer is None else tracer * u
        flux = flux * a
        if region is not None:
            flux = flux.where(region)
        flux_dim = {"u": "x", "v": "y", "w": "z"}[axis]
        spatial_dims = [d for d in flux.dims if d != time_dim]
        out[f"flux_{flux_dim}"] = flux.sum(dim=spatial_dims, skipna=True)

    return xr.Dataset(out)


__all__ = ["boundary_flux"]
