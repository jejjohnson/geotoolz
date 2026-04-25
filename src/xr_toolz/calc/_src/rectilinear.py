"""Finite-difference operators on rectilinear (non-uniform 1-D) coords.

Each differentiation dimension carries its own 1-D coordinate; spacing
along that coordinate may vary point-to-point. We get the derivative on
the unit-step index grid via :func:`finitediffx.difference` and then
scale by ``1 / (dx/di)`` — itself computed with the same stencil — to
get the chain-ruled physical derivative.

If the coordinate happens to be uniformly spaced we delegate to the
cartesian fast path.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xr_toolz.calc._src import cartesian


def _is_uniform(values: np.ndarray, *, rtol: float) -> bool:
    if values.size < 2:
        return False
    diffs = np.diff(values)
    return bool(np.allclose(diffs, diffs[0], rtol=rtol, atol=0.0))


def rectilinear_partial(
    da: xr.DataArray,
    dim: str,
    *,
    accuracy: int = 1,
    method: str = "central",
    uniform_rtol: float = 1e-6,
) -> xr.DataArray:
    """Partial derivative ``∂da/∂<dim>`` on a non-uniform 1-D coord.

    Args:
        da: Input field with a 1-D coordinate for ``dim``.
        dim: Dimension along which to differentiate.
        accuracy: Stencil accuracy order (forwarded to fdx).
        method: ``"central"`` | ``"forward"`` | ``"backward"``.
        uniform_rtol: If the coord is uniform within this tolerance, we
            delegate to the cartesian implementation (cheaper and avoids
            the chain-rule step).

    Returns:
        DataArray with the same dims/coords as ``da``.
    """
    if dim not in da.dims:
        raise ValueError(
            f"Dimension {dim!r} not present on DataArray with dims={da.dims}."
        )
    coord_values = np.asarray(da[dim].values)
    if coord_values.size < 2:
        raise ValueError(
            f"Coordinate {dim!r} has {coord_values.size} sample(s); "
            "need at least 2 to compute a finite-difference step."
        )
    if _is_uniform(coord_values, rtol=uniform_rtol):
        return cartesian.cartesian_partial(
            da, dim, accuracy=accuracy, method=method, uniform_rtol=uniform_rtol
        )

    axis = da.get_axis_num(dim)
    df_di = cartesian._difference(
        da.values, axis=axis, step_size=1.0, accuracy=accuracy, method=method
    )
    dxdi = cartesian._difference(
        coord_values, axis=0, step_size=1.0, accuracy=accuracy, method=method
    )
    shape = [1] * da.ndim
    shape[axis] = coord_values.size
    out = df_di / dxdi.reshape(shape)

    return xr.DataArray(
        out,
        dims=da.dims,
        coords={k: da.coords[k] for k in da.coords},
        name=cartesian._output_name(da, dim),
        attrs=dict(da.attrs),
    )


def rectilinear_gradient(
    da: xr.DataArray,
    *,
    dims: tuple[str, ...] | None = None,
    accuracy: int | tuple[int, ...] = 1,
    method: str = "central",
    uniform_rtol: float = 1e-6,
) -> xr.Dataset:
    """Gradient ``∇da`` on rectilinear (per-axis) coordinates."""
    target_dims = tuple(da.dims) if dims is None else tuple(dims)
    if isinstance(accuracy, int):
        per_dim = (accuracy,) * len(target_dims)
    else:
        per_dim = tuple(accuracy)
        if len(per_dim) != len(target_dims):
            raise ValueError(
                f"accuracy tuple length ({len(per_dim)}) does not match "
                f"number of dims ({len(target_dims)})."
            )
    base = da.name or "f"
    out: dict[str, xr.DataArray] = {}
    for dim, acc in zip(target_dims, per_dim, strict=True):
        out[f"d{base}_d{dim}"] = rectilinear_partial(
            da,
            dim,
            accuracy=acc,
            method=method,
            uniform_rtol=uniform_rtol,
        )
    return xr.Dataset(out)
