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

from xrgrad._src import cartesian


def _is_uniform(values: np.ndarray, *, rtol: float) -> bool:
    if values.size < 2:
        return False
    diffs = np.diff(values)
    return bool(np.allclose(diffs, diffs[0], rtol=rtol, atol=0.0))


def rectilinear_partial(
    da: xr.DataArray,
    dim: str,
    *,
    order: int = 1,
    accuracy: int = 1,
    method: str = "central",
    uniform_rtol: float = 1e-6,
) -> xr.DataArray:
    """Partial derivative ``∂da/∂<dim>`` on a non-uniform 1-D coord.

    Args:
        da: Input field with a 1-D coordinate for ``dim``.
        dim: Dimension along which to differentiate.
        order: Derivative order. Only ``1`` is supported on a genuinely
            non-uniform coordinate; a uniform coordinate delegates to
            the cartesian backend, which supports any order.
        accuracy: Stencil accuracy order (forwarded to fdx).
        method: ``"central"`` | ``"forward"`` | ``"backward"``.
        uniform_rtol: If the coord is uniform within this tolerance, we
            delegate to the cartesian implementation (cheaper and avoids
            the chain-rule step).

    Returns:
        DataArray with the same dims/coords as ``da``.

    Raises:
        ValueError: if ``dim`` is absent from ``da.dims``, carries no
            coordinate, has fewer than two samples, or is not strictly
            monotonic.
        NotImplementedError: if ``order > 1`` on a non-uniform
            coordinate.
    """
    cartesian._validate_order(order)
    if dim not in da.dims:
        raise ValueError(
            f"Dimension {dim!r} not present on DataArray with dims={da.dims}."
        )
    cartesian._require_coord(da, dim)
    cartesian._require_1d_coord(da[dim])
    coord_values = np.asarray(da[dim].values)
    if coord_values.size < 2:
        raise ValueError(
            f"Coordinate {dim!r} has {coord_values.size} sample(s); "
            "need at least 2 to compute a finite-difference step."
        )
    diffs = np.diff(coord_values)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(
            f"Coordinate {dim!r} must be strictly monotonic for "
            "geometry='rectilinear'; found repeated or reversing values "
            f"(min step {float(diffs.min())}, max step {float(diffs.max())}). "
            "Sort or deduplicate the coordinate first."
        )
    if _is_uniform(coord_values, rtol=uniform_rtol):
        return cartesian.cartesian_partial(
            da,
            dim,
            order=order,
            accuracy=accuracy,
            method=method,
            uniform_rtol=uniform_rtol,
        )
    if order > 1:
        raise NotImplementedError(
            f"order={order} is not supported on the non-uniform coordinate "
            f"{dim!r}: the index-grid chain rule used here is only valid for "
            "first derivatives. Use a uniform coordinate (which delegates to "
            "the cartesian backend), or compose xrgrad.laplacian for a second "
            "derivative."
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
    per_dim = cartesian._per_dim_accuracy(accuracy, target_dims)
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
