"""Grid-to-grid value resampling.

Deterministic refinement (:func:`refine`), aggregation
(:func:`coarsen`), and target-grid resampling (:func:`regrid_like`)
along one or more dimensions. Learned counterparts
(``Downscale``/``Upscale``) live in :mod:`.downscale`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import xarray as xr


_RESIZE_MODES = frozenset({"reflect", "constant", "edge", "symmetric", "wrap"})


def coarsen(
    ds: xr.Dataset | xr.DataArray,
    factor: dict[str, int],
    method: str = "mean",
    boundary: str = "trim",
) -> xr.Dataset | xr.DataArray:
    """Coarsen ``ds`` along one or more dimensions by integer factors.

    Thin wrapper around ``xr.Dataset.coarsen``.
    """
    coarsened = ds.coarsen(dim=factor, boundary=boundary)
    return getattr(coarsened, method)()


def coarsen_conservative(
    ds: xr.Dataset | xr.DataArray,
    factor: Mapping[str, int],
    *,
    lat: str = "lat",
    boundary: Literal["trim", "exact", "pad"] = "trim",
) -> xr.Dataset | xr.DataArray:
    """Area-weighted coarsen using cosine-of-latitude weights.

    This preserves cosine-latitude-weighted integrals for aligned, integer
    coarsening of regular latitude/longitude grids. Non-latitude dimensions
    use uniform weights, and missing values are skipped with weights
    renormalized within each block.

    Args:
        ds: Dataset or data array to coarsen.
        factor: Mapping from dimension name to integer coarsen factor.
        lat: Latitude dimension name. Values are expected to be cell centers in
            degrees within the usual [-90, 90] latitude range.
        boundary: Boundary mode forwarded to :meth:`xarray.DataArray.coarsen`.

    Returns:
        Coarsened dataset or data array with the same dimension names.

    Examples:
        >>> coarsen_conservative(da, {"lat": 4, "lon": 4})
        >>> coarsen_conservative(ds, {"latitude": 2}, lat="latitude")
    """
    factor_dict = _validate_coarsen_factor(factor)
    if isinstance(ds, xr.Dataset):
        return ds.map(
            lambda da: _coarsen_conservative_dataset_variable(
                da, factor_dict, lat=lat, boundary=boundary
            )
        )
    return _coarsen_conservative_dataarray(ds, factor_dict, lat=lat, boundary=boundary)


def _coarsen_conservative_dataset_variable(
    da: xr.DataArray,
    factor: dict[str, int],
    *,
    lat: str,
    boundary: Literal["trim", "exact", "pad"],
) -> xr.DataArray:
    variable_factor = {dim: value for dim, value in factor.items() if dim in da.dims}
    if not variable_factor:
        return da
    return _coarsen_conservative_dataarray(
        da, variable_factor, lat=lat, boundary=boundary
    )


def _coarsen_conservative_dataarray(
    da: xr.DataArray,
    factor: dict[str, int],
    *,
    lat: str,
    boundary: Literal["trim", "exact", "pad"],
) -> xr.DataArray:
    if lat not in da.dims or lat not in factor:
        return da.coarsen(dim=factor, boundary=boundary).mean()

    _validate_lat_chunks(da, factor[lat], lat=lat)

    cos_lat = np.cos(np.deg2rad(da[lat]))
    mask = xr.apply_ufunc(np.isfinite, da, dask="allowed")
    # Single mask multiplication: zero-out NaN cells in da, then weight.
    weights = cos_lat * mask
    numerator = (
        (da.where(mask, 0.0) * cos_lat).coarsen(dim=factor, boundary=boundary).sum()
    )
    denominator = weights.coarsen(dim=factor, boundary=boundary).sum()
    # Mask zero denominators before dividing so we never trigger 0/0 warnings.
    safe_den = denominator.where(denominator > 0)
    return numerator / safe_den


def _validate_coarsen_factor(factor: Mapping[str, int]) -> dict[str, int]:
    factor_dict: dict[str, int] = {}
    for dim, value in factor.items():
        # Accept numpy integer types (np.int64 etc.) by reducing through __index__.
        try:
            int_value = (
                int(value.__index__()) if hasattr(value, "__index__") else int(value)
            )
            int_match = hasattr(value, "__index__")
        except (TypeError, ValueError):
            int_match = False
            int_value = 0
        if not int_match or int_value < 1 or isinstance(value, bool):
            raise ValueError(
                f"coarsen factor for {dim!r} must be a positive integer "
                f"(>= 1), got {value!r}."
            )
        factor_dict[dim] = int_value
    return factor_dict


def _validate_lat_chunks(da: xr.DataArray, factor: int, *, lat: str) -> None:
    chunks = da.chunks
    if chunks is None:
        return
    axis = da.get_axis_num(lat)
    bad_chunks = [chunk for chunk in chunks[axis] if chunk % factor != 0]
    if bad_chunks:
        raise ValueError(
            f"conservative coarsen requires chunks along {lat!r} to be multiples "
            f"of factor {factor}; got chunks {chunks[axis]!r}."
        )


def refine(
    ds: xr.Dataset | xr.DataArray,
    factor: dict[str, int],
    method: str = "linear",
) -> xr.Dataset | xr.DataArray:
    """Refine ``ds`` along one or more dimensions by integer factors.

    Produces a ``factor[dim]``-times-denser grid along each dimension
    via :meth:`xr.Dataset.interp`.
    """
    new_coords: dict[str, Sequence[float]] = {}
    for dim, f in factor.items():
        old = ds[dim].values
        if f <= 0:
            raise ValueError(f"refinement factor for {dim!r} must be positive.")
        new_coords[dim] = np.linspace(old.min(), old.max(), (len(old) - 1) * f + 1)
    return ds.interp(new_coords, method=method)


def refine_2d(
    da: xr.DataArray,
    *,
    factor: Mapping[str, int | float],
    lat: str = "lat",
    lon: str = "lon",
    order: int = 3,
    anti_aliasing: bool | None = None,
    mode: Literal["reflect", "constant", "edge", "symmetric", "wrap"] = "reflect",
    cval: float = 0.0,
) -> xr.DataArray:
    """Resize a 2-D ``(lat, lon)`` plate via ``skimage.transform.resize``.

    Order follows scikit-image's spline convention: ``0`` nearest, ``1``
    bilinear, ``2`` biquadratic, ``3`` bicubic, ``4`` biquartic, and ``5``
    biquintic. Leading dimensions are broadcast independently with
    :func:`xarray.apply_ufunc`.

    Args:
        da: Input data with ``lat`` and ``lon`` dimensions.
        factor: Per-axis resize factors. Must include both ``lat`` and ``lon``.
        lat: Name of the latitude-like dimension.
        lon: Name of the longitude-like dimension.
        order: Spline interpolation order from 0 to 5.
        anti_aliasing: Whether to apply scikit-image's anti-aliasing filter.
            ``None`` uses scikit-image's default.
        mode: Boundary extension mode passed to scikit-image.
        cval: Fill value used when ``mode="constant"``.

    Returns:
        Resized data with updated ``lat`` and ``lon`` coordinates.

    Raises:
        ImportError: If scikit-image is not installed.
        ValueError: If required dims or factors are missing, ``order`` is
            outside 0..5, or either resize factor is non-positive.

    Examples:
        >>> refined = refine_2d(da, factor={"lat": 2, "lon": 2}, order=3)
    """
    resize = _get_skimage_resize()
    if lat not in da.dims or lon not in da.dims:
        raise ValueError(f"da must have dims {lat!r} and {lon!r}.")
    if lat not in factor or lon not in factor:
        raise ValueError(f"factor must include both {lat!r} and {lon!r}.")
    # bool is an int subclass, but True/False are not meaningful spline orders.
    if isinstance(order, bool) or not isinstance(order, int):
        raise ValueError(f"order must be an integer in 0..5, got {order!r}.")
    if order not in range(6):
        raise ValueError(f"order must be in 0..5, got {order!r}.")
    if mode not in _RESIZE_MODES:
        valid_modes = sorted(_RESIZE_MODES)
        raise ValueError(f"mode must be one of {valid_modes!r}, got {mode!r}.")

    f_lat = factor[lat]
    f_lon = factor[lon]
    for d, f in ((lat, f_lat), (lon, f_lon)):
        if isinstance(f, bool):
            raise ValueError(f"refinement factor for {d!r} must not be a boolean.")
        if f <= 0:
            raise ValueError(f"refinement factor for {d!r} must be positive.")

    # Match :func:`refine` semantics for integer factors: (n-1)*f + 1 preserves
    # the original endpoints on the refined grid. Non-integer factors fall back
    # to round(size * factor) for backwards-compat with skimage's resize.
    def _new_size(size: int, factor_value: int | float) -> int:
        if isinstance(factor_value, int) or float(factor_value).is_integer():
            return (size - 1) * int(factor_value) + 1
        return max(1, round(size * factor_value))

    n_lat = _new_size(da.sizes[lat], f_lat)
    n_lon = _new_size(da.sizes[lon], f_lon)
    new_lat = _interp_coord(da[lat].values, n_lat)
    new_lon = _interp_coord(da[lon].values, n_lon)

    def _resize_slice(arr2d: np.ndarray) -> np.ndarray:
        arr2d = np.asarray(arr2d, dtype=np.float64)
        return resize(
            arr2d,
            (n_lat, n_lon),
            order=order,
            anti_aliasing=anti_aliasing,
            mode=mode,
            cval=cval,
            preserve_range=True,
        )

    out = xr.apply_ufunc(
        _resize_slice,
        da,
        input_core_dims=[[lat, lon]],
        output_core_dims=[[lat, lon]],
        exclude_dims={lat, lon},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {lat: n_lat, lon: n_lon},
            "allow_rechunk": False,
        },
    )
    return out.assign_coords({lat: new_lat, lon: new_lon})


def _get_skimage_resize() -> Callable[..., np.ndarray]:
    # importlib keeps ty (typecheck) from resolving the optional [image] extra
    # at static-analysis time.
    import importlib

    try:
        transform = importlib.import_module("skimage.transform")
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "refine_2d requires scikit-image. "
            "Install with: pip install 'xr_toolz[image]'"
        ) from exc
    return transform.resize


def _interp_coord(coord: np.ndarray, size: int) -> np.ndarray:
    old = np.asarray(coord)
    old_idx = np.arange(len(old))
    new_idx = np.linspace(0, len(old) - 1, size)
    return np.interp(new_idx, old_idx, old)


def regrid_like(
    ds: xr.Dataset | xr.DataArray,
    target: xr.Dataset | xr.DataArray,
    *,
    dims: Iterable[str] = ("lat", "lon"),
    method: str = "linear",
) -> xr.Dataset | xr.DataArray:
    """Resample ``ds`` onto ``target``'s coordinates along ``dims``.

    Thin :meth:`xr.Dataset.interp` wrapper for the common
    "regrid model output to observation grid" step. Coordinates listed
    in ``dims`` must exist on both ``ds`` and ``target``; values along
    other dims pass through.
    """
    dim_list = list(dims)
    missing_target = [d for d in dim_list if d not in target.coords]
    if missing_target:
        raise ValueError(
            f"target is missing requested dims {missing_target!r} as coords; "
            f"got coords {tuple(target.coords)}. Pass `dims=` explicitly to "
            "regrid only the dims that are actually shared."
        )
    missing_source = [d for d in dim_list if d not in ds.coords]
    if missing_source:
        raise ValueError(
            f"input is missing requested dims {missing_source!r} as coords; "
            f"got coords {tuple(ds.coords)}."
        )
    target_coords = {d: target[d] for d in dim_list}
    return ds.interp(target_coords, method=method)
