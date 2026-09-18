"""Grid-to-grid value resampling.

Deterministic refinement (:func:`refine`), aggregation
(:func:`coarsen`), target-grid resampling (:func:`regrid_like`) and
first-order conservative regridding (:func:`regrid_conservative`)
along one or more dimensions. Learned counterparts
(``Downscale``/``Upscale``) live in :mod:`.downscale`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import xarray as xr
from jaxtyping import Float
from scipy import sparse

from xrtoolz.interpolate._src.grid_to_points import _check_dask_chunks
from xrtoolz.utils._src.finite import _finite_mask_da
from xrtoolz.utils._src.optional_imports import _require_optional
from xrtoolz.utils._src.validation import _validate_coarsen_factor


_RESIZE_MODES = frozenset({"reflect", "constant", "edge", "symmetric", "wrap"})

Geometry = Literal["spherical", "planar"]
RegridMode = Literal["mean", "sum"]
Normalize = Literal["destarea", "fracarea"]
BoundsLike = xr.DataArray | np.ndarray
GridLike = xr.Dataset | xr.DataArray | Mapping[str, Sequence[float] | np.ndarray]

#: Longitude period assumed by :func:`regrid_conservative` when unwrapping
#: target longitude bounds into the source's range (degrees).
_LON_WRAP = 360.0


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
    renormalized within each block. For non-integer factors or an
    arbitrary rectilinear target grid use :func:`regrid_conservative`,
    which reduces to this function for aligned integer factors.

    Args:
        ds: Dataset or data array to coarsen.
        factor: Mapping from dimension name to integer coarsen factor.
        lat: Latitude dimension name. Values are expected to be cell centers in
            degrees within the usual [-90, 90] latitude range.
        boundary: Boundary mode forwarded to :meth:`xarray.DataArray.coarsen`.

    Returns:
        Coarsened dataset or data array with the same dimension names.

    Examples:
        ```pycon
        >>> coarsen_conservative(da, {"lat": 4, "lon": 4})
        >>> coarsen_conservative(ds, {"latitude": 2}, lat="latitude")
        ```
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
    mask = _finite_mask_da(da)
    # Single mask multiplication: zero-out NaN cells in da, then weight.
    # mask on the left keeps ty's inference DataArray-shaped (cos_lat is ndarray).
    weights = mask * cos_lat
    numerator = (
        (da.where(mask, 0.0) * cos_lat).coarsen(dim=factor, boundary=boundary).sum()
    )
    denominator = weights.coarsen(dim=factor, boundary=boundary).sum()  # ty: ignore[unresolved-attribute]
    # Mask zero denominators before dividing so we never trigger 0/0 warnings.
    safe_den = denominator.where(denominator > 0)
    return numerator / safe_den


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
        ```pycon
        >>> refined = refine_2d(da, factor={"lat": 2, "lon": 2}, order=3)
        ```
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
    transform = _require_optional(
        "skimage.transform",
        extra="image",
        feature="refine_2d",
        package="scikit-image",
    )
    return transform.resize


def _interp_coord(
    coord: Float[np.ndarray, "n"], size: int
) -> Float[np.ndarray, "size"]:
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

    Assumes ``ds`` and ``target`` share a coordinate space — if they are
    in different CRSs, use :func:`xrtoolz.geo.reproject_match` instead,
    which is CRS-aware and matches ``target``'s pixel grid exactly.
    Interpolation does not preserve integrals; for fluxes, inventories
    and other fields whose area integral must survive the regrid use
    :func:`regrid_conservative`.
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


# ---------- conservative regrid -------------------------------------------


def overlap_weights_1d(
    src_bounds: Float[np.ndarray, "n+1"], tgt_bounds: Float[np.ndarray, "m+1"]
) -> Float[np.ndarray, "m n"]:
    """Overlap length of each source interval with each target interval.

    Entry ``(j, i)`` is ``max(0, min(hi_i, hi_j) - max(lo_i, lo_j))`` for
    source interval ``i`` and target interval ``j``. Bounds may be
    ascending or descending (each interval is normalised to ``lo < hi``),
    and in any unit — pass ``sin(lat)`` bounds to obtain spherical
    latitude weights.

    Args:
        src_bounds: ``n + 1`` cell edges of the source axis.
        tgt_bounds: ``m + 1`` cell edges of the target axis.

    Returns:
        ``(m, n)`` array of non-negative overlap lengths.

    Raises:
        ValueError: If either bounds array is not 1-D with at least two
            edges.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from xrtoolz.interpolate import overlap_weights_1d
        >>> src, tgt = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.5, 3.0])
        >>> overlap_weights_1d(src, tgt)
        array([[1. , 0.5, 0. ],
               [0. , 0.5, 1. ]])

        ```
    """
    src = np.asarray(src_bounds, dtype=float)
    tgt = np.asarray(tgt_bounds, dtype=float)
    if src.ndim != 1 or tgt.ndim != 1 or src.size < 2 or tgt.size < 2:
        raise ValueError(
            "src_bounds and tgt_bounds must be 1-D arrays of at least two edges; "
            f"got shapes {src.shape} and {tgt.shape}"
        )
    return _overlap_intervals(
        _edges_to_intervals(src), _edges_to_intervals(tgt)
    ).toarray()


def _edges_to_intervals(edges: np.ndarray) -> np.ndarray:
    """``(n + 1,)`` edges to ``(n, 2)`` ``(lo, hi)`` intervals."""
    return np.stack([edges[:-1], edges[1:]], axis=1)


def _overlap_intervals(src: np.ndarray, tgt: np.ndarray) -> sparse.csr_matrix:
    """Sparse ``(m, n)`` overlap lengths between source and target intervals.

    ``src`` is ``(n, 2)`` and ``tgt`` is ``(m, 2)``; each interval is
    normalised to ``lo <= hi``. Candidate pairs come from two
    ``searchsorted`` sweeps over the source intervals sorted by ``lo``
    (against the running maximum of ``hi``, so overlapping source cells
    are handled too), so construction is linear in the number of
    non-zero overlaps rather than in ``m * n``.
    """
    src_lo, src_hi = np.min(src, axis=1), np.max(src, axis=1)
    tgt_lo, tgt_hi = np.min(tgt, axis=1), np.max(tgt, axis=1)
    order = np.argsort(src_lo, kind="stable")
    lo_sorted = src_lo[order]
    hi_running = np.maximum.accumulate(src_hi[order])
    start = np.searchsorted(hi_running, tgt_lo, side="right")
    stop = np.maximum(np.searchsorted(lo_sorted, tgt_hi, side="left"), start)
    counts = stop - start
    rows = np.repeat(np.arange(tgt.shape[0]), counts)
    # Source positions (in sorted order) of every candidate pair.
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    cols = order[np.repeat(start, counts) + offsets]
    overlap = np.minimum(src_hi[cols], tgt_hi[rows]) - np.maximum(
        src_lo[cols], tgt_lo[rows]
    )
    keep = overlap > 0.0
    return sparse.csr_matrix(
        (overlap[keep], (rows[keep], cols[keep])), shape=(tgt.shape[0], src.shape[0])
    )


def _bounds_from_centres(centres: np.ndarray, name: str) -> np.ndarray:
    """Cell edges at the midpoints between centres, end cells symmetric."""
    c = np.asarray(centres, dtype=float)
    if c.ndim != 1:
        raise ValueError(f"coordinate {name!r} must be 1-D, got shape {c.shape}")
    if c.size < 2:
        raise ValueError(
            f"cannot infer cell bounds for {name!r} from a single centre; "
            "pass bounds explicitly"
        )
    d = np.diff(c)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError(f"coordinate {name!r} must be strictly monotone")
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[2.0 * c[0] - mid[0]], mid, [2.0 * c[-1] - mid[-1]]])


def _as_intervals(bounds: BoundsLike, name: str, n: int) -> np.ndarray:
    """Normalise ``(n + 1,)`` edges or CF ``(n, 2)`` bounds to ``(n, 2)``
    ``(lo, hi)`` intervals. CF bounds need not be contiguous."""
    b = np.asarray(bounds.values if isinstance(bounds, xr.DataArray) else bounds)
    b = b.astype(float)
    if b.ndim == 2 and b.shape == (n, 2):
        return b
    if b.ndim == 1 and b.size == n + 1:
        return _edges_to_intervals(b)
    raise ValueError(
        f"bounds for {name!r} must have shape ({n + 1},) or ({n}, 2); got {b.shape}"
    )


def _centres(grid: GridLike, dim: str, *, what: str) -> np.ndarray:
    if isinstance(grid, xr.Dataset | xr.DataArray):
        if dim not in grid.coords:
            raise ValueError(
                f"{what} is missing a coordinate for {dim!r}; got coords "
                f"{tuple(grid.coords)}"
            )
        return np.asarray(grid[dim].values, dtype=float)
    if dim not in grid:
        raise ValueError(f"{what} mapping is missing {dim!r}; got {tuple(grid)}")
    return np.asarray(grid[dim], dtype=float)


def _cf_bounds_variable(grid: GridLike, dim: str) -> xr.DataArray | None:
    """The CF ``bounds`` variable named by ``grid[dim].attrs``, if present."""
    if not isinstance(grid, xr.Dataset | xr.DataArray):
        return None
    bounds_name = grid[dim].attrs.get("bounds")
    variables = grid.variables if isinstance(grid, xr.Dataset) else grid.coords
    if bounds_name and bounds_name in variables:
        return grid[bounds_name]
    return None


def _resolve_intervals(
    grid: GridLike,
    dim: str,
    explicit: Mapping[str, BoundsLike] | None,
    *,
    what: str,
) -> np.ndarray:
    """``(n, 2)`` cell intervals for ``dim``: explicit > CF ``bounds``
    attribute > midpoints between centres."""
    centres = _centres(grid, dim, what=what)
    n = centres.size
    if explicit is not None and dim in explicit:
        return _as_intervals(explicit[dim], dim, n)
    cf_bounds = _cf_bounds_variable(grid, dim)
    if cf_bounds is not None:
        return _as_intervals(np.asarray(cf_bounds.values), dim, n)
    return _edges_to_intervals(_bounds_from_centres(centres, dim))


def _lon_overlap_weights(src: np.ndarray, tgt: np.ndarray) -> sparse.csr_matrix:
    """Longitude overlap (radians) with the target unwrapped onto the source.

    The target intervals are shifted by a multiple of ``_LON_WRAP`` so they
    start inside the source's range, and the source is tiled one period
    either side, so a target cell straddling the source's seam picks up
    both parts.
    """
    for name, intervals in (("source", src), ("target", tgt)):
        span = intervals.max() - intervals.min()
        if span > _LON_WRAP * (1.0 + 1e-9):
            raise ValueError(
                f"{name} longitude bounds span {span:g} degrees, more than one "
                f"period of {_LON_WRAP:g}"
            )
    shift = np.floor((tgt.min() - src.min()) / _LON_WRAP) * _LON_WRAP
    tgt = tgt - shift
    weights = (
        _overlap_intervals(src - _LON_WRAP, tgt)
        + _overlap_intervals(src, tgt)
        + _overlap_intervals(src + _LON_WRAP, tgt)
    )
    return weights * np.deg2rad(1.0)


def _sin_lat(intervals: np.ndarray) -> np.ndarray:
    return np.sin(np.deg2rad(np.clip(intervals, -90.0, 90.0)))


def _cell_lengths(intervals: np.ndarray) -> np.ndarray:
    return np.abs(intervals[:, 1] - intervals[:, 0])


def regrid_conservative(
    ds: xr.Dataset | xr.DataArray,
    target: GridLike,
    *,
    dims: tuple[str, str] = ("lat", "lon"),
    geometry: Geometry = "spherical",
    bounds: Mapping[str, BoundsLike] | None = None,
    target_bounds: Mapping[str, BoundsLike] | None = None,
    mode: RegridMode = "mean",
    normalize: Normalize = "fracarea",
    skipna: bool = True,
) -> xr.Dataset | xr.DataArray:
    r"""First-order conservative regrid along two rectilinear dims.

    Source cells $i$ (area $a_i$, value $f_i$) and target cells $j$ overlap
    on areas $w_{ij} = |c_i \cap c_j|$. Because both grids are rectilinear
    the overlap factorises into one 1-D weight matrix per axis
    (:func:`overlap_weights_1d`), so the whole regrid is two matrix
    products per field:

    $$\text{mean: } \bar f_j = \frac{\sum_i w_{ij} f_i}{\sum_i w_{ij}\,[f_i
    \text{ valid}]},\qquad \text{sum: } F_j = \sum_i \frac{w_{ij}}{a_i} f_i$$

    ``mode="mean"`` preserves *intensive* fields (concentration, mixing
    ratio, flux density). With ``normalize="destarea"`` the area integral
    $\sum_j A_j \bar f_j = \sum_i a_i f_i$ is conserved whenever the target
    covers the source; with ``normalize="fracarea"`` (default) it is
    conserved only over target cells that are fully covered by unmasked
    source cells — a partially covered or partially masked target cell
    holds the mean of its valid part, which over-counts that cell's
    contribution to the integral. ``mode="sum"`` preserves *extensive*
    per-cell fields (mass, emission per cell): each source cell's value
    is split among the target cells in proportion to the overlapped
    fraction, so $\sum_j F_j = \sum_i f_i$.

    On a ``"spherical"`` geometry ``dims[0]`` is latitude and ``dims[1]``
    longitude, both in degrees, and cell areas are $R^2 \Delta\lambda
    (\sin\varphi_+ - \sin\varphi_-)$ ($R$ cancels). Longitude is periodic
    with period 360 degrees: the target bounds are unwrapped into the
    source's range, so a source on 0–360 regrids onto a target on
    −180–180 (and vice versa) without pre-rolling. On a ``"planar"``
    geometry areas are $\Delta x\,\Delta y$ in coordinate units and nothing
    is periodic.

    Cell bounds default to the midpoints between centres, with the end
    cells extrapolated symmetrically. A CF ``bounds`` attribute on a
    coordinate (naming an ``(n, 2)`` variable) is honoured, and ``bounds``
    / ``target_bounds`` override both, as ``(n + 1,)`` edges or CF
    ``(n, 2)`` bounds per dim. CF bounds need not be contiguous: cells
    may leave gaps (or overlap), and only the overlap with each cell's
    own ``(lo, hi)`` interval counts.

    Missing values: with ``skipna=True`` (default) NaN source cells carry
    no weight. ``normalize="fracarea"`` divides by the *valid* overlap, so
    a partially masked (or partially covered) target cell keeps the mean
    of its valid part; ``"destarea"`` divides by the full target cell area
    (ESMF / xESMF naming), scaling the mean down by the valid fraction.
    A target cell with no valid overlap is NaN in every mode. With
    ``skipna=False`` any NaN source cell that touches a target cell makes
    it NaN. Complex fields are regridded component-wise (a cell is valid
    when both parts are finite) and stay complex.

    Extra dims (``time``, ``level``) broadcast; dask inputs stay lazy
    along them, while the two regridded dims must each sit in a single
    chunk. Datasets are regridded per variable: variables without either
    dim pass through, CF bounds variables of the source grid are dropped
    (those of the target grid, when it is a dataset / data array carrying
    them, are attached), and a variable carrying only one of the two dims
    raises. A data array output has no room for a bounds variable, so its
    coordinates carry no dangling ``bounds`` attribute.

    Args:
        ds: Source dataset or data array with 1-D, strictly monotone
            coordinates for both ``dims``.
        target: Target grid — a dataset / data array whose coordinates
            include ``dims``, or a mapping ``{dim: centres}``.
        dims: The two dims to regrid, ``(lat, lon)`` order on a spherical
            geometry.
        geometry: ``"spherical"`` (degrees, sin-latitude weighting,
            periodic longitude) or ``"planar"``.
        bounds: Explicit source cell bounds per dim.
        target_bounds: Explicit target cell bounds per dim.
        mode: ``"mean"`` (intensive) or ``"sum"`` (extensive).
        normalize: ``"fracarea"`` or ``"destarea"``; only affects
            ``mode="mean"``.
        skipna: Whether NaN source cells are skipped (default) or
            propagate.

    Returns:
        ``ds`` on the target grid, with the target centres as coordinates
        along ``dims`` and every other dim unchanged. Values are
        ``float64`` (``complex128`` for complex input).

    Raises:
        ValueError: If an option is unknown, a dim or its coordinate is
            missing, a coordinate is not strictly monotone, bounds have the
            wrong shape, longitude bounds span more than one period, a dask
            input is chunked along a regridded dim, or a Dataset variable
            carries only one of the two dims.

    Example:
        Move a 1-degree per-cell emission inventory (0–360 longitudes) onto
        a 2-degree grid on −180–180; the global total is preserved to
        round-off:

        ```pycon
        >>> import numpy as np, xarray as xr
        >>> from xrtoolz.interpolate import regrid_conservative
        >>> lat, lon = np.arange(-89.5, 90.0, 1.0), np.arange(0.5, 360.0, 1.0)
        >>> rng = np.random.default_rng(0)
        >>> emis = xr.DataArray(
        ...     rng.uniform(0.0, 1.0, size=(lat.size, lon.size)),
        ...     dims=("lat", "lon"),
        ...     coords={"lat": lat, "lon": lon},
        ... )
        >>> target = {
        ...     "lat": np.arange(-89.0, 90.0, 2.0),
        ...     "lon": np.arange(-179.0, 180.0, 2.0),
        ... }
        >>> out = regrid_conservative(emis, target, mode="sum")
        >>> out.sizes
        Frozen({'lat': 90, 'lon': 180})
        >>> bool(np.isclose(float(out.sum()), float(emis.sum()), rtol=1e-12))
        True

        ```
    """
    if geometry not in ("spherical", "planar"):
        raise ValueError(f"geometry must be 'spherical' or 'planar', got {geometry!r}")
    if mode not in ("mean", "sum"):
        raise ValueError(f"mode must be 'mean' or 'sum', got {mode!r}")
    if normalize not in ("destarea", "fracarea"):
        raise ValueError(
            f"normalize must be 'destarea' or 'fracarea', got {normalize!r}"
        )
    dims = tuple(dims)
    if len(dims) != 2:
        raise ValueError(f"dims must name exactly two dimensions, got {dims!r}")
    missing = [d for d in dims if d not in ds.dims]
    if missing:
        raise ValueError(f"input is missing dims {missing!r}; got {tuple(ds.dims)}")

    src_iv = {d: _resolve_intervals(ds, d, bounds, what="input") for d in dims}
    tgt_iv = {
        d: _resolve_intervals(target, d, target_bounds, what="target") for d in dims
    }
    tgt_centres = {d: _centres(target, d, what="target") for d in dims}
    d0, d1 = dims
    if geometry == "spherical":
        src_len = {d0: _sin_lat(src_iv[d0]), d1: np.deg2rad(src_iv[d1])}
        tgt_len = {d0: _sin_lat(tgt_iv[d0]), d1: np.deg2rad(tgt_iv[d1])}
        w0 = _overlap_intervals(src_len[d0], tgt_len[d0])
        w1 = _lon_overlap_weights(src_iv[d1], tgt_iv[d1])
    else:
        src_len, tgt_len = src_iv, tgt_iv
        w0 = _overlap_intervals(src_len[d0], tgt_len[d0])
        w1 = _overlap_intervals(src_len[d1], tgt_len[d1])
    src_area = np.outer(_cell_lengths(src_len[d0]), _cell_lengths(src_len[d1]))
    tgt_area = np.outer(_cell_lengths(tgt_len[d0]), _cell_lengths(tgt_len[d1]))

    # A Dataset output carries the target's CF bounds variables that were
    # actually used (not overridden by ``target_bounds``) alongside the
    # ``bounds`` attr; otherwise the attr would dangle, so it is dropped.
    keep_bounds = isinstance(ds, xr.Dataset)
    tgt_bounds_vars: dict[str, xr.DataArray] = {}
    new_coords = {}
    for d in dims:
        attrs = (
            dict(target[d].attrs)
            if isinstance(target, xr.Dataset | xr.DataArray)
            else {}
        )
        cf_bounds = _cf_bounds_variable(target, d)
        overridden = target_bounds is not None and d in target_bounds
        if keep_bounds and cf_bounds is not None and not overridden:
            tgt_bounds_vars[str(cf_bounds.name)] = cf_bounds
        else:
            attrs.pop("bounds", None)
        new_coords[d] = xr.DataArray(tgt_centres[d], dims=(d,), attrs=attrs)

    def _regrid(da: xr.DataArray) -> xr.DataArray:
        return _regrid_conservative_dataarray(
            da,
            dims=dims,
            w0=w0,
            w1=w1,
            src_area=src_area,
            tgt_area=tgt_area,
            new_coords=new_coords,
            mode=mode,
            normalize=normalize,
            skipna=skipna,
        )

    if isinstance(ds, xr.DataArray):
        return _regrid(ds)

    bounds_vars = {ds[d].attrs.get("bounds") for d in dims} - {None}
    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if name in bounds_vars:
            continue
        present = [d for d in dims if d in da.dims]
        if not present:
            out_vars[str(name)] = da
        elif len(present) == 2:
            out_vars[str(name)] = _regrid(da)
        else:
            raise ValueError(
                f"variable {name!r} carries only {present[0]!r} of the regridded "
                f"dims {dims!r}; drop it or regrid it separately"
            )
    out = xr.Dataset(out_vars, attrs=dict(ds.attrs))
    for name, bnds in tgt_bounds_vars.items():
        out[name] = bnds.variable
    extra_coords = {
        name: coord
        for name, coord in ds.coords.items()
        if name not in out.coords and not set(dims) & set(coord.dims)
    }
    return out.assign_coords(extra_coords) if extra_coords else out


def _regrid_conservative_dataarray(
    da: xr.DataArray,
    *,
    dims: tuple[str, str],
    w0: sparse.csr_matrix,
    w1: sparse.csr_matrix,
    src_area: np.ndarray,
    tgt_area: np.ndarray,
    new_coords: Mapping[str, xr.DataArray],
    mode: RegridMode,
    normalize: Normalize,
    skipna: bool,
) -> xr.DataArray:
    _check_dask_chunks(da, dims, func_name="regrid_conservative")
    d0, d1 = dims
    w0 = sparse.csr_matrix(w0)
    w1 = sparse.csr_matrix(w1)
    out_dtype = np.complex128 if np.iscomplexobj(da.data) else np.float64

    def _weights(x: np.ndarray) -> np.ndarray:
        """``w0 @ x @ w1.T`` over the two trailing axes of ``x``."""
        return _contract(w0, _contract(w1, x, -1), -2)

    def _kernel(field: np.ndarray) -> np.ndarray:
        f = np.asarray(field)
        f = f if np.iscomplexobj(f) else np.asarray(f, dtype=float)
        # ``np.where(..., np.nan)`` on complex data gives ``nan+0j``: fill
        # uncovered cells with NaN in both parts instead.
        fill = complex(np.nan, np.nan) if np.iscomplexobj(f) else np.nan
        valid = np.isfinite(f)
        f0 = np.where(valid, f, 0.0)
        if mode == "sum":
            f0 = np.divide(f0, src_area, out=np.zeros_like(f0), where=src_area > 0)
        num = _weights(f0)
        valid_w = _weights(valid.astype(float))
        covered = valid_w > 0
        if mode == "mean":
            den = valid_w if normalize == "fracarea" else tgt_area
            out = np.where(covered, num / np.where(covered, den, 1.0), fill)
        else:
            out = np.where(covered, num, fill)
        if not skipna:
            invalid_w = _weights((~valid).astype(float))
            out = np.where(invalid_w > 0, fill, out)
        return out

    out = xr.apply_ufunc(
        _kernel,
        da,
        input_core_dims=[[d0, d1]],
        output_core_dims=[[d0, d1]],
        exclude_dims={d0, d1},
        dask="parallelized",
        output_dtypes=[out_dtype],
        keep_attrs=True,
        dask_gufunc_kwargs={
            "output_sizes": {d0: w0.shape[0], d1: w1.shape[0]},
            "allow_rechunk": False,
        },
    )
    return out.assign_coords(new_coords).transpose(*da.dims)


def _contract(w: sparse.csr_matrix, x: np.ndarray, axis: int) -> np.ndarray:
    """Apply the sparse ``(m, n)`` matrix ``w`` along ``axis`` of ``x``.

    ``x`` is flattened to ``(n, -1)`` with ``axis`` in front so the product
    is a single sparse-dense matmul; the result has size ``m`` along
    ``axis`` and every other axis unchanged.
    """
    front = np.moveaxis(x, axis, 0)
    rest = front.shape[1:]
    y = w @ front.reshape(front.shape[0], -1)
    return np.moveaxis(np.asarray(y).reshape((w.shape[0], *rest)), 0, axis)
