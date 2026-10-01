"""Tier-A array primitives for cross-modality coregistration.

The two-tier discipline (``_src/array.py`` pure numpy / scipy; the
sibling ``_src/operators.py`` carrier-aware) mirrors the rest of
geotoolz. Operators wrap these primitives and add GeoTensor /
GeoDataFrame / xvec metadata handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import einx
import numpy as np
import rasterio.windows
from georeader import read
from georeader.geotensor import GeoTensor

from geotoolz._src.shape import require_ndim
from geotoolz.geom._src.array import resolve_resampling


if TYPE_CHECKING:
    from affine import Affine
    from jaxtyping import Float, Num
    from scipy.spatial import KDTree


def _require_axis_aligned(transform: Any, op_name: str) -> None:
    """Reject affine transforms with rotation / shear terms.

    ``_pixel_center_coords`` and the bin-edge logic in
    `points_to_raster_binned` derive ``x``/``y`` from only the ``a/c``
    and ``e/f`` affine terms. For non-axis-aligned grids (``b != 0`` or
    ``d != 0``) that's silently wrong — the world coordinates would
    drop the rotation/shear contribution. Fail loudly upfront
    rather than emit misregistered output.
    """
    if transform.b != 0.0 or transform.d != 0.0:
        raise ValueError(
            f"{op_name} requires an axis-aligned affine "
            "(rotation/shear terms b and d must be 0); got "
            f"b={transform.b!r}, d={transform.d!r}. Reproject to an "
            "axis-aligned grid first (e.g. via "
            "`gz.geom.Reproject(dst_crs=...)`)."
        )


def _pixel_center_coords(
    transform: Any, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(x_centers, y_centers)`` for a raster from its affine.

    Returns 1-D arrays of length ``W`` and ``H`` respectively — the
    pixel-center coordinates suitable for an xarray-backed extraction.
    Requires an axis-aligned affine (no rotation/shear) — call
    ``_require_axis_aligned`` first.
    """
    h, w = shape[-2], shape[-1]
    x = transform.c + transform.a * (np.arange(w) + 0.5)
    y = transform.f + transform.e * (np.arange(h) + 0.5)
    return x, y


def _pixel_center_xy(transform: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Row-major ``(H * W, 2)`` XY array of every pixel center."""
    x_centers, y_centers = _pixel_center_coords(transform, shape)
    yy, xx = np.meshgrid(y_centers, x_centers, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _pixel_center_tree(transform: Any, shape: tuple[int, ...]) -> KDTree:
    """KDTree over the pixel centers (row-major flat pixel indices)."""
    from scipy.spatial import KDTree

    return KDTree(_pixel_center_xy(transform, shape))


def _values_to_dataarray(values: np.ndarray, transform: Any, op_name: str) -> Any:
    """Wrap ``(H, W)`` / ``(C, H, W)`` values in a pixel-center-coord DataArray.

    The result has ``x`` / ``y`` 1-D coords at pixel centers and a
    ``band`` dim for 3-D input. Errors (non-axis-aligned affine,
    unsupported rank) name ``op_name``.
    """
    import xarray as xr

    _require_axis_aligned(transform, op_name)
    arr = np.asarray(values)
    require_ndim(arr, (2, 3), op_name)
    x, y = _pixel_center_coords(transform, arr.shape)
    if arr.ndim == 2:
        return xr.DataArray(arr, dims=("y", "x"), coords={"x": x, "y": y})
    return xr.DataArray(
        arr,
        dims=("band", "y", "x"),
        coords={"band": np.arange(arr.shape[0]), "x": x, "y": y},
    )


def reproject_like(
    src_values: Num[np.ndarray, "*batch h w"],
    src_transform: Affine,
    src_crs: Any,
    *,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    dst_crs: Any,
    resampling: Literal[
        "nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average"
    ] = "bilinear",
    fill_value: float | None = None,
) -> Num[np.ndarray, "*batch dst_h dst_w"]:
    """Reproject ``src`` onto the destination grid in one shot.

    Composes the (reproject CRS) + (resample to dst pixel grid)
    steps that ``Reproject`` + ``Resample`` would do back-to-back.
    Delegates to :func:`georeader.read.read_reproject`, so a grid that
    only differs by an integer pixel offset is read without warping.

    Args:
        src_values: Source pixel values, ``(H, W)`` or ``(..., H, W)``.
        src_transform: Affine transform of the source grid.
        src_crs: CRS of the source grid.
        dst_shape: Destination spatial shape ``(H', W')``.
        dst_transform: Affine transform of the destination grid.
        dst_crs: CRS of the destination grid.
        resampling: Rasterio resampling kernel name.
        fill_value: Nodata value of the source, also written to
            destination pixels no source pixel covers. ``None`` means
            ``NaN`` for floating input and ``0`` otherwise.

    Returns:
        The warped values on the destination grid, same leading dims
        and dtype as ``src_values``.
    """
    values = np.asarray(src_values)
    if fill_value is None:
        fill_value = np.nan if np.issubdtype(values.dtype, np.floating) else 0
    src = GeoTensor(
        values, transform=src_transform, crs=src_crs, fill_value_default=fill_value
    )
    height, width = dst_shape
    return np.asarray(
        read.read_reproject(
            src,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            window_out=rasterio.windows.Window(
                col_off=0, row_off=0, width=width, height=height
            ),
            resampling=resolve_resampling(resampling),
            return_only_data=True,
            dst_nodata=fill_value,
        )
    )


def points_to_raster_binned(
    points_xy: Float[np.ndarray, "n 2"],
    point_values: Num[np.ndarray, " n"],
    *,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    stat: Literal["mean", "median", "sum", "count", "max", "min"] = "mean",
) -> Float[np.ndarray, "h w"]:
    """Bin scattered point measurements onto a regular grid.

    Uses ``scipy.stats.binned_statistic_2d`` with the destination
    grid's pixel edges as bins. North-up (negative ``e``) and west-up
    (negative ``a``) grids are handled: the output keeps the raster
    convention (row 0 = first row of the affine, col 0 = first column).

    Args:
        points_xy: Point coordinates as an ``(N, 2)`` XY array in the
            destination grid's CRS.
        point_values: Parallel ``(N,)`` array of measurement values.
        dst_shape: Destination spatial shape ``(H, W)``.
        dst_transform: Affine transform of the destination grid (must
            be axis-aligned).
        stat: Per-cell reduction statistic.

    Returns:
        The binned ``float64`` ``(H, W)`` grid; cells with no points
        are ``NaN`` (``0`` for ``stat="count"``).

    Raises:
        ValueError: If ``dst_transform`` has rotation / shear terms.
    """
    from scipy.stats import binned_statistic_2d

    _require_axis_aligned(dst_transform, "points_to_raster_binned")
    xy = np.asarray(points_xy, dtype=np.float64)
    h, w = dst_shape
    # The affine maps pixel (col, row) corners → CRS coords; an H x W
    # grid has W+1 x edges and H+1 y edges.
    x_edges = dst_transform.c + dst_transform.a * np.arange(w + 1)
    y_edges = dst_transform.f + dst_transform.e * np.arange(h + 1)
    # binned_statistic_2d requires monotonically-increasing edges.
    # rasterio's "north-up" affine has negative `e`, which makes y_edges
    # decrease; "west-up" rasters with negative `a` likewise make x_edges
    # decrease. Sort each axis independently and flip the corresponding
    # output dimension so the result keeps the raster convention.
    flip_y = y_edges[0] > y_edges[-1]
    flip_x = x_edges[0] > x_edges[-1]
    y_edges_sorted = y_edges[::-1] if flip_y else y_edges
    x_edges_sorted = x_edges[::-1] if flip_x else x_edges

    stat_out, _, _, _ = binned_statistic_2d(
        xy[:, 0],
        xy[:, 1],
        point_values,
        statistic=stat,
        bins=[x_edges_sorted, y_edges_sorted],
    )
    # `binned_statistic_2d` returns shape (n_x_bins, n_y_bins); swap to
    # the (H, W) raster convention, with rows = y.
    result = einx.id("x y -> y x", stat_out)
    if flip_y:
        result = result[::-1, :]
    if flip_x:
        result = result[:, ::-1]
    return result.astype(np.float64)


def points_to_raster_idw(
    points_xy: Float[np.ndarray, "n 2"],
    point_values: Num[np.ndarray, " n"],
    *,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    k: int = 8,
    power: float = 2.0,
    max_radius: float | None = None,
) -> Float[np.ndarray, "h w"]:
    """Inverse-distance-weight scattered points onto a regular grid.

    Each pixel center takes the IDW mean of its ``k`` nearest points
    (KDTree), with weights ``1 / max(d ** power, 1e-12)``.

    Args:
        points_xy: Point coordinates as an ``(N, 2)`` XY array in the
            destination grid's CRS.
        point_values: Parallel ``(N,)`` array of measurement values.
        dst_shape: Destination spatial shape ``(H, W)``.
        dst_transform: Affine transform of the destination grid (must
            be axis-aligned).
        k: Number of nearest points per pixel (clamped to ``N``).
        power: Inverse-distance exponent.
        max_radius: Optional distance ceiling in CRS units; pixels whose
            nearest point is farther than this are ``NaN``.

    Returns:
        The ``float64`` ``(H, W)`` grid; all-``NaN`` for an empty cloud.

    Raises:
        ValueError: If ``dst_transform`` has rotation / shear terms.
    """
    from scipy.spatial import KDTree

    _require_axis_aligned(dst_transform, "points_to_raster_idw")
    xy = np.asarray(points_xy, dtype=np.float64)
    values = np.asarray(point_values)
    h, w = dst_shape
    # Empty cloud → all-NaN raster (well-defined rather than a KDTree
    # crash on zero-row input).
    if xy.shape[0] == 0:
        return np.full((h, w), np.nan, dtype=np.float64)

    tree = KDTree(xy)
    eff_k = min(k, xy.shape[0])
    distances, indices = tree.query(_pixel_center_xy(dst_transform, (h, w)), k=eff_k)
    if eff_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    sampled = values[indices]  # (H*W, eff_k)
    weights = 1.0 / np.maximum(distances**power, 1e-12)
    num = (sampled * weights).sum(axis=1)
    den = weights.sum(axis=1)
    result = (num / den).reshape(h, w).astype(np.float64)

    if max_radius is not None:
        # Mask pixels whose nearest point exceeds the radius.
        nearest_dist = distances[:, 0].reshape(h, w)
        result = np.where(nearest_dist > max_radius, np.nan, result)
    return result


def raster_to_point_cloud(
    raster_values: Num[np.ndarray, "*batch h w"],
    raster_transform: Affine,
    cloud_xy: Float[np.ndarray, "n 2"],
    *,
    k: int = 1,
    max_radius: float | None = None,
    method: Literal["nearest", "bilinear", "idw"] = "nearest",
    power: float = 2.0,
) -> Float[np.ndarray, "*batch n"]:
    """Sample raster values onto each point in a point cloud.

    ``"nearest"`` and ``"idw"`` query a KDTree built on the pixel
    centers; ``"bilinear"`` interpolates between pixel centers with
    ``xarray.DataArray.interp`` (points outside the pixel-center hull
    are ``NaN``).

    Args:
        raster_values: Raster pixel values, ``(H, W)`` or ``(C, H, W)``.
        raster_transform: Affine transform of the raster grid (must be
            axis-aligned).
        cloud_xy: Point coordinates as an ``(N, 2)`` XY array in the
            raster's CRS (extra columns are ignored).
        k: Number of nearest pixels per point (``k > 1`` is only
            meaningful for ``method="idw"``; clamped to ``H * W``).
        max_radius: Optional distance ceiling in CRS units; points
            farther than this from any pixel centre get ``NaN``.
        method: ``"nearest"``, ``"bilinear"``, or ``"idw"``.
        power: Inverse-distance exponent for ``method="idw"``.

    Returns:
        Sampled values, ``(N,)`` for 2-D input or ``(C, N)`` for
        multi-band input. ``float64``, except that an empty cloud keeps
        the raster dtype and ``"bilinear"`` without ``max_radius`` keeps
        xarray's interpolation dtype.

    Raises:
        ValueError: If the raster is not 2-D / 3-D or its affine has
            rotation / shear terms.
    """
    arr = np.asarray(raster_values)
    xy = np.asarray(cloud_xy, dtype=np.float64)[:, :2]
    if method == "bilinear":
        return _sample_bilinear(arr, raster_transform, xy, max_radius)
    _require_axis_aligned(raster_transform, "raster_to_point_cloud")
    require_ndim(arr, (2, 3), "raster_to_point_cloud")
    if xy.shape[0] == 0:
        return np.zeros((*arr.shape[:-2], 0), dtype=arr.dtype)

    tree = _pixel_center_tree(raster_transform, arr.shape)
    # Clamp k to the number of available pixels — querying KDTree with
    # k > N returns placeholder out-of-bounds indices that would crash
    # the `flat[indices]` lookup.
    eff_k = min(k, tree.n)
    distances, indices = tree.query(xy, k=eff_k)
    # `KDTree.query` returns 1-D arrays for k=1; promote to (N, 1) so
    # the IDW path is uniform.
    if eff_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Flatten the spatial dims for fancy-index lookup:
    # (H, W) → (H*W,) and (C, H, W) → (C, H*W); gather → (..., N, k).
    flat = arr.reshape(*arr.shape[:-2], -1)
    gathered = flat[..., indices]

    if method == "nearest" or eff_k == 1:
        result = gathered[..., 0].astype(np.float64)
    else:
        weights = 1.0 / np.maximum(distances**power, 1e-12)
        num = (gathered * weights).sum(axis=-1)
        den = weights.sum(axis=-1)
        result = (num / den).astype(np.float64)

    # Out-of-radius gate: use the *closest* pixel's distance.
    if max_radius is not None:
        result[..., distances[:, 0] > max_radius] = np.nan
    return result


def _sample_bilinear(
    arr: np.ndarray, transform: Any, xy: np.ndarray, max_radius: float | None
) -> np.ndarray:
    """Bilinear branch of `raster_to_point_cloud`."""
    da = _values_to_dataarray(arr, transform, "raster_to_point_cloud")
    if xy.shape[0] == 0:
        return np.zeros((*arr.shape[:-2], 0), dtype=arr.dtype)
    interp = da.interp(x=("point", xy[:, 0]), y=("point", xy[:, 1]))
    values = np.asarray(interp.values)
    if max_radius is not None:
        # Mask out points whose nearest pixel center is beyond the radius.
        distances, _ = _pixel_center_tree(transform, arr.shape).query(xy, k=1)
        values = values.astype(np.float64, copy=True)
        values[..., distances > max_radius] = np.nan
    return values
