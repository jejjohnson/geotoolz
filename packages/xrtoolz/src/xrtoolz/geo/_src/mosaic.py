"""Multi-granule mosaicking — merge georeferenced tiles into one cube.

Thin wrapper around :mod:`rioxarray.merge`. Inputs that carry differing
CRSs are reprojected onto the first input's CRS before the merge, so a
mixed-CRS granule list produces one coherent output rather than a
silently misaligned union.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import rioxarray  # noqa: F401  — needed so ds.rio is populated
import xarray as xr


MosaicMethod = Literal["first", "last", "min", "max", "sum"]

_VALID_METHODS: tuple[str, ...] = ("first", "last", "min", "max", "sum")


def _require_crs(obj: xr.DataArray | xr.Dataset, index: int):
    """Return ``obj``'s CRS, raising if it has none."""
    crs = obj.rio.crs
    if crs is None:
        raise ValueError(
            f"input {index} has no CRS; every mosaic input must carry one "
            "(set it with `xrtoolz.geo.assign_crs` or `ds.rio.write_crs`)."
        )
    return crs


def _require_matching_variables(items: list) -> None:
    """Reject Dataset inputs whose ``data_vars`` sets differ.

    ``merge_datasets`` treats the first input as the representative
    schema: a variable present only in a later tile is silently dropped,
    and one missing from a later tile fails with a bare ``KeyError``.
    Neither is a useful outcome, so demand identical schemas up front.
    """
    reference = set(items[0].data_vars)
    for index, item in enumerate(items[1:], start=1):
        variables = set(item.data_vars)
        if variables != reference:
            missing = sorted(reference - variables)
            extra = sorted(variables - reference)
            raise ValueError(
                f"input {index} has a different variable set than input 0; "
                f"missing {missing}, unexpected {extra}. Every Dataset in a "
                "mosaic must carry the same data_vars — subset them to a "
                "common set first."
            )


def _widen_integers_for_sum(obj):
    """Promote integer data to ``int64`` so ``method="sum"`` cannot wrap.

    ``merge_*`` allocates the output in the first input's dtype, so summing
    overlapping ``uint8`` tiles silently wraps (200 + 200 → 144). Every
    rasterio-supported integer type widens losslessly into ``int64``.
    """
    if isinstance(obj, xr.Dataset):
        widened = obj.map(
            lambda da: (
                da.astype("int64") if np.issubdtype(da.dtype, np.integer) else da
            ),
            keep_attrs=True,
        )
        return widened.rio.write_crs(obj.rio.crs)
    if np.issubdtype(obj.dtype, np.integer):
        return obj.astype("int64").rio.write_crs(obj.rio.crs)
    return obj


def spatial_mosaic(
    datasets: Sequence[xr.DataArray] | Sequence[xr.Dataset],
    *,
    method: MosaicMethod = "first",
    resampling: str = "nearest",
    nodata: float | None = None,
) -> xr.DataArray | xr.Dataset:
    """Merge overlapping / adjacent georeferenced inputs into a single cube.

    Wraps :func:`rioxarray.merge.merge_arrays` (DataArray inputs) or
    :func:`rioxarray.merge.merge_datasets` (Dataset inputs). All inputs
    must carry a CRS (``ds.rio.crs``); any input whose CRS differs from
    the first input's is reprojected onto it before merging. Overlapping
    pixels are resolved by ``method``. The result spans the union of the
    input extents.

    Args:
        datasets: One or more DataArrays *or* one or more Datasets — the
            sequence must be homogeneous, and Datasets must all carry the
            same ``data_vars``. Each must have a CRS attached. A
            single-element sequence is returned unchanged.
        method: Overlap resolution rule — one of ``"first"``, ``"last"``,
            ``"min"``, ``"max"``, ``"sum"``. ``"first"`` keeps the value
            from the earliest input covering a pixel. ``"sum"`` promotes
            integer data to ``int64`` so overlapping values cannot wrap.
        resampling: Name of the :class:`rasterio.enums.Resampling` member
            used when harmonising inputs whose CRS differs from the first
            input's — they are reprojected at the first input's CRS *and*
            resolution, so this governs the final values. Ignored when
            all inputs already share a CRS. Default ``"nearest"`` (safe
            for categorical data; use ``"bilinear"`` for continuous
            fields).
        nodata: Fill value for cells covered by no input. ``None`` defers
            to each input's own ``rio.nodata``.

    Returns:
        A single DataArray (for DataArray inputs) or Dataset (for Dataset
        inputs) spanning the union of the input extents, in the first
        input's CRS.

    Note:
        Inputs must be north-up (descending ``y``); ``rasterio.merge``
        rejects "upside down" rasters outright.

    Raises:
        ValueError: If ``datasets`` is empty, mixes DataArrays and
            Datasets, names an unknown ``method`` / ``resampling``, holds
            Datasets with mismatched ``data_vars``, or if any input lacks
            a CRS.

    Example:
        >>> import numpy as np, xarray as xr
        >>> import rioxarray  # noqa: F401
        >>> from xrtoolz.geo import spatial_mosaic
        >>> def tile(x0):
        ...     # North-up: y descends, as in every georeferenced raster.
        ...     return xr.DataArray(
        ...         np.ones((4, 4)),
        ...         dims=("y", "x"),
        ...         coords={"y": np.arange(3.0, -1.0, -1.0), "x": x0 + np.arange(4.0)},
        ...     ).rio.write_crs("EPSG:4326")
        >>> mosaic = spatial_mosaic([tile(0.0), tile(4.0)])
        >>> int(mosaic.sizes["x"])
        8

    """
    from rasterio.enums import Resampling
    from rioxarray.merge import merge_arrays, merge_datasets

    items = list(datasets)
    if not items:
        raise ValueError("spatial_mosaic requires at least one input dataset.")

    if not hasattr(Resampling, resampling):
        valid = [r.name for r in Resampling]
        raise ValueError(f"Unknown resampling {resampling!r}; expected one of {valid}.")

    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}; expected one of {list(_VALID_METHODS)}."
        )

    is_dataset = [isinstance(item, xr.Dataset) for item in items]
    if any(is_dataset) and not all(is_dataset):
        raise ValueError(
            "spatial_mosaic requires a homogeneous sequence — either all "
            "DataArrays or all Datasets, not a mix."
        )
    if all(is_dataset):
        _require_matching_variables(items)

    target_crs = _require_crs(items[0], 0)
    # Reproject differing-CRS inputs at the first tile's *resolution* as well
    # as its CRS. Letting rioxarray derive the resolution from the source
    # would leave a cell-size mismatch for merge_* to resolve, and that
    # merge-time resample is hard-coded to nearest — so `resampling` would
    # not actually govern the output values.
    target_resolution = items[0].rio.resolution()
    harmonised = [items[0]]
    for index, item in enumerate(items[1:], start=1):
        crs = _require_crs(item, index)
        if crs == target_crs:
            harmonised.append(item)
        else:
            harmonised.append(
                item.rio.reproject(
                    target_crs,
                    resolution=target_resolution,
                    resampling=getattr(Resampling, resampling),
                )
            )

    if len(harmonised) == 1:
        return harmonised[0]

    if method == "sum":
        harmonised = [_widen_integers_for_sum(item) for item in harmonised]

    merge = merge_datasets if all(is_dataset) else merge_arrays
    return merge(harmonised, method=method, nodata=nodata)


__all__ = ["MosaicMethod", "spatial_mosaic"]
