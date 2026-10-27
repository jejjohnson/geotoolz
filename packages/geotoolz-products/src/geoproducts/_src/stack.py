"""Put several readers on one grid.

Sensors ship bands at different resolutions (GOES ABI: one channel per
file at 0.5, 1 or 2 km, L2 products at 2, 4 or 10 km), and analyses mix
products from several readers. :func:`stack` reads each reader only over
the target area and warps it onto a reference grid, without the operator
library. georeader computes the windows and the warps (the internal
engine); readers go in and a `geoarray.GeoArray` comes out.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import geoarray
import numpy as np
from geoarray import FrozenAttrs, GeoArray, RasterSource
from geoarray.coords import SpectralAxis
from rasterio.enums import Resampling

from geoproducts._src.carrier import require_source


__all__ = ["stack"]


def stack(
    readers: Sequence[RasterSource],
    *,
    bounds: tuple[float, float, float, float] | None = None,
    crs_bounds: Any = None,
    like: int = 0,
    resampling: str | None = None,
) -> GeoArray:
    """Stack readers' bands on the grid of ``readers[like]``.

    The reference reader is read over ``bounds`` (or loaded whole); every
    other reader is read only where it overlaps that grid and warped onto
    it. The bands are concatenated in reader order, with their
    ``band_names`` (and ``units`` / ``wavelengths`` when every reader
    carries them).

    Args:
        readers: Readers to stack: any `geoarray.RasterSource` — a
            ``ProductReader`` (``goes.Reader``, ``goes.L2Reader``, …), a
            ``geocloud.cog.CogSource`` or a loaded ``GeoArray``.
        bounds: ``(minx, miny, maxx, maxy)`` area to read. Default: the
            reference reader's whole extent.
        crs_bounds: CRS of ``bounds``, e.g. ``"EPSG:4326"``. Default: the
            reference reader's CRS.
        like: Index of the reference reader whose grid the output takes.
        resampling: A rasterio resampling name for every warp. Default:
            chosen per reader — ``"average"`` from a finer float grid,
            ``"bilinear"`` from a coarser one, ``"mode"`` / ``"nearest"``
            for integer (mask / class) readers.

    Returns:
        A ``(sum of bands, H, W)`` ``GeoArray`` on the reference grid. It
        keeps an integer dtype when every reader shares one integer dtype
        and fill; otherwise it is ``float32`` with ``NaN`` fill, and every
        reader's nodata pixels (its own fill, integer or float such as
        ``-9999.0``, and non-finite values) become ``NaN``.

    Raises:
        TypeError: A reader is a georeader ``GeoTensor`` / ``GeoData``
            (convert it with ``geoarray.interop.georeader.from_georeader`` /
            ``source_from_georeader``) or not a `geoarray.RasterSource`.
        ValueError: ``readers`` is empty or ``like`` is out of range.

    Examples:
        True-colour inputs at 1 km, from three L1b files::

            blue, red, veggie = (goes.Reader(p, calibration="reflectance")
                                 for p in (c01_path, c02_path, c03_path))
            rgb_in = geoproducts.stack([blue, red, veggie], bounds=aoi,
                                       crs_bounds="EPSG:4326")  # (3, h, w)
    """
    if not readers:
        raise ValueError("stack needs at least one reader.")
    for reader in readers:
        require_source(reader, "geoproducts.stack")
    if not 0 <= like < len(readers):
        raise ValueError(f"like={like} is out of range for {len(readers)} readers.")
    from geoarray.interop.georeader import (
        from_georeader,
        source_to_georeader,
        to_georeader,
    )
    from georeader import read

    reference = readers[like]
    if bounds is None:
        ref = reference.load()
    else:
        ref = from_georeader(
            read.read_from_bounds(
                source_to_georeader(reference), bounds, crs_bounds=crs_bounds
            )
        )
    ref_res = abs(ref.grid.affine.a)
    ref_engine = to_georeader(ref)
    parts: list[GeoArray] = []
    for i, reader in enumerate(readers):
        if i == like:
            parts.append(ref)
            continue
        method = resampling or _auto_resampling(reader, ref_res)
        warped = read.read_reproject_like(
            source_to_georeader(reader), ref_engine, resampling=Resampling[method]
        )
        parts.append(from_georeader(warped))
    return _concatenate(ref, parts, [r.attrs for r in readers], list(readers))


def _auto_resampling(reader: RasterSource, ref_res: float) -> str:
    finer = abs(reader.grid.affine.a) < 0.99 * ref_res
    if np.issubdtype(np.dtype(reader.dtype), np.integer):
        return "mode" if finer else "nearest"
    return "average" if finer else "bilinear"


def _concatenate(
    ref: GeoArray,
    parts: list[GeoArray],
    part_attrs: list[Any],
    sources: list[Any] | None = None,
) -> GeoArray:
    """Concatenate ``parts`` along bands with `geoarray.concat`.

    georeader's warp drops ``attrs`` and coordinates, so each part is first
    rebuilt on the reference grid with its reader's band axis (``coords``)
    and per-band attrs; `geoarray.concat` then joins both, so names,
    wavelengths and FWHM survive whenever every reader declares them.
    """
    keys = {(np.dtype(p.dtype), _fill_key(p.fill_value)) for p in parts}
    integer = len(keys) == 1 and np.issubdtype(next(iter(keys))[0], np.integer)
    fill = parts[0].fill_value if integer else np.nan
    sources = sources if sources is not None and len(sources) == len(parts) else []
    normalized = []
    for i, (part, part_attr) in enumerate(zip(parts, part_attrs, strict=True)):
        values = np.asarray(part) if integer else _nan_filled(part)
        count = values.shape[0]
        normalized.append(
            GeoArray(
                values,
                ref.transform,
                ref.crs,
                fill_value=fill,
                # the readers' 0.2 per-band keys, where they fit this part
                attrs=FrozenAttrs(
                    {
                        key: tuple(part_attr[key])
                        for key in ("band_names", "units", "wavelengths")
                        if key in part_attr and len(part_attr[key]) == count
                    }
                ),
                dims=parts[0].dims,
                coords=_band_coords(sources[i] if sources else None, count),
            )
        )
    out = geoarray.concat(normalized, "band")
    # geoarray's per-band keys exclude `units` (a carrier's units are
    # usually one string); the readers declare one per band, so join them
    # (concat keeps the first part's, so drop it unless every part has them)
    units = [g.attrs.get("units") for g in normalized]
    if all(isinstance(u, tuple) for u in units):
        return out.with_attrs(units=tuple(item for u in units for item in u))
    if "units" in out.attrs:
        return out.with_data(out.data, attrs=out.attrs.drop("units"))
    return out


def _band_coords(source: Any, count: int) -> tuple[SpectralAxis, ...]:
    """``source``'s `SpectralAxis` when it has one of ``count`` bands, else ``()``."""
    axes = [c for c in getattr(source, "coords", ()) if isinstance(c, SpectralAxis)]
    return (axes[0],) if len(axes) == 1 and axes[0].size == count else ()


def _nan_filled(part: GeoArray) -> np.ndarray:
    """``part`` as ``float32`` with ``NaN`` wherever geoarray reads a band as
    nodata (non-finite or equal to the part's own fill), band by band, so a
    ``-9999.0`` or integer fill is not read as data under the ``NaN`` fill."""
    values = np.asarray(part).astype(np.float32)
    for k in range(values.shape[0]):
        values[k][~np.asarray(part[k : k + 1].valid_mask())] = np.nan
    return values


def _fill_key(fill: Any) -> Any:
    """A hashable fill that treats every ``NaN`` as equal."""
    return (
        "nan" if fill is None or (isinstance(fill, float) and np.isnan(fill)) else fill
    )
