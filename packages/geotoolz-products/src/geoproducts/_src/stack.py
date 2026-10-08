"""Put several readers on one grid.

Sensors ship bands at different resolutions (GOES ABI: one channel per
file at 0.5, 1 or 2 km, L2 products at 2, 4 or 10 km), and analyses mix
products from several readers. :func:`stack` reads each reader only over
the target area and warps it onto a reference grid with georeader, without
the operator library.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from georeader import read
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from rasterio.enums import Resampling

from geoproducts._src.base import ProductReader


__all__ = ["stack"]


def stack(
    readers: Sequence[GeoData],
    *,
    bounds: tuple[float, float, float, float] | None = None,
    crs_bounds: Any = None,
    like: int = 0,
    resampling: str | None = None,
) -> GeoTensor:
    """Stack readers' bands on the grid of ``readers[like]``.

    The reference reader is read over ``bounds`` (or loaded whole); every
    other reader is read only where it overlaps that grid and warped onto
    it. The bands are concatenated in reader order, with their
    ``band_names`` (and ``units`` / ``wavelengths`` when every reader
    carries them).

    Args:
        readers: Readers to stack (any ``ProductReader`` — ``goes.Reader``,
            ``goes.L2Reader``, … — or any georeader ``GeoData``).
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
        A ``(sum of bands, H, W)`` ``GeoTensor`` on the reference grid. It
        keeps an integer dtype when every reader shares one integer dtype
        and fill; otherwise it is ``float32`` with ``NaN`` fill (integer
        fills become ``NaN``).

    Raises:
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
    if not 0 <= like < len(readers):
        raise ValueError(f"like={like} is out of range for {len(readers)} readers.")
    reference = readers[like]
    if bounds is None:
        ref = reference.load()
    else:
        # A GeoData read returns a GeoTensor (the ndarray arm of the
        # annotation is for return_only_data reads).
        ref = cast(
            GeoTensor, read.read_from_bounds(reference, bounds, crs_bounds=crs_bounds)
        )
    ref_res = abs(ref.transform.a)
    parts: list[GeoTensor] = []
    for i, reader in enumerate(readers):
        if i == like:
            parts.append(ref)
            continue
        method = resampling or _auto_resampling(reader, ref_res)
        parts.append(
            read.read_reproject_like(reader, ref, resampling=Resampling[method])
        )
    return _concatenate(ref, parts, [_reader_attrs(r) for r in readers])


def _auto_resampling(reader: GeoData, ref_res: float) -> str:
    finer = abs(reader.transform.a) < 0.99 * ref_res
    if np.issubdtype(np.dtype(reader.dtype), np.integer):
        return "mode" if finer else "nearest"
    return "average" if finer else "bilinear"


def _reader_attrs(reader: GeoData) -> dict[str, Any]:
    """Per-band attrs of a reader (georeader's warp drops ``attrs``)."""
    if isinstance(reader, ProductReader):
        return {"band_names": reader.bands, **reader._band_attrs()}
    bands = getattr(reader, "bands", None)
    return {"band_names": tuple(bands)} if bands else {}


def _concatenate(
    ref: GeoTensor, parts: list[GeoTensor], part_attrs: list[dict[str, Any]]
) -> GeoTensor:
    keys = {(np.dtype(p.dtype), _fill_key(p.fill_value_default)) for p in parts}
    integer = len(keys) == 1 and np.issubdtype(next(iter(keys))[0], np.integer)
    arrays = []
    for part in parts:
        values = np.asarray(part)
        if not integer and not np.issubdtype(values.dtype, np.floating):
            filled = values == part.fill_value_default
            values = values.astype(np.float32)
            values[filled] = np.nan
        arrays.append(values.astype(values.dtype if integer else np.float32))
    attrs: dict[str, Any] = {}
    for key in ("band_names", "units", "wavelengths"):
        values: list[Any] = []
        for part, part_attr in zip(parts, part_attrs, strict=True):
            per_band = part_attr.get(key)
            if per_band is None or len(per_band) != np.asarray(part).shape[0]:
                break
            values.extend(per_band)
        else:
            attrs[key] = tuple(values)
    return GeoTensor(
        np.concatenate(arrays, axis=0),
        transform=ref.transform,
        crs=ref.crs,
        fill_value_default=parts[0].fill_value_default if integer else np.nan,
        attrs=attrs,
    )


def _fill_key(fill: Any) -> Any:
    """A hashable fill that treats every ``NaN`` as equal."""
    return (
        "nan" if fill is None or (isinstance(fill, float) and np.isnan(fill)) else fill
    )
