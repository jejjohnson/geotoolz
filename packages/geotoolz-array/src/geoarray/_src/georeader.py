"""The georeader adapter: `GeoTensor` ↔ `GeoArray`, NumPy only.

georeader is imported here and nowhere else in geoarray (the ``[georeader]``
extra), and only when a function runs. The adapter maps public attributes
one to one; it shares no code with georeader.

| GeoTensor | GeoArray |
| --- | --- |
| ``values`` (``np.ndarray`` view) | ``data`` (the same memory) |
| ``transform`` (``affine.Affine``) | ``transform`` ((6,) float64) |
| ``crs`` | ``crs`` (``pyproj.CRS``) |
| ``fill_value_default`` | ``fill_value`` |
| ``attrs`` (dict) | ``attrs`` (`FrozenAttrs`) |

Lazy readers cross the same way: `source_from_georeader` wraps a georeader
``GeoData`` (a ``RasterioReader``, …) as a `RasterSource`, and
`source_to_georeader` presents a `RasterSource` to georeader's ``read``
functions as a ``GeoData``.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
from affine import Affine

from geoarray._src.array import GeoArray, _normalize_fill
from geoarray._src.attrs import FrozenAttrs
from geoarray._src.grid import Grid, PixelWindow, crs_string
from geoarray._src.source import RasterSource


__all__ = [
    "from_georeader",
    "source_from_georeader",
    "source_to_georeader",
    "to_georeader",
]

_HINT = "the georeader adapter needs georeader: pip install 'geotoolz-array[georeader]'"


def _geotensor_module() -> ModuleType:
    try:
        from georeader import geotensor
    except ImportError as exc:  # pragma: no cover - exercised by the base-install job
        raise ImportError(_HINT) from exc
    return geotensor


def from_georeader(gt: Any) -> GeoArray:
    """A NumPy `GeoArray` from a georeader ``GeoTensor``, sharing its memory.

    Args:
        gt: A ``georeader.geotensor.GeoTensor`` (2-D to 4-D). A lazy
            reader (``GeoData``) must be loaded first: ``gt.load()``.

    Returns:
        The carrier with the same values (no copy), transform, CRS, fill
        value and attrs (frozen: lists become tuples).

    Raises:
        ImportError: georeader is not installed.
        TypeError: ``gt`` is not a ``GeoTensor``, or an attr value cannot
            be frozen (the error names the key).

    Example:
        >>> import numpy as np
        >>> from affine import Affine
        >>> from georeader.geotensor import GeoTensor
        >>> gt = GeoTensor(
        ...     np.zeros((2, 4, 4), np.float32), Affine(20, 0, 600000, 0, -20, 3550000),
        ...     "EPSG:32613", fill_value_default=np.nan,
        ... )
        >>> g = from_georeader(gt)
        >>> g.shape, g.grid.bounds[0], np.shares_memory(g.data, gt.values)
        ((2, 4, 4), 600000.0, True)
    """
    geotensor = _geotensor_module()
    if not isinstance(gt, geotensor.GeoTensor):
        hint = "; load a lazy reader first with .load()" if hasattr(gt, "load") else ""
        raise TypeError(
            f"from_georeader takes a GeoTensor, got {type(gt).__name__}{hint}"
        )
    return GeoArray(
        np.asarray(gt.values),
        gt.transform,
        gt.crs,
        fill_value=gt.fill_value_default,
        attrs=FrozenAttrs(dict(gt.attrs or {})),
    )


def to_georeader(g: GeoArray) -> Any:
    """A georeader ``GeoTensor`` from a NumPy `GeoArray`, sharing its memory.

    Args:
        g: A NumPy carrier, not batched. Move a JAX carrier to the host
            first with `GeoArray.to_numpy`.

    Returns:
        A ``GeoTensor`` with the same values (no copy), an ``affine.Affine``
        transform, the CRS as ``"AUTH:CODE"`` when that is exact (else
        WKT), the fill value (``None`` stays ``None``; georeader's default
        of 0 is never assumed) and the attrs as a plain dict.

    Raises:
        ImportError: georeader is not installed.
        TypeError: ``g`` is not a GeoArray, or holds JAX data.
        ValueError: ``g`` is batched (GeoTensor has one transform).

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> g = GeoArray(np.ones((4, 4)), (20, 0, 600000, 0, -20, 3550000), 32613)
        >>> gt = to_georeader(g)
        >>> gt.crs, gt.transform.c, gt.fill_value_default is None
        ('EPSG:32613', 600000.0, True)
    """
    geotensor = _geotensor_module()
    if not isinstance(g, GeoArray):
        raise TypeError(f"to_georeader takes a GeoArray, got {type(g).__name__}")
    if g.backend != "numpy":
        raise TypeError(
            "GeoTensor holds NumPy data: move the carrier to the host first "
            "with g.to_numpy()"
        )
    if g.batch_shape:
        raise ValueError(
            f"a batched GeoArray (batch {g.batch_shape}) has one transform per "
            "element; convert each element, e.g. [to_georeader(g[i]) for i ...]"
        )
    return geotensor.GeoTensor(
        g.data,
        transform=Affine(*np.asarray(g.transform).tolist()),
        crs=crs_string(g.crs),
        fill_value_default=g.fill_value,
        attrs=g.attrs.to_dict(),
    )


def _whole_number(value: Any, name: str) -> int:
    """``value`` as an int, refusing a fractional window offset or size."""
    as_int = round(float(value))
    if abs(float(value) - as_int) > 1e-9:
        raise ValueError(
            f"window {name}={value} is not a whole number of pixels; round the "
            "window first (rasterio.windows.Window.round_offsets / round_lengths)"
        )
    return as_int


class _GeoreaderSource:
    """A georeader ``GeoData`` as a `RasterSource` (`source_from_georeader`)."""

    def __init__(self, reader: Any) -> None:
        self.reader = reader

    @property
    def grid(self) -> Grid:
        return Grid(
            self.reader.transform, self.reader.crs, tuple(self.reader.shape[-2:])
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.reader.shape)

    @property
    def dtype(self) -> Any:
        return np.dtype(self.reader.dtype)

    @property
    def fill_value(self) -> Any:
        return _normalize_fill(self.reader.fill_value_default)

    @property
    def attrs(self) -> FrozenAttrs:
        return FrozenAttrs(dict(getattr(self.reader, "attrs", None) or {}))

    def read_window(self, window: PixelWindow, *, boundless: bool = True) -> GeoArray:
        from rasterio.windows import Window

        rows, cols = window.to_slices()
        rio_window = Window.from_slices(rows, cols, boundless=True)
        out = self.reader.read_from_window(rio_window, boundless=boundless)
        return self._carrier(out, boundless)

    def load(self) -> GeoArray:
        return self._carrier(self.reader.load(boundless=True), True)

    def _carrier(self, out: Any, boundless: bool) -> GeoArray:
        geotensor = _geotensor_module()
        if not isinstance(out, geotensor.GeoTensor):
            out = out.load(boundless=boundless)  # a lazy reader of the window
        g = from_georeader(out)
        attrs = self.attrs
        return g.with_attrs(**attrs) if attrs and not g.attrs else g

    def __repr__(self) -> str:
        return f"source_from_georeader({self.reader!r})"


def source_from_georeader(reader: Any) -> RasterSource:
    """A georeader ``GeoData`` (or ``GeoTensor``) as a `RasterSource`.

    Args:
        reader: A lazy georeader reader (``RasterioReader``, …), read on
            demand; or a ``GeoTensor``, converted at once (zero-copy).

    Returns:
        A `RasterSource` whose reads return `GeoArray`. The wrapped reader
        is its ``reader`` attribute.

    Raises:
        ImportError: georeader is not installed.
        TypeError: ``reader`` has no georeader read surface.

    Example:
        >>> from georeader.rasterio_reader import RasterioReader  # doctest: +SKIP
        >>> reader = RasterioReader("scene.tif")  # doctest: +SKIP
        >>> source = source_from_georeader(reader)  # doctest: +SKIP
        >>> source.grid.shape, source.read_window(window).shape  # doctest: +SKIP
    """
    geotensor = _geotensor_module()
    if isinstance(reader, geotensor.GeoTensor):
        return from_georeader(reader)
    missing = [
        name
        for name in ("transform", "crs", "shape", "read_from_window", "load")
        if not hasattr(reader, name)
    ]
    if missing:
        raise TypeError(
            f"source_from_georeader takes a georeader GeoData; "
            f"{type(reader).__name__} lacks {missing}"
        )
    return _GeoreaderSource(reader)


_GEODATA_CLASS: type | None = None


def _geodata_class() -> type:
    """The ``GeoData`` view of a `RasterSource`, built once georeader loads."""
    global _GEODATA_CLASS
    if _GEODATA_CLASS is not None:
        return _GEODATA_CLASS
    _geotensor_module()
    from georeader.abstract_reader import GeoData

    class SourceAsGeoData(GeoData):
        """A `RasterSource` presented to georeader as a ``GeoData``."""

        def __init__(self, source: Any) -> None:
            self.source = source

        @property
        def transform(self) -> Affine:
            return self.source.grid.affine

        @property
        def crs(self) -> Any:
            return crs_string(self.source.grid.crs)

        @property
        def shape(self) -> tuple[int, ...]:
            return tuple(self.source.shape)

        @property
        def width(self) -> int:
            return int(self.source.shape[-1])

        @property
        def height(self) -> int:
            return int(self.source.shape[-2])

        @property
        def dtype(self) -> Any:
            return self.source.dtype

        @property
        def dims(self) -> list[str]:
            return list(("time", "band", "y", "x")[-len(self.shape) :])

        @property
        def fill_value_default(self) -> Any:
            return self.source.fill_value

        @property
        def attrs(self) -> dict[str, Any]:
            return dict(FrozenAttrs(self.source.attrs).to_dict())

        @property
        def values(self) -> np.ndarray:
            return np.asarray(self.source.load().data)

        def load(self, boundless: bool = True) -> Any:
            return to_georeader(self.source.load())

        def read_from_window(self, window: Any, boundless: bool = True) -> Any:
            pixel_window = PixelWindow(
                col_off=_whole_number(window.col_off, "col_off"),
                row_off=_whole_number(window.row_off, "row_off"),
                width=_whole_number(window.width, "width"),
                height=_whole_number(window.height, "height"),
            )
            return to_georeader(
                self.source.read_window(pixel_window, boundless=boundless)
            )

        def __repr__(self) -> str:
            return f"source_to_georeader({self.source!r})"

    _GEODATA_CLASS = SourceAsGeoData
    return SourceAsGeoData


def source_to_georeader(source: RasterSource) -> Any:
    """A `RasterSource` as a georeader ``GeoData``, for georeader's functions.

    georeader's ``read`` module (``read_from_bounds``, ``read_reproject``,
    …) works on any ``GeoData``; this view lets it read a geotoolz reader
    window by window. A `GeoArray` becomes a ``GeoTensor``
    (`to_georeader`); a `source_from_georeader` source gives back the
    reader it wraps.

    Args:
        source: Any `RasterSource` with NumPy reads.

    Returns:
        A ``GeoData`` whose reads return ``GeoTensor``.

    Raises:
        ImportError: georeader is not installed.
        TypeError: ``source`` is not a `RasterSource`.

    Example:
        >>> from georeader import read  # doctest: +SKIP
        >>> view = source_to_georeader(reader)  # doctest: +SKIP
        >>> tile = read.read_from_bounds(view, bounds)  # doctest: +SKIP
        >>> from_georeader(tile)  # doctest: +SKIP
    """
    if isinstance(source, GeoArray):
        return to_georeader(source)
    if isinstance(source, _GeoreaderSource):
        return source.reader
    if not isinstance(source, RasterSource):
        raise TypeError(
            f"source_to_georeader takes a RasterSource, got {type(source).__name__}"
        )
    return _geodata_class()(source)
