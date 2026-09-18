"""Coordinate reference system utilities.

Thin wrappers around :mod:`pyproj` and :mod:`rioxarray`. For full
reprojection / raster I/O, delegate to ``rioxarray`` directly.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import rioxarray  # noqa: F401  — needed so ds.rio is populated
import xarray as xr
from jaxtyping import Float
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


GridResolution = Literal["one_degree", "quarter_degree", "twelfth_degree", "other"]


def assign_crs(ds: xr.Dataset, crs: str = "EPSG:4326") -> xr.Dataset:
    """Attach a CRS to ``ds`` via :mod:`rioxarray`.

    Args:
        ds: Input dataset.
        crs: Any CRS specifier accepted by :class:`pyproj.CRS`.

    Returns:
        Dataset with the CRS attached (``ds.rio.crs``).
    """
    return ds.rio.write_crs(crs)


def get_crs(ds: xr.Dataset) -> CRS | None:
    """Return ``ds.rio.crs`` if set, otherwise ``None``."""
    crs = ds.rio.crs
    return CRS(crs) if crs is not None else None


def reproject(
    ds: xr.Dataset,
    target_crs: str,
    resolution: float | None = None,
    resampling: str = "bilinear",
) -> xr.Dataset:
    """Reproject a raster dataset to ``target_crs`` via :mod:`rioxarray`.

    Args:
        ds: Input dataset with a CRS attached.
        target_crs: Any CRS specifier accepted by :class:`pyproj.CRS`.
        resolution: If provided, resample to this output cell size.
        resampling: Name of the ``rasterio.enums.Resampling`` member.

    Returns:
        Reprojected dataset.
    """
    from rasterio.enums import Resampling

    if not hasattr(Resampling, resampling):
        valid = [r.name for r in Resampling]
        raise ValueError(f"Unknown resampling {resampling!r}; expected one of {valid}.")
    return ds.rio.reproject(
        target_crs,
        resolution=resolution,
        resampling=getattr(Resampling, resampling),
    )


def reproject_match(
    ds: xr.Dataset | xr.DataArray,
    target: xr.Dataset | xr.DataArray,
    *,
    resampling: str = "bilinear",
) -> xr.Dataset | xr.DataArray:
    """Reproject ``ds`` onto ``target``'s CRS, transform and shape.

    Wraps :meth:`rioxarray.raster_array.RasterArray.reproject_match`.
    Unlike :func:`xrtoolz.interpolate.regrid_like` — which is
    :meth:`xr.Dataset.interp` over shared coordinate names — this is
    CRS-aware and aligns the output pixel grid exactly with ``target``.
    Use it whenever ``ds`` and ``target`` live in different CRSs, where
    ``regrid_like`` would interpolate mismatched coordinate spaces and
    return a silently wrong result.

    Args:
        ds: Source raster with a CRS attached (``ds.rio.crs``).
        target: Raster whose CRS, affine transform and shape define the
            output grid. Must also carry a CRS.
        resampling: Name of the :class:`rasterio.enums.Resampling`
            member. Default ``"bilinear"`` (use ``"nearest"`` for
            categorical data).

    Returns:
        ``ds`` on ``target``'s grid — identical CRS, transform, shape and
        spatial coordinates.

    Raises:
        ValueError: If ``resampling`` is not a ``Resampling`` member name.

    Example:
        >>> import numpy as np, xarray as xr
        >>> import rioxarray  # noqa: F401
        >>> from xrtoolz.geo import reproject_match
        >>> src = xr.DataArray(
        ...     np.ones((4, 4)),
        ...     dims=("y", "x"),
        ...     coords={"y": np.arange(3.0, -1.0, -1.0), "x": np.arange(4.0)},
        ... ).rio.write_crs("EPSG:4326")
        >>> tgt = xr.DataArray(
        ...     np.zeros((2, 2)),
        ...     dims=("y", "x"),
        ...     coords={"y": np.arange(3.0, -1.0, -2.0), "x": np.arange(0.0, 4.0, 2.0)},
        ... ).rio.write_crs("EPSG:4326")
        >>> out = reproject_match(src, tgt)
        >>> out.shape == tgt.shape
        True

    """
    from rasterio.enums import Resampling

    if not hasattr(Resampling, resampling):
        valid = [r.name for r in Resampling]
        raise ValueError(f"Unknown resampling {resampling!r}; expected one of {valid}.")
    return ds.rio.reproject_match(
        target,
        resampling=getattr(Resampling, resampling),
    )


def lonlat_to_xy(
    crs: str,
    lon: Sequence[float] | np.ndarray,
    lat: Sequence[float] | np.ndarray,
) -> tuple[Float[np.ndarray, "..."], Float[np.ndarray, "..."]]:
    """Convert WGS-84 lon/lat to ``crs`` x/y coordinates (same shape as input)."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def xy_to_lonlat(
    crs: str,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> tuple[Float[np.ndarray, "..."], Float[np.ndarray, "..."]]:
    """Convert ``crs`` x/y coordinates back to WGS-84 lon/lat (same shape)."""
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return np.asarray(lon), np.asarray(lat)


# UTM covers 80°S–84°N (the polar caps use UPS instead).
_UTM_LAT_MIN = -80.0
_UTM_LAT_MAX = 84.0


def _utm_zone(lon: float, lat: float) -> int:
    """UTM zone number for ``(lon, lat)`` including the Norway / Svalbard
    exceptions. A zone-boundary meridian belongs to the zone east of it."""
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        return 32
    if 72.0 <= lat <= 84.0 and 0.0 <= lon < 42.0:
        if lon < 9.0:
            return 31
        if lon < 21.0:
            return 33
        if lon < 33.0:
            return 35
        return 37
    return int((lon + 180.0) // 6.0) + 1


def utm_crs_for(lon: float, lat: float, *, datum: str = "WGS 84") -> str:
    """CRS authority code of the UTM zone containing ``(lon, lat)``.

    The zone number follows the standard 6° bands plus the Norway
    (zone 32 widened over 56–64°N) and Svalbard (zones 31/33/35/37 over
    72–84°N) exceptions. A zone-boundary meridian belongs to the zone
    east of it (``lon=-102.0`` → zone 14), longitudes are wrapped into
    ``[-180, 180)`` first (so ``0–360`` grids work), and ``lat=0`` counts
    as the northern hemisphere. The code itself comes from
    :func:`pyproj.database.query_utm_crs_info` for ``datum`` (a point
    :class:`~pyproj.aoi.AreaOfInterest` on the zone's central meridian);
    when the database has no entry the WGS 84 convention
    ``"EPSG:326NN"`` (north) / ``"EPSG:327NN"`` (south) is used.

    Args:
        lon: Longitude in degrees east.
        lat: Latitude in degrees north.
        datum: Datum name as spelled in the PROJ database
            (e.g. ``"WGS 84"``, ``"NAD83"``, ``"ETRS89"``).

    Returns:
        ``"<AUTH>:<code>"`` string, e.g. ``"EPSG:32613"``.

    Raises:
        ValueError: If ``lat`` is outside ``[-80, 84]`` (the UTM domain —
            use a polar stereographic CRS instead), or if ``datum`` is not
            ``"WGS 84"`` and the database has no UTM CRS for that zone.

    Example:
        >>> from xrtoolz.geo import utm_crs_for
        >>> utm_crs_for(-102.5, 31.5)
        'EPSG:32613'
        >>> utm_crs_for(151.2, -33.9)
        'EPSG:32756'

    """
    lat = float(lat)
    # Chained comparison is False for NaN, so NaN is rejected too.
    if not _UTM_LAT_MIN <= lat <= _UTM_LAT_MAX:
        raise ValueError(
            "utm_crs_for: UTM is only defined for "
            f"{_UTM_LAT_MIN}° <= lat <= {_UTM_LAT_MAX}°, got lat={lat}. "
            "Use a polar stereographic CRS for the poles."
        )
    lon = (float(lon) + 180.0) % 360.0 - 180.0
    zone = _utm_zone(lon, lat)
    hemisphere = "N" if lat >= 0.0 else "S"
    central_meridian = (zone - 1) * 6.0 - 180.0 + 3.0
    infos = query_utm_crs_info(
        datum_name=datum,
        area_of_interest=AreaOfInterest(central_meridian, lat, central_meridian, lat),
    )
    suffix = f"zone {zone}{hemisphere}"
    for info in infos:
        if info.name.endswith(suffix):
            return f"{info.auth_name}:{info.code}"
    if datum != "WGS 84":
        raise ValueError(
            f"utm_crs_for: no UTM {suffix} CRS found for datum {datum!r} in the "
            "PROJ database."
        )
    return f"EPSG:{326 if hemisphere == 'N' else 327}{zone:02d}"


def _check_metric_crs(crs: str) -> None:
    """Raise ``ValueError`` unless ``crs`` is projected with metre axes."""
    parsed = CRS(crs)
    units = sorted({axis.unit_name for axis in parsed.axis_info})
    if not parsed.is_projected:
        raise ValueError(
            f"LocalFrame: crs {crs!r} is not a projected CRS (axes in {units}); "
            "offsets would not be metres. Pass a projected CRS such as a UTM zone."
        )
    if any(axis.unit_conversion_factor != 1.0 for axis in parsed.axis_info):
        raise ValueError(
            f"LocalFrame: crs {crs!r} has axes in {units}, not metres; "
            "offsets would be mislabelled as metres. Pass a metre-based "
            "projected CRS such as a UTM zone."
        )


@functools.cache
def _frame_transformers(crs: str) -> tuple[Transformer, Transformer]:
    """``(lonlat → crs, crs → lonlat)`` transformers, cached per CRS string.

    Validates the CRS once per string (see :func:`_check_metric_crs`).
    """
    _check_metric_crs(crs)
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return forward, inverse


@dataclass(frozen=True)
class LocalFrame:
    """A metric frame: projected CRS plus an origin expressed in that CRS.

    ``to_xy`` returns metres east / north of the origin in ``crs``; ``to_lonlat``
    inverts it. The frame holds only plain strings and floats, so it is
    hashable and JSON-serialisable (``to_dict`` / ``from_dict``); the
    :class:`pyproj.Transformer` pair is cached at module level per CRS.

    Attributes:
        crs: A projected CRS with metre axes, as any specifier accepted by
            :class:`pyproj.CRS`, kept in string form (e.g. ``"EPSG:32613"``).
        origin_lon: WGS-84 longitude of the origin, degrees east.
        origin_lat: WGS-84 latitude of the origin, degrees north.
        origin_xy: ``(x0, y0)`` of the origin projected into ``crs``,
            computed at construction.

    Raises:
        ValueError: If ``crs`` is geographic or its axes are not in metres
            (e.g. ``"EPSG:2263"``, US survey feet).

    Example:
        >>> from xrtoolz.geo import local_frame
        >>> frame = local_frame(-102.5, 31.5)
        >>> frame.crs
        'EPSG:32613'
        >>> x, y = frame.to_xy(-102.5, 31.5)
        >>> float(x), float(y)
        (0.0, 0.0)

    """

    crs: str
    origin_lon: float
    origin_lat: float
    origin_xy: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        forward, _ = _frame_transformers(self.crs)
        x0, y0 = forward.transform(self.origin_lon, self.origin_lat)
        object.__setattr__(self, "origin_xy", (float(x0), float(y0)))

    def to_xy(
        self,
        lon: float | Sequence[float] | np.ndarray,
        lat: float | Sequence[float] | np.ndarray,
    ) -> tuple[Float[np.ndarray, "..."], Float[np.ndarray, "..."]]:
        """Project WGS-84 ``lon``/``lat`` to metres relative to the origin.

        Args:
            lon: Longitude(s) in degrees east (any shape).
            lat: Latitude(s) in degrees north (same shape as ``lon``).

        Returns:
            ``(x, y)`` arrays of metres east / north of the origin.
        """
        forward, _ = _frame_transformers(self.crs)
        x, y = forward.transform(lon, lat)
        x0, y0 = self.origin_xy
        return np.asarray(x, dtype=float) - x0, np.asarray(y, dtype=float) - y0

    def to_lonlat(
        self,
        x: float | Sequence[float] | np.ndarray,
        y: float | Sequence[float] | np.ndarray,
    ) -> tuple[Float[np.ndarray, "..."], Float[np.ndarray, "..."]]:
        """Inverse of :meth:`to_xy`: metres from the origin to WGS-84 lon/lat.

        Args:
            x: Metres east of the origin (any shape).
            y: Metres north of the origin (same shape as ``x``).

        Returns:
            ``(lon, lat)`` arrays in degrees.
        """
        _, inverse = _frame_transformers(self.crs)
        x0, y0 = self.origin_xy
        lon, lat = inverse.transform(
            np.asarray(x, dtype=float) + x0, np.asarray(y, dtype=float) + y0
        )
        return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable ``{"crs", "origin_lon", "origin_lat"}`` payload."""
        return {
            "crs": self.crs,
            "origin_lon": float(self.origin_lon),
            "origin_lat": float(self.origin_lat),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LocalFrame:
        """Rebuild a frame from a :meth:`to_dict` payload.

        Args:
            d: Mapping with ``"crs"``, ``"origin_lon"`` and ``"origin_lat"``
                keys; ``origin_xy`` is recomputed, never read.

        Returns:
            The reconstructed :class:`LocalFrame`.
        """
        return cls(
            crs=str(d["crs"]),
            origin_lon=float(d["origin_lon"]),
            origin_lat=float(d["origin_lat"]),
        )


def local_frame(
    origin_lon: float, origin_lat: float, *, crs: str | None = None
) -> LocalFrame:
    """Build a :class:`LocalFrame` anchored at ``(origin_lon, origin_lat)``.

    Args:
        origin_lon: WGS-84 longitude of the origin, degrees east.
        origin_lat: WGS-84 latitude of the origin, degrees north.
        crs: Projected CRS to use. ``None`` selects the UTM zone containing
            the origin via :func:`utm_crs_for`.

    Returns:
        The frame; ``frame.to_xy(origin_lon, origin_lat)`` is ``(0, 0)``.

    Raises:
        ValueError: If ``crs`` is ``None`` and ``origin_lat`` is outside
            ``[-80, 84]``, or if ``crs`` is not a metric projected CRS.
    """
    if crs is None:
        crs = utm_crs_for(origin_lon, origin_lat)
    return LocalFrame(
        crs=crs, origin_lon=float(origin_lon), origin_lat=float(origin_lat)
    )


def assign_local_xy(
    ds: xr.Dataset,
    frame: LocalFrame,
    *,
    lon: str = "lon",
    lat: str = "lat",
    x: str = "x",
    y: str = "y",
) -> xr.Dataset:
    """Add metric ``x``/``y`` coordinates (metres from ``frame``'s origin).

    Rectilinear grids, swaths and point tracks are supported. Because a
    UTM grid is not axis-aligned with lon/lat, gridded output is 2-D: for
    1-D ``lon``/``lat`` on *different* dims they are broadcast onto
    ``(lat_dim, lon_dim)``; for 2-D ``lon``/``lat`` (e.g.
    ``(scanline, ground_pixel)``) they share the input dims. Paired 1-D
    ``lon``/``lat`` on the *same* dim (along-track points, e.g.
    ``lon(obs)``/``lat(obs)``) are transformed elementwise and yield 1-D
    ``x``/``y`` on that dim.

    Args:
        ds: Dataset carrying geographic coordinates.
        frame: Local frame that defines the CRS and origin.
        lon: Longitude coordinate name.
        lat: Latitude coordinate name.
        x: Name of the output east coordinate.
        y: Name of the output north coordinate.

    Returns:
        ``ds`` with ``x``/``y`` coordinates (``units="m"``; 2-D for grids
        and swaths, 1-D for point tracks) and
        ``ds.attrs["local_frame"] = frame.to_dict()``.

    Raises:
        ValueError: If ``lon``/``lat`` are not both 1-D or both 2-D on the
            same dimensions.
    """
    lon_da, lat_da = ds[lon], ds[lat]
    if lon_da.ndim == 1 and lat_da.ndim == 1 and lon_da.dims == lat_da.dims:
        # Along-track points: one (lon, lat) pair per element, no meshgrid.
        lon2d, lat2d = lon_da.values, lat_da.values
        dims: tuple[str, ...] = (str(lon_da.dims[0]),)
    elif lon_da.ndim == 1 and lat_da.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon_da.values, lat_da.values)
        dims = (str(lat_da.dims[0]), str(lon_da.dims[0]))
    elif lon_da.ndim == 2 and lon_da.dims == lat_da.dims:
        lon2d, lat2d = lon_da.values, lat_da.values
        dims = tuple(str(d) for d in lon_da.dims)
    else:
        raise ValueError(
            f"assign_local_xy expects {lon!r}/{lat!r} to be both 1-D or both 2-D "
            f"on the same dims, got {lon!r}: {lon_da.dims} and "
            f"{lat!r}: {lat_da.dims}."
        )
    xx, yy = frame.to_xy(lon2d, lat2d)
    out = ds.assign_coords(
        {
            x: (dims, xx, {"units": "m", "long_name": "distance east of origin"}),
            y: (dims, yy, {"units": "m", "long_name": "distance north of origin"}),
        }
    )
    return out.assign_attrs(local_frame=frame.to_dict())


def calc_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Add 2-D ``latitude`` / ``longitude`` coords computed from x/y.

    Assumes ``ds`` has 1-D ``x`` and ``y`` coordinates and a CRS attached
    (``ds.rio.crs``). Inf values from the transform are replaced with
    NaN so that downstream masking handles them cleanly. This is the
    inverse direction of :func:`assign_local_xy` for gridded rasters:
    projected ``x``/``y`` axes in, geographic coordinates out.

    Args:
        ds: Input dataset.

    Returns:
        Dataset with additional 2-D ``latitude`` / ``longitude``
        coordinates along ``(y, x)``.
    """
    if ds.rio.crs is None:
        raise ValueError(
            "assign a CRS with assign_crs(ds, ...) before calling calc_latlon."
        )

    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    lons, lats = xy_to_lonlat(str(ds.rio.crs), xx, yy)
    lons = np.where(np.isfinite(lons), lons, np.nan)
    lats = np.where(np.isfinite(lats), lats, np.nan)

    ds = ds.assign_coords(
        longitude=(("y", "x"), lons),
        latitude=(("y", "x"), lats),
    )
    ds["longitude"].attrs["units"] = "degrees_east"
    ds["latitude"].attrs["units"] = "degrees_north"
    return ds


def get_dataset_resolution(
    ds: xr.Dataset,
    *,
    lon: str = "lon",
    lat: str = "lat",
    rtol: float = 0.05,
) -> GridResolution:
    """Classify a regular lat/lon grid against canonical OceanBench resolutions.

    Returns one of ``"one_degree"``, ``"quarter_degree"``,
    ``"twelfth_degree"``, or ``"other"`` (no canonical resolution
    matches within ``rtol`` of *both* axes). Useful for dispatch — e.g.
    picking the right MDT URL or the right reference dataset for a
    given grid.

    The classifier checks the median absolute spacing of the ``lon``
    and ``lat`` coordinates **independently** — an anisotropic grid
    (e.g. ``dlon=1.1`` but ``dlat=0.9``) is rejected even though its
    averaged spacing would round to 1°. Curvilinear (2-D) lon/lat
    coordinates are rejected with a ``ValueError``: ``np.diff`` on a
    multi-dim array would silently mix endpoints across rows and
    misclassify the grid.

    Args:
        ds: Dataset carrying 1-D ``lon`` / ``lat`` coordinates.
        lon: Longitude coordinate name.
        lat: Latitude coordinate name.
        rtol: Relative tolerance for the canonical-resolution match,
            applied per-axis.

    Returns:
        Resolution label as a ``Literal[...]`` string.

    Raises:
        ValueError: If ``ds[lon]`` or ``ds[lat]`` is not 1-D — the
            classifier only operates on rectilinear grids; curvilinear
            datasets should be regridded first.
    """
    lon_da = ds[lon]
    lat_da = ds[lat]
    if lon_da.ndim != 1:
        raise ValueError(
            f"get_dataset_resolution expects a 1-D {lon!r} coord, got "
            f"shape {tuple(lon_da.shape)}. Curvilinear / 2-D grids are not "
            "supported — regrid onto a rectilinear axis first."
        )
    if lat_da.ndim != 1:
        raise ValueError(
            f"get_dataset_resolution expects a 1-D {lat!r} coord, got "
            f"shape {tuple(lat_da.shape)}. Curvilinear / 2-D grids are not "
            "supported — regrid onto a rectilinear axis first."
        )
    dlon = float(np.median(np.abs(np.diff(np.asarray(lon_da.values, dtype=float)))))
    dlat = float(np.median(np.abs(np.diff(np.asarray(lat_da.values, dtype=float)))))
    candidates: list[tuple[GridResolution, float]] = [
        ("one_degree", 1.0),
        ("quarter_degree", 0.25),
        ("twelfth_degree", 1.0 / 12.0),
    ]
    for label, target in candidates:
        # Require BOTH axes to match the canonical target — averaging
        # would let an anisotropic grid (dlon=1.1, dlat=0.9 → mean=1.0)
        # masquerade as canonical even though neither axis is.
        if abs(dlon - target) < rtol * target and abs(dlat - target) < rtol * target:
            return label
    return "other"
