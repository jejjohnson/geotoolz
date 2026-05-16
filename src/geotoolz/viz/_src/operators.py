"""Tier-B visualization Operators wrapping display primitives.

The display palette: band-selection composites (``TrueColor``,
``FalseColor``, ``SWIRComposite``, generic ``Composite``),
display-stretch + uint8 cast (``StretchToUint8``, ``GammaCorrect``),
colormaps (``ApplyColormap``, ``ApplyDiscreteColormap``), terrain
shading (``Hillshade``, ``ShadedRelief``), and alpha overlays
(``Overlay``, ``AnnotatePolygons``, ``AnnotatePoints``).

These operators are *visualization* primitives — they preserve the
carrier's ``transform`` / ``crs`` (the spatial footprint is the same
pre- and post-render) but typically change the band axis from
N-band reflectance to 3-band RGB or 4-band RGBA ``uint8``. They sit
downstream of :mod:`geotoolz.radiometry` (``PercentileClip``,
``MinMax``, ``Gamma``) which already does the float contrast stretch
— the viz operators add the band-selection and byte-cast steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from geotoolz.core import Operator
from geotoolz.viz._src.array import (
    Color,
    blend_rgba,
    composite,
    ensure_rgba,
    gamma_correct_display,
    hillshade,
    rgba_from_categories,
    rgba_from_scalar,
    stretch_to_uint8,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


BandRef = int | str


class Composite(Operator):
    """Build a multi-band composite by arbitrary band reference.

    The generic band-selection operator the named composites
    (`TrueColor`, `FalseColor`, `SWIRComposite`) wrap. Bands may be
    referenced by integer position along ``axis`` or by name when the
    carrier carries a ``bands`` / ``band_names`` / ``descriptions``
    entry in ``attrs``. Output has ``len(bands)`` slices along ``axis``
    and the same spatial footprint as the input — ``transform`` and
    ``crs`` round-trip unchanged.

    Args:
        bands: Sequence of band references. Each entry is either an
            integer position or a string name resolved against the
            carrier's ``attrs``.
        axis: Band axis. Default ``0`` (``(C, H, W)`` convention).

    Examples:
        >>> import geotoolz as gz
        >>> # Sentinel-2 natural-colour composite by name.
        >>> rgb = gz.viz.Composite(bands=["B04", "B03", "B02"])(s2_geotensor)
        >>> # Or by integer position when the carrier has no band names.
        >>> rgb = gz.viz.Composite(bands=[3, 2, 1])(s2_geotensor)
    """

    def __init__(self, *, bands: Sequence[BandRef], axis: int = 0) -> None:
        self.bands = list(bands)
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        indices = _resolve_bands(gt, self.bands)
        return gt.array_as_geotensor(composite(np.asarray(gt), indices, axis=self.axis))

    def get_config(self) -> dict[str, Any]:
        return {"bands": list(self.bands), "axis": self.axis}


class TrueColor(Composite):
    """Build an RGB composite from explicit red, green, and blue band refs.

    Convenience wrapper over `Composite` that keeps the band ordering
    self-documenting. Output is shaped ``(3, H, W)`` in (R, G, B) order
    — the matplotlib / PIL display convention. Geo-metadata is
    preserved.

    Args:
        red: Red-band reference (int index or string name).
        green: Green-band reference.
        blue: Blue-band reference.
        axis: Band axis. Default ``0``.

    Examples:
        >>> import geotoolz as gz
        >>> # Named Sentinel-2 natural-colour RGB.
        >>> rgb = gz.viz.TrueColor(red="B04", green="B03", blue="B02")(s2)
        >>> # Stretched display-ready pipeline.
        >>> display = (
        ...     gz.viz.TrueColor(red="B04", green="B03", blue="B02")
        ...     | gz.viz.StretchToUint8(lower=2.0, upper=98.0)
        ... )
        >>> rgb_uint8 = display(s2)
    """

    def __init__(
        self, *, red: BandRef, green: BandRef, blue: BandRef, axis: int = 0
    ) -> None:
        super().__init__(bands=[red, green, blue], axis=axis)
        self.red = red
        self.green = green
        self.blue = blue

    def get_config(self) -> dict[str, Any]:
        return {
            "red": self.red,
            "green": self.green,
            "blue": self.blue,
            "axis": self.axis,
        }


class FalseColor(Composite):
    """Build a NIR-red-green false-colour composite.

    Vegetation visualisation preset: healthy vegetation shows red
    because NIR reflectance is high. Equivalent to
    ``Composite(bands=[nir, red, green])`` with self-documenting kwargs.

    Args:
        nir: Near-infrared band reference (rendered as red).
        red: Red band reference (rendered as green).
        green: Green band reference (rendered as blue).
        axis: Band axis. Default ``0``.

    Examples:
        >>> import geotoolz as gz
        >>> # Sentinel-2 vegetation false colour (8/4/3).
        >>> vis = gz.viz.FalseColor(nir="B08", red="B04", green="B03")(s2)
    """

    def __init__(
        self, *, nir: BandRef, red: BandRef, green: BandRef, axis: int = 0
    ) -> None:
        super().__init__(bands=[nir, red, green], axis=axis)
        self.nir = nir
        self.red = red
        self.green = green

    def get_config(self) -> dict[str, Any]:
        return {
            "nir": self.nir,
            "red": self.red,
            "green": self.green,
            "axis": self.axis,
        }


class SWIRComposite(Composite):
    """Build a SWIR2-NIR-red composite.

    Burn-scar / urban / geology visualisation preset where SWIR2 is
    rendered as red. Equivalent to ``Composite(bands=[swir2, nir, red])``.

    Args:
        swir2: SWIR2 band reference (rendered as red).
        nir: NIR band reference (rendered as green).
        red: Red band reference (rendered as blue).
        axis: Band axis. Default ``0``.

    Examples:
        >>> import geotoolz as gz
        >>> # Landsat-8 SWIR2/NIR/red burn-scar composite (7/5/4).
        >>> vis = gz.viz.SWIRComposite(swir2="B7", nir="B5", red="B4")(l8)
    """

    def __init__(
        self, *, swir2: BandRef, nir: BandRef, red: BandRef, axis: int = 0
    ) -> None:
        super().__init__(bands=[swir2, nir, red], axis=axis)
        self.swir2 = swir2
        self.nir = nir
        self.red = red

    def get_config(self) -> dict[str, Any]:
        return {
            "swir2": self.swir2,
            "nir": self.nir,
            "red": self.red,
            "axis": self.axis,
        }


class StretchToUint8(Operator):
    """Percentile-stretch display data to ``uint8``.

    The display-ready counterpart of
    :class:`geotoolz.radiometry.PercentileClip`: it uses the same
    percentile-based stretch but is NaN-safe and casts the unit-
    interval output to byte range so the result is ready for
    PIL / matplotlib. Pair this with
    :class:`geotoolz.radiometry.PercentileClip` + ``MinMax`` if you
    instead need float outputs for further math.

    Args:
        lower: Lower percentile. Default ``2.0``.
        upper: Upper percentile. Default ``98.0``.
        per_band: Compute percentiles independently per band. Default
            ``True``.

    Examples:
        >>> import geotoolz as gz
        >>> # Classic satellite display pipeline.
        >>> pipe = (
        ...     gz.viz.TrueColor(red="B04", green="B03", blue="B02")
        ...     | gz.viz.StretchToUint8(lower=2.0, upper=98.0)
        ... )
        >>> rgb_uint8 = pipe(s2_geotensor)
    """

    def __init__(
        self, *, lower: float = 2.0, upper: float = 98.0, per_band: bool = True
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.per_band = per_band

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = stretch_to_uint8(
            np.asarray(gt),
            lower=self.lower,
            upper=self.upper,
            per_band=self.per_band,
        )
        return gt.array_as_geotensor(out, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {"lower": self.lower, "upper": self.upper, "per_band": self.per_band}


class GammaCorrect(Operator):
    """Apply display gamma correction.

    Power-law correction for display-range arrays — brightens midtones
    when ``gamma > 1``. Distinct from
    :class:`geotoolz.radiometry.Gamma` in that it operates on already
    display-prepped (``[0, 1]`` float or byte-range integer) arrays.
    Integer carriers are normalised to ``[0, 1]`` before the exponent
    and scaled back to their dtype maximum so the transform is display-
    correct rather than a raw ``256 ** (1 / gamma) = 16`` on uint8. For
    the radiometry-stage gamma correction, use
    ``geotoolz.radiometry.Gamma``.

    Args:
        gamma: Gamma factor (must be strictly positive). Default ``1.0``.
        inplace_norm: Normalise integer inputs to ``[0, 1]`` before the
            exponent (default ``True``). Set ``False`` only when the
            carrier has been pre-normalised upstream.

    Examples:
        >>> import geotoolz as gz
        >>> # Brighten midtones on a percentile-stretched RGB.
        >>> pipe = (
        ...     gz.viz.TrueColor(red="B04", green="B03", blue="B02")
        ...     | gz.viz.StretchToUint8()
        ...     | gz.viz.GammaCorrect(gamma=1.2)
        ... )
    """

    def __init__(self, *, gamma: float = 1.0, inplace_norm: bool = True) -> None:
        self.gamma = gamma
        self.inplace_norm = inplace_norm

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            gamma_correct_display(
                np.asarray(gt), gamma=self.gamma, inplace_norm=self.inplace_norm
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {"gamma": self.gamma, "inplace_norm": self.inplace_norm}


class ToDisplayRange(StretchToUint8):
    """Alias for `StretchToUint8` — percentile clip + ``uint8`` cast.

    Kept for naming familiarity. Prefer `StretchToUint8` in new code.

    Examples:
        >>> import geotoolz as gz
        >>> uint8 = gz.viz.ToDisplayRange()(reflectance_geotensor)
    """


class ApplyColormap(Operator):
    """Map a single-band raster to a four-band RGBA GeoTensor.

    Looks up ``name`` in matplotlib's registry (or in cmocean if the
    name is prefixed with ``"cmocean."``). The colormap is referenced
    by *string name*, not by the live `Colormap` object — so
    ``get_config()`` round-trips through JSON / YAML cleanly.
    Geo-metadata (``transform`` / ``crs``) is preserved.

    Args:
        name: Colormap registry name (e.g. ``"viridis"``, ``"terrain"``,
            ``"cmocean.balance"``).
        vmin: Optional explicit lower bound. ``None`` auto-detects.
        vmax: Optional explicit upper bound. ``None`` auto-detects.
        nan_color: RGBA tuple in ``[0, 1]`` to paint NaN pixels.
            Default fully transparent.

    Examples:
        >>> import geotoolz as gz
        >>> # Render an NDVI raster as a viridis RGBA overlay.
        >>> rgba = gz.viz.ApplyColormap(name="viridis", vmin=-1, vmax=1)(ndvi_gt)
        >>> # Use a cmocean ocean colourmap with the prefix.
        >>> rgba = gz.viz.ApplyColormap(name="cmocean.thermal")(sst_gt)
    """

    def __init__(
        self,
        *,
        name: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        nan_color: Color = (0.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.name = name
        self.vmin = vmin
        self.vmax = vmax
        self.nan_color = nan_color

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        cmap = _get_colormap(self.name)
        out = rgba_from_scalar(
            np.asarray(gt),
            cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            nan_color=self.nan_color,
        )
        return gt.array_as_geotensor(out, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "nan_color": list(self.nan_color),
        }


class ApplyDiscreteColormap(Operator):
    """Map integer categories to a four-band RGBA GeoTensor.

    Classic categorical-label visualisation (land-cover classes, cloud
    masks, etc). Unmapped pixels render fully transparent. Pixels not
    in the mapping fall back to that default colour.

    Args:
        mapping: ``{class_id: (r, g, b, a)}`` lookup table. The class
            IDs are integers; the RGBA components are floats in
            ``[0, 1]``. ``get_config()`` stringifies the integer keys
            so the config is JSON-safe.

    Examples:
        >>> import geotoolz as gz
        >>> # Render a 3-class land-cover mask.
        >>> cmap = {
        ...     0: (0.0, 0.0, 0.0, 0.0),  # nodata -> transparent
        ...     1: (0.1, 0.5, 0.1, 1.0),  # forest
        ...     2: (0.7, 0.7, 0.2, 1.0),  # cropland
        ... }
        >>> rgba = gz.viz.ApplyDiscreteColormap(mapping=cmap)(lulc_gt)
    """

    def __init__(self, *, mapping: Mapping[int, Color]) -> None:
        self.mapping = {int(k): tuple(v) for k, v in mapping.items()}

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            rgba_from_categories(np.asarray(gt), self.mapping),
            fill_value_default=0,
        )

    def get_config(self) -> dict[str, Any]:
        # JSON object keys must be strings; cast int class IDs and
        # convert Color tuples to lists so the config round-trips
        # through json / yaml / hydra-zen cleanly.
        return {
            "mapping": {str(k): list(v) for k, v in self.mapping.items()},
        }


class Hillshade(Operator):
    """Compute a single-band ``uint8`` hillshade from a DEM.

    GDAL-style hillshade: combines slope and aspect from the DEM with
    a sun position (azimuth + altitude). The output is a single-band
    ``(H, W)`` ``uint8`` raster, ready to compose with a colour relief
    (see `ShadedRelief`). Pixel sizes come from the carrier's
    ``transform`` so units are correct in physical projections.

    Args:
        azimuth_deg: Sun azimuth in degrees clockwise from north.
            Default ``315`` (NW — the cartographic convention).
        altitude_deg: Sun elevation in degrees above horizon. Default
            ``45``. Values ``>= 90`` short-circuit to flat 255.
        z_factor: Vertical exaggeration. Default ``1.0``.

    Examples:
        >>> import geotoolz as gz
        >>> shade = gz.viz.Hillshade(azimuth_deg=315, altitude_deg=45)(dem_gt)
    """

    def __init__(
        self,
        *,
        azimuth_deg: float = 315.0,
        altitude_deg: float = 45.0,
        z_factor: float = 1.0,
    ) -> None:
        self.azimuth_deg = azimuth_deg
        self.altitude_deg = altitude_deg
        self.z_factor = z_factor

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = hillshade(
            np.asarray(gt),
            x_resolution=float(abs(gt.transform.a)),
            y_resolution=float(abs(gt.transform.e)),
            azimuth_deg=self.azimuth_deg,
            altitude_deg=self.altitude_deg,
            z_factor=self.z_factor,
        )
        return gt.array_as_geotensor(out, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {
            "azimuth_deg": self.azimuth_deg,
            "altitude_deg": self.altitude_deg,
            "z_factor": self.z_factor,
        }


class ShadedRelief(Operator):
    """Apply an elevation colormap and modulate RGB with hillshade.

    The composite terrain visualisation: run the DEM through
    `ApplyColormap` and modulate the RGB channels by a `Hillshade`
    so the result reads like a cartographer's shaded-relief map.
    Alpha channel is preserved from the colormap.

    Args:
        azimuth_deg: Sun azimuth (degrees clockwise from north).
            Default ``315``.
        altitude_deg: Sun elevation (degrees). Default ``45``.
        colormap: Name of the elevation colormap. Default ``"terrain"``.
        z_factor: Vertical exaggeration for the hillshade. Default
            ``1.0``.

    Examples:
        >>> import geotoolz as gz
        >>> shaded = gz.viz.ShadedRelief(colormap="terrain")(dem_gt)
    """

    def __init__(
        self,
        *,
        azimuth_deg: float = 315.0,
        altitude_deg: float = 45.0,
        colormap: str = "terrain",
        z_factor: float = 1.0,
    ) -> None:
        self.azimuth_deg = azimuth_deg
        self.altitude_deg = altitude_deg
        self.colormap = colormap
        self.z_factor = z_factor

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        rgba = np.asarray(ApplyColormap(name=self.colormap)(gt)).copy()
        shade = (
            np.asarray(
                Hillshade(
                    azimuth_deg=self.azimuth_deg,
                    altitude_deg=self.altitude_deg,
                    z_factor=self.z_factor,
                )(gt),
                dtype=np.float32,
            )
            / 255.0
        )
        rgba[:3] = (rgba[:3].astype(np.float32) * shade).astype(np.uint8)
        return gt.array_as_geotensor(rgba, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {
            "azimuth_deg": self.azimuth_deg,
            "altitude_deg": self.altitude_deg,
            "colormap": self.colormap,
            "z_factor": self.z_factor,
        }


class Overlay(Operator):
    """Blend background and foreground GeoTensors on the same grid.

    Two-input Operator: ``Overlay()(background, foreground)``. Both
    carriers must share ``transform`` and ``crs`` (this is a viz
    primitive, not a reprojector). Output is always 4-band ``uint8``
    RGBA — even when ``alpha=0`` — so downstream code sees a
    consistent shape.

    Args:
        alpha: Foreground opacity in ``[0, 1]``. Default ``0.6``.
        mode: Blend mode. One of ``"alpha"``, ``"multiply"``, or
            ``"screen"``. Default ``"alpha"``.

    Examples:
        >>> import geotoolz as gz
        >>> # Overlay a cloud mask at 50 % opacity.
        >>> blended = gz.viz.Overlay(alpha=0.5)(rgb_uint8, cloud_rgba)
    """

    def __init__(self, *, alpha: float = 0.6, mode: str = "alpha") -> None:
        self.alpha = alpha
        self.mode = mode

    def _apply(self, background: GeoTensor, foreground: GeoTensor) -> GeoTensor:
        if background.transform != foreground.transform or str(background.crs) != str(
            foreground.crs
        ):
            raise ValueError("background and foreground must share transform and CRS")
        if self.alpha == 0.0:
            # Consistency: always return RGBA, even when no blending occurs.
            return background.array_as_geotensor(
                ensure_rgba(np.asarray(background)), fill_value_default=0
            )
        out = blend_rgba(
            np.asarray(background),
            np.asarray(foreground),
            alpha=self.alpha,
            mode=self.mode,
        )
        return background.array_as_geotensor(out, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "mode": self.mode}


class AnnotatePolygons(Operator):
    """Rasterize polygon outlines into a display GeoTensor.

    Burn polygon boundaries (buffered by ``width`` pixels) into the
    RGBA carrier. The geometries are reprojected into the carrier's
    CRS when they ship as a GeoDataFrame with a CRS set. Flagged
    ``forbid_in_yaml = True`` because the geometries themselves are
    runtime objects and don't round-trip through YAML.

    Args:
        geometries: Iterable of Shapely geometries or a GeoDataFrame.
        color: RGBA tuple in ``[0, 1]``. Default red opaque.
        width: Outline width in pixels. ``0`` is a no-op. Default ``2``.

    Examples:
        >>> import geotoolz as gz
        >>> from shapely.geometry import Polygon
        >>> field = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        >>> annotated = gz.viz.AnnotatePolygons(geometries=[field])(rgb_gt)
    """

    forbid_in_yaml = True

    def __init__(
        self,
        *,
        geometries: Any,
        color: Color = (1.0, 0.0, 0.0, 1.0),
        width: int = 2,
    ) -> None:
        self.geometries = geometries
        self.color = color
        self.width = width

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        from rasterio.features import rasterize

        rgba = ensure_rgba(np.asarray(gt))
        geometries = _iter_geometries(self.geometries, dst_crs=gt.crs)
        if not geometries or self.width <= 0:
            return gt.array_as_geotensor(rgba, fill_value_default=0)
        pixel_size = max(abs(float(gt.transform.a)), abs(float(gt.transform.e)))
        half_width = self.width * pixel_size / 2.0
        shapes = [(geom.boundary.buffer(half_width), 1) for geom in geometries]
        mask = rasterize(
            shapes,
            out_shape=rgba.shape[-2:],
            transform=gt.transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
        rgba[:, mask] = _color_to_uint8(self.color)[:, None]
        return gt.array_as_geotensor(rgba, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {
            "geometries": repr(self.geometries),
            "color": list(self.color),
            "width": self.width,
        }


class AnnotatePoints(Operator):
    """Draw circular point markers into a display GeoTensor.

    Renders disk-shaped markers at the supplied points into the RGBA
    carrier. Accepts either an ``(N, 2)`` array of ``(x, y)`` map
    coordinates or a GeoDataFrame (which is reprojected when it carries
    a CRS). Flagged ``forbid_in_yaml = True`` because the points are
    runtime objects.

    Args:
        points: ``(N, 2)`` array of map coords or a GeoDataFrame of
            Point geometries.
        radius: Marker radius in pixels. ``0`` paints a single pixel.
            Default ``3``.
        color: RGBA tuple in ``[0, 1]``. Default opaque yellow.

    Examples:
        >>> import geotoolz as gz
        >>> import numpy as np
        >>> annotated = gz.viz.AnnotatePoints(
        ...     points=np.array([[12.5, 41.9]]), radius=4
        ... )(rgb_gt)
    """

    forbid_in_yaml = True

    def __init__(
        self,
        *,
        points: Any,
        radius: int = 3,
        color: Color = (1.0, 1.0, 0.0, 1.0),
    ) -> None:
        self.points = points
        self.radius = radius
        self.color = color

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        from rasterio.transform import rowcol

        rgba = ensure_rgba(np.asarray(gt))
        coords = _point_coords(self.points, dst_crs=gt.crs)
        if coords.size == 0:
            return gt.array_as_geotensor(rgba, fill_value_default=0)
        rows, cols = rowcol(gt.transform, coords[:, 0], coords[:, 1])
        yy, xx = np.ogrid[: rgba.shape[-2], : rgba.shape[-1]]
        marker = np.zeros(rgba.shape[-2:], dtype=bool)
        radius = max(int(self.radius), 0)
        for row, col in zip(rows, cols, strict=True):
            marker |= (yy - row) ** 2 + (xx - col) ** 2 <= radius**2
        rgba[:, marker] = _color_to_uint8(self.color)[:, None]
        return gt.array_as_geotensor(rgba, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {
            "points": repr(self.points),
            "radius": self.radius,
            "color": list(self.color),
        }


def _resolve_bands(gt: GeoTensor, refs: Sequence[BandRef]) -> list[int]:
    names = _band_names(gt)
    indices: list[int] = []
    for ref in refs:
        if isinstance(ref, int):
            indices.append(ref)
            continue
        if ref not in names:
            raise ValueError(f"band {ref!r} not found in GeoTensor attrs")
        indices.append(names.index(ref))
    return indices


def _band_names(gt: GeoTensor) -> list[str]:
    for key in ("bands", "band_names", "descriptions"):
        value = gt.attrs.get(key)
        if value is not None:
            return [str(v) for v in value]
    return []


def _get_colormap(name: str) -> Any:
    if name.startswith("cmocean."):
        try:
            import cmocean.cm as cmocean_cm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("cmocean colormaps require installing cmocean") from exc
        return getattr(cmocean_cm, name.split(".", 1)[1])

    from matplotlib import colormaps

    return colormaps[name]


def _iter_geometries(geometries: Any, *, dst_crs: Any) -> list[Any]:
    if hasattr(geometries, "geometry"):
        gdf = geometries
        if getattr(gdf, "crs", None) is not None and dst_crs is not None:
            gdf = gdf.to_crs(dst_crs)
        return [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return [geom for geom in geometries if geom is not None and not geom.is_empty]


def _point_coords(points: Any, *, dst_crs: Any) -> np.ndarray:
    if hasattr(points, "geometry"):
        gdf = points
        if getattr(gdf, "crs", None) is not None and dst_crs is not None:
            gdf = gdf.to_crs(dst_crs)
        return np.asarray([[geom.x, geom.y] for geom in gdf.geometry], dtype=np.float64)
    coords = np.asarray(points, dtype=np.float64)
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"points must be shaped (N, 2); got {coords.shape}")
    return coords


def _color_to_uint8(color: Color) -> np.ndarray:
    return np.clip(np.asarray(color, dtype=np.float32) * 255.0, 0.0, 255.0).astype(
        np.uint8
    )
