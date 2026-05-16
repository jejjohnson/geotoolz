"""Tier-B visualization Operators wrapping display primitives."""

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


class TrueColor(Operator):
    """Build an RGB composite from explicit red, green, and blue band refs."""

    def __init__(
        self, *, red: BandRef, green: BandRef, blue: BandRef, axis: int = 0
    ) -> None:
        self.red = red
        self.green = green
        self.blue = blue
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        bands = _resolve_bands(gt, [self.red, self.green, self.blue])
        return gt.array_as_geotensor(composite(np.asarray(gt), bands, axis=self.axis))

    def get_config(self) -> dict[str, Any]:
        return {
            "red": self.red,
            "green": self.green,
            "blue": self.blue,
            "axis": self.axis,
        }


class FalseColor(Operator):
    """Build a NIR-red-green false-colour composite."""

    def __init__(
        self, *, nir: BandRef, red: BandRef, green: BandRef, axis: int = 0
    ) -> None:
        self.nir = nir
        self.red = red
        self.green = green
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        bands = _resolve_bands(gt, [self.nir, self.red, self.green])
        return gt.array_as_geotensor(composite(np.asarray(gt), bands, axis=self.axis))

    def get_config(self) -> dict[str, Any]:
        return {
            "nir": self.nir,
            "red": self.red,
            "green": self.green,
            "axis": self.axis,
        }


class SWIRComposite(Operator):
    """Build a SWIR2-NIR-red composite."""

    def __init__(
        self, *, swir2: BandRef, nir: BandRef, red: BandRef, axis: int = 0
    ) -> None:
        self.swir2 = swir2
        self.nir = nir
        self.red = red
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        bands = _resolve_bands(gt, [self.swir2, self.nir, self.red])
        return gt.array_as_geotensor(composite(np.asarray(gt), bands, axis=self.axis))

    def get_config(self) -> dict[str, Any]:
        return {
            "swir2": self.swir2,
            "nir": self.nir,
            "red": self.red,
            "axis": self.axis,
        }


class StretchToUint8(Operator):
    """Percentile-stretch display data to ``uint8``."""

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
    """Apply display gamma correction."""

    def __init__(self, *, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            gamma_correct_display(np.asarray(gt), gamma=self.gamma)
        )

    def get_config(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class ToDisplayRange(StretchToUint8):
    """Alias for a percentile clip plus min-max scale plus ``uint8`` cast."""


class ApplyColormap(Operator):
    """Map a single-band raster to a four-band RGBA GeoTensor."""

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
            "nan_color": self.nan_color,
        }


class ApplyDiscreteColormap(Operator):
    """Map integer categories to a four-band RGBA GeoTensor."""

    def __init__(self, *, mapping: Mapping[int, Color]) -> None:
        self.mapping = dict(mapping)

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            rgba_from_categories(np.asarray(gt), self.mapping),
            fill_value_default=0,
        )

    def get_config(self) -> dict[str, Any]:
        return {"mapping": self.mapping}


class Hillshade(Operator):
    """Compute a single-band ``uint8`` hillshade from a DEM."""

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
    """Apply an elevation colormap and modulate RGB with hillshade."""

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
    """Blend background and foreground GeoTensors on the same grid."""

    def __init__(self, *, alpha: float = 0.6, mode: str = "alpha") -> None:
        self.alpha = alpha
        self.mode = mode

    def _apply(self, background: GeoTensor, foreground: GeoTensor) -> GeoTensor:
        if background.transform != foreground.transform or str(background.crs) != str(
            foreground.crs
        ):
            raise ValueError("background and foreground must share transform and CRS")
        if self.alpha == 0.0:
            return background.array_as_geotensor(np.asarray(background).copy())
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
    """Rasterize polygon outlines into a display GeoTensor."""

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
        if not geometries:
            return gt.array_as_geotensor(rgba, fill_value_default=0)
        pixel_size = max(abs(float(gt.transform.a)), abs(float(gt.transform.e)))
        half_width = max(self.width, 1) * pixel_size / 2.0
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
            "color": self.color,
            "width": self.width,
        }


class AnnotatePoints(Operator):
    """Draw circular point markers into a display GeoTensor."""

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
        return {"points": repr(self.points), "radius": self.radius, "color": self.color}


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
