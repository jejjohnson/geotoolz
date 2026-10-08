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
``MinMaxStretch``, ``Gamma``) which already does the float contrast stretch
— the composites are :class:`geotoolz.spectral.SelectBands` presets and
``StretchToUint8`` / ``GammaCorrect`` reuse radiometry's stretch / gamma
and add the (rounded) byte cast.

Per-pixel display ops (composites by integer index, stretch, gamma,
colormaps, overlay blending) accept either a ``GeoTensor`` or a plain
``np.ndarray`` and return the same carrier kind. Ops that need the
geotransform / CRS to be meaningful (``Hillshade`` without explicit
resolutions, ``ShadedRelief``, ``AnnotatePolygons``, ``AnnotatePoints``)
require a georeferenced GeoTensor and raise ``TypeError`` otherwise.

Time stacks: composites select along the band axis (``-3`` by default),
so a ``(T, C, H, W)`` stack gives a ``(T, 3, H, W)`` composite;
colormaps and terrain shading render each frame (``(T, 4, H, W)`` RGBA,
``(T, 1, H, W)`` hillshade).

Nodata: a pixel is invalid when any band is non-finite or equals the
carrier's ``fill_value_default`` (:mod:`geotoolz._src.valid`). Stretches
and colormaps compute their percentiles / ranges over valid pixels only;
their ``uint8`` outputs carry ``fill_value_default=0`` and write ``0``
(RGBA: fully transparent, alpha ``0``) into invalid pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pipekit import Operator

from geotoolz._src.bands import BandRef, band_names
from geotoolz._src.config import (
    as_tuple,
    mapping_from_pairs,
    mapping_to_pairs,
)
from geotoolz._src.geo import (
    ground_pixel_size,
    require_geotensor,
    require_projected_crs,
)
from geotoolz._src.optional import import_optional
from geotoolz._src.shape import over_frames
from geotoolz._src.valid import (
    carried_fill,
    invalid_values,
    mask_invalid_to_nan,
    restore_fill,
    valid_pixels,
)
from geotoolz._src.wrap import wrap_like
from geotoolz.radiometry._src.operators import PercentileClip
from geotoolz.spectral._src.array import evaluate_band_math
from geotoolz.spectral._src.operators import SelectBands
from geotoolz.viz._src.array import (
    Color,
    _unit_to_uint8,
    blend_rgba,
    ensure_rgba,
    gamma_correct_display,
    hillshade,
    rgb_recipe,
    rgba_from_categories,
    rgba_from_scalar,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


class Composite(SelectBands):
    """Build a multi-band composite by arbitrary band reference.

    A thin :class:`geotoolz.spectral.SelectBands` subclass spelled with
    ``bands`` — the generic band-selection operator the named composites
    (`TrueColor`, `FalseColor`, `SWIRComposite`) wrap. Bands may be
    referenced by integer position along ``axis`` or by name, resolved
    against the carrier's ``attrs`` by the package-wide resolver
    (``band_names``, then ``descriptions``, then ``bands``). Output has
    ``len(bands)`` slices along ``axis`` and the same spatial footprint
    as the input — ``transform`` and ``crs`` round-trip unchanged. The
    selected bands' per-band attrs (``band_names``, ``wavelengths``, ...)
    travel with the output in output order, in a fresh ``attrs`` dict.
    Plain ``np.ndarray`` carriers are supported with integer band
    references (returning a plain array); string names need a carrier
    with band names in ``attrs``. Fill values pass through unchanged
    (pure band selection).

    Args:
        bands: Sequence of band references. Each entry is either an
            integer position or a string name resolved against the
            carrier's ``attrs``.
        axis: Band axis. Default ``-3`` (``(C, H, W)`` convention).

    Examples:
        >>> import geotoolz as gz
        >>> # Sentinel-2 natural-colour composite by name.
        >>> rgb = gz.viz.Composite(bands=["B04", "B03", "B02"])(s2_geotensor)
        >>> # Or by integer position when the carrier has no band names.
        >>> rgb = gz.viz.Composite(bands=[3, 2, 1])(s2_geotensor)
    """

    def __init__(self, *, bands: Sequence[BandRef], axis: int = -3) -> None:
        super().__init__(bands=list(bands), axis=axis)


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
        axis: Band axis. Default ``-3``.

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
        self, *, red: BandRef, green: BandRef, blue: BandRef, axis: int = -3
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
        axis: Band axis. Default ``-3``.

    Examples:
        >>> import geotoolz as gz
        >>> # Sentinel-2 vegetation false colour (8/4/3).
        >>> vis = gz.viz.FalseColor(nir="B08", red="B04", green="B03")(s2)
    """

    def __init__(
        self, *, nir: BandRef, red: BandRef, green: BandRef, axis: int = -3
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
        axis: Band axis. Default ``-3``.

    Examples:
        >>> import geotoolz as gz
        >>> # Landsat-8 SWIR2/NIR/red burn-scar composite (7/5/4).
        >>> vis = gz.viz.SWIRComposite(swir2="B7", nir="B5", red="B4")(l8)
    """

    def __init__(
        self, *, swir2: BandRef, nir: BandRef, red: BandRef, axis: int = -3
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

    Exactly :class:`geotoolz.radiometry.PercentileClip` (same ``lower`` /
    ``upper`` / ``reduce_axes``) followed by a rounded byte cast, so the result
    is ready for PIL / matplotlib. Use ``PercentileClip`` directly if you
    instead need the ``[0, 1]`` floats for further math.
    Metadata-independent: plain ``np.ndarray`` carriers pass through as
    plain arrays.

    Nodata pixels (any band non-finite or equal to the carrier's
    ``fill_value_default``) are excluded from the percentiles, so a
    ``-9999`` fill cannot dominate the stretch, and are written as ``0``
    -- the output's ``fill_value_default``.

    Args:
        lower: Lower percentile. Default ``2.0``.
        upper: Upper percentile. Default ``98.0``; must exceed ``lower``.
        reduce_axes: Axes the percentiles are computed over. Default
            ``(-2, -1)`` stretches each band (and frame) independently;
            ``None`` uses one global pair of thresholds.

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
        self,
        *,
        lower: float = 2.0,
        upper: float = 98.0,
        reduce_axes: int | tuple[int, ...] | None = (-2, -1),
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.reduce_axes = as_tuple(reduce_axes)

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        # Nodata comes back NaN from PercentileClip and casts to the 0 fill.
        stretched = PercentileClip(
            lower=self.lower, upper=self.upper, reduce_axes=self.reduce_axes
        )(gt)
        return wrap_like(gt, _unit_to_uint8(stretched), fill_value_default=0)


class GammaCorrect(Operator):
    """Apply display gamma correction.

    Power-law correction for display-range arrays — brightens midtones
    when ``gamma > 1``. The exponent is
    :func:`geotoolz.radiometry.gamma_correct` (the math behind
    :class:`geotoolz.radiometry.Gamma`); this operator adds the display
    range handling: float carriers are clipped to ``[0, 1]`` before the
    exponent, and integer carriers are normalised to ``[0, 1]`` by their
    dtype maximum and rounded back so the transform is display-correct
    rather than a raw ``256 ** (1 / gamma) = 16`` on uint8.
    Metadata-independent: plain ``np.ndarray`` carriers pass through as
    plain arrays. Elementwise: fill (and non-finite) values are passed
    through unchanged rather than gamma-corrected.

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = gamma_correct_display(
            np.asarray(gt), gamma=self.gamma, inplace_norm=self.inplace_norm
        )
        valid = ~invalid_values(gt)
        fill = carried_fill(gt, np.asarray(out).dtype)
        return wrap_like(gt, restore_fill(out, valid, fill), fill_value_default=fill)


class RGBRecipe(Operator):
    """Build a display RGB from three band expressions with fixed stretches.

    The operational satellite RGB products (the CIRA / EUMETSAT "quick
    guide" composites: day cloud phase, fire temperature, natural colour,
    dust, air mass, ...) are all one recipe: each colour channel is a band
    or a small band expression, linearly stretched between a fixed
    ``vmin`` and ``vmax``, clipped to ``[0, 1]`` and raised to
    ``1 / gamma``. This operator is that recipe; see
    :func:`geotoolz.viz.rgb_recipe` for the maths.

    ``red`` / ``green`` / ``blue`` are each an integer band position, a band
    name resolved against the carrier's ``attrs`` (``band_names``, then
    ``descriptions``, then ``bands``), or an arithmetic expression over band
    names in :class:`geotoolz.spectral.BandMath` grammar
    (``"0.45 * C02 + 0.10 * C03 + 0.45 * C01"``). Invalid input pixels
    (non-finite or equal to the fill) become ``NaN`` per band, so a channel
    is ``NaN`` wherever one of its bands is. Output is ``(3, H, W)``
    ``float32`` in ``[0, 1]`` with ``band_names=("red", "green", "blue")``
    and ``NaN`` fill; follow with ``StretchToUint8`` or multiply by 255 for
    bytes.

    Args:
        red: Red channel: band position, band name or expression.
        green: Green channel.
        blue: Blue channel.
        vmin: Value mapped to 0 — one for all channels or one per channel.
            Set above ``vmax`` to invert a channel. Default ``0.0``.
        vmax: Value mapped to 1. Default ``1.0``.
        gamma: Display gamma (``> 0``), one or per channel. Default ``1.0``.
        axis: Band axis of the input. Default ``-3``.

    Raises:
        ValueError: A ``vmin`` equals its ``vmax``, a gamma is not
            positive, or a per-channel sequence does not have 3 entries.

    Examples:
        >>> import geotoolz as gz
        >>> # CIRA day cloud phase distinction from ABI channels.
        >>> dcp = gz.viz.RGBRecipe(
        ...     red="C13", green="C02", blue="C05",
        ...     vmin=(280.65, 0.0, 0.01), vmax=(219.65, 0.78, 0.59),
        ... )
        >>> rgb = dcp(abi_stack)  # (3, H, W) float32 in [0, 1]
    """

    def __init__(
        self,
        *,
        red: BandRef,
        green: BandRef,
        blue: BandRef,
        vmin: float | Sequence[float] = 0.0,
        vmax: float | Sequence[float] = 1.0,
        gamma: float | Sequence[float] = 1.0,
        axis: int = -3,
    ) -> None:
        self.red = red
        self.green = green
        self.blue = blue
        self.vmin = _per_channel(vmin, "vmin")
        self.vmax = _per_channel(vmax, "vmax")
        self.gamma = _per_channel(gamma, "gamma")
        self.axis = axis
        if any(lo == hi for lo, hi in zip(self.vmin, self.vmax, strict=True)):
            raise ValueError(
                f"vmin and vmax must differ per channel; got {self.vmin}, {self.vmax}."
            )
        if any(g <= 0 for g in self.gamma):
            raise ValueError(f"gamma must be positive; got {self.gamma}.")

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        work = np.where(invalid_values(gt), np.nan, arr.astype(np.float32))
        names = band_names(gt) or []
        bands = {
            name: np.take(work, idx, axis=self.axis)
            for idx, name in enumerate(names)
            if name is not None
        }
        channels = np.stack(
            [
                self._channel(ref, work, names, bands)
                for ref in (self.red, self.green, self.blue)
            ]
        )
        out = rgb_recipe(channels, vmin=self.vmin, vmax=self.vmax, gamma=self.gamma)
        return wrap_like(
            gt,
            np.moveaxis(out, 0, self.axis) if arr.ndim > 2 else out,
            fill_value_default=np.nan,
            attrs={"band_names": ("red", "green", "blue")},
        )

    def _channel(
        self,
        ref: BandRef,
        work: np.ndarray,
        names: list[str | None],
        bands: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if isinstance(ref, int | np.integer):
            return np.take(work, int(ref), axis=self.axis)
        if ref in bands:
            return bands[ref]
        if not bands:
            raise ValueError(
                f"RGBRecipe: {ref!r} needs band names in the carrier's attrs."
            )
        value = np.asarray(evaluate_band_math(ref, bands), dtype=np.float32)
        return np.broadcast_to(value, next(iter(bands.values())).shape)

    def get_config(self) -> dict[str, Any]:
        return {
            "red": self.red,
            "green": self.green,
            "blue": self.blue,
            "vmin": list(self.vmin),
            "vmax": list(self.vmax),
            "gamma": list(self.gamma),
            "axis": self.axis,
        }


def _per_channel(value: float | Sequence[float], name: str) -> tuple[float, ...]:
    """A scalar or a 3-sequence as a 3-tuple of floats."""
    if np.ndim(value) > 0:
        values = tuple(float(v) for v in np.asarray(value, dtype=float).ravel())
        if len(values) != 3:
            raise ValueError(f"{name} needs 1 or 3 values; got {len(values)}.")
        return values
    return (float(np.asarray(value)),) * 3


class ApplyColormap(Operator):
    """Map a single-band raster to a four-band RGBA GeoTensor.

    Looks up ``name`` in matplotlib's registry (or in cmocean if the
    name is prefixed with ``"cmocean."``). The colormap is referenced
    by *string name*, not by the live `Colormap` object — so
    ``get_config()`` round-trips through JSON / YAML cleanly.
    Geo-metadata (``transform`` / ``crs``) is preserved.
    Metadata-independent: plain ``np.ndarray`` carriers pass through
    as plain arrays.

    Args:
        name: Colormap registry name (e.g. ``"viridis"``, ``"terrain"``,
            ``"cmocean.balance"``).
        vmin: Optional explicit lower bound. ``None`` auto-detects.
        vmax: Optional explicit upper bound. ``None`` auto-detects.
        nan_color: RGBA tuple in ``[0, 1]`` to paint nodata pixels
            (non-finite, or equal to the carrier's
            ``fill_value_default``). Default fully transparent, i.e. the
            output fill ``0``. Nodata pixels never enter the auto-detected
            ``vmin`` / ``vmax``.

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

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        cmap = _get_colormap(self.name)
        valid = valid_pixels(gt)
        out = rgba_from_scalar(
            np.asarray(gt) if valid.all() else mask_invalid_to_nan(gt, valid=valid),
            cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            nan_color=self.nan_color,
        )
        return wrap_like(gt, out, fill_value_default=0)

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
    in the mapping fall back to that default colour. Nodata pixels
    (equal to the carrier's ``fill_value_default``) render fully
    transparent (``0``) even if the fill value appears in ``mapping``.
    Metadata-independent: plain ``np.ndarray`` carriers pass through
    as plain arrays.

    Args:
        mapping: ``{class_id: (r, g, b, a)}`` lookup table. The class
            IDs are integers; the RGBA components are floats in
            ``[0, 1]``. Also accepts the ``[[class_id, rgba], ...]``
            pairs that ``get_config()`` emits.

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

    def __init__(
        self, *, mapping: Mapping[int, Color] | Sequence[Sequence[Any]]
    ) -> None:
        pairs = mapping_from_pairs(mapping) or {}
        self.mapping = {int(k): tuple(v) for k, v in pairs.items()}

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = rgba_from_categories(np.asarray(gt), self.mapping)
        out = restore_fill(out, valid_pixels(gt), 0)
        return wrap_like(gt, out, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {"mapping": mapping_to_pairs(self.mapping)}


class Hillshade(Operator):
    """Compute a single-band ``uint8`` hillshade from a DEM.

    GDAL-style hillshade: combines slope and aspect from the DEM with
    a sun position (azimuth + altitude). The output is a single-band
    ``(H, W)`` ``uint8`` raster, ready to compose with a colour relief
    (see `ShadedRelief`). Pixel sizes come from the carrier's
    ``transform`` as ground step lengths (``hypot(a, d)`` /
    ``hypot(b, e)``), so units are correct in physical projections and on
    rotated grids (the azimuth is then relative to the grid's "up"
    direction); a sheared grid or a geographic CRS (degree pixel sizes
    against linear elevations) raises ``ValueError`` -- reproject first,
    or pass both resolutions explicitly.
    Geo-dependent by default: plain ``np.ndarray`` DEMs are accepted
    only when ``x_resolution`` and ``y_resolution`` are given
    explicitly; otherwise a ``TypeError`` is raised. Nodata DEM pixels
    (non-finite or equal to ``fill_value_default``) do not leak into
    their neighbours' slope and are written as ``0`` (the output fill).

    Args:
        azimuth_deg: Sun azimuth in degrees clockwise from north.
            Default ``315`` (NW — the cartographic convention).
        altitude_deg: Sun elevation in degrees above horizon. Default
            ``45``. Values ``>= 90`` short-circuit to flat 255.
        z_factor: Vertical exaggeration. Default ``1.0``.
        x_resolution: Explicit pixel width in map units. ``None``
            (default) reads it from the carrier's ``transform``.
        y_resolution: Explicit pixel height in map units. ``None``
            (default) reads it from the carrier's ``transform``.

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
        x_resolution: float | None = None,
        y_resolution: float | None = None,
    ) -> None:
        self.azimuth_deg = azimuth_deg
        self.altitude_deg = altitude_deg
        self.z_factor = z_factor
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        x_resolution = self.x_resolution
        y_resolution = self.y_resolution
        if x_resolution is None or y_resolution is None:
            transform = require_geotensor(
                gt,
                "Hillshade",
                hint="Pass x_resolution/y_resolution explicitly to hillshade "
                "plain arrays.",
            ).transform
            require_projected_crs(gt, "Hillshade", what="slopes", metres=False)
            row_step, col_step = ground_pixel_size(transform, "Hillshade")
            if x_resolution is None:
                x_resolution = col_step
            if y_resolution is None:
                y_resolution = row_step
        out = hillshade(
            np.asarray(gt),
            x_resolution=x_resolution,
            y_resolution=y_resolution,
            azimuth_deg=self.azimuth_deg,
            altitude_deg=self.altitude_deg,
            z_factor=self.z_factor,
            valid=valid_pixels(gt),
        )
        return wrap_like(gt, out, fill_value_default=0)


class ShadedRelief(Operator):
    """Apply an elevation colormap and modulate RGB with hillshade.

    The composite terrain visualisation: run the DEM through
    `ApplyColormap` and modulate the RGB channels by a `Hillshade`
    so the result reads like a cartographer's shaded-relief map.
    Alpha channel is preserved from the colormap, so nodata DEM pixels
    stay fully transparent (see `ApplyColormap` / `Hillshade`). Geo-dependent: the
    hillshade pixel size comes from the carrier's ``transform``, so a
    georeferenced GeoTensor input in a projected CRS is required (a
    geographic CRS raises ``ValueError``; see `Hillshade`).

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

    @over_frames
    def _apply(self, gt: GeoTensor) -> GeoTensor:
        require_geotensor(gt, "ShadedRelief")
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
        rgba[:3] = np.rint(rgba[:3].astype(np.float32) * shade).astype(np.uint8)
        return wrap_like(gt, rgba, fill_value_default=0)


class Overlay(Operator):
    """Blend background and foreground GeoTensors on the same grid.

    Two-input Operator: ``Overlay()(background, foreground)``. When
    both carriers are georeferenced they must share ``transform`` and
    ``crs`` (this is a viz primitive, not a reprojector); plain
    ``np.ndarray`` carriers skip the metadata check and blend on pixel
    grids alone, returning the background's carrier kind. Output is
    always 4-band ``uint8`` RGBA — even when ``alpha=0`` — so
    downstream code sees a consistent shape.

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

    def _apply(
        self,
        background: GeoTensor | np.ndarray,
        foreground: GeoTensor | np.ndarray,
    ) -> GeoTensor | np.ndarray:
        bg_transform = getattr(background, "transform", None)
        fg_transform = getattr(foreground, "transform", None)
        if (
            bg_transform is not None
            and fg_transform is not None
            and (
                bg_transform != fg_transform
                or str(background.crs) != str(foreground.crs)
            )
        ):
            raise ValueError("background and foreground must share transform and CRS")
        if self.alpha == 0.0:
            # Consistency: always return RGBA, even when no blending occurs.
            return wrap_like(
                background, ensure_rgba(np.asarray(background)), fill_value_default=0
            )
        out = blend_rgba(
            np.asarray(background),
            np.asarray(foreground),
            alpha=self.alpha,
            mode=self.mode,
        )
        return wrap_like(background, out, fill_value_default=0)


class AnnotatePolygons(Operator):
    """Rasterize polygon outlines into a display GeoTensor.

    Burn polygon boundaries (buffered by ``width`` pixels) into the
    RGBA carrier. The geometries are reprojected into the carrier's
    CRS when they ship as a GeoDataFrame with a CRS set. Flagged
    ``forbid_in_yaml = True`` because the geometries themselves are
    runtime objects and don't round-trip through YAML. Geo-dependent:
    rasterisation needs the carrier's ``transform`` / ``crs``, so a
    georeferenced GeoTensor input is required. A ``(T, C, H, W)`` stack
    is annotated frame by frame.

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

    forbid_in_yaml: ClassVar[bool] = True

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

    @over_frames
    def _apply(self, gt: GeoTensor) -> GeoTensor:
        import geopandas as gpd
        from georeader.rasterize import rasterize_geopandas_like

        require_geotensor(gt, "AnnotatePolygons")
        rgba = ensure_rgba(np.asarray(gt), name="AnnotatePolygons")
        geometries = _iter_geometries(self.geometries, dst_crs=gt.crs)
        if not geometries or self.width <= 0:
            return wrap_like(gt, rgba, fill_value_default=0)
        pixel_size = max(abs(float(gt.transform.a)), abs(float(gt.transform.e)))
        half_width = self.width * pixel_size / 2.0
        # Buffer in the carrier's CRS (pixel-width outlines), then burn
        # through georeader onto the carrier's grid.
        outlines = gpd.GeoDataFrame(
            {"burn": np.ones(len(geometries), dtype=np.uint8)},
            geometry=[geom.boundary.buffer(half_width) for geom in geometries],
            crs=gt.crs,
        )
        mask = np.asarray(
            rasterize_geopandas_like(
                outlines, gt, "burn", fill=0, all_touched=True, return_only_data=True
            )
        ).astype(bool)
        rgba[:, mask] = _color_to_uint8(self.color)[:, None]
        return wrap_like(gt, rgba, fill_value_default=0)

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
    runtime objects. Geo-dependent: locating the markers needs the
    carrier's ``transform`` / ``crs``, so a georeferenced GeoTensor
    input is required. A ``(T, C, H, W)`` stack is annotated frame by
    frame.

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

    forbid_in_yaml: ClassVar[bool] = True

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

    @over_frames
    def _apply(self, gt: GeoTensor) -> GeoTensor:
        from rasterio.transform import rowcol

        require_geotensor(gt, "AnnotatePoints")
        rgba = ensure_rgba(np.asarray(gt), name="AnnotatePoints")
        coords = _point_coords(self.points, dst_crs=gt.crs)
        if coords.size == 0:
            return wrap_like(gt, rgba, fill_value_default=0)
        rows, cols = rowcol(gt.transform, coords[:, 0], coords[:, 1])
        yy, xx = np.ogrid[: rgba.shape[-2], : rgba.shape[-1]]
        marker = np.zeros(rgba.shape[-2:], dtype=bool)
        radius = max(int(self.radius), 0)
        for row, col in zip(rows, cols, strict=True):
            marker |= (yy - row) ** 2 + (xx - col) ** 2 <= radius**2
        rgba[:, marker] = _color_to_uint8(self.color)[:, None]
        return wrap_like(gt, rgba, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {
            "points": repr(self.points),
            "radius": self.radius,
            "color": list(self.color),
        }


def _get_colormap(name: str) -> Any:
    if name.startswith("cmocean."):
        try:
            import cmocean.cm as cmocean_cm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                f"Colormap {name!r} requires cmocean: `pip install cmocean`."
            ) from exc
        return getattr(cmocean_cm, name.split(".", 1)[1])

    matplotlib = import_optional("matplotlib", "viz", feature=f"Colormap {name!r}")

    return matplotlib.colormaps[name]


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
    return _unit_to_uint8(np.asarray(color, dtype=np.float32))
