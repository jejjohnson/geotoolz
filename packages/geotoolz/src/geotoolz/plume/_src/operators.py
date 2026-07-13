"""Carrier-aware plume operators for retrieval post-processing.

These Tier-B operators wrap the Tier-A primitives in ``array.py``.
Per-pixel operators accept a ``georeader.GeoTensor`` or a plain
``np.ndarray`` and return the same carrier kind; geo-dependent operators
(advection geometry, pixel-area math, footprints) require a
georeferenced ``GeoTensor`` and raise ``TypeError`` otherwise. Vector
outputs are ``geopandas.GeoDataFrame``. Algorithms follow the literature
cited in :mod:`geotoolz.plume._src.array`:

- ``SBMP`` — Varon et al. (2021) Sentinel-2 SWIR ratio retrieval.
- ``PlumeMask`` / ``PlumeContours`` / ``PlumeFootprint`` — Frankenberg
  et al. (2016) detection / segmentation conventions.
- ``IMEEstimate`` — Varon et al. (2018) integrated mass enhancement.
- ``CrossSectionalFlux`` — Krings et al. (2011) / Varon (2018) appendix.

The IME and cross-section operators expect the enhancement carrier to be
in kg/m^2. Use :class:`ColumnToMass` to convert from ppm m or mol/m^2.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pipekit import Operator
from pyproj import CRS, Transformer
from rasterio import features
from shapely.geometry import LineString, shape
from shapely.ops import unary_union
from skimage.measure import regionprops_table

from geotoolz._src.wrap import wrap_like
from geotoolz.plume._src.array import (
    ColumnUnit,
    Connectivity,
    ThresholdMode,
    convert_column_units,
    label_components,
    pixel_area,
    pixel_centers,
    plume_length,
    plume_mask,
    squeeze_single_band,
    wind_advection_cone,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


S2_BAND_TO_INDEX = {
    "B1": 0,
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8": 7,
    "B8A": 8,
    "B9": 9,
    "B11": 10,
    "B12": 11,
}

PLUME_REGIONPROPS: tuple[str, ...] = (
    "label",
    "area",
    "area_convex",
    "area_filled",
    "centroid",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "eccentricity",
    "solidity",
    "perimeter",
    "bbox",
    "inertia_tensor_eigvals",
)


def _band_index(band: int | str) -> int:
    if isinstance(band, str):
        key = band.upper()
        if key not in S2_BAND_TO_INDEX:
            raise ValueError(f"unknown Sentinel-2 band name {band!r}")
        return S2_BAND_TO_INDEX[key]
    return int(band)


def _extract_and_clip_band(arr: np.ndarray, band: int | str, axis: int) -> np.ndarray:
    return np.maximum(np.take(arr, _band_index(band), axis=axis), 0.0)


class SBMP(Operator):
    r"""Single-band multi-pass Sentinel-2 SWIR ratio retrieval.

    Emits a unitless methane "enhancement score" that highlights pixels
    where SWIR-2 (B12, ~2200 nm — a strong CH4 absorption window) is
    depressed relative to SWIR-1 (B11, ~1600 nm — weakly absorbing).

    With a ``reference_scene`` (a clean-air acquisition over the same
    geography) the operator computes the log-ratio change

    .. math::

        \Delta \;=\; \log\!\Bigl(\tfrac{\rho_{B11} + \varepsilon}
                                       {\rho_{B12} + \varepsilon}\Bigr)
                   - \log\!\Bigl(\tfrac{\rho^{ref}_{B11} + \varepsilon}
                                       {\rho^{ref}_{B12} + \varepsilon}\Bigr)

    following Varon et al. (2021) — *Satellite discovery of anomalously
    large methane point sources from oil/gas production* (RSE, 251). The
    reference-scene formulation cancels static surface signatures so
    plume pixels dominate :math:`\Delta`.

    Without a reference scene the operator falls back to the same-scene
    normalized SWIR contrast
    ``(SWIR1 - SWIR2) / (SWIR1 + SWIR2 + eps)``, which is a cheap
    detection prefilter (Ehret et al., 2022, TGRS) but is *not* a
    quantitative column retrieval — pair it with ``ColumnToMass`` only
    when calibrated against a per-scene ppm m mapping.

    Inputs are radiance or reflectance arrays with band axis ``axis``;
    negative values are clipped to zero before the log to keep the
    operator finite over noisy radiances. The output is single-band.

    Args:
        swir1: Index or Sentinel-2 band name of the SWIR-1 channel.
        swir2: Index or Sentinel-2 band name of the SWIR-2 channel.
        reference_scene: Optional clean-air ``GeoTensor`` or plain array
            with the same band layout. When supplied, returns log-ratio
            change.
        axis: Band axis of the input. Default ``0``.
        eps: Numerical guard against division by zero. Default ``1e-10``.

    Examples:
        Reference-scene retrieval over a single source::

            import geotoolz as gz

            score = gz.plume.SBMP(reference_scene=pre_event_s2)(post_event_s2)
            mask = gz.plume.PlumeMask(threshold="percentile:99.5")(score)
    """

    def __init__(
        self,
        *,
        swir1: int | str = "B11",
        swir2: int | str = "B12",
        reference_scene: GeoTensor | np.ndarray | None = None,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.swir1 = swir1
        self.swir2 = swir2
        self.reference_scene = reference_scene
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt, dtype=float)
        swir1 = _extract_and_clip_band(arr, self.swir1, self.axis)
        swir2 = _extract_and_clip_band(arr, self.swir2, self.axis)
        ratio = np.log((swir1 + self.eps) / (swir2 + self.eps))
        if self.reference_scene is not None:
            ref = np.asarray(self.reference_scene, dtype=float)
            ref_swir1 = _extract_and_clip_band(ref, self.swir1, self.axis)
            ref_swir2 = _extract_and_clip_band(ref, self.swir2, self.axis)
            ref_ratio = np.log((ref_swir1 + self.eps) / (ref_swir2 + self.eps))
            out = ratio - ref_ratio
        else:
            out = (swir1 - swir2) / (swir1 + swir2 + self.eps)
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "swir1": self.swir1,
            "swir2": self.swir2,
            "reference_scene": None,
            "axis": self.axis,
            "eps": self.eps,
        }
        if self.reference_scene is not None:
            config["reference_scene"] = {
                "shape": list(np.asarray(self.reference_scene).shape),
                "dtype": str(np.asarray(self.reference_scene).dtype),
            }
        return config


class PlumeMask(Operator):
    """Binary plume mask from a single-band score or enhancement map.

    Thresholds the input (absolute number, Otsu, or percentile) and
    drops connected components smaller than ``min_area`` pixels — the
    Frankenberg et al. (2016) detection convention, also used by Varon
    et al. (2018, 2021) for S2/AVIRIS plume identification.

    Args:
        threshold: ``float`` (absolute), ``"otsu"``, or
            ``"percentile:<p>"`` with ``p`` in [0, 100].
        min_area: Minimum component size in pixels.
        connectivity: 4 or 8 connectivity for component labelling.

    Examples:
        >>> mask = gz.plume.PlumeMask(
        ...     threshold="percentile:99.5", min_area=50,
        ... )(enhancement)
    """

    def __init__(
        self,
        *,
        threshold: ThresholdMode = "otsu",
        min_area: int = 50,
        connectivity: Connectivity = 8,
    ) -> None:
        self.threshold = threshold
        self.min_area = min_area
        self.connectivity = connectivity

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        mask = plume_mask(
            np.asarray(gt),
            threshold=self.threshold,
            min_area=self.min_area,
            connectivity=self.connectivity,
        )
        return wrap_like(gt, mask)

    def get_config(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "min_area": self.min_area,
            "connectivity": self.connectivity,
        }


class PlumeContours(Operator):
    """Connected-component labelling of a plume mask.

    Returns either an int32 label image (default) or a boolean mask of
    the retained components. Component size threshold and connectivity
    follow :func:`label_components`.

    Examples:
        >>> labels = gz.plume.PlumeContours(min_area=50)(mask)
    """

    def __init__(
        self,
        *,
        min_area: int = 50,
        return_labels: bool = True,
        connectivity: Connectivity = 8,
    ) -> None:
        self.min_area = min_area
        self.return_labels = return_labels
        self.connectivity = connectivity

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        labels = label_components(
            squeeze_single_band(np.asarray(gt)).astype(bool),
            min_area=self.min_area,
            connectivity=self.connectivity,
        )
        out = labels if self.return_labels else labels > 0
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "min_area": self.min_area,
            "return_labels": self.return_labels,
            "connectivity": self.connectivity,
        }


class PlumeFootprint(Operator):
    """Vectorize a plume mask into polygons with per-plume metadata.

    Implemented over :func:`skimage.measure.regionprops_table`. Returns a
    ``GeoDataFrame`` with one row per surviving plume: ``geometry``, ``area_m2``,
    ``centroid``,
    ``mean_enhancement`` / ``max_enhancement`` (if ``enhancement`` is
    supplied), ``n_pixels``, ``label_id``, and skimage region properties such as
    ``major_axis_length``, ``orientation``, ``eccentricity``, ``solidity``,
    ``perimeter``, ``bbox-*``, and ``inertia_tensor_eigvals-*``.

    Geo-dependent: polygonisation needs the carrier's transform and CRS,
    so the input must be a georeferenced ``GeoTensor`` (plain arrays
    raise ``TypeError``).

    Args:
        min_area_m2: Drop polygons smaller than this area.
        simplify_tolerance: Douglas-Peucker tolerance (in CRS units) for
            polygon simplification; ``None`` to skip.
        enhancement: Optional enhancement ``GeoTensor`` aligned with the
            mask; enables ``mean_enhancement`` / ``max_enhancement``
            statistics per polygon.
        properties: Region properties forwarded to ``regionprops_table``.
        extra_properties: User-defined ``regionprops_table`` callables.

    Examples:
        >>> gdf = gz.plume.PlumeFootprint(
        ...     min_area_m2=500.0,
        ...     enhancement=kg_m2,
        ... )(mask)
    """

    def __init__(
        self,
        *,
        min_area_m2: float = 500.0,
        simplify_tolerance: float | None = 15.0,
        enhancement: GeoTensor | None = None,
        properties: Sequence[str] | None = None,
        extra_properties: Sequence[Callable[..., Any]] | None = None,
    ) -> None:
        self.min_area_m2 = min_area_m2
        self.simplify_tolerance = simplify_tolerance
        self.enhancement = enhancement
        self.properties = tuple(PLUME_REGIONPROPS if properties is None else properties)
        self.extra_properties = (
            None if extra_properties is None else tuple(extra_properties)
        )

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        if getattr(gt, "transform", None) is None:
            raise TypeError(
                "PlumeFootprint requires a georeferenced GeoTensor input; "
                "got a plain array"
            )
        mask_arr = squeeze_single_band(np.asarray(gt))
        if mask_arr.dtype == bool:
            labels = label_components(mask_arr, min_area=1, connectivity=8)
        else:
            labels = mask_arr.astype(np.int32, copy=False)
        enh = (
            None
            if self.enhancement is None
            else squeeze_single_band(np.asarray(self.enhancement))
        )
        properties = list(self.properties)
        if enh is not None:
            properties.extend(
                prop
                for prop in ("centroid_weighted", "mean_intensity", "max_intensity")
                if prop not in properties
            )
        props = pd.DataFrame(
            regionprops_table(
                labels,
                intensity_image=enh,
                properties=properties,
                extra_properties=self.extra_properties,
            )
        )
        # Cast keys to int defensively: skimage returns numpy-int labels and
        # pandas to_dict("index") propagates that dtype. Python int hashes
        # equal in CPython but the explicit cast makes the lookup contract
        # obvious and survives dtype changes upstream.
        #
        # When the caller's ``properties`` omits ``"label"`` we still need a
        # per-component lookup, so synthesise it from ``regionprops_table``'s
        # row order (which mirrors ``sorted(unique(labels) - {0})``).
        if props.empty:
            props_by_label = {}
        else:
            if "label" in props.columns:
                indexed = props.set_index("label", drop=False)
            else:
                synthetic_labels = sorted(int(v) for v in np.unique(labels) if v != 0)
                indexed = props.copy()
                indexed.index = pd.Index(synthetic_labels[: len(indexed)])
            props_by_label = {int(k): v for k, v in indexed.to_dict("index").items()}

        rows: list[dict[str, Any]] = []
        labels_i32 = labels.astype(np.int32, copy=False)
        for label_id in sorted(int(v) for v in np.unique(labels) if v != 0):
            component = labels == label_id
            n_pixels = int(component.sum())
            # ``mask=component`` already restricts the output to this
            # connected component, so no value-based filter is needed.
            geometries = [
                shape(geom)
                for geom, _ in features.shapes(
                    labels_i32, mask=component, transform=gt.transform
                )
            ]
            if not geometries:
                continue
            geometry = unary_union(geometries)
            if self.simplify_tolerance is not None:
                geometry = geometry.simplify(
                    self.simplify_tolerance, preserve_topology=True
                )
            area_m2 = float(geometry.area)
            if area_m2 < self.min_area_m2:
                continue
            component_values = None if enh is None else enh[component]
            region_props = props_by_label.get(label_id, {})

            # Bind region_props / component_values via default args to avoid
            # the late-binding closure trap that ruff B023 flags.
            def _nan_safe(
                reducer: Callable[[np.ndarray], float],
                key: str,
                rp: dict[str, Any] = region_props,
                vals: np.ndarray | None = component_values,
            ) -> float:
                # Prefer the regionprops_table value, but fall back to the
                # NaN-aware reducer when skimage propagates NaN through
                # ``mean_intensity`` / ``max_intensity`` (it does not ignore
                # NaNs internally). ``vals`` is non-None here.
                rp_val = rp.get(key)
                if rp_val is None or not np.isfinite(float(rp_val)):
                    return float(reducer(vals))
                return float(rp_val)

            mean_enhancement = (
                None
                if component_values is None
                else _nan_safe(np.nanmean, "mean_intensity")
            )
            max_enhancement = (
                None
                if component_values is None
                else _nan_safe(np.nanmax, "max_intensity")
            )
            rows.append(
                {
                    **region_props,
                    "geometry": geometry,
                    "area_m2": area_m2,
                    "centroid": geometry.centroid,
                    "mean_enhancement": mean_enhancement,
                    "max_enhancement": max_enhancement,
                    "n_pixels": n_pixels,
                    "label_id": label_id,
                }
            )
        if not rows:
            # No surviving components (empty mask, or all polygons filtered
            # out by ``min_area_m2``). Construct an empty GeoDataFrame with
            # the expected schema so downstream consumers see a stable
            # column layout instead of a constructor error.
            return gpd.GeoDataFrame(
                {
                    "geometry": [],
                    "area_m2": [],
                    "centroid": [],
                    "mean_enhancement": [],
                    "max_enhancement": [],
                    "n_pixels": [],
                    "label_id": [],
                },
                geometry="geometry",
                crs=gt.crs,
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=gt.crs)

    def get_config(self) -> dict[str, Any]:
        return {
            "min_area_m2": self.min_area_m2,
            "simplify_tolerance": self.simplify_tolerance,
            "enhancement": None
            if self.enhancement is None
            else {
                "shape": list(np.asarray(self.enhancement).shape),
                "dtype": str(np.asarray(self.enhancement).dtype),
            },
            "properties": list(self.properties),
            "extra_properties": None
            if self.extra_properties is None
            else [
                getattr(func, "__name__", repr(func)) for func in self.extra_properties
            ],
        }


class WindAdvectionCone(Operator):
    """Mask likely downwind plume extent from a source and wind vector.

    Rasterizes a circular sector centered at ``source``, oriented along
    ``(wind_u, wind_v)``, with half-angle ``half_angle_deg`` and radius
    ``max_distance``. The mask is useful as a prior for plume
    segmentation when the source location and wind are known (e.g. from
    a reanalysis product), following the practice of Varon et al. (2018,
    2021).

    Geo-dependent: the cone is rasterised in CRS coordinates via the
    carrier's transform, so the input must be a georeferenced
    ``GeoTensor`` (plain arrays raise ``TypeError``).

    Args:
        source: ``(x, y)`` source coordinates. If ``crs`` is supplied,
            ``source`` is interpreted in that CRS and reprojected to the
            carrier CRS.
        wind_u: Eastward wind component (m/s).
        wind_v: Northward wind component (m/s).
        half_angle_deg: Half-angle of the cone (degrees).
        max_distance: Radius of the cone (m, in carrier CRS units).
        crs: Optional CRS of ``source``.

    Examples:
        >>> cone = gz.plume.WindAdvectionCone(
        ...     source=(x0, y0), wind_u=3.0, wind_v=1.0,
        ...     half_angle_deg=30.0, max_distance=5000.0,
        ... )(enhancement)
    """

    def __init__(
        self,
        *,
        source: tuple[float, float],
        wind_u: float,
        wind_v: float,
        half_angle_deg: float = 30.0,
        max_distance: float = 5000.0,
        crs: str | None = None,
    ) -> None:
        self.source = source
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.half_angle_deg = half_angle_deg
        self.max_distance = max_distance
        self.crs = crs

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        if getattr(gt, "transform", None) is None:
            raise TypeError(
                "WindAdvectionCone requires a georeferenced GeoTensor input; "
                "got a plain array"
            )
        arr = squeeze_single_band(np.asarray(gt))
        source = self.source
        if self.crs is not None and gt.crs is not None:
            src_crs = CRS.from_user_input(self.crs)
            dst_crs = CRS.from_user_input(gt.crs)
            if src_crs != dst_crs:
                transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                source = transformer.transform(*source)
        mask = wind_advection_cone(
            arr.shape,
            gt.transform,
            source=source,
            wind_u=self.wind_u,
            wind_v=self.wind_v,
            half_angle_deg=self.half_angle_deg,
            max_distance=self.max_distance,
        )
        return wrap_like(gt, mask)

    def get_config(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "wind_u": self.wind_u,
            "wind_v": self.wind_v,
            "half_angle_deg": self.half_angle_deg,
            "max_distance": self.max_distance,
            "crs": self.crs,
        }


class IMEEstimate(Operator):
    r"""Integrated Mass Enhancement (IME) flux estimator.

    Implements the IME method of Varon et al. (2018), AMT
    *Quantifying methane point sources from fine-scale satellite
    observations of atmospheric methane plumes*:

    .. math::

        \mathrm{IME} \;=\; \sum_{i \in \mathrm{plume}}
                            \Omega_i \,\Delta A_i, \qquad
        Q \;=\; \frac{U_{\mathrm{eff}}\,\mathrm{IME}}{L}

    where :math:`\Omega_i` is the column enhancement (kg/m^2) of pixel
    :math:`i`, :math:`\Delta A_i` is its area (m^2), :math:`L` is the
    effective plume length (m) — see :func:`plume_length` — and
    :math:`U_{\mathrm{eff}}` is the effective wind speed (m/s). Varon
    et al. (2018) calibrate :math:`U_{\mathrm{eff}}` against LES
    simulations; this operator takes the calibrated value as
    ``wind_speed``.

    The ``enhancement`` carrier MUST be in kg/m^2. Use
    :class:`ColumnToMass` upstream to convert from ppm m or mol/m^2:
    passing other units silently produces nonsense.

    Geo-dependent: pixel area and plume length are derived from the
    carrier's transform, so the input must be a georeferenced
    ``GeoTensor`` (plain arrays raise ``TypeError``).

    Args:
        plume_mask: Boolean ``GeoTensor`` selecting plume pixels.
        wind_speed: Effective wind speed :math:`U_{\mathrm{eff}}` in m/s.
        length_method: Length estimator: ``"max_axis"``, ``"convex_hull"``,
            or ``"skeleton"``. See :func:`plume_length`.
        pixel_area_m2: Override the pixel area; default is taken from the
            input transform determinant (correct for an equal-area CRS).
        return_uncertainty: Append ``emission_rate_uncertainty_kg_s``.
        uncertainty_fraction: Fractional 1-sigma uncertainty on Q. The
            default 0.5 follows Varon et al. (2018) Table 3.

    Returns:
        Dict with keys ``ime_kg``, ``length_m``, ``wind_speed_m_s``,
        ``emission_rate_kg_s``, and optionally
        ``emission_rate_uncertainty_kg_s``.

    Examples:
        Estimate Q in kg/s from a ppm m enhancement map::

            import geotoolz as gz

            kg_m2 = gz.plume.ColumnToMass(units_in="ppm_m")(enhancement)
            mask = gz.plume.PlumeMask(threshold="percentile:99")(kg_m2)
            result = gz.plume.IMEEstimate(
                plume_mask=mask,
                wind_speed=3.5,
                length_method="convex_hull",
            )(kg_m2)
            print(result["emission_rate_kg_s"])
    """

    def __init__(
        self,
        *,
        plume_mask: GeoTensor,
        wind_speed: float,
        length_method: Literal["max_axis", "convex_hull", "skeleton"] = "max_axis",
        pixel_area_m2: float | None = None,
        return_uncertainty: bool = True,
        uncertainty_fraction: float = 0.5,
    ) -> None:
        self.plume_mask = plume_mask
        self.wind_speed = wind_speed
        self.length_method = length_method
        self.pixel_area_m2 = pixel_area_m2
        self.return_uncertainty = return_uncertainty
        if uncertainty_fraction < 0.0:
            raise ValueError("uncertainty_fraction must be non-negative")
        self.uncertainty_fraction = uncertainty_fraction

    def _apply(self, gt: GeoTensor) -> dict[str, float]:
        if getattr(gt, "transform", None) is None:
            raise TypeError(
                "IMEEstimate requires a georeferenced GeoTensor input; "
                "got a plain array"
            )
        mask = squeeze_single_band(np.asarray(self.plume_mask)).astype(bool)
        enhancement = squeeze_single_band(np.asarray(gt, dtype=float))
        area = (
            self.pixel_area_m2
            if self.pixel_area_m2 is not None
            else pixel_area(gt.transform)
        )
        ime_kg = float(np.nansum(enhancement[mask] * area))
        length_m = plume_length(mask, gt.transform, method=self.length_method)
        rate = 0.0 if length_m == 0.0 else ime_kg * self.wind_speed / length_m
        out = {
            "ime_kg": ime_kg,
            "length_m": float(length_m),
            "wind_speed_m_s": float(self.wind_speed),
            "emission_rate_kg_s": float(rate),
        }
        if self.return_uncertainty:
            out["emission_rate_uncertainty_kg_s"] = float(
                abs(rate) * self.uncertainty_fraction
            )
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "plume_mask": {
                "shape": list(np.asarray(self.plume_mask).shape),
                "dtype": str(np.asarray(self.plume_mask).dtype),
            },
            "wind_speed": self.wind_speed,
            "length_method": self.length_method,
            "pixel_area_m2": self.pixel_area_m2,
            "return_uncertainty": self.return_uncertainty,
            "uncertainty_fraction": self.uncertainty_fraction,
        }


class CrossSectionalFlux(Operator):
    r"""Estimate cross-sectional flux at downwind transects.

    For each transect at downwind distance :math:`d_k`, the operator
    integrates column mass across the plume:

    .. math::

        Q_k \;=\; U \int_{-\infty}^{\infty}
                       \Omega(d_k, y) \, dy
            \;\approx\; U \sum_{i \in \mathrm{strip}_k}
                       \Omega_i \,\Delta y_i

    where :math:`\mathrm{strip}_k` is the set of plume pixels whose
    along-wind distance to the source falls within half a pixel of
    :math:`d_k`. This is the cross-section method of Krings et al. (2011)
    and Varon et al. (2018, appendix); a flat downwind profile of
    :math:`Q_k` versus :math:`d_k` confirms a steady-state estimate.

    The enhancement carrier MUST be in kg/m^2 (see :class:`ColumnToMass`).
    The implementation assumes (approximately) square pixels: it uses
    :math:`\Delta y = \sqrt{\mathrm{pixel\_area}}` as the across-wind
    pixel size. For strongly anisotropic pixel grids, reproject first.

    Geo-dependent: transect placement and pixel size come from the
    carrier's transform and CRS, so the input must be a georeferenced
    ``GeoTensor`` (plain arrays raise ``TypeError``).

    Args:
        plume_mask: Boolean plume ``GeoTensor``.
        source: ``(x, y)`` source coordinates in the carrier CRS.
        wind_u: Eastward wind component (m/s).
        wind_v: Northward wind component (m/s).
        n_transects: Number of downwind transects to evaluate.
        transect_spacing_m: Along-wind distance between transects (m).

    Returns:
        ``GeoDataFrame`` with one row per transect: ``geometry``
        (across-wind ``LineString``), ``transect_id``, ``distance_m``,
        ``flux_kg_s``, ``n_pixels``.

    Examples:
        Evaluate fluxes at 100 m, 200 m, 300 m downwind::

            gdf = gz.plume.CrossSectionalFlux(
                plume_mask=mask, source=(x0, y0),
                wind_u=3.0, wind_v=1.0,
                n_transects=3, transect_spacing_m=100.0,
            )(kg_m2)
    """

    def __init__(
        self,
        *,
        plume_mask: GeoTensor,
        source: tuple[float, float],
        wind_u: float,
        wind_v: float,
        n_transects: int = 5,
        transect_spacing_m: float = 100.0,
    ) -> None:
        self.plume_mask = plume_mask
        self.source = source
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.n_transects = n_transects
        self.transect_spacing_m = transect_spacing_m

    def _apply(self, gt: GeoTensor) -> gpd.GeoDataFrame:
        if getattr(gt, "transform", None) is None:
            raise TypeError(
                "CrossSectionalFlux requires a georeferenced GeoTensor input; "
                "got a plain array"
            )
        wind_norm = float(np.hypot(self.wind_u, self.wind_v))
        if wind_norm == 0.0:
            raise ValueError("wind vector must be non-zero")
        mask = squeeze_single_band(np.asarray(self.plume_mask)).astype(bool)
        enhancement = squeeze_single_band(np.asarray(gt, dtype=float))
        area = pixel_area(gt.transform)
        width = float(np.sqrt(area))
        xs, ys = pixel_centers(mask.shape, gt.transform)
        dx = xs - self.source[0]
        dy = ys - self.source[1]
        along = (dx * self.wind_u + dy * self.wind_v) / wind_norm
        across = (-dx * self.wind_v + dy * self.wind_u) / wind_norm
        rows: list[dict[str, Any]] = []
        for idx in range(self.n_transects):
            distance = (idx + 1) * self.transect_spacing_m
            on_transect = mask & (np.abs(along - distance) <= width / 2.0)
            flux = float(np.nansum(enhancement[on_transect] * wind_norm * width))
            half_len = max(
                float(np.max(np.abs(across[mask]))) if mask.any() else width,
                width,
            )
            cx = self.source[0] + (self.wind_u / wind_norm) * distance
            cy = self.source[1] + (self.wind_v / wind_norm) * distance
            px = -self.wind_v / wind_norm
            py = self.wind_u / wind_norm
            line = LineString(
                [
                    (cx - px * half_len, cy - py * half_len),
                    (cx + px * half_len, cy + py * half_len),
                ]
            )
            rows.append(
                {
                    "geometry": line,
                    "transect_id": idx + 1,
                    "distance_m": float(distance),
                    "flux_kg_s": flux,
                    "n_pixels": int(on_transect.sum()),
                }
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=gt.crs)

    def get_config(self) -> dict[str, Any]:
        return {
            "plume_mask": {
                "shape": list(np.asarray(self.plume_mask).shape),
                "dtype": str(np.asarray(self.plume_mask).dtype),
            },
            "source": self.source,
            "wind_u": self.wind_u,
            "wind_v": self.wind_v,
            "n_transects": self.n_transects,
            "transect_spacing_m": self.transect_spacing_m,
        }


class ColumnToMass(Operator):
    """Convert column enhancement units (ppm m / mol m^-2 / kg m^-2).

    Thin wrapper over :func:`convert_column_units`. The ppm m conversion
    assumes a standard molar volume (298.15 K, 1 atm); see that
    function's docstring for the exact relation and caveats.

    Args:
        gas: ``"CH4"`` (default) or ``"CO2"``.
        units_in: Units of the input carrier.
        units_out: Desired output units.

    Examples:
        >>> kg_m2 = gz.plume.ColumnToMass(
        ...     gas="CH4", units_in="ppm_m", units_out="kg_m2",
        ... )(enhancement)
    """

    def __init__(
        self,
        *,
        gas: str = "CH4",
        units_in: ColumnUnit = "ppm_m",
        units_out: ColumnUnit = "kg_m2",
    ) -> None:
        self.gas = gas
        self.units_in = units_in
        self.units_out = units_out

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = convert_column_units(
            np.asarray(gt),
            gas=self.gas,
            units_in=self.units_in,
            units_out=self.units_out,
        )
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "gas": self.gas,
            "units_in": self.units_in,
            "units_out": self.units_out,
        }


def _iter_instances(
    labels: np.ndarray,
) -> list[tuple[int, np.ndarray, tuple[int, int, int, int]]]:
    """Return ``(label, compact_mask, bbox)`` per non-zero label."""
    from scipy import ndimage

    work = np.where(labels > 0, labels, 0).astype(np.int64, copy=False)
    ids = np.unique(work)
    ids = ids[ids > 0]
    if ids.size == 0:
        return []
    slices = ndimage.find_objects(work)
    out: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []
    for lbl_value in ids:
        lbl = int(lbl_value)
        sl = slices[lbl - 1]
        if sl is None:
            continue
        y_sl, x_sl = sl
        compact = work[sl] == lbl
        bbox = (int(y_sl.start), int(x_sl.start), int(y_sl.stop), int(x_sl.stop))
        out.append((lbl, compact, bbox))
    return out


class PlumeShapeFilter(Operator):
    """Drop instance labels that fail a fiber-shape morphology gate.

    Adapted from Pérez Carrasco et al. (2026), ``physical_constraints.
    apply_physical_constraints`` — a post-Mask R-CNN gate that distinguishes
    elongated plume features from compact false positives. Per instance:

    1. ``area = mask.sum()``; drop when below ``min_area``.
    2. ``fiber_length`` = skeleton-graph diameter (see
       :class:`geotoolz.measure.SkeletonLength`).
    3. ``major_axis`` = ``regionprops.axis_major_length`` — the PCA-based
       major axis of the equivalent ellipse. The original paper uses
       ``cv2.minAreaRect`` (rotated minimum-area rectangle), avoided here
       to keep the optional-dependency surface unchanged; the two measures
       agree closely for typical plume morphologies.
    4. ``fiber_width = area / fiber_length``.
    5. Keep iff
       ``min_fiber_to_major_ratio <= fiber_length / major_axis
       <= max_fiber_to_major_ratio`` AND
       ``min_fiber_width <= fiber_width <= max_fiber_width``.

    Surviving instances keep their original integer labels; rejected
    instances are mapped to background (``0``).

    Args:
        min_area: Minimum pixel count to consider an instance.
        min_fiber_to_major_ratio: Lower bound on ``fiber/major``.
        max_fiber_to_major_ratio: Upper bound on ``fiber/major``.
        min_fiber_width: Lower bound on ``area / fiber_length``.
        max_fiber_width: Upper bound on ``area / fiber_length``.
    """

    def __init__(
        self,
        *,
        min_area: int = 50,
        min_fiber_to_major_ratio: float = 1.0,
        max_fiber_to_major_ratio: float = 10.0,
        min_fiber_width: float = 1.0,
        max_fiber_width: float = 100.0,
    ) -> None:
        if min_area < 0:
            raise ValueError("min_area must be non-negative")
        if min_fiber_to_major_ratio < 0 or max_fiber_to_major_ratio <= 0:
            raise ValueError("fiber-to-major ratio bounds must be non-negative")
        if min_fiber_to_major_ratio > max_fiber_to_major_ratio:
            raise ValueError("min_fiber_to_major_ratio must be <= max")
        if min_fiber_width < 0 or max_fiber_width <= 0:
            raise ValueError("fiber width bounds must be non-negative")
        if min_fiber_width > max_fiber_width:
            raise ValueError("min_fiber_width must be <= max")
        self.min_area = int(min_area)
        self.min_fiber_to_major_ratio = float(min_fiber_to_major_ratio)
        self.max_fiber_to_major_ratio = float(max_fiber_to_major_ratio)
        self.min_fiber_width = float(min_fiber_width)
        self.max_fiber_width = float(max_fiber_width)

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        from geotoolz.measure._src.operators import _skeleton_diameter_pixels

        labels = squeeze_single_band(np.asarray(gt)).astype(np.int64, copy=False)
        kept: set[int] = set()
        for lbl, compact, _bbox in _iter_instances(labels):
            area = int(compact.sum())
            if area < self.min_area:
                continue
            fiber_length = _skeleton_diameter_pixels(compact)
            if fiber_length <= 0:
                continue
            major_axis = _major_axis_length(compact)
            if major_axis <= 0:
                continue
            ratio = fiber_length / major_axis
            fiber_width = area / fiber_length
            if not (
                self.min_fiber_to_major_ratio <= ratio <= self.max_fiber_to_major_ratio
            ):
                continue
            if not self.min_fiber_width <= fiber_width <= self.max_fiber_width:
                continue
            kept.add(lbl)

        if not kept:
            out = np.zeros_like(labels, dtype=np.int32)
        else:
            keep_mask = np.isin(labels, list(kept))
            out = np.where(keep_mask, labels, 0).astype(np.int32, copy=False)
        return wrap_like(gt, out, fill_value_default=0)

    def get_config(self) -> dict[str, Any]:
        return {
            "min_area": self.min_area,
            "min_fiber_to_major_ratio": self.min_fiber_to_major_ratio,
            "max_fiber_to_major_ratio": self.max_fiber_to_major_ratio,
            "min_fiber_width": self.min_fiber_width,
            "max_fiber_width": self.max_fiber_width,
        }


def _major_axis_length(compact_mask: np.ndarray) -> float:
    """PCA-based major-axis length of a compact binary mask (in pixels)."""
    props = regionprops_table(
        compact_mask.astype(np.int32, copy=False),
        properties=("axis_major_length",),
    )
    arr = props.get("axis_major_length")
    if arr is None or len(arr) == 0:
        return 0.0
    return float(arr[0])


def _in_out_mask_stats(
    values: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """``(mean_plume, bg_mean, bg_std, contrast, z_score, ipa)`` from
    Pérez Carrasco et al. (2026), ``rfc_utils.xch4_metrics``."""
    plume_data = values[mask]
    plume_data = plume_data[np.isfinite(plume_data)]
    if plume_data.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan
    bg = values[~mask]
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        # Fallback used in the paper: synthesise a background by shifting
        # the plume distribution down by one std. Keeps the metric
        # well-defined when the mask covers the entire valid area.
        bg = plume_data - float(np.std(plume_data))
    bg_mean = float(np.mean(bg))
    bg_std = float(np.std(bg))
    mean_plume = float(np.mean(plume_data))
    contrast = mean_plume / bg_mean if bg_mean != 0 else float("nan")
    z_score = (mean_plume - bg_mean) / bg_std if bg_std > 0 else float("nan")
    ipa = float(np.sum(plume_data) - bg_mean * plume_data.size) / plume_data.size
    return mean_plume, bg_mean, bg_std, contrast, z_score, ipa


class PlumeColumnStats(Operator):
    """Per-instance in-mask vs background statistics of a column field.

    Adapted from Pérez Carrasco et al. (2026), ``rfc_utils.xch4_metrics``.
    For every non-zero label in the input map, returns a row with the
    in-mask mean, the out-of-mask (background) mean / std, and three
    derived metrics used by the paper's Random-Forest false-plume
    classifier:

    - ``contrast`` = ``mean_plume / mean_background``
    - ``z_score`` = ``(mean_plume - mean_background) / std_background``
    - ``intensity_per_area`` = ``(sum_plume - mean_background * n_plume)
      / n_plume``

    Args:
        column: Carrier with the column field (single-band ``GeoTensor``
            or plain array, same shape as the input label map).
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, column: GeoTensor | np.ndarray) -> None:
        self.column = column

    def _apply(self, gt: GeoTensor | np.ndarray) -> pd.DataFrame:
        labels = squeeze_single_band(np.asarray(gt)).astype(np.int64, copy=False)
        col_arr = squeeze_single_band(np.asarray(self.column)).astype(float, copy=False)
        if col_arr.shape != labels.shape:
            raise ValueError(
                "column shape "
                f"{col_arr.shape} does not match label shape {labels.shape}"
            )
        rows: list[dict[str, Any]] = []
        for lbl, compact, bbox in _iter_instances(labels):
            y0, x0, y1, x1 = bbox
            full_mask = np.zeros_like(col_arr, dtype=bool)
            full_mask[y0:y1, x0:x1] = compact
            mean_plume, bg_mean, bg_std, contrast, z_score, ipa = _in_out_mask_stats(
                col_arr, full_mask
            )
            rows.append(
                {
                    "label": lbl,
                    "area": int(compact.sum()),
                    "mean_enhancement": mean_plume,
                    "background_mean": bg_mean,
                    "background_std": bg_std,
                    "contrast": contrast,
                    "z_score": z_score,
                    "intensity_per_area": ipa,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "label",
                    "area",
                    "mean_enhancement",
                    "background_mean",
                    "background_std",
                    "contrast",
                    "z_score",
                    "intensity_per_area",
                ]
            )
        return pd.DataFrame(rows)

    def get_config(self) -> dict[str, Any]:
        return {
            "column": {
                "shape": list(np.asarray(self.column).shape),
                "dtype": str(np.asarray(self.column).dtype),
            },
        }


def _qnd_polynomial(
    values: np.ndarray,
    mask: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Quantile-Normality-Deviation polynomial coefficients.

    Mirrors Pérez Carrasco et al. (2026), ``rfc_utils.get_QND_poly``: at
    each percentile ``q in [0.01, 0.99]``, take the absolute difference
    between the empirical CDF of the in-mask values and the CDF of a
    Gaussian fit to *all finite values in ``values``*, then fit a polynomial
    of the given degree. Returns coefficients in ``np.polyfit`` order
    (highest power first). Returns ``np.full(degree + 1, nan)`` when fewer
    than 10 finite in-mask samples are available.
    """
    from scipy import stats

    in_mask = values[mask]
    in_mask = in_mask[np.isfinite(in_mask)]
    if in_mask.size < 10:
        return np.full(degree + 1, np.nan)
    scene = values[np.isfinite(values)]
    if scene.size == 0:
        return np.full(degree + 1, np.nan)
    quantiles = np.linspace(0.01, 0.99, 100)
    q_values = np.percentile(in_mask, quantiles * 100.0)
    theoretical_cdf = stats.norm(loc=scene.mean(), scale=scene.std()).cdf(q_values)
    sorted_in = np.sort(in_mask)
    empirical_cdf = np.searchsorted(sorted_in, q_values, side="right") / sorted_in.size
    ks = np.abs(empirical_cdf - theoretical_cdf)
    return np.polyfit(quantiles, ks, degree)


def _dbscan_clump_count(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    perc_threshold: float,
) -> int:
    """Number of DBSCAN clumps among in-mask pixels above ``perc_threshold``.

    Wraps ``rfc_utils.DBSCAN_clusters`` from the reference implementation.
    Returns ``0`` when no in-mask pixels clear the percentile gate.

    Counts unique non-noise DBSCAN cluster labels directly rather than
    re-rasterising to a 4-connected canvas — DBSCAN's ``eps`` already
    encodes the proximity rule the paper relies on, and a separate
    connected-component pass would split a single density-connected
    DBSCAN cluster into multiple "clumps" whenever its members are within
    ``eps`` but not 4-adjacent.
    """
    from sklearn.cluster import DBSCAN

    in_mask = values[mask]
    in_mask = in_mask[np.isfinite(in_mask)]
    if in_mask.size == 0:
        return 0
    threshold = float(np.percentile(in_mask, perc_threshold))
    high_mask = (values > threshold) & mask & np.isfinite(values)
    coords = np.argwhere(high_mask)
    if coords.shape[0] == 0:
        return 0
    labels_db = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    non_noise = labels_db[labels_db != -1]
    return int(np.unique(non_noise).size)


class PlumeQNDFeatures(Operator):
    """Per-instance Quantile-Normality-Deviation features for plume QA.

    Adapted from Pérez Carrasco et al. (2026) — the feature pipeline that
    feeds the Random-Forest false-plume classifier
    (``rfc_utils.get_QND_df_cms`` / ``get_plume_info``). For each non-zero
    label in the input map this operator emits one row containing:

    - ``qnd_column_*`` — polynomial coefficients of the
      Kolmogorov-Smirnov distance between the in-mask CDF of the column
      field and a Gaussian fit to the full scene's column distribution,
      sampled across 100 percentiles in ``[0.01, 0.99]``.
    - ``qnd_albedo_*`` — same construction over the optional albedo field.
    - ``n_clumps`` — DBSCAN cluster count of in-mask pixels above the
      ``perc_threshold`` percentile (per the paper's clump-count signal).
    - ``contrast`` / ``z_score`` / ``intensity_per_area`` — the same stats
      that :class:`PlumeColumnStats` emits, included so a single call
      assembles the full feature vector used by the paper.

    Args:
        column: Column-enhancement carrier (same shape as the label map).
        albedo: Optional albedo carrier; when omitted the ``qnd_albedo_*``
            coefficients are filled with NaN.
        degree: Degree of the QND polynomial (paper default: 6).
        eps: DBSCAN ``eps`` (pixels).
        min_samples: DBSCAN ``min_samples``.
        perc_threshold: Percentile (within the instance mask) above which
            pixels are eligible for clustering.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        column: GeoTensor | np.ndarray,
        albedo: GeoTensor | np.ndarray | None = None,
        degree: int = 6,
        eps: float = 50.0,
        min_samples: int = 50,
        perc_threshold: float = 98.0,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if not 0.0 < perc_threshold < 100.0:
            raise ValueError("perc_threshold must be in (0, 100)")
        self.column = column
        self.albedo = albedo
        self.degree = int(degree)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.perc_threshold = float(perc_threshold)

    def _apply(self, gt: GeoTensor | np.ndarray) -> pd.DataFrame:
        labels = squeeze_single_band(np.asarray(gt)).astype(np.int64, copy=False)
        col_arr = squeeze_single_band(np.asarray(self.column)).astype(float, copy=False)
        if col_arr.shape != labels.shape:
            raise ValueError(
                "column shape "
                f"{col_arr.shape} does not match label shape {labels.shape}"
            )
        if self.albedo is None:
            alb_arr = None
        else:
            alb_arr = squeeze_single_band(np.asarray(self.albedo)).astype(
                float, copy=False
            )
            if alb_arr.shape != labels.shape:
                raise ValueError(
                    "albedo shape "
                    f"{alb_arr.shape} does not match label shape {labels.shape}"
                )

        col_keys = [f"qnd_column_c{i}" for i in range(self.degree + 1)]
        alb_keys = [f"qnd_albedo_c{i}" for i in range(self.degree + 1)]
        rows: list[dict[str, Any]] = []
        for lbl, compact, bbox in _iter_instances(labels):
            y0, x0, y1, x1 = bbox
            full_mask = np.zeros_like(col_arr, dtype=bool)
            full_mask[y0:y1, x0:x1] = compact
            qnd_col = _qnd_polynomial(col_arr, full_mask, self.degree)
            qnd_alb = (
                _qnd_polynomial(alb_arr, full_mask, self.degree)
                if alb_arr is not None
                else np.full(self.degree + 1, np.nan)
            )
            n_clumps = _dbscan_clump_count(
                col_arr,
                full_mask,
                eps=self.eps,
                min_samples=self.min_samples,
                perc_threshold=self.perc_threshold,
            )
            _mp, _bm, _bs, contrast, z_score, ipa = _in_out_mask_stats(
                col_arr, full_mask
            )
            row: dict[str, Any] = {
                "label": lbl,
                "area": int(compact.sum()),
                "n_clumps": n_clumps,
                "contrast": contrast,
                "z_score": z_score,
                "intensity_per_area": ipa,
            }
            row.update(dict(zip(col_keys, qnd_col.tolist(), strict=True)))
            row.update(dict(zip(alb_keys, qnd_alb.tolist(), strict=True)))
            rows.append(row)

        if not rows:
            empty_cols = [
                "label",
                "area",
                "n_clumps",
                "contrast",
                "z_score",
                "intensity_per_area",
                *col_keys,
                *alb_keys,
            ]
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame(rows)

    def get_config(self) -> dict[str, Any]:
        return {
            "column": {
                "shape": list(np.asarray(self.column).shape),
                "dtype": str(np.asarray(self.column).dtype),
            },
            "albedo": None
            if self.albedo is None
            else {
                "shape": list(np.asarray(self.albedo).shape),
                "dtype": str(np.asarray(self.albedo).dtype),
            },
            "degree": self.degree,
            "eps": self.eps,
            "min_samples": self.min_samples,
            "perc_threshold": self.perc_threshold,
        }
