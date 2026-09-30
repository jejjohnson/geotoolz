"""Tier-B Operators — carrier-aware radiometric transforms.

Each Operator wraps a primitive in
:mod:`geotoolz.radiometry._src.array`. Operators accept either a
``GeoTensor`` or a plain ``np.ndarray`` and return the same carrier
kind — the rewrap is centralised in
:func:`geotoolz._src.wrap.wrap_like`. The only geo-dependent ops are
`RadianceToReflectance` / `ReflectanceToRadiance` when the solar
geometry must be derived from the footprint (no ``sza_deg`` /
``center_coords`` given); those require a GeoTensor in that mode.

Nodata: pixels that are non-finite or equal the input's
``fill_value_default`` in any band (see :mod:`geotoolz._src.valid`) hold
the output's fill value. Unit conversions carry the input's values, so
they keep its fill when the input is floating point -- as georeader's
own radiance / reflectance conversions do -- and switch to ``NaN`` when
integer DN are promoted to float, where an integer fill such as ``0``
would collide with real data (:func:`geotoolz._src.valid.carried_fill`).
The display stretches (:class:`MinMax`, :class:`PercentileClip`,
:class:`Gamma`) map onto ``[0, 1]``, so they always use ``NaN``.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from georeader.reflectance import (
    integrated_irradiance,
    load_thuillier_irradiance,
    radiance_to_reflectance,
    reflectance_to_radiance,
    srf,
    transform_to_srf,
)
from pipekit import Operator

from geotoolz._src.bands import resolve_wavelengths, strip_band_attrs
from geotoolz._src.config import as_tuple, jsonable
from geotoolz._src.geo import require_geotensor
from geotoolz._src.shape import over_frames
from geotoolz._src.valid import (
    carried_fill,
    invalid_values,
    mask_invalid_to_nan,
    restore_fill,
    wrap_filled,
)
from geotoolz._src.wrap import wrap_like
from geotoolz.radiometry._src.array import (
    _broadcast_to_band_axis,
    bt_from_radiance,
    dn_to_radiance,
    dn_to_reflectance,
    dos1,
    gamma_correct,
    min_max_normalize,
    percentile_clip,
    radiance_to_dn,
)
from geotoolz.radiometry._src.solar import (
    compute_sza,
    earth_sun_distance_correction_factor,
    observation_date_correction_factor,
)


def _rewrap_carried(gt: Any, out: Any, **kwargs: Any) -> Any:
    """Rewrap a unit conversion of ``gt`` with the carried fill at nodata pixels."""
    out = np.asarray(out)
    if out.ndim < 2:
        return wrap_like(gt, out, **kwargs)
    return wrap_filled(
        gt, out, fill_value_default=carried_fill(gt, out.dtype), **kwargs
    )


def _rewrap_stretch(gt: Any, out: Any) -> Any:
    """Rewrap a ``[0, 1]`` display stretch of ``gt`` with ``NaN`` at nodata pixels."""
    out = np.asarray(out)
    if out.ndim < 2:
        return wrap_like(gt, out, fill_value_default=np.nan)
    return wrap_filled(gt, out, fill_value_default=np.nan)


def _parse_datetime(value: datetime | str) -> datetime:
    """Accept either a ``datetime`` or an ISO-8601 string."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


class ToFloat32(Operator):
    """Cast the carrier's values to ``float32``.

    The first stop on most radiometric pipelines: sensor DN are usually
    ``uint16``, and downstream arithmetic (division for indices, gain
    application, etc.) wants float. ``float32`` is the right default for
    imagery — half the memory of ``float64`` with plenty of dynamic
    range for reflectance.

    Integer input with a fill value comes back with
    ``fill_value_default=NaN`` and ``NaN`` at its nodata pixels (an
    integer fill such as ``0`` would be a valid float value); float input
    keeps its fill.

    Examples:
        >>> import geotoolz as gz
        >>> # Standard first stage of a TOA-reflectance pipeline:
        >>> pipe = (
        ...     gz.radiometry.ToFloat32()
        ...     | gz.radiometry.DNToReflectance(scale=1e-4)
        ... )
        >>> reflectance = pipe(uint16_dn_geotensor)
    """

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        # `GeoTensor.astype` would alias the input's attrs dict via
        # __array_finalize__; rewrap so the output gets its own copy.
        return _rewrap_carried(gt, np.asarray(gt).astype(np.float32))


class DNToRadiance(Operator):
    r"""Convert DN to at-sensor radiance.

    .. math::

        L \;=\; \text{gain} \cdot DN + \text{offset}

    Pure linear decode; gain and offset come from sensor metadata. Pass
    scalars for uniform per-pixel coefficients, or 1-D sequences with
    one entry per band to apply different coefficients along the band
    axis.

    See :func:`~geotoolz.radiometry._src.array.dn_to_radiance` for the
    physics. The Operator additionally reshapes 1-D coefficients so
    they broadcast along the configured ``axis``.

    Args:
        gain: Slope of the DN→L decode. Scalar or per-band sequence.
        offset: Intercept. Default ``0.0``.
        axis: Position of the band axis when ``gain`` / ``offset`` are
            per-band sequences. Default ``-3``.

    Examples:
        >>> import numpy as np
        >>> from geotoolz.radiometry import DNToRadiance
        >>> # Landsat-8 OLI per-band radiance multiplicative + additive
        >>> # constants from the MTL file (just illustrative numbers):
        >>> gains   = np.array([0.012, 0.013, 0.011, 0.009])
        >>> offsets = np.array([-60.0, -61.0, -55.0, -45.0])
        >>> dn_to_rad = DNToRadiance(gain=gains, offset=offsets)
        >>> radiance = dn_to_rad(dn_geotensor)  # shape preserved
    """

    def __init__(
        self,
        *,
        gain: float | np.ndarray | list,
        offset: float | np.ndarray | list = 0.0,
        scale: float | np.ndarray | list = 1.0,
        axis: int = -3,
    ) -> None:
        self.gain = gain
        self.offset = offset
        self.scale = scale
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        gain = _broadcast_to_band_axis(self.gain, n_bands, self.axis, arr.ndim)
        offset = _broadcast_to_band_axis(self.offset, n_bands, self.axis, arr.ndim)
        scale = _broadcast_to_band_axis(self.scale, n_bands, self.axis, arr.ndim)
        out = dn_to_radiance(arr, gain, offset, scale)
        return _rewrap_carried(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "gain": jsonable(np.asarray(self.gain, dtype=float)),
            "offset": jsonable(np.asarray(self.offset, dtype=float)),
            "scale": jsonable(np.asarray(self.scale, dtype=float)),
            "axis": self.axis,
        }


class RadianceToDN(Operator):
    r"""Convert at-sensor radiance back to raw DN — inverse of `DNToRadiance`.

    .. math::

        DN \;=\; (L - \text{offset}) \cdot \text{scale} / \text{gain}

    The exact algebraic inverse of `DNToRadiance`; mostly useful when
    you've manipulated radiance in physical units and want to round-trip
    back to a sensor-native integer encoding (e.g. for re-quantisation
    studies or for round-trip testing of an end-to-end pipeline).

    Args:
        gain: Slope used by the forward decode. Scalar or per-band.
        offset: Intercept used by the forward decode. Default ``0.0``.
        scale: DN scale divisor used by the forward decode.
            Default ``1.0``.
        axis: Position of the band axis for per-band coefficients.
            Default ``-3``.

    Examples:
        >>> import numpy as np
        >>> from geotoolz.radiometry import DNToRadiance, RadianceToDN
        >>> # Round-trip a synthetic radiance back to DN.
        >>> radiance = DNToRadiance(gain=0.012, offset=-60.0)(dn_geotensor)
        >>> dn = RadianceToDN(gain=0.012, offset=-60.0)(radiance)
        >>> np.allclose(np.asarray(dn), np.asarray(dn_geotensor))
        True
    """

    def __init__(
        self,
        *,
        gain: float | np.ndarray | list,
        offset: float | np.ndarray | list = 0.0,
        scale: float | np.ndarray | list = 1.0,
        axis: int = -3,
    ) -> None:
        self.gain = gain
        self.offset = offset
        self.scale = scale
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        gain = _broadcast_to_band_axis(self.gain, n_bands, self.axis, arr.ndim)
        offset = _broadcast_to_band_axis(self.offset, n_bands, self.axis, arr.ndim)
        scale = _broadcast_to_band_axis(self.scale, n_bands, self.axis, arr.ndim)
        out = radiance_to_dn(arr, gain, offset, scale)
        return _rewrap_carried(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "gain": jsonable(np.asarray(self.gain, dtype=float)),
            "offset": jsonable(np.asarray(self.offset, dtype=float)),
            "scale": jsonable(np.asarray(self.scale, dtype=float)),
            "axis": self.axis,
        }


class DNToReflectance(Operator):
    r"""Convert DN to TOA / surface reflectance via a linear affine decode.

    .. math::

        \rho \;=\; \text{scale} \cdot DN + \text{offset}

    ``scale`` is the slope (reflectance per DN unit) and ``offset`` is
    the *reflectance-units* intercept — the canonical
    ``y = m * x + b`` form. Matches Landsat Collection-2 SR verbatim
    and absorbs the Sentinel-2 L1C ``RADIO_ADD_OFFSET`` after
    multiplying it through the scale.

    For sensors without a pre-scaled-reflectance product (raw radiance
    only), use `DNToRadiance` then call
    `georeader.reflectance.radiance_to_reflectance` (which handles
    solar geometry properly).

    Args:
        scale: Quantification slope (reflectance per DN unit). Scalar
            or per-band 1-D sequence.
        offset: Reflectance-units intercept. Default ``0.0``.
        axis: Band axis when ``scale`` / ``offset`` are per-band. Default ``-3``.

    Examples:
        >>> from geotoolz.radiometry import DNToReflectance
        >>> # Sentinel-2 L1C pre-2022: a single global scale, no offset.
        >>> op = DNToReflectance(scale=1e-4)
        >>> reflectance = op(s2_l1c_dn_geotensor)
        >>>
        >>> # Post-2022 S2 L1C: RADIO_ADD_OFFSET=-1000 in DN units
        >>> # collapses to -0.1 in reflectance units (-1000 * 1e-4).
        >>> op_v2 = DNToReflectance(scale=1e-4, offset=-0.1)
        >>> reflectance = op_v2(s2_l1c_modern_dn_geotensor)
        >>>
        >>> # Landsat-8/9 Collection-2 surface reflectance.
        >>> op_l8 = DNToReflectance(scale=2.75e-5, offset=-0.2)
        >>> reflectance = op_l8(landsat_c2_sr_geotensor)
    """

    def __init__(
        self,
        *,
        scale: float | np.ndarray | list,
        offset: float | np.ndarray | list = 0.0,
        axis: int = -3,
    ) -> None:
        self.scale = scale
        self.offset = offset
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        scale = _broadcast_to_band_axis(self.scale, n_bands, self.axis, arr.ndim)
        offset = _broadcast_to_band_axis(self.offset, n_bands, self.axis, arr.ndim)
        out = dn_to_reflectance(arr, scale, offset)
        return _rewrap_carried(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "scale": jsonable(np.asarray(self.scale, dtype=float)),
            "offset": jsonable(np.asarray(self.offset, dtype=float)),
            "axis": self.axis,
        }


class RadianceToReflectance(Operator):
    r"""Convert at-sensor radiance to TOA reflectance with solar geometry.

    .. math::

        \rho \;=\; \frac{L \, \pi \, d^{2}}{E_{\text{sun}} \, \cos(\theta_z)}

    where ``L`` is at-sensor radiance, ``E_sun`` is per-band TOA solar
    irradiance in :math:`W \cdot m^{-2} \cdot nm^{-1}`, ``d`` is the
    Earth–Sun distance (AU) for the acquisition date and ``θ_z`` is the
    solar zenith angle. The factor ``π·d²/cos(θ_z)`` is the
    *observation-date correction factor* — pass ``sza_deg`` to provide
    it from metadata, or pass ``center_coords`` and let ``pysolar``
    derive it from the location and UTC time.

    This is a carrier-aware wrapper around
    :func:`georeader.reflectance.radiance_to_reflectance`; it does not
    duplicate the unit-conversion or geometry logic, only the JSON-safe
    ``get_config`` and the ``sza_deg`` shortcut.

    Geo-dependence: when neither ``sza_deg`` nor ``center_coords`` is
    provided, the solar geometry is derived from the footprint, so the
    input must be a georeferenced GeoTensor (plain arrays raise
    ``TypeError`` in that mode).

    Args:
        solar_irradiance: Per-band TOA solar irradiance in W/m²/nm,
            shape ``(C,)``.
        acquisition_date: UTC datetime of acquisition.
        center_coords: Optional ``(lon, lat)`` for the SZA computation.
            Inferred from the GeoTensor transform if omitted.
        sza_deg: Optional pre-computed solar zenith angle in degrees.
            If provided, ``center_coords`` and ``pysolar`` are skipped.
        crs_coords: CRS of ``center_coords``. ``None`` → EPSG:4326.
        units: Radiance units. One of ``"W/m2/sr/nm"``,
            ``"mW/m2/sr/nm"``, ``"uW/cm^2/SR/nm"``.

    Examples:
        >>> from datetime import datetime
        >>> import numpy as np
        >>> from geotoolz.radiometry import RadianceToReflectance
        >>> op = RadianceToReflectance(
        ...     solar_irradiance=np.array([1.95, 1.85, 1.55]),
        ...     acquisition_date=datetime(2024, 6, 21, 10, 30),
        ...     sza_deg=27.0,
        ...     units="W/m2/sr/nm",
        ... )
        >>> reflectance = op(radiance_geotensor)
    """

    def __init__(
        self,
        *,
        solar_irradiance: np.ndarray | list,
        acquisition_date: datetime | str,
        center_coords: tuple[float, float] | None = None,
        sza_deg: float | None = None,
        crs_coords: str | None = None,
        units: str = "W/m2/sr/nm",
    ) -> None:
        self.solar_irradiance = solar_irradiance
        self.acquisition_date = _parse_datetime(acquisition_date)
        self.center_coords = center_coords
        self.sza_deg = sza_deg
        self.crs_coords = crs_coords
        self.units = units

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        obs_factor = observation_date_correction_factor(
            self.acquisition_date,
            sza_deg=self.sza_deg,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
        )
        if obs_factor is None:
            require_geotensor(
                gt,
                "RadianceToReflectance",
                hint="The solar geometry is derived from the footprint when "
                "neither `sza_deg` nor `center_coords` is provided.",
            )
        out = radiance_to_reflectance(
            gt,
            solar_irradiance=self.solar_irradiance,
            date_of_acquisition=self.acquisition_date,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
            observation_date_corr_factor=obs_factor,
            units=self.units,
        )
        # georeader rebuilds the GeoTensor without ``attrs``; restore them.
        return _rewrap_carried(gt, np.asarray(out))

    def get_config(self) -> dict[str, Any]:
        return {
            "solar_irradiance": jsonable(
                np.asarray(self.solar_irradiance, dtype=float)
            ),
            "acquisition_date": jsonable(self.acquisition_date),
            "center_coords": (
                list(self.center_coords) if self.center_coords is not None else None
            ),
            "sza_deg": self.sza_deg,
            "crs_coords": self.crs_coords,
            "units": self.units,
        }


class ReflectanceToRadiance(Operator):
    r"""Convert TOA reflectance back to at-sensor radiance — inverse of `RadianceToReflectance`.

    .. math::

        L \;=\; \frac{\rho \, E_{\text{sun}} \, \cos(\theta_z)}{\pi \, d^{2}}

    Same solar-geometry inputs as the forward direction: pass
    ``sza_deg`` for a metadata-driven SZA or ``center_coords`` to let
    ``pysolar`` derive it.

    Geo-dependence: when neither ``sza_deg`` nor ``center_coords`` is
    provided, the solar geometry is derived from the footprint, so the
    input must be a georeferenced GeoTensor (plain arrays raise
    ``TypeError`` in that mode).

    Args:
        solar_irradiance: Per-band TOA solar irradiance in W/m²/nm.
        acquisition_date: UTC datetime of acquisition.
        center_coords: Optional ``(lon, lat)`` for SZA.
        sza_deg: Optional pre-computed solar zenith angle (degrees).
        crs_coords: CRS of ``center_coords``. ``None`` → EPSG:4326.

    Examples:
        >>> from datetime import datetime
        >>> import numpy as np
        >>> from geotoolz.radiometry import (
        ...     RadianceToReflectance,
        ...     ReflectanceToRadiance,
        ... )
        >>> solar = np.array([1.95, 1.85])
        >>> date = datetime(2024, 7, 14, 11, 32)
        >>> fwd = RadianceToReflectance(
        ...     solar_irradiance=solar, acquisition_date=date, sza_deg=30.0
        ... )
        >>> inv = ReflectanceToRadiance(
        ...     solar_irradiance=solar, acquisition_date=date, sza_deg=30.0
        ... )
        >>> # Round-trip preserves radiance up to float precision.
        >>> radiance_out = inv(fwd(radiance_geotensor))
    """

    def __init__(
        self,
        *,
        solar_irradiance: np.ndarray | list,
        acquisition_date: datetime | str,
        center_coords: tuple[float, float] | None = None,
        sza_deg: float | None = None,
        crs_coords: str | None = None,
    ) -> None:
        self.solar_irradiance = solar_irradiance
        self.acquisition_date = _parse_datetime(acquisition_date)
        self.center_coords = center_coords
        self.sza_deg = sza_deg
        self.crs_coords = crs_coords

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        obs_factor = observation_date_correction_factor(
            self.acquisition_date,
            sza_deg=self.sza_deg,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
        )
        if obs_factor is None:
            require_geotensor(
                gt,
                "ReflectanceToRadiance",
                hint="The solar geometry is derived from the footprint when "
                "neither `sza_deg` nor `center_coords` is provided.",
            )
        out = reflectance_to_radiance(
            gt,
            solar_irradiance=self.solar_irradiance,
            date_of_acquisition=self.acquisition_date,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
            observation_date_corr_factor=obs_factor,
        )
        # georeader rebuilds the GeoTensor without ``attrs``; restore them.
        return _rewrap_carried(gt, np.asarray(out))

    def get_config(self) -> dict[str, Any]:
        return {
            "solar_irradiance": jsonable(
                np.asarray(self.solar_irradiance, dtype=float)
            ),
            "acquisition_date": jsonable(self.acquisition_date),
            "center_coords": (
                list(self.center_coords) if self.center_coords is not None else None
            ),
            "sza_deg": self.sza_deg,
            "crs_coords": self.crs_coords,
        }


class EarthSunDistanceCorrection(Operator):
    r"""Compute the Earth–Sun distance ``d`` (in AU) for an acquisition date.

    Graph source: computed from its configuration alone. An optional
    input is accepted and ignored, so it can close a ``Sequential``.

    .. math::

        d \;=\; 1 - 0.01673 \cdot \cos\!\bigl(0.0172 \cdot (t - 4)\bigr)

    Perihelion (~Jan 3-4) → ``d ≈ 0.983 AU``; aphelion (~Jul 4) →
    ``d ≈ 1.017 AU``. The irradiance reaching Earth scales as
    ``1/d²`` (inverse square law); the ``d²`` factor appears in the
    TOA-reflectance equation.

    Returns a scalar — useful as a building block inside `Graph`
    pipelines that need to thread the value through to a downstream
    operator (e.g. an in-graph reflectance conversion).

    Args:
        acquisition_date: UTC datetime; only the day of year is used.

    Examples:
        >>> from datetime import datetime
        >>> from geotoolz.radiometry import EarthSunDistanceCorrection
        >>> d = EarthSunDistanceCorrection(acquisition_date=datetime(2024, 1, 3))()
        >>> # Perihelion -> ~0.983 AU.
        >>> round(d, 3)
        0.983

    References:
        Spencer, J. W. (1971). Fourier series representation of the
        position of the sun. *Search* 2(5), 172.
    """

    _terminal: ClassVar[bool] = True

    def __init__(self, *, acquisition_date: datetime | str) -> None:
        self.acquisition_date = _parse_datetime(acquisition_date)

    def _apply(self, _input: Any | None = None) -> float:
        return earth_sun_distance_correction_factor(self.acquisition_date)

    def get_config(self) -> dict[str, Any]:
        return {"acquisition_date": jsonable(self.acquisition_date)}


class ComputeSZA(Operator):
    """Compute the solar zenith angle (degrees) for a location and UTC datetime.

    Graph source: computed from its configuration alone. An optional
    input is accepted and ignored, so it can close a ``Sequential``.

    Thin wrapper over :func:`geotoolz.radiometry._src.solar.compute_sza`
    (which delegates to ``pysolar``). The SZA is the complement of the
    solar altitude — ``SZA = 90° - altitude`` — and enters the TOA
    reflectance equation through ``cos(θ_z)``.

    Args:
        center_coords: ``(x, y)`` location. If ``crs_coords`` is
            ``None``, interpreted as ``(lon, lat)`` in EPSG:4326.
        acquisition_date: UTC datetime of acquisition.
        crs_coords: CRS of ``center_coords``. ``None`` → EPSG:4326.

    Examples:
        >>> from datetime import datetime
        >>> from geotoolz.radiometry import ComputeSZA
        >>> # Summer solstice noon at San Francisco.
        >>> op = ComputeSZA(
        ...     center_coords=(-122.4, 37.8),
        ...     acquisition_date=datetime(2024, 6, 21, 20, 0),  # UTC
        ... )
        >>> sza_deg = op()  # ~16° (close to local solar noon)
    """

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        center_coords: tuple[float, float],
        acquisition_date: datetime | str,
        crs_coords: str | None = None,
    ) -> None:
        self.center_coords = center_coords
        self.acquisition_date = _parse_datetime(acquisition_date)
        self.crs_coords = crs_coords

    def _apply(self, _input: Any | None = None) -> float:
        return compute_sza(
            self.center_coords,
            self.acquisition_date,
            crs_coords=self.crs_coords,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "center_coords": list(self.center_coords),
            "acquisition_date": jsonable(self.acquisition_date),
            "crs_coords": self.crs_coords,
        }


class IntegratedIrradiance(Operator):
    r"""Compute band-integrated TOA solar irradiance from an SRF table.

    Graph source: computed from its configuration alone. An optional
    input is accepted and ignored, so it can close a ``Sequential``.

    .. math::

        E_k \;=\; \frac{\displaystyle \int E_{\text{sun}}(\lambda) \, R_k(\lambda) \, d\lambda}
                       {\displaystyle \int R_k(\lambda) \, d\lambda}

    Convolves a TOA solar spectrum (Thuillier 2003 by default) with a
    per-band spectral response function ``R_k(λ)`` to yield the
    band-effective irradiance ``E_k`` that feeds
    `RadianceToReflectance`. Output units match the input solar
    spectrum (``mW/m²/nm`` for the default Thuillier table — divide by
    1000 before handing to the reflectance equation, which expects
    SI ``W/m²/nm``).

    Holds a ``pandas.DataFrame`` — not YAML-serialisable, hence
    ``forbid_in_yaml = True``.

    Args:
        srf: Spectral response DataFrame. Index = wavelength (nm),
            columns = band names. Shape ``(N, K)``.
        solar_irradiance: Optional solar-spectrum DataFrame with
            columns ``["Nanometer", "Radiance(mW/m2/nm)"]``. Defaults
            to Thuillier 2003.
        epsilon_srf: SRF threshold below which a band's contribution is
            ignored. Default ``1e-4``.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from geotoolz.radiometry import IntegratedIrradiance
        >>> # Toy 3-wavelength SRF for a single band.
        >>> srf_df = pd.DataFrame({"B1": [1.0, 1.0, 1.0]}, index=[499.0, 500.0, 501.0])
        >>> # Flat 2 mW/m²/nm solar spectrum.
        >>> solar = pd.DataFrame(
        ...     {"Nanometer": [499.0, 500.0, 501.0],
        ...      "Radiance(mW/m2/nm)": [2.0, 2.0, 2.0]}
        ... )
        >>> e_band = IntegratedIrradiance(srf=srf_df, solar_irradiance=solar)()
        >>> e_band
        array([2.])

    References:
        Thuillier, G. et al. (2003). The Solar Spectral Irradiance from
        200 to 2400 nm as Measured by the SOLSPEC Spectrometer. *Solar
        Physics* 214(1), 1–22.
    """

    # Holds a ``pandas.DataFrame``, not YAML-serialisable.
    forbid_in_yaml: ClassVar[bool] = True
    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        srf: pd.DataFrame,
        solar_irradiance: pd.DataFrame | None = None,
        epsilon_srf: float = 1e-4,
    ) -> None:
        self.srf = srf
        self.solar_irradiance = solar_irradiance
        self.epsilon_srf = epsilon_srf

    def _apply(self, _input: Any | None = None) -> np.ndarray:
        solar_irradiance = (
            load_thuillier_irradiance()
            if self.solar_irradiance is None
            else self.solar_irradiance
        )
        return integrated_irradiance(
            self.srf,
            solar_irradiance=solar_irradiance,
            epsilon_srf=self.epsilon_srf,
        )

    def get_config(self) -> dict[str, Any]:
        # Debug payload: summarise the DataFrames instead of embedding them.
        def summary(frame: pd.DataFrame | None) -> dict[str, Any] | None:
            if frame is None:
                return None
            return {"columns": [str(c) for c in frame.columns], "n_rows": len(frame)}

        return {
            "srf": summary(self.srf),
            "solar_irradiance": summary(self.solar_irradiance),
            "epsilon_srf": self.epsilon_srf,
        }


#: Spacing (nm) of the wavelength grid the SRFs are integrated on.
_SRF_GRID_STEP_NM = 1.0


def _srf_grid(source_wavelengths: np.ndarray, *, extrapolate: bool) -> np.ndarray:
    """The 1-nm integration grid over the source wavelength range.

    Integer nanometres from ``floor(min)`` to ``ceil(max)``. Without
    ``extrapolate``, points outside ``[min, max]`` (only possible for
    fractional source wavelengths) are dropped: georeader maps each grid
    point to its nearest source band and cannot do so outside the range.
    """
    lo, hi = float(source_wavelengths.min()), float(source_wavelengths.max())
    grid = np.arange(np.floor(lo), np.ceil(hi) + 1, _SRF_GRID_STEP_NM)
    if not extrapolate:
        grid = grid[(grid >= lo) & (grid <= hi)]
    return grid


def _check_srf_targets(
    centers: np.ndarray,
    fwhm: np.ndarray,
    source_wavelengths: np.ndarray,
    grid: np.ndarray,
    *,
    epsilon_srf: float,
) -> None:
    """Reject target bands the source cube cannot represent.

    Raises:
        ValueError: If the target arrays disagree in shape, a FWHM is not
            strictly positive and finite, fewer than two source bands are
            given, a target centre lies outside the source range, or a
            target's Gaussian SRF has no grid weight above ``epsilon_srf``
            (FWHM far below the 1-nm grid spacing) -- georeader would
            otherwise divide by a zero SRF sum and return garbage.
    """
    if centers.ndim != 1 or centers.shape != fwhm.shape:
        raise ValueError(
            "target_center_wavelengths and target_fwhm must be 1-D and the same "
            f"length; got shapes {centers.shape} and {fwhm.shape}"
        )
    if not np.all(np.isfinite(fwhm) & (fwhm > 0)):
        raise ValueError(f"target_fwhm must be strictly positive; got {fwhm.tolist()}")
    if source_wavelengths.size < 2 or not np.all(np.isfinite(source_wavelengths)):
        raise ValueError(
            "ApplySRF needs at least two finite source wavelengths; got "
            f"{source_wavelengths.tolist()}"
        )
    lo, hi = float(source_wavelengths.min()), float(source_wavelengths.max())
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    for k, (center, width, s) in enumerate(zip(centers, fwhm, sigma, strict=True)):
        if not lo <= center <= hi:
            raise ValueError(
                f"target band {k} (centre {center:g} nm) lies outside the source "
                f"wavelength range [{lo:g}, {hi:g}] nm"
            )
        # Same Gaussian as georeader.reflectance.srf, normalised on the grid.
        response = np.exp(-((grid - center) ** 2) / (2.0 * s**2))
        total = response.sum()
        if total <= 0 or response.max() / total <= epsilon_srf:
            raise ValueError(
                f"target band {k} (centre {center:g} nm, FWHM {width:g} nm) has no "
                f"SRF weight above epsilon_srf={epsilon_srf:g} on the "
                f"{_SRF_GRID_STEP_NM:g}-nm integration grid; the FWHM must be "
                f"comparable to or wider than {_SRF_GRID_STEP_NM:g} nm"
            )


class ApplySRF(Operator):
    r"""Convolve hyperspectral bands to target Gaussian-SRF multispectral bands.

    For each target band ``k`` with center wavelength ``λ_k`` and FWHM
    ``Δλ_k``, build a normalised Gaussian SRF (:func:`georeader.reflectance.srf`)
    on a 1-nm wavelength grid spanning the source range, then integrate
    with :func:`georeader.reflectance.transform_to_srf`, which maps each
    grid wavelength to its nearest source band:

    .. math::

        L_k \;=\; \frac{\displaystyle \int L(\lambda) \, R_k(\lambda) \, d\lambda}
                       {\displaystyle \int R_k(\lambda) \, d\lambda}

    The 1-nm grid keeps the SRF well defined when a target FWHM is much
    narrower than the source band spacing (the target then reads its
    nearest source band). Targets the grid cannot represent -- a centre
    outside the source range, or a FWHM far below 1 nm -- raise
    ``ValueError`` instead of returning an all-zero band.

    Nodata: the GeoTensor goes straight to ``transform_to_srf``, which
    marks a target pixel missing when any source band *that target reads*
    (weight above ``epsilon_srf``) equals the input's
    ``fill_value_default``; non-finite source values propagate through the
    weighted sum the same way. Those pixels hold the carried fill
    (:func:`geotoolz._src.valid.carried_fill`: the input's fill for float
    input, ``NaN`` for integer DN promoted to float). Source bands a
    target does not read never invalidate it.

    Output ``attrs``: source per-band keys are dropped; ``wavelengths``
    holds the target centres and ``band_names`` the given ``band_names``.
    A ``(T, C, H, W)`` stack is convolved frame by frame. Plain
    ``np.ndarray`` input ``(B, H, W)`` returns a plain float32 array
    (``NaN`` at non-finite pixels; there is no fill value to match).

    Args:
        target_center_wavelengths: ``λ_k`` for each target band (nm).
        target_fwhm: FWHM for each target band (nm).
        source_wavelengths: Source hyperspectral band centres (nm).
            Default ``None`` reads ``gt.attrs["wavelengths"]``.
        band_names: Optional names for the target bands, written to the
            output's ``attrs["band_names"]``. Default ``None`` (no names).
        epsilon_srf: SRF threshold below which contribution is ignored.
            Default ``1e-4``.
        extrapolate: Let georeader map grid wavelengths outside the source
            range to the nearest edge band (only matters for fractional
            source wavelengths). Default ``False``.

    Raises:
        ValueError: On inconsistent target arrays, a non-positive FWHM, a
            target the source range / 1-nm grid cannot represent, or
            source wavelengths missing or not matching the band count.

    Examples:
        >>> from geotoolz.radiometry import ApplySRF
        >>> # Collapse a 5-band hyperspectral cube to two 20-nm-FWHM
        >>> # Gaussian bands at 500 nm and 520 nm.
        >>> op = ApplySRF(
        ...     target_center_wavelengths=[500.0, 520.0],
        ...     target_fwhm=[20.0, 20.0],
        ...     source_wavelengths=[480.0, 490.0, 500.0, 510.0, 520.0],
        ...     band_names=["b500", "b520"],
        ... )
        >>> multispectral = op(hyperspectral_geotensor)
        >>> # Source wavelengths can come from gt.attrs["wavelengths"].
        >>> s2_like = ApplySRF(
        ...     target_center_wavelengths=[490.0, 665.0, 842.0],
        ...     target_fwhm=[65.0, 30.0, 115.0],
        ... )(emit_geotensor)
    """

    def __init__(
        self,
        *,
        target_center_wavelengths: np.ndarray | list[float],
        target_fwhm: np.ndarray | list[float],
        source_wavelengths: np.ndarray | list[float] | None = None,
        band_names: list[str] | None = None,
        epsilon_srf: float = 1e-4,
        extrapolate: bool = False,
    ) -> None:
        self.target_center_wavelengths = target_center_wavelengths
        self.target_fwhm = target_fwhm
        self.source_wavelengths = source_wavelengths
        self.band_names = band_names
        self.epsilon_srf = epsilon_srf
        self.extrapolate = extrapolate

    @over_frames
    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        n_bands = np.shape(gt)[0] if np.ndim(gt) == 3 else None
        if n_bands is None:
            raise ValueError(
                f"ApplySRF expects a (B, H, W) cube or a (T, B, H, W) stack; "
                f"got shape {np.shape(gt)}"
            )
        source_wavelengths = resolve_wavelengths(
            gt, self.source_wavelengths, n_bands=n_bands, name="source_wavelengths"
        )
        centers = np.asarray(self.target_center_wavelengths, dtype=float)
        fwhm = np.asarray(self.target_fwhm, dtype=float)
        if self.band_names is not None and len(self.band_names) != centers.size:
            raise ValueError(
                f"band_names has {len(self.band_names)} entries but there are "
                f"{centers.size} target band(s)"
            )
        grid = _srf_grid(source_wavelengths, extrapolate=self.extrapolate)
        _check_srf_targets(
            centers, fwhm, source_wavelengths, grid, epsilon_srf=self.epsilon_srf
        )
        with warnings.catch_warnings():
            # georeader normalises with ``np.divide(..., where=sum > 0)`` and
            # no ``out``; every column has a positive sum (checked above), so
            # the "uninitialized memory" warning does not apply.
            warnings.filterwarnings("ignore", message="'where' used without 'out'")
            srf_df = pd.DataFrame(srf(centers, fwhm, grid), index=grid)
        # A NaN fill_value_default makes every georeader-detected missing
        # pixel NaN, so it is found below together with pixels made
        # non-finite by NaN / inf source values.
        out = np.asarray(
            transform_to_srf(
                gt,
                srf_df,
                source_wavelengths.tolist(),
                fill_value_default=np.nan,
                epsilon_srf=self.epsilon_srf,
                extrapolate=self.extrapolate,
            )
        )
        fill = carried_fill(gt, out.dtype)
        out = restore_fill(out, np.isfinite(out), fill)
        attrs = strip_band_attrs(getattr(gt, "attrs", None))
        attrs["wavelengths"] = jsonable(centers)
        return wrap_like(
            gt,
            out,
            fill_value_default=fill,
            attrs=attrs,
            band_names=self.band_names,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "target_center_wavelengths": jsonable(
                np.asarray(self.target_center_wavelengths, dtype=float)
            ),
            "target_fwhm": jsonable(np.asarray(self.target_fwhm, dtype=float)),
            "source_wavelengths": (
                None
                if self.source_wavelengths is None
                else jsonable(np.asarray(self.source_wavelengths, dtype=float))
            ),
            "band_names": None if self.band_names is None else list(self.band_names),
            "epsilon_srf": self.epsilon_srf,
            "extrapolate": self.extrapolate,
        }


class BTFromRadiance(Operator):
    r"""Convert thermal at-sensor radiance to brightness temperature (Kelvin).

    Inversion of the Planck blackbody radiation law, in the standard
    Landsat/MTL form:

    .. math::

        T_B \;=\; \frac{K_2}{\ln\!\bigl(K_1 / L + 1\bigr)}

    where ``L`` is at-sensor thermal radiance and ``K1``/``K2`` are
    sensor-specific pre-computed Planck constants supplied by the
    product metadata (Landsat-8 OLI/TIRS Band 10 example:
    ``K1=774.8853``, ``K2=1321.0789``). The output ``T_B`` is in
    Kelvin and represents the temperature a blackbody would need to
    emit the observed radiance — not the true surface temperature,
    which additionally requires emissivity and atmospheric correction.

    Nodata pixels hold the output fill (see the module docstring).

    Args:
        K1: Per-band Planck constant ``K1``. Scalar or per-band 1-D
            sequence (in radiance units, matching ``L``).
        K2: Per-band Planck constant ``K2``. Scalar or per-band 1-D
            sequence (Kelvin).
        axis: Position of the band axis for per-band ``K1``/``K2``.
            Default ``-3``.

    Examples:
        >>> from geotoolz.radiometry import BTFromRadiance
        >>> # Landsat-8 TIRS Band 10 published constants.
        >>> op = BTFromRadiance(K1=774.8853, K2=1321.0789)
        >>> brightness_temp = op(radiance_geotensor)  # Kelvin

    References:
        Planck, M. (1901). Ueber das Gesetz der Energieverteilung im
        Normalspectrum. *Annalen der Physik* 309, 553–563.

        USGS Landsat-8 Data Users Handbook, §5.1 (TIRS thermal
        constants).
    """

    def __init__(
        self,
        *,
        K1: float | np.ndarray | list,
        K2: float | np.ndarray | list,
        axis: int = -3,
    ) -> None:
        self.K1 = K1
        self.K2 = K2
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        k1 = _broadcast_to_band_axis(self.K1, n_bands, self.axis, arr.ndim)
        k2 = _broadcast_to_band_axis(self.K2, n_bands, self.axis, arr.ndim)
        # NaN out nodata before the log so we don't pollute valid pixels
        # with -inf / 0; the output fill is written back afterwards.
        work = np.where(invalid_values(gt), np.nan, arr.astype(float, copy=False))
        out = bt_from_radiance(work, k1, k2)
        return _rewrap_carried(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "K1": jsonable(np.asarray(self.K1, dtype=float)),
            "K2": jsonable(np.asarray(self.K2, dtype=float)),
            "axis": self.axis,
        }


class DOS1(Operator):
    r"""Apply a Chavez (1988) Dark-Object Subtraction (DOS1) approximation.

    The simplest atmospheric correction in the DOS family: for each
    band, estimate the path-radiance / haze contribution as the
    darkest pixel (or low-percentile value) in the scene, then
    subtract it. Pixels in deep shadow or clear water should approach
    zero reflectance in the absence of atmospheric scattering, so any
    observed positive minimum is taken as the haze offset.

    .. math::

        \rho_{\text{surface},\,k}(x, y) \;=\; \max\!\bigl(\,\rho_{\text{TOA},\,k}(x, y) - \rho_{\text{dark},\,k},\; 0\bigr)

    where ``ρ_dark,k`` is the configured low-percentile of band ``k``
    over the spatial axes. DOS1 assumes a single-scattering atmosphere
    with no transmission correction — the more accurate variants
    (DOS2, DOS3, DOS4 from Chavez 1996) layer atmospheric transmission
    and Rayleigh modelling on top.

    Fill-value pixels are excluded from the dark-object percentile and
    hold the output fill (see the module docstring).

    Args:
        dark_percentile: Spatial percentile in ``[0, 100]`` taken as
            the dark-object value per band. Default ``1.0`` (1st
            percentile is more robust than the absolute minimum
            against single-pixel sensor noise).

    Examples:
        >>> from geotoolz.radiometry import DOS1
        >>> # Subtract the 1st-percentile haze per band from TOA
        >>> # reflectance to approximate surface reflectance.
        >>> op = DOS1(dark_percentile=1.0)
        >>> surface = op(toa_reflectance_geotensor)

    References:
        Chavez, P. S. (1988). An improved dark-object subtraction
        technique for atmospheric scattering correction of multispectral
        data. *Remote Sensing of Environment* 24(3), 459–479.

        Chavez, P. S. (1996). Image-based atmospheric corrections —
        Revisited and improved. *Photogrammetric Engineering & Remote
        Sensing* 62(9), 1025–1036.
    """

    def __init__(self, *, dark_percentile: float = 1.0) -> None:
        self.dark_percentile = dark_percentile

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        # NaN out nodata so the percentile reflects only valid data.
        work = mask_invalid_to_nan(gt, dtype=float)
        out = dos1(work, dark_percentile=self.dark_percentile, axis=(-2, -1))
        return _rewrap_carried(gt, out)


class SimpleAtmosphericCorrection(Operator):
    """Dispatch simple BOA approximations; currently supports ``method='dos1'``.

    Convenience entry point that picks an atmospheric-correction
    strategy by name. Only ``"dos1"`` is implemented today; the
    ``aod`` parameter is accepted but ignored — it's reserved for
    future DOS2/DOS3/SMAC variants that will layer aerosol optical
    depth on top of the dark-object estimate.

    Args:
        method: Correction strategy. Currently only ``"dos1"``.
        dark_percentile: Forwarded to `DOS1` when ``method='dos1'``.
        aod: Aerosol optical depth (reserved; currently ignored).

    Examples:
        >>> from geotoolz.radiometry import SimpleAtmosphericCorrection
        >>> op = SimpleAtmosphericCorrection(method="dos1", dark_percentile=1.0)
        >>> surface = op(toa_reflectance_geotensor)

    References:
        See `DOS1` for the Chavez 1988/1996 references.
    """

    def __init__(
        self,
        *,
        method: str = "dos1",
        dark_percentile: float = 1.0,
        aod: float | None = None,
    ) -> None:
        self.method = method
        self.dark_percentile = dark_percentile
        self.aod = aod

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        if self.method.lower() != "dos1":
            raise NotImplementedError(
                "SimpleAtmosphericCorrection currently supports only "
                f"method='dos1'; got {self.method!r}"
            )
        return DOS1(dark_percentile=self.dark_percentile)(gt)


class MinMax(Operator):
    r"""Linear contrast stretch into ``[0, 1]``.

    .. math::

        y \;=\; \frac{x - v_{\min}}{v_{\max} - v_{\min}}

    Display-prep — no physics; just a fixed-bound rescale. See
    `PercentileClip` for the more robust per-scene variant.

    Args:
        vmin: Lower bound (maps to 0).
        vmax: Upper bound (maps to 1). Must be strictly greater than
            ``vmin``.
        clip: Whether to clamp output to ``[0, 1]``. Default ``True``.

    Examples:
        >>> from geotoolz.radiometry import MinMax
        >>> # Stretch reflectance in [0, 0.3] to display range.
        >>> op = MinMax(vmin=0.0, vmax=0.3)
        >>> display_ready = op(reflectance_geotensor)
    """

    def __init__(self, *, vmin: float, vmax: float, clip: bool = True) -> None:
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = min_max_normalize(np.asarray(gt), self.vmin, self.vmax, clip=self.clip)
        return _rewrap_stretch(gt, out)


class PercentileClip(Operator):
    r"""Per-band robust contrast stretch using percentile thresholds.

    Computes :math:`v_{lo} = P_{p_{\min}}(\text{arr})` and
    :math:`v_{hi} = P_{p_{\max}}(\text{arr})` over the configured
    ``axis`` and rescales each slice into ``[0, 1]``.

    Robust against bright outliers (cumulus, specular reflection, sensor
    saturation): a tiny sub-percent population of bright pixels won't
    crush the rest of the histogram the way a true min/max would.

    Args:
        p_min: Lower percentile. Default ``2.0``.
        p_max: Upper percentile. Default ``98.0``.
        axis: Axis (or tuple) to compute percentiles over.
            ``(-2, -1)`` is per-band/-time. ``None`` is global.

    Examples:
        >>> from geotoolz.radiometry import PercentileClip
        >>> # Standard "satellite RGB" stretch -- per-band 2-98 %.
        >>> op = PercentileClip(p_min=2.0, p_max=98.0)
        >>> rgb = op(reflectance_geotensor)
    """

    def __init__(
        self,
        *,
        p_min: float = 2.0,
        p_max: float = 98.0,
        axis: int | tuple[int, ...] | None = (-2, -1),
    ) -> None:
        self.p_min = p_min
        self.p_max = p_max
        self.axis = as_tuple(axis)

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        # Nodata never enters the percentiles (``nanpercentile``).
        values = mask_invalid_to_nan(gt) if np.ndim(gt) >= 2 else np.asarray(gt)
        out = percentile_clip(
            values, p_min=self.p_min, p_max=self.p_max, axis=self.axis
        )
        return _rewrap_stretch(gt, out)


class Gamma(Operator):
    r"""Power-law gamma correction.

    .. math::

        y \;=\; x^{1/\gamma}

    Display-prep — ``g > 1`` brightens midtones, ``g < 1`` darkens
    them. A gentle ``g = 1.2`` is the workhorse "satellite RGB pop"
    default; sRGB encoding uses ``g ≈ 2.2``.

    Args:
        g: Gamma factor (must be strictly positive). Default ``1.2``.

    Examples:
        >>> import geotoolz as gz
        >>> # Classic display pipeline: stretch, then gamma-brighten.
        >>> pipe = (
        ...     gz.radiometry.PercentileClip()
        ...     | gz.radiometry.Gamma(g=1.4)
        ... )
        >>> rgb = pipe(reflectance_geotensor)
    """

    def __init__(self, *, g: float = 1.2) -> None:
        self.g = g

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = gamma_correct(np.asarray(gt), g=self.g)
        return _rewrap_stretch(gt, out)
