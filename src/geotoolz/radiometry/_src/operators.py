"""Tier-B Operators — carrier-aware radiometric transforms.

Each Operator wraps a primitive in
:mod:`geotoolz.radiometry._src.array`. Operators accept either a
``GeoTensor`` or a plain ``np.ndarray`` and return the same carrier
kind — the rewrap is centralised in
:func:`geotoolz._src.wrap.wrap_like`. The only geo-dependent ops are
`RadianceToReflectance` / `ReflectanceToRadiance` when the solar
geometry must be derived from the footprint (no ``sza_deg`` /
``center_coords`` given); those require a GeoTensor in that mode.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import einx
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

from geotoolz._src.config import jsonable
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


def _datetime_as_jsonable(value: datetime | None) -> str | None:
    """Coerce a ``datetime`` config field into an ISO-8601 string.

    Hydra-zen / OmegaConf YAML serialisers can't round-trip arbitrary
    ``datetime`` objects, so we always emit ISO-8601 strings from
    ``get_config()`` and accept either form on the way back in.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def _parse_datetime(value: datetime | str) -> datetime:
    """Accept either a ``datetime`` or an ISO-8601 string."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _coef_as_jsonable(coef: Any) -> float | list[float]:
    """Coerce a gain / offset coefficient into a JSON-safe scalar or list.

    Keeps scalars as Python ``float``; converts numpy scalars, ndarrays
    and any sequence (list / tuple) of numerics into a *flat*
    ``list[float]`` (via ``ravel``) — stricter semantics than the shared
    :func:`geotoolz._src.config.jsonable`, which it builds on. Used by
    `DNToRadiance` / `DNToReflectance`'s ``get_config()`` so Hydra /
    YAML round-trips don't choke on ndarray leaves.
    """
    arr = np.asarray(coef, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return jsonable(arr.ravel())


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


class ToFloat32(Operator):
    """Cast the carrier's values to ``float32``.

    The first stop on most radiometric pipelines: sensor DN are usually
    ``uint16``, and downstream arithmetic (division for indices, gain
    application, etc.) wants float. ``float32`` is the right default for
    imagery — half the memory of ``float64`` with plenty of dynamic
    range for reflectance.

    Pure ufunc — `GeoTensor.__array_ufunc__` preserves transform / CRS /
    fill-value automatically.

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
        # `astype` on a GeoTensor returns a GeoTensor with metadata
        # preserved via __array_finalize__; on a plain ndarray it
        # returns a plain ndarray — carrier kind is preserved either way.
        return gt.astype(np.float32)

    def get_config(self) -> dict[str, Any]:
        return {}


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
            per-band sequences. Default ``0``.

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
        axis: int = 0,
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
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "gain": _coef_as_jsonable(self.gain),
            "offset": _coef_as_jsonable(self.offset),
            "scale": _coef_as_jsonable(self.scale),
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
            Default ``0``.

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
        axis: int = 0,
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
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "gain": _coef_as_jsonable(self.gain),
            "offset": _coef_as_jsonable(self.offset),
            "scale": _coef_as_jsonable(self.scale),
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
        axis: Band axis when ``scale`` / ``offset`` are per-band.

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
        axis: int = 0,
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
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "scale": _coef_as_jsonable(self.scale),
            "offset": _coef_as_jsonable(self.offset),
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

    # Holds a ``datetime`` instance — emit ISO-8601 from ``get_config``
    # so hydra-zen / OmegaConf YAML round-trips succeed.
    forbid_in_yaml: ClassVar[bool] = False

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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        obs_factor = observation_date_correction_factor(
            self.acquisition_date,
            sza_deg=self.sza_deg,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
        )
        if obs_factor is None and not hasattr(gt, "transform"):
            raise TypeError(
                "RadianceToReflectance requires a georeferenced GeoTensor "
                "input to derive the solar geometry from the footprint when "
                "neither `sza_deg` nor `center_coords` is provided; got a "
                "plain array"
            )
        return radiance_to_reflectance(
            gt,
            solar_irradiance=self.solar_irradiance,
            date_of_acquisition=self.acquisition_date,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
            observation_date_corr_factor=obs_factor,
            units=self.units,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "solar_irradiance": _coef_as_jsonable(self.solar_irradiance),
            "acquisition_date": _datetime_as_jsonable(self.acquisition_date),
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

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        obs_factor = observation_date_correction_factor(
            self.acquisition_date,
            sza_deg=self.sza_deg,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
        )
        if obs_factor is None and not hasattr(gt, "transform"):
            raise TypeError(
                "ReflectanceToRadiance requires a georeferenced GeoTensor "
                "input to derive the solar geometry from the footprint when "
                "neither `sza_deg` nor `center_coords` is provided; got a "
                "plain array"
            )
        return reflectance_to_radiance(
            gt,
            solar_irradiance=self.solar_irradiance,
            date_of_acquisition=self.acquisition_date,
            center_coords=self.center_coords,
            crs_coords=self.crs_coords,
            observation_date_corr_factor=obs_factor,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "solar_irradiance": _coef_as_jsonable(self.solar_irradiance),
            "acquisition_date": _datetime_as_jsonable(self.acquisition_date),
            "center_coords": (
                list(self.center_coords) if self.center_coords is not None else None
            ),
            "sza_deg": self.sza_deg,
            "crs_coords": self.crs_coords,
        }


class EarthSunDistanceCorrection(Operator):
    r"""Compute the Earth–Sun distance ``d`` (in AU) for an acquisition date.

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

    def __init__(self, *, acquisition_date: datetime | str) -> None:
        self.acquisition_date = _parse_datetime(acquisition_date)

    def _apply(self, _input: Any | None = None) -> float:
        return earth_sun_distance_correction_factor(self.acquisition_date)

    def get_config(self) -> dict[str, Any]:
        return {"acquisition_date": _datetime_as_jsonable(self.acquisition_date)}


class ComputeSZA(Operator):
    """Compute the solar zenith angle (degrees) for a location and UTC datetime.

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
            "acquisition_date": _datetime_as_jsonable(self.acquisition_date),
            "crs_coords": self.crs_coords,
        }


class IntegratedIrradiance(Operator):
    r"""Compute band-integrated TOA solar irradiance from an SRF table.

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
        # The DataFrame leaves are kept as-is for debugging; ``forbid_in_yaml``
        # signals to YAML loaders that this op cannot be serialised verbatim.
        return {
            "srf": self.srf,
            "solar_irradiance": self.solar_irradiance,
            "epsilon_srf": self.epsilon_srf,
        }


class ApplySRF(Operator):
    r"""Convolve hyperspectral bands to target Gaussian-SRF multispectral bands.

    For each target band ``k`` with center wavelength ``λ_k`` and FWHM
    ``Δλ_k``, build a normalised Gaussian SRF on a 1-nm wavelength grid
    spanning the source range, then integrate:

    .. math::

        L_k \;=\; \frac{\displaystyle \int L(\lambda) \, R_k(\lambda) \, d\lambda}
                       {\displaystyle \int R_k(\lambda) \, d\lambda}

    Carrier-aware wrapper around
    :func:`georeader.reflectance.transform_to_srf`. Source-pixel
    fill-value locations are propagated into the target bands (any
    pixel that was fill in *any* source band stays fill in *every*
    target band).

    This class holds the top-level ``geotoolz.ApplySRF`` name. The
    deliberately distinct :class:`geotoolz.spectral.ApplySRF` variant
    skips the fill propagation but adds band-name / wavelength attrs
    bookkeeping.

    Args:
        target_center_wavelengths: ``λ_k`` for each target band (nm).
        target_fwhm: FWHM for each target band (nm).
        source_wavelengths: Source hyperspectral band centres (nm).
        epsilon_srf: SRF threshold below which contribution is ignored.
            Default ``1e-4``.
        extrapolate: Whether to extrapolate when target FWHM extends
            outside the source range. Default ``False``.

    Examples:
        >>> from geotoolz.radiometry import ApplySRF
        >>> # Collapse a 5-band hyperspectral cube to two 20-nm-FWHM
        >>> # Gaussian bands at 500 nm and 520 nm.
        >>> op = ApplySRF(
        ...     target_center_wavelengths=[500.0, 520.0],
        ...     target_fwhm=[20.0, 20.0],
        ...     source_wavelengths=[480.0, 490.0, 500.0, 510.0, 520.0],
        ... )
        >>> multispectral = op(hyperspectral_geotensor)
    """

    def __init__(
        self,
        *,
        target_center_wavelengths: np.ndarray | list,
        target_fwhm: np.ndarray | list,
        source_wavelengths: np.ndarray | list,
        epsilon_srf: float = 1e-4,
        extrapolate: bool = False,
    ) -> None:
        self.target_center_wavelengths = target_center_wavelengths
        self.target_fwhm = target_fwhm
        self.source_wavelengths = source_wavelengths
        self.epsilon_srf = epsilon_srf
        self.extrapolate = extrapolate

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        source_wavelengths = np.asarray(self.source_wavelengths, dtype=float)
        # 1-nm grid spanning the source range — fine enough for sensor-
        # realistic Gaussian convolutions.
        wavelengths = np.arange(
            np.floor(source_wavelengths.min()),
            np.ceil(source_wavelengths.max()) + 1,
        )
        srf_values = srf(
            self.target_center_wavelengths,
            self.target_fwhm,
            wavelengths,
        )
        srf_df = pd.DataFrame(srf_values, index=wavelengths)
        # Plain ndarray carriers have no fill value — skip the fill
        # bookkeeping entirely in that case.
        fill_value = getattr(gt, "fill_value_default", None)
        out = transform_to_srf(
            np.asarray(gt),
            srf_df,
            source_wavelengths.tolist(),
            fill_value_default=0.0 if fill_value is None else fill_value,
            epsilon_srf=self.epsilon_srf,
            extrapolate=self.extrapolate,
        )
        # Propagate fill-value masks per target band. Each target band's
        # SRF only depends on a subset of source bands (those covered by
        # wavelengths with non-negligible SRF weight); a target band
        # must become fill only when one of its *contributing* source
        # bands is fill at that pixel — not when any unrelated source
        # band happens to be fill.
        if fill_value is not None:
            src = np.asarray(gt)
            src_invalid = src == fill_value  # (n_src, H, W)
            support = self._source_band_support(srf_df, source_wavelengths)
            # invalid[j, h, w] = any contributing source band is fill
            invalid = (
                einx.dot(
                    "j i, i h w -> j h w",
                    support.astype(np.int64),
                    src_invalid.astype(np.int64),
                )
                > 0
            )
            out[invalid] = fill_value
        return wrap_like(gt, out)

    def _source_band_support(
        self,
        srf_df: pd.DataFrame,
        source_wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Boolean (n_target, n_source) matrix of SRF support.

        ``support[j, i]`` is ``True`` iff source band ``i`` is the
        nearest source band for at least one SRF wavelength where the
        target band ``j`` has weight above ``epsilon_srf``. This
        mirrors the per-target source-band selection inside
        ``transform_to_srf``.
        """
        n_target = srf_df.shape[1]
        n_source = source_wavelengths.shape[0]
        # Nearest source-band index per SRF wavelength.
        nearest = np.abs(
            srf_df.index.to_numpy()[:, None] - source_wavelengths[None, :]
        ).argmin(axis=1)
        weights = srf_df.to_numpy()  # (n_wavelength, n_target)
        support = np.zeros((n_target, n_source), dtype=bool)
        for j in range(n_target):
            active = weights[:, j] > self.epsilon_srf
            if not np.any(active):
                continue
            support[j, np.unique(nearest[active])] = True
        return support

    def get_config(self) -> dict[str, Any]:
        return {
            "target_center_wavelengths": _coef_as_jsonable(
                self.target_center_wavelengths
            ),
            "target_fwhm": _coef_as_jsonable(self.target_fwhm),
            "source_wavelengths": _coef_as_jsonable(self.source_wavelengths),
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

    Fill-value pixels are propagated through unchanged.

    Args:
        K1: Per-band Planck constant ``K1``. Scalar or per-band 1-D
            sequence (in radiance units, matching ``L``).
        K2: Per-band Planck constant ``K2``. Scalar or per-band 1-D
            sequence (Kelvin).
        axis: Position of the band axis for per-band ``K1``/``K2``.
            Default ``0``.

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
        axis: int = 0,
    ) -> None:
        self.K1 = K1
        self.K2 = K2
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        arr = np.asarray(gt)
        n_bands = arr.shape[self.axis] if arr.ndim > 2 else 1
        k1 = _broadcast_to_band_axis(self.K1, n_bands, self.axis, arr.ndim)
        k2 = _broadcast_to_band_axis(self.K2, n_bands, self.axis, arr.ndim)
        fill_value = getattr(gt, "fill_value_default", None)
        # Replace fill with NaN before the log so we don't pollute valid
        # pixels with -inf / 0; restore the original fill value after.
        work = arr.astype(float, copy=False)
        if fill_value is not None:
            work = np.where(arr == fill_value, np.nan, work)
        out = bt_from_radiance(work, k1, k2)
        if fill_value is not None:
            out[arr == fill_value] = fill_value
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "K1": _coef_as_jsonable(self.K1),
            "K2": _coef_as_jsonable(self.K2),
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
    propagated through unchanged.

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
        arr = np.asarray(gt, dtype=float)
        fill_value = getattr(gt, "fill_value_default", None)
        valid = None if fill_value is None else arr != fill_value
        # NaN out fill pixels so the percentile reflects only valid data.
        work = arr if valid is None else np.where(valid, arr, np.nan)
        out = dos1(work, dark_percentile=self.dark_percentile, axis=(-2, -1))
        if valid is not None:
            out[~valid] = fill_value
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"dark_percentile": self.dark_percentile}


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

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "dark_percentile": self.dark_percentile,
            "aod": self.aod,
        }


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
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"vmin": self.vmin, "vmax": self.vmax, "clip": self.clip}


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
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = percentile_clip(
            np.asarray(gt), p_min=self.p_min, p_max=self.p_max, axis=self.axis
        )
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"p_min": self.p_min, "p_max": self.p_max, "axis": self.axis}


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
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"g": self.g}
