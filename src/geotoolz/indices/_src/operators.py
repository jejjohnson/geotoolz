"""Tier-B Operators — carrier-aware wrappers around the indices primitives.

Each Operator here:

1. Takes its parameters as **keyword-only** ctor args so YAML / Hydra
   configs are unambiguous.
2. Calls into the matching primitive in ``array.py`` for the math.
3. Wraps the result back into a ``GeoTensor`` via the carrier's
   ``array_as_geotensor`` (which propagates ``transform``, ``crs``,
   and ``fill_value_default`` — see `georeader.geotensor`).
4. Returns its config via ``get_config()`` so the Operator round-trips
   through ``hydra_zen.builds``.

The wrap discipline: index primitives collapse the channel axis but
preserve the trailing two spatial axes ``(H, W)``. That matches
``array_as_geotensor``'s contract exactly — it accepts any result whose
last two dims agree with the input's. Carriers' transforms therefore
survive unchanged through every index operator here.

See geotoolz design report §4.1 (two-tier delegation chain) for the
overall pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from geotoolz.core import Operator
from geotoolz.indices._src.array import (
    arvi,
    bais2,
    bsi,
    ciri,
    clay_minerals,
    evi,
    evi2,
    gci,
    iron_oxide,
    kndvi,
    mndwi,
    nbr,
    nbr2,
    ndbi,
    ndmi,
    ndsi,
    ndvi,
    ndwi_mcfeeters,
    normalized_difference,
    savi,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


BandRef = int | str


def _resolve_band(gt: GeoTensor, ref: BandRef) -> int:
    if isinstance(ref, int):
        return ref

    for key in ("descriptions", "band_names", "bands"):
        names = gt.attrs.get(key)
        if names is not None:
            try:
                return list(names).index(ref)
            except ValueError:
                continue

    raise ValueError(
        f"Band {ref!r} was not found in GeoTensor attrs['descriptions'], "
        "attrs['band_names'], or attrs['bands']."
    )


def _configured_ref(value: BandRef | None, fallback: BandRef | None) -> BandRef:
    if value is not None:
        return value
    if fallback is None:
        raise ValueError("A band reference is required.")
    return fallback


def _geotensor_grid_matches(a: GeoTensor, b: GeoTensor) -> bool:
    return (
        a.shape[-2:] == b.shape[-2:] and a.transform == b.transform and a.crs == b.crs
    )


class NormalizedDifference(Operator):
    """Generic normalized-difference index ``(a - b) / (a + b + eps)``.

    Use this when none of the named indices (NDVI/NDWI/NDBI/NBR/...)
    matches your band pair. Equivalent in math to all of them; the
    named subclasses are pinned-argument convenience wrappers.

    Args:
        a_idx: Index of the "high" band (numerator-positive term).
        b_idx: Index of the "low" band.
        axis: Position of the band axis in the carrier. Default ``0``.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geotoolz.indices import NormalizedDifference
        >>> # A custom water/snow index using SWIR vs Green.
        >>> swsi = NormalizedDifference(a_idx=2, b_idx=10)  # Green=2, SWIR1=10
        >>> result = swsi(reflectance_geotensor)
    """

    def __init__(
        self,
        *,
        a: BandRef | None = None,
        b: BandRef | None = None,
        a_idx: int | None = None,
        b_idx: int | None = None,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.a_idx = _configured_ref(a, a_idx)
        self.b_idx = _configured_ref(b, b_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = normalized_difference(
            np.asarray(gt),
            _resolve_band(gt, self.a_idx),
            _resolve_band(gt, self.b_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "a_idx": self.a_idx,
            "b_idx": self.b_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class NDVI(Operator):
    r"""Normalized Difference Vegetation Index — Rouse et al. 1974.

    .. math::

        \mathrm{NDVI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
                                {\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}} + \varepsilon}

    Returns a single-channel ``GeoTensor`` of NDVI values in
    ``[-1, +1]``. See :func:`~geotoolz.indices._src.array.ndvi` for the
    physics; this Operator just plumbs the primitive through carriers.

    The default ``nir_idx=3, red_idx=2`` matches a 4-band ``BGRN``
    convention. For Sentinel-2 imagery stacked in band-number order use
    ``NDVI(nir_idx=7, red_idx=3)`` (B8 NIR, B4 Red after 0-indexing past
    B1/B2/B3); for Landsat-8 use ``NDVI(nir_idx=4, red_idx=3)``
    (B5 NIR, B4 Red).

    Args:
        nir_idx: Band-axis index of the NIR reflectance. Default ``3``.
        red_idx: Band-axis index of the Red reflectance. Default ``2``.
        axis: Position of the band axis. Default ``0``.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geotoolz.indices import NDVI
        >>> ndvi_op = NDVI(nir_idx=7, red_idx=3)         # Sentinel-2 band order
        >>> ndvi_map = ndvi_op(reflectance_geotensor)    # GeoTensor (H, W)
        >>>
        >>> # Chain with reflectance conversion:
        >>> import geotoolz as gz
        >>> pipe = (
        ...     gz.radiometry.DNToReflectance(scale=1e-4)
        ...     | NDVI(nir_idx=7, red_idx=3)
        ... )
        >>> ndvi_map = pipe(dn_geotensor)
    """

    def __init__(
        self,
        *,
        red: BandRef | None = None,
        nir: BandRef | None = None,
        nir_idx: int | None = 3,
        red_idx: int | None = 2,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.red_idx = _configured_ref(red, red_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = ndvi(
            np.asarray(gt),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.red_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "nir_idx": self.nir_idx,
            "red_idx": self.red_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class NDWI(Operator):
    r"""Normalized Difference Water Index — McFeeters 1996.

    .. math::

        \mathrm{NDWI} \;=\; \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{NIR}}}
                                 {\rho_{\mathrm{Green}} + \rho_{\mathrm{NIR}} + \varepsilon}

    For *surface water* delineation; not to be confused with Gao's
    leaf-water NDWI (SWIR/NIR). High over open water, low over
    vegetation. See :func:`~geotoolz.indices._src.array.ndwi_mcfeeters`
    for physics.

    Args:
        green_idx: Band index of Green reflectance. Default ``1``.
        nir_idx: Band index of NIR reflectance. Default ``3``.
        axis: Position of the band axis. Default ``0``.
        eps: Denominator stabiliser.

    Examples:
        >>> from geotoolz.indices import NDWI
        >>> water_idx = NDWI(green_idx=2, nir_idx=7)  # Sentinel-2 B3, B8
        >>> ndwi_map = water_idx(reflectance_geotensor)
        >>> water_mask = (ndwi_map > 0).values  # rough surface-water mask
    """

    def __init__(
        self,
        *,
        green: BandRef | None = None,
        nir: BandRef | None = None,
        green_idx: int | None = 1,
        nir_idx: int | None = 3,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.green_idx = _configured_ref(green, green_idx)
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = ndwi_mcfeeters(
            np.asarray(gt),
            _resolve_band(gt, self.green_idx),
            _resolve_band(gt, self.nir_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "green_idx": self.green_idx,
            "nir_idx": self.nir_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class NDBI(Operator):
    r"""Normalized Difference Built-up Index — Zha et al. 2003.

    .. math::

        \mathrm{NDBI} \;=\; \frac{\rho_{\mathrm{SWIR}} - \rho_{\mathrm{NIR}}}
                                 {\rho_{\mathrm{SWIR}} + \rho_{\mathrm{NIR}} + \varepsilon}

    High over impervious surfaces (concrete, asphalt). Pair with NDVI
    as ``NDBI - NDVI`` to suppress bare-soil confounders. See
    :func:`~geotoolz.indices._src.array.ndbi` for physics.

    Args:
        swir_idx: Band index of SWIR-1 reflectance. Default ``5``.
        nir_idx: Band index of NIR reflectance. Default ``3``.
        axis: Position of the band axis. Default ``0``.
        eps: Denominator stabiliser.

    Examples:
        >>> from geotoolz.indices import NDBI, NDVI
        >>> ndbi_op = NDBI(swir_idx=10, nir_idx=7)  # Sentinel-2 B11, B8
        >>> built_up_raw = ndbi_op(reflectance_geotensor)
        >>>
        >>> # Zha's recipe: subtract NDVI to suppress soil confounders.
        >>> import numpy as np
        >>> ndvi_op = NDVI(nir_idx=7, red_idx=3)
        >>> built_up = np.asarray(ndbi_op(reflectance_geotensor)) - np.asarray(
        ...     ndvi_op(reflectance_geotensor)
        ... )
    """

    def __init__(
        self,
        *,
        swir: BandRef | None = None,
        nir: BandRef | None = None,
        swir_idx: int | None = 5,
        nir_idx: int | None = 3,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.swir_idx = _configured_ref(swir, swir_idx)
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = ndbi(
            np.asarray(gt),
            _resolve_band(gt, self.swir_idx),
            _resolve_band(gt, self.nir_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "swir_idx": self.swir_idx,
            "nir_idx": self.nir_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class NBR(Operator):
    r"""Normalized Burn Ratio — Key & Benson 2006.

    .. math::

        \mathrm{NBR} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{SWIR2}}}
                                {\rho_{\mathrm{NIR}} + \rho_{\mathrm{SWIR2}} + \varepsilon}

    High over healthy vegetation, low / negative over recently-burned
    surfaces. dNBR (``pre - post``) is the standard burn-severity
    quantitative measure. See
    :func:`~geotoolz.indices._src.array.nbr` for physics.

    Args:
        nir_idx: NIR band index. Default ``3``.
        swir2_idx: SWIR-2 band index. Default ``6``.
        axis: Position of the band axis. Default ``0``.
        eps: Denominator stabiliser.

    Examples:
        >>> from geotoolz.indices import NBR
        >>> import numpy as np
        >>> nbr_op = NBR(nir_idx=7, swir2_idx=11)  # Sentinel-2 B8, B12
        >>>
        >>> # dNBR — burn severity from pre- vs post-fire scenes.
        >>> dnbr = np.asarray(nbr_op(pre_fire_geotensor)) - np.asarray(
        ...     nbr_op(post_fire_geotensor)
        ... )
        >>> # Key & Benson thresholds: > 0.66 high severity, 0.27-0.44 low.
    """

    def __init__(
        self,
        *,
        nir: BandRef | None = None,
        swir2: BandRef | None = None,
        nir_idx: int | None = 3,
        swir2_idx: int | None = 6,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.swir2_idx = _configured_ref(swir2, swir2_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = nbr(
            np.asarray(gt),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.swir2_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "nir_idx": self.nir_idx,
            "swir2_idx": self.swir2_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class SAVI(Operator):
    r"""Soil-Adjusted Vegetation Index — Huete 1988.

    .. math::

        \mathrm{SAVI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
                                 {\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}} + L}
                            \,(1 + L)

    Works in sparsely-vegetated scenes where NDVI overweights soil
    brightness. ``L`` shifts the soil line. ``L=0`` recovers NDVI;
    ``L=0.5`` (Huete default) is good for intermediate cover; ``L=1``
    for very sparse cover.

    See :func:`~geotoolz.indices._src.array.savi` for physics.

    Args:
        nir_idx: NIR band index. Default ``3``.
        red_idx: Red band index. Default ``2``.
        L: Soil-adjustment factor in ``[0, 1]``. Default ``0.5``.
        axis: Position of the band axis. Default ``0``.

    Examples:
        >>> from geotoolz.indices import SAVI
        >>> # Default L=0.5 for intermediate cover.
        >>> savi_op = SAVI(nir_idx=7, red_idx=3, L=0.5)
        >>> v = savi_op(reflectance_geotensor)
        >>>
        >>> # Sparse drylands -> larger L:
        >>> dryland_savi = SAVI(nir_idx=7, red_idx=3, L=1.0)
    """

    def __init__(
        self,
        *,
        red: BandRef | None = None,
        nir: BandRef | None = None,
        nir_idx: int | None = 3,
        red_idx: int | None = 2,
        L: float = 0.5,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.red_idx = _configured_ref(red, red_idx)
        self.L = L
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = savi(
            np.asarray(gt),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.red_idx),
            L=self.L,
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "nir_idx": self.nir_idx,
            "red_idx": self.red_idx,
            "L": self.L,
            "axis": self.axis,
            "eps": self.eps,
        }


class EVI(Operator):
    r"""Enhanced Vegetation Index — Huete et al. 2002.

    .. math::

        \mathrm{EVI} \;=\; G \cdot \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
            {\rho_{\mathrm{NIR}} + C_1 \rho_{\mathrm{Red}} - C_2 \rho_{\mathrm{Blue}} + L}

    Designed for MODIS to address NDVI saturation in dense canopies
    and atmospheric-aerosol bias in the red band. The Blue-band
    correction cancels Rayleigh-scattering noise; the L-term
    decouples canopy background.

    Standard MODIS / S2 / L8 coefficients: ``G=2.5, C1=6, C2=7.5, L=1``.
    See :func:`~geotoolz.indices._src.array.evi` for physics.

    Args:
        nir_idx: NIR band index. Default ``3``.
        red_idx: Red band index. Default ``2``.
        blue_idx: Blue band index. Default ``0``.
        G: Gain factor. Default ``2.5``.
        C1: Red aerosol-resistance coefficient. Default ``6``.
        C2: Blue aerosol-resistance coefficient. Default ``7.5``.
        L: Canopy-background correction. Default ``1``.
        axis: Position of the band axis. Default ``0``.

    Examples:
        >>> from geotoolz.indices import EVI
        >>> # Sentinel-2: B2=Blue, B4=Red, B8=NIR.
        >>> evi_op = EVI(nir_idx=7, red_idx=3, blue_idx=1)
        >>> v = evi_op(boa_reflectance_geotensor)
        >>>
        >>> # EVI assumes surface reflectance (BOA). Apply atmospheric
        >>> # correction first for cross-scene comparability.
    """

    def __init__(
        self,
        *,
        blue: BandRef | None = None,
        red: BandRef | None = None,
        nir: BandRef | None = None,
        nir_idx: int | None = 3,
        red_idx: int | None = 2,
        blue_idx: int | None = 0,
        G: float = 2.5,
        C1: float = 6.0,
        C2: float = 7.5,
        L: float = 1.0,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.red_idx = _configured_ref(red, red_idx)
        self.blue_idx = _configured_ref(blue, blue_idx)
        self.G = G
        self.C1 = C1
        self.C2 = C2
        self.L = L
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = evi(
            np.asarray(gt),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.red_idx),
            _resolve_band(gt, self.blue_idx),
            G=self.G,
            C1=self.C1,
            C2=self.C2,
            L=self.L,
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "nir_idx": self.nir_idx,
            "red_idx": self.red_idx,
            "blue_idx": self.blue_idx,
            "G": self.G,
            "C1": self.C1,
            "C2": self.C2,
            "L": self.L,
            "axis": self.axis,
            "eps": self.eps,
        }


class EVI2(Operator):
    """Two-band Enhanced Vegetation Index."""

    def __init__(
        self,
        *,
        red: BandRef | None = None,
        nir: BandRef | None = None,
        red_idx: int | None = 2,
        nir_idx: int | None = 3,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.red_idx = _configured_ref(red, red_idx)
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = evi2(
            np.asarray(gt),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.red_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "red_idx": self.red_idx,
            "nir_idx": self.nir_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class ARVI(Operator):
    """Atmospherically Resistant Vegetation Index."""

    def __init__(
        self,
        *,
        blue: BandRef | None = None,
        red: BandRef | None = None,
        nir: BandRef | None = None,
        blue_idx: int | None = 0,
        red_idx: int | None = 2,
        nir_idx: int | None = 3,
        gamma: float = 1.0,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.blue_idx = _configured_ref(blue, blue_idx)
        self.red_idx = _configured_ref(red, red_idx)
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.gamma = gamma
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = arvi(
            np.asarray(gt),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.red_idx),
            _resolve_band(gt, self.blue_idx),
            gamma=self.gamma,
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "blue_idx": self.blue_idx,
            "red_idx": self.red_idx,
            "nir_idx": self.nir_idx,
            "gamma": self.gamma,
            "axis": self.axis,
            "eps": self.eps,
        }


class GCI(Operator):
    """Green Chlorophyll Index."""

    def __init__(
        self,
        *,
        green: BandRef | None = None,
        nir: BandRef | None = None,
        green_idx: int | None = 1,
        nir_idx: int | None = 3,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.green_idx = _configured_ref(green, green_idx)
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = gci(
            np.asarray(gt),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.green_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "green_idx": self.green_idx,
            "nir_idx": self.nir_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class kNDVI(Operator):
    """Kernel NDVI (Camps-Valls et al. 2021)."""

    def __init__(
        self,
        *,
        red: BandRef | None = None,
        nir: BandRef | None = None,
        red_idx: int | None = 2,
        nir_idx: int | None = 3,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.red_idx = _configured_ref(red, red_idx)
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = kndvi(
            np.asarray(gt),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.red_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "red_idx": self.red_idx,
            "nir_idx": self.nir_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class MNDWI(Operator):
    """Modified Normalized Difference Water Index."""

    def __init__(
        self,
        *,
        green: BandRef | None = None,
        swir: BandRef | None = None,
        green_idx: int | None = 1,
        swir_idx: int | None = 5,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.green_idx = _configured_ref(green, green_idx)
        self.swir_idx = _configured_ref(swir, swir_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = mndwi(
            np.asarray(gt),
            _resolve_band(gt, self.green_idx),
            _resolve_band(gt, self.swir_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "green_idx": self.green_idx,
            "swir_idx": self.swir_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class NDMI(Operator):
    """Normalized Difference Moisture Index."""

    def __init__(
        self,
        *,
        nir: BandRef | None = None,
        swir1: BandRef | None = None,
        nir_idx: int | None = 3,
        swir1_idx: int | None = 5,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.swir1_idx = _configured_ref(swir1, swir1_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = ndmi(
            np.asarray(gt),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.swir1_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "nir_idx": self.nir_idx,
            "swir1_idx": self.swir1_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class NDSI(Operator):
    """Normalized Difference Snow Index."""

    def __init__(
        self,
        *,
        green: BandRef | None = None,
        swir: BandRef | None = None,
        green_idx: int | None = 1,
        swir_idx: int | None = 5,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.green_idx = _configured_ref(green, green_idx)
        self.swir_idx = _configured_ref(swir, swir_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = ndsi(
            np.asarray(gt),
            _resolve_band(gt, self.green_idx),
            _resolve_band(gt, self.swir_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "green_idx": self.green_idx,
            "swir_idx": self.swir_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class NBR2(Operator):
    """Normalized Burn Ratio 2."""

    def __init__(
        self,
        *,
        swir1: BandRef | None = None,
        swir2: BandRef | None = None,
        swir1_idx: int | None = 5,
        swir2_idx: int | None = 6,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.swir1_idx = _configured_ref(swir1, swir1_idx)
        self.swir2_idx = _configured_ref(swir2, swir2_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = nbr2(
            np.asarray(gt),
            _resolve_band(gt, self.swir1_idx),
            _resolve_band(gt, self.swir2_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "swir1_idx": self.swir1_idx,
            "swir2_idx": self.swir2_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class BAIS2(Operator):
    """Burned Area Index for Sentinel-2."""

    def __init__(
        self,
        *,
        red: BandRef | None = None,
        red_edge1: BandRef | None = None,
        red_edge2: BandRef | None = None,
        nir: BandRef | None = None,
        swir2: BandRef | None = None,
        red_idx: int | None = 2,
        red_edge1_idx: int | None = 4,
        red_edge2_idx: int | None = 5,
        nir_idx: int | None = 7,
        swir2_idx: int | None = 9,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.red_idx = _configured_ref(red, red_idx)
        self.red_edge1_idx = _configured_ref(red_edge1, red_edge1_idx)
        self.red_edge2_idx = _configured_ref(red_edge2, red_edge2_idx)
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.swir2_idx = _configured_ref(swir2, swir2_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = bais2(
            np.asarray(gt),
            _resolve_band(gt, self.red_idx),
            _resolve_band(gt, self.red_edge1_idx),
            _resolve_band(gt, self.red_edge2_idx),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.swir2_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "red_idx": self.red_idx,
            "red_edge1_idx": self.red_edge1_idx,
            "red_edge2_idx": self.red_edge2_idx,
            "nir_idx": self.nir_idx,
            "swir2_idx": self.swir2_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class dNBR(Operator):
    """Difference of two NBR rasters: ``pre - post``."""

    def _apply(self, pre: GeoTensor, post: GeoTensor) -> GeoTensor:
        if not _geotensor_grid_matches(pre, post):
            raise ValueError("dNBR inputs must share shape, transform, and CRS.")
        return pre.array_as_geotensor(np.asarray(pre) - np.asarray(post))


class BSI(Operator):
    """Bare Soil Index."""

    def __init__(
        self,
        *,
        blue: BandRef | None = None,
        red: BandRef | None = None,
        nir: BandRef | None = None,
        swir: BandRef | None = None,
        blue_idx: int | None = 0,
        red_idx: int | None = 2,
        nir_idx: int | None = 3,
        swir_idx: int | None = 5,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.blue_idx = _configured_ref(blue, blue_idx)
        self.red_idx = _configured_ref(red, red_idx)
        self.nir_idx = _configured_ref(nir, nir_idx)
        self.swir_idx = _configured_ref(swir, swir_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = bsi(
            np.asarray(gt),
            _resolve_band(gt, self.blue_idx),
            _resolve_band(gt, self.red_idx),
            _resolve_band(gt, self.nir_idx),
            _resolve_band(gt, self.swir_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "blue_idx": self.blue_idx,
            "red_idx": self.red_idx,
            "nir_idx": self.nir_idx,
            "swir_idx": self.swir_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class IronOxide(Operator):
    """Iron oxide ratio."""

    def __init__(
        self,
        *,
        red: BandRef | None = None,
        blue: BandRef | None = None,
        red_idx: int | None = 2,
        blue_idx: int | None = 0,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.red_idx = _configured_ref(red, red_idx)
        self.blue_idx = _configured_ref(blue, blue_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = iron_oxide(
            np.asarray(gt),
            _resolve_band(gt, self.red_idx),
            _resolve_band(gt, self.blue_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "red_idx": self.red_idx,
            "blue_idx": self.blue_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class ClayMinerals(Operator):
    """Clay minerals ratio."""

    def __init__(
        self,
        *,
        swir1: BandRef | None = None,
        swir2: BandRef | None = None,
        swir1_idx: int | None = 5,
        swir2_idx: int | None = 6,
        axis: int = 0,
        eps: float = 1e-10,
    ) -> None:
        self.swir1_idx = _configured_ref(swir1, swir1_idx)
        self.swir2_idx = _configured_ref(swir2, swir2_idx)
        self.axis = axis
        self.eps = eps

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = clay_minerals(
            np.asarray(gt),
            _resolve_band(gt, self.swir1_idx),
            _resolve_band(gt, self.swir2_idx),
            axis=self.axis,
            eps=self.eps,
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {
            "swir1_idx": self.swir1_idx,
            "swir2_idx": self.swir2_idx,
            "axis": self.axis,
            "eps": self.eps,
        }


class CIRI(Operator):
    """Cirrus Reflectance Index."""

    def __init__(
        self,
        *,
        cirrus: BandRef | None = None,
        cirrus_idx: int | None = None,
        axis: int = 0,
    ) -> None:
        self.cirrus_idx = _configured_ref(cirrus, cirrus_idx)
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = ciri(np.asarray(gt), _resolve_band(gt, self.cirrus_idx), axis=self.axis)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"cirrus_idx": self.cirrus_idx, "axis": self.axis}


class AppendIndex(Operator):
    """Run an index operator and concatenate its output back as a new band.

    Most index operators *collapse* the band axis: NDVI of a ``(C, H, W)``
    carrier returns ``(H, W)``. `AppendIndex` runs the wrapped operator,
    expands the result to ``(1, H, W)``, and concatenates it back onto
    the original carrier along the band axis — so the output has shape
    ``(C+1, H, W)`` with the index sitting as the last channel.

    Useful when the index is a feature for a downstream model that
    expects a fixed multi-channel input.

    Args:
        index_op: An index operator (e.g. ``NDVI()``, ``EVI()``). Must
            return a `GeoTensor` whose band-axis is collapsed.
        axis: Position of the band axis. Default ``0``. Must match the
            ``axis`` configured on ``index_op``.

    Examples:
        >>> from geotoolz.indices import AppendIndex, NDVI
        >>> bands_plus_ndvi = AppendIndex(index_op=NDVI(nir_idx=3, red_idx=2))
        >>> stacked = bands_plus_ndvi(reflectance_geotensor)  # (C+1, H, W)
    """

    def __init__(self, *, index_op: Operator, axis: int = 0) -> None:
        self.index_op = index_op
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        index = self.index_op(gt)
        index_arr = np.asarray(index)
        # Expand back to (..., 1, ..., H, W) along the configured axis so
        # concatenation lines up. np.expand_dims handles negative axes
        # correctly.
        index_3d = np.expand_dims(index_arr, axis=self.axis)
        stacked = np.concatenate([np.asarray(gt), index_3d], axis=self.axis)
        return gt.array_as_geotensor(stacked)

    def get_config(self) -> dict[str, Any]:
        # `index_op` is a nested Operator — emit the JSON-safe nested form
        # (matches `Sequential` / `Branch`'s pattern) instead of leaking
        # the raw instance into config.
        return {
            "index_op": {
                "class": type(self.index_op).__name__,
                "config": self.index_op.get_config(),
            },
            "axis": self.axis,
        }
