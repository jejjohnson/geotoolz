"""Tier-B Operators — carrier-aware wrappers around the indices primitives.

Each Operator here:

1. Takes its parameters as **keyword-only** ctor args so YAML / Hydra
   configs are unambiguous.
2. Calls into the matching primitive in ``array.py`` for the math.
3. Rewraps the result to match the input carrier via
   `geoops._src.wrap.wrap_like`: a ``GeoArray`` input comes back as
   a ``GeoArray`` (``transform``, ``crs`` and a
   copy of ``attrs`` propagated, ``fill_value=NaN``, with per-band keys such as
   ``band_names`` dropped because the band axis collapsed); a plain ``np.ndarray``
   comes back as a plain ndarray. Named-band references (``red="B04"``)
   need carrier metadata and therefore require a GeoArray input;
   integer band indices work on both carriers.
4. Returns its config via ``get_config()`` so the Operator round-trips
   through ``hydra_zen.builds``.

The wrap discipline: index primitives collapse the channel axis (``-3``
by default) but preserve the trailing two spatial axes ``(H, W)``. On a
``(T, C, H, W)`` time stack the result is ``(T, 1, H, W)``: one index map
per frame, with a singleton band axis so the time axis is not read as
bands. That matches
``wrap_like``'s contract exactly — it accepts any result whose
last two dims agree with the input's. Carriers' transforms therefore
survive unchanged through every index operator here.

Each operator delegates its maths to the matching Tier-A primitive in
``indices/_src/array.py``; this module only resolves bands, judges
nodata and rewraps the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pipekit import Operator

from geoops._src.bands import (
    BAND_NAME_KEYS,
    BandRef,
    per_band_values,
    resolve_band as _resolve_band,
    strip_band_attrs,
)
from geoops._src.config import nested_config
from geoops._src.geo import grid_matches
from geoops._src.shape import band_dim, dim_axis, keep_band_axis
from geoops._src.valid import (
    carried_fill,
    carrier_fill_value,
    invalid_values,
    restore_fill,
    wrap_filled,
)
from geoops._src.wrap import wrap_like
from geoops.indices._src.array import (
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
    from geoarray import GeoArray


def _wrap_index(
    gt: GeoArray | np.ndarray,
    out: np.ndarray,
    *,
    axis: int,
    bands: tuple[BandRef, ...],
) -> GeoArray | np.ndarray:
    """Rewrap an index result, writing ``NaN`` into nodata pixels.

    A pixel is nodata when *any band the index reads* (``bands``) is
    non-finite or equals ``gt.fill_value`` (see
    :mod:`geoops._src.valid`); its index value is meaningless (e.g. NDVI
    of ``-9999 / -9999``), so it is replaced by ``NaN``. An index is a new
    quantity -- the input's fill (``0`` especially) can be a real index
    value -- so the output always declares ``fill_value=NaN``.
    Bands the index does not read do not affect validity.

    On a ``(T, C, H, W)`` stack the ``(T, H, W)`` result keeps a singleton
    band axis, ``(T, 1, H, W)``, so it stays a time series.
    """
    idx = [_resolve_band(gt, band) for band in bands]
    used = np.take(np.asarray(gt), idx, axis=axis)
    valid = ~invalid_values(used, fill_value=carrier_fill_value(gt)).any(axis=axis)
    out = keep_band_axis(out, gt)
    return wrap_filled(gt, out, fill_value=np.nan, valid=valid)


class NormalizedDifference(Operator):
    """Generic normalized-difference index ``(a - b) / (a + b + eps)``.

    Use this when none of the named indices (NDVI/NDWI/NDBI/NBR/...)
    matches your band pair. Equivalent in math to all of them; the
    named subclasses are pinned-argument convenience wrappers.

    The package's one normalized-difference operator (also the top-level
    ``geoops.NormalizedDifference``). Bands ``a`` / ``b`` are each an
    integer position or a band name resolved against ``gt.attrs`` by the
    package-wide resolver (:func:`geoops._src.bands.resolve_band`). Names
    need a GeoArray carrying band-name attrs; on a plain ``np.ndarray``
    a name raises ``TypeError`` and integer positions work.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``. On a ``(T, C, H, W)`` stack the result is
    ``(T, 1, H, W)``.

    Args:
        a: Index or name of the "high" band (numerator-positive term).
        b: Index or name of the "low" band.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``, the package-wide
            value (see :mod:`geoops.indices._src.array`).

    Examples:
        >>> from geoops.indices import NormalizedDifference
        >>> # A custom water/snow index using SWIR vs Green.
        >>> swsi = NormalizedDifference(a=2, b=10)  # Green=2, SWIR1=10
        >>> result = swsi(reflectance_geoarray)
        >>> # By band name (resolved against gt.attrs["band_names"]).
        >>> ndvi = NormalizedDifference(a="B8", b="B4")(reflectance_geoarray)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        a: BandRef,
        b: BandRef,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.a = a
        self.b = b
        self.dim = band_dim(dim, axis, "NormalizedDifference")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="NormalizedDifference")
        out = normalized_difference(
            np.asarray(gt),
            _resolve_band(gt, self.a),
            _resolve_band(gt, self.b),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.a, self.b))


class NDVI(Operator):
    r"""Normalized Difference Vegetation Index — Rouse et al. 1974.

    .. math::

        \mathrm{NDVI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
                                {\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}} + \varepsilon}

    Returns a single-channel ``GeoArray`` of NDVI values in
    ``[-1, +1]``. See :func:`~geoops.indices._src.array.ndvi` for the
    physics; this Operator just plumbs the primitive through carriers.

    The default ``nir=3, red=2`` matches a 4-band ``BGRN``
    convention. For Sentinel-2 imagery stacked in band-number order use
    ``NDVI(nir=7, red=3)`` (B8 NIR, B4 Red after 0-indexing past
    B1/B2/B3); for Landsat-8 use ``NDVI(nir=4, red=3)``
    (B5 NIR, B4 Red).

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        nir: NIR band: an integer position or a band name. Default ``3``.
        red: Red band: an integer position or a band name. Default ``2``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import NDVI
        >>> ndvi_op = NDVI(nir=7, red=3)         # Sentinel-2 band order
        >>> ndvi_map = ndvi_op(reflectance_geoarray)    # GeoArray (H, W)
        >>>
        >>> # Chain with reflectance conversion:
        >>> import geoops as gz
        >>> pipe = (
        ...     gz.radiometry.DNToReflectance(scale=1e-4)
        ...     | NDVI(nir=7, red=3)
        ... )
        >>> ndvi_map = pipe(dn_geoarray)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        red: BandRef = 2,
        nir: BandRef = 3,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.nir = nir
        self.red = red
        self.dim = band_dim(dim, axis, "NDVI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="NDVI")
        out = ndvi(
            np.asarray(gt),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.red),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.nir, self.red))


class NDWI(Operator):
    r"""Normalized Difference Water Index — McFeeters 1996.

    .. math::

        \mathrm{NDWI} \;=\; \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{NIR}}}
                                 {\rho_{\mathrm{Green}} + \rho_{\mathrm{NIR}} + \varepsilon}

    For *surface water* delineation; not to be confused with Gao's
    leaf-water NDWI (SWIR/NIR). High over open water, low over
    vegetation. See :func:`~geoops.indices._src.array.ndwi_mcfeeters`
    for physics.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        green: Green band: an integer position or a band name. Default ``1``.
        nir: NIR band: an integer position or a band name. Default ``3``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser.

    Examples:
        >>> from geoops.indices import NDWI
        >>> water_idx = NDWI(green=2, nir=7)  # Sentinel-2 B3, B8
        >>> ndwi_map = water_idx(reflectance_geoarray)
        >>> water_mask = np.asarray(ndwi_map) > 0  # rough surface-water mask
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        green: BandRef = 1,
        nir: BandRef = 3,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.green = green
        self.nir = nir
        self.dim = band_dim(dim, axis, "NDWI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="NDWI")
        out = ndwi_mcfeeters(
            np.asarray(gt),
            _resolve_band(gt, self.green),
            _resolve_band(gt, self.nir),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.green, self.nir))


class NDBI(Operator):
    r"""Normalized Difference Built-up Index — Zha et al. 2003.

    .. math::

        \mathrm{NDBI} \;=\; \frac{\rho_{\mathrm{SWIR}} - \rho_{\mathrm{NIR}}}
                                 {\rho_{\mathrm{SWIR}} + \rho_{\mathrm{NIR}} + \varepsilon}

    High over impervious surfaces (concrete, asphalt). Pair with NDVI
    as ``NDBI - NDVI`` to suppress bare-soil confounders. See
    :func:`~geoops.indices._src.array.ndbi` for physics.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        swir: SWIR-1 band: an integer position or a band name. Default ``5``.
        nir: NIR band: an integer position or a band name. Default ``3``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser.

    Examples:
        >>> from geoops.indices import NDBI, NDVI
        >>> ndbi_op = NDBI(swir=10, nir=7)  # Sentinel-2 B11, B8
        >>> built_up_raw = ndbi_op(reflectance_geoarray)
        >>>
        >>> # Zha's recipe: subtract NDVI to suppress soil confounders.
        >>> import numpy as np
        >>> ndvi_op = NDVI(nir=7, red=3)
        >>> built_up = np.asarray(ndbi_op(reflectance_geoarray)) - np.asarray(
        ...     ndvi_op(reflectance_geoarray)
        ... )
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        swir: BandRef = 5,
        nir: BandRef = 3,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.swir = swir
        self.nir = nir
        self.dim = band_dim(dim, axis, "NDBI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="NDBI")
        out = ndbi(
            np.asarray(gt),
            _resolve_band(gt, self.swir),
            _resolve_band(gt, self.nir),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.swir, self.nir))


class NBR(Operator):
    r"""Normalized Burn Ratio — Key & Benson 2006.

    .. math::

        \mathrm{NBR} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{SWIR2}}}
                                {\rho_{\mathrm{NIR}} + \rho_{\mathrm{SWIR2}} + \varepsilon}

    High over healthy vegetation, low / negative over recently-burned
    surfaces. dNBR (``pre - post``) is the standard burn-severity
    quantitative measure. See
    :func:`~geoops.indices._src.array.nbr` for physics.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        nir: NIR band: an integer position or a band name. Default ``3``.
        swir2: SWIR-2 band: an integer position or a band name. Default ``6``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser.

    Examples:
        >>> from geoops.indices import NBR
        >>> import numpy as np
        >>> nbr_op = NBR(nir=7, swir2=11)  # Sentinel-2 B8, B12
        >>>
        >>> # dNBR — burn severity from pre- vs post-fire scenes.
        >>> dnbr = np.asarray(nbr_op(pre_fire_geoarray)) - np.asarray(
        ...     nbr_op(post_fire_geoarray)
        ... )
        >>> # Key & Benson thresholds: > 0.66 high severity, 0.27-0.44 low.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        nir: BandRef = 3,
        swir2: BandRef = 6,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.nir = nir
        self.swir2 = swir2
        self.dim = band_dim(dim, axis, "NBR")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="NBR")
        out = nbr(
            np.asarray(gt),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.swir2),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.nir, self.swir2))


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

    See :func:`~geoops.indices._src.array.savi` for physics.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        nir: NIR band: an integer position or a band name. Default ``3``.
        red: Red band: an integer position or a band name. Default ``2``.
        L: Soil-adjustment factor in ``[0, 1]``. Default ``0.5``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.

    Examples:
        >>> from geoops.indices import SAVI
        >>> # Default L=0.5 for intermediate cover.
        >>> savi_op = SAVI(nir=7, red=3, L=0.5)
        >>> v = savi_op(reflectance_geoarray)
        >>>
        >>> # Sparse drylands -> larger L:
        >>> dryland_savi = SAVI(nir=7, red=3, L=1.0)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        red: BandRef = 2,
        nir: BandRef = 3,
        L: float = 0.5,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.nir = nir
        self.red = red
        self.L = L
        self.dim = band_dim(dim, axis, "SAVI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="SAVI")
        out = savi(
            np.asarray(gt),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.red),
            L=self.L,
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.nir, self.red))


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
    See :func:`~geoops.indices._src.array.evi` for physics.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        nir: NIR band: an integer position or a band name. Default ``3``.
        red: Red band: an integer position or a band name. Default ``2``.
        blue: Blue band: an integer position or a band name. Default ``0``.
        G: Gain factor. Default ``2.5``.
        C1: Red aerosol-resistance coefficient. Default ``6``.
        C2: Blue aerosol-resistance coefficient. Default ``7.5``.
        L: Canopy-background correction. Default ``1``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.

    Examples:
        >>> from geoops.indices import EVI
        >>> # Sentinel-2: B2=Blue, B4=Red, B8=NIR.
        >>> evi_op = EVI(nir=7, red=3, blue=1)
        >>> v = evi_op(boa_reflectance_geoarray)
        >>>
        >>> # EVI assumes surface reflectance (BOA). Apply atmospheric
        >>> # correction first for cross-scene comparability.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        blue: BandRef = 0,
        red: BandRef = 2,
        nir: BandRef = 3,
        G: float = 2.5,
        C1: float = 6.0,
        C2: float = 7.5,
        L: float = 1.0,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.nir = nir
        self.red = red
        self.blue = blue
        self.G = G
        self.C1 = C1
        self.C2 = C2
        self.L = L
        self.dim = band_dim(dim, axis, "EVI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="EVI")
        out = evi(
            np.asarray(gt),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.red),
            _resolve_band(gt, self.blue),
            G=self.G,
            C1=self.C1,
            C2=self.C2,
            L=self.L,
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.nir, self.red, self.blue))


class EVI2(Operator):
    r"""Two-band Enhanced Vegetation Index — Jiang et al. 2008.

    .. math::

        \mathrm{EVI2} \;=\; 2.5 \cdot
            \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
                 {\rho_{\mathrm{NIR}} + 2.4\,\rho_{\mathrm{Red}} + 1}

    Drops EVI's Blue-band aerosol-resistance term so the index can be
    computed when Blue is unavailable or noisy (e.g. AVHRR, sensors
    with poor blue calibration). The fixed coefficients ``2.4`` and
    ``1`` are tuned to track EVI within a few percent on most cover
    types. See :func:`~geoops.indices._src.array.evi2` for physics.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        red: Red band: an integer position or a band name. Default ``2``.
        nir: NIR band: an integer position or a band name. Default ``3``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import EVI2
        >>> evi2_op = EVI2(nir=7, red=3)  # S2 band order
        >>> v = evi2_op(reflectance_geoarray)

    References:
        Jiang, Z., Huete, A. R., Didan, K., & Miura, T. (2008).
        "Development of a two-band enhanced vegetation index without a
        blue band." *Remote Sensing of Environment*, 112(10), 3833–3845.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        red: BandRef = 2,
        nir: BandRef = 3,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.red = red
        self.nir = nir
        self.dim = band_dim(dim, axis, "EVI2")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="EVI2")
        out = evi2(
            np.asarray(gt),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.red),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.nir, self.red))


class ARVI(Operator):
    r"""Atmospherically Resistant Vegetation Index — Kaufman & Tanre 1992.

    .. math::

        \mathrm{ARVI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{rb}}
                                 {\rho_{\mathrm{NIR}} + \rho_{rb} + \varepsilon}
        \quad\text{where}\quad
        \rho_{rb} = \rho_{\mathrm{Red}} - \gamma\,(\rho_{\mathrm{Blue}}
                                                 - \rho_{\mathrm{Red}})

    Extends NDVI with a Blue-band correction (``rb``) that cancels the
    aerosol-driven inflation of the red signal. ``gamma=1`` is the
    standard value derived from MODIS Rayleigh-scattering simulations.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        blue: Blue band: an integer position or a band name. Default ``0``.
        red: Red band: an integer position or a band name. Default ``2``.
        nir: NIR band: an integer position or a band name. Default ``3``.
        gamma: Aerosol-correction strength. Default ``1.0``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import ARVI
        >>> # Sentinel-2: B02=Blue (1), B04=Red (3), B08=NIR (7).
        >>> arvi_op = ARVI(blue=1, red=3, nir=7)
        >>> v = arvi_op(toa_reflectance_geoarray)

    References:
        Kaufman, Y. J., & Tanre, D. (1992). "Atmospherically resistant
        vegetation index (ARVI) for EOS-MODIS." *IEEE Trans. Geosci.
        Remote Sens.*, 30(2), 261–270.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        blue: BandRef = 0,
        red: BandRef = 2,
        nir: BandRef = 3,
        gamma: float = 1.0,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.blue = blue
        self.red = red
        self.nir = nir
        self.gamma = gamma
        self.dim = band_dim(dim, axis, "ARVI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="ARVI")
        out = arvi(
            np.asarray(gt),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.red),
            _resolve_band(gt, self.blue),
            gamma=self.gamma,
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.nir, self.red, self.blue))


class GCI(Operator):
    r"""Green Chlorophyll Index — Gitelson et al. 2003.

    .. math::

        \mathrm{GCI} \;=\; \frac{\rho_{\mathrm{NIR}}}
                                {\rho_{\mathrm{Green}} + \varepsilon} - 1

    Linearly proportional to canopy chlorophyll content over a broader
    dynamic range than NDVI. Saturates much later — useful for dense
    crops and forests where NDVI plateaus.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        green: Green band: an integer position or a band name. Default ``1``.
        nir: NIR band: an integer position or a band name. Default ``3``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import GCI
        >>> gci_op = GCI(green=2, nir=7)  # S2 band order
        >>> v = gci_op(reflectance_geoarray)

    References:
        Gitelson, A. A., Vina, A., Arkebauer, T. J., Rundquist, D. C.,
        Keydan, G., & Leavitt, B. (2003). "Remote estimation of leaf
        area index and green leaf biomass in maize canopies."
        *Geophys. Res. Lett.*, 30(5).
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        green: BandRef = 1,
        nir: BandRef = 3,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.green = green
        self.nir = nir
        self.dim = band_dim(dim, axis, "GCI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="GCI")
        out = gci(
            np.asarray(gt),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.green),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.nir, self.green))


class kNDVI(Operator):
    r"""Kernel NDVI — Camps-Valls et al. 2021.

    .. math::

        \mathrm{kNDVI} \;=\; \tanh\!\bigl(\mathrm{NDVI}^2\bigr)

    A kernel-method-inspired non-linear transform of NDVI that is more
    resilient to saturation, more linearly related to gross primary
    productivity, and more robust to atmospheric noise than NDVI on
    most cover types.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        red: Red band: an integer position or a band name. Default ``2``.
        nir: NIR band: an integer position or a band name. Default ``3``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import kNDVI
        >>> kndvi_op = kNDVI(nir=7, red=3)
        >>> v = kndvi_op(reflectance_geoarray)

    References:
        Camps-Valls, G., Campos-Taberner, M., Moreno-Martinez, A.,
        et al. (2021). "A unified vegetation index for quantifying the
        terrestrial biosphere." *Science Advances*, 7(9), eabc7447.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        red: BandRef = 2,
        nir: BandRef = 3,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.red = red
        self.nir = nir
        self.dim = band_dim(dim, axis, "kNDVI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="kNDVI")
        out = kndvi(
            np.asarray(gt),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.red),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.nir, self.red))


class MNDWI(Operator):
    r"""Modified Normalized Difference Water Index — Xu 2006.

    .. math::

        \mathrm{MNDWI} \;=\; \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{SWIR1}}}
                                  {\rho_{\mathrm{Green}} + \rho_{\mathrm{SWIR1}}
                                   + \varepsilon}

    Replaces NDWI's NIR with SWIR-1, which is far more strongly
    absorbed by water and far more reflective over built-up land. The
    result is sharper water/non-water contrast and reduced confusion
    with urban surfaces. *Same arithmetic form as NDSI* — the
    Green/SWIR1 ratio happens to separate snow from rock just as it
    separates water from soil, so the two indices share a formula but
    are interpreted differently.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        green: Green band: an integer position or a band name. Default ``1``.
        swir: SWIR-1 band: an integer position or a band name. Default ``5``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import MNDWI
        >>> mndwi_op = MNDWI(green=2, swir=10)  # S2 B3, B11
        >>> v = mndwi_op(reflectance_geoarray)

    References:
        Xu, H. (2006). "Modification of normalised difference water
        index (NDWI) to enhance open water features in remotely sensed
        imagery." *International Journal of Remote Sensing*, 27(14),
        3025–3033.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        green: BandRef = 1,
        swir: BandRef = 5,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.green = green
        self.swir = swir
        self.dim = band_dim(dim, axis, "MNDWI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="MNDWI")
        out = mndwi(
            np.asarray(gt),
            _resolve_band(gt, self.green),
            _resolve_band(gt, self.swir),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.green, self.swir))


class NDMI(Operator):
    r"""Normalized Difference Moisture Index — Gao 1996.

    .. math::

        \mathrm{NDMI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{SWIR1}}}
                                 {\rho_{\mathrm{NIR}} + \rho_{\mathrm{SWIR1}}
                                  + \varepsilon}

    Tracks vegetation *liquid-water content*, not surface water — high
    over moist canopies, low over water-stressed or dry vegetation.
    Sometimes also called "Gao's NDWI"; we use ``NDMI`` here to keep
    the McFeeters surface-water ``NDWI`` distinct.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        nir: NIR band: an integer position or a band name. Default ``3``.
        swir1: SWIR-1 band: an integer position or a band name. Default ``5``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import NDMI
        >>> ndmi_op = NDMI(nir=7, swir1=10)  # S2 B8, B11
        >>> v = ndmi_op(reflectance_geoarray)

    References:
        Gao, B. C. (1996). "NDWI - A normalized difference water index
        for remote sensing of vegetation liquid water from space."
        *Remote Sensing of Environment*, 58(3), 257–266.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        nir: BandRef = 3,
        swir1: BandRef = 5,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.nir = nir
        self.swir1 = swir1
        self.dim = band_dim(dim, axis, "NDMI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="NDMI")
        out = ndmi(
            np.asarray(gt),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.swir1),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.nir, self.swir1))


class NDSI(Operator):
    r"""Normalized Difference Snow Index — Hall et al. 1995.

    .. math::

        \mathrm{NDSI} \;=\; \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{SWIR1}}}
                                 {\rho_{\mathrm{Green}} + \rho_{\mathrm{SWIR1}}
                                  + \varepsilon}

    Snow is highly reflective in visible green but strongly absorbs in
    SWIR-1 (~1.6 µm); the Green/SWIR1 ratio therefore lights up snow
    and ice while suppressing clouds, which are bright in both. NDSI >
    0.4 is the MODIS / Sentinel-2 default snow threshold. Shares the
    arithmetic form of MNDWI — same formula, different physics target.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        green: Green band: an integer position or a band name. Default ``1``.
        swir: SWIR-1 band: an integer position or a band name. Default ``5``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import NDSI
        >>> ndsi_op = NDSI(green=2, swir=10)  # S2 B3, B11
        >>> snow_mask = (ndsi_op(reflectance_geoarray) > 0.4)

    References:
        Hall, D. K., Riggs, G. A., & Salomonson, V. V. (1995).
        "Development of methods for mapping global snow cover using
        moderate resolution imaging spectroradiometer data."
        *Remote Sensing of Environment*, 54(2), 127–140.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        green: BandRef = 1,
        swir: BandRef = 5,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.green = green
        self.swir = swir
        self.dim = band_dim(dim, axis, "NDSI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="NDSI")
        out = ndsi(
            np.asarray(gt),
            _resolve_band(gt, self.green),
            _resolve_band(gt, self.swir),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.green, self.swir))


class NBR2(Operator):
    r"""Normalized Burn Ratio 2 — USGS Landsat product.

    .. math::

        \mathrm{NBR2} \;=\; \frac{\rho_{\mathrm{SWIR1}} - \rho_{\mathrm{SWIR2}}}
                                 {\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{SWIR2}}
                                  + \varepsilon}

    Complements NBR by sharpening sensitivity to burned-area moisture
    differences in the SWIR window. Published by USGS alongside NBR as
    part of the Landsat Analysis-Ready burn-severity stack.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        swir1: SWIR-1 band: an integer position or a band name. Default ``5``.
        swir2: SWIR-2 band: an integer position or a band name. Default ``6``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import NBR2
        >>> nbr2_op = NBR2(swir1=10, swir2=11)  # S2 B11, B12
        >>> v = nbr2_op(reflectance_geoarray)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        swir1: BandRef = 5,
        swir2: BandRef = 6,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.swir1 = swir1
        self.swir2 = swir2
        self.dim = band_dim(dim, axis, "NBR2")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="NBR2")
        out = nbr2(
            np.asarray(gt),
            _resolve_band(gt, self.swir1),
            _resolve_band(gt, self.swir2),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.swir1, self.swir2))


class BAIS2(Operator):
    r"""Burned Area Index for Sentinel-2 — Filipponi 2018.

    .. math::

        \mathrm{BAIS2} \;=\;
        \Bigl(1 - \sqrt{\tfrac{\rho_{6}\,\rho_{7}\,\rho_{8\mathrm{A}}}
                              {\rho_{4}}}\Bigr)
        \cdot
        \Bigl(\tfrac{\rho_{12} - \rho_{8\mathrm{A}}}
                    {\sqrt{\rho_{12} + \rho_{8\mathrm{A}}}} + 1\Bigr)

    Filipponi designed BAIS2 to maximise contrast between recently
    burned and unburned land using the rich red-edge sampling
    available on Sentinel-2 (B05/B06/B07) plus the narrow-NIR (B8A)
    and SWIR-2 (B12). The first factor exploits the post-fire collapse
    of red-edge reflectance; the second factor uses the
    SWIR-2/narrowNIR shift characteristic of charred surfaces.

    Defaults assume a Sentinel-2 stack ordered as
    ``B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12`` (10 bands,
    skipping the cirrus/aerosol bands). For different stacking
    conventions, pass explicit indices or named bands.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        red: Red band: an integer position or a band name. Default ``2``.
        red_edge1: first red-edge band: an integer position or a band name. Default ``4``.
        red_edge2: second red-edge band: an integer position or a band name. Default ``5``.
        nir: narrow-NIR band: an integer position or a band name. Default ``7``.
        swir2: SWIR-2 band: an integer position or a band name. Default ``9``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import BAIS2
        >>> bais2_op = BAIS2()  # S2-stack default ordering
        >>> burn_score = bais2_op(reflectance_geoarray)

    References:
        Filipponi, F. (2018). "BAIS2: Burned Area Index for
        Sentinel-2." *Proceedings*, 2(7), 364.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        red: BandRef = 2,
        red_edge1: BandRef = 4,
        red_edge2: BandRef = 5,
        nir: BandRef = 7,
        swir2: BandRef = 9,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.red = red
        self.red_edge1 = red_edge1
        self.red_edge2 = red_edge2
        self.nir = nir
        self.swir2 = swir2
        self.dim = band_dim(dim, axis, "BAIS2")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="BAIS2")
        out = bais2(
            np.asarray(gt),
            _resolve_band(gt, self.red),
            _resolve_band(gt, self.red_edge1),
            _resolve_band(gt, self.red_edge2),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.swir2),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(
            gt,
            out,
            axis=axis,
            bands=(
                self.red,
                self.red_edge1,
                self.red_edge2,
                self.nir,
                self.swir2,
            ),
        )


class dNBR(Operator):
    r"""Differenced Normalized Burn Ratio — Key & Benson 2006.

    .. math::

        \mathrm{dNBR} \;=\; \mathrm{NBR}_{\mathrm{pre}} -
                             \mathrm{NBR}_{\mathrm{post}}

    Standard quantitative burn-severity index. Pre- and post-fire NBR
    rasters must share grid (shape, transform, CRS) — co-register
    upstream (e.g. with `geoops.geom.ReprojectLike`)
    before passing them in. The output `GeoArray` inherits the
    pre-fire raster's spatial metadata. Plain ``np.ndarray`` inputs are
    also accepted (only shapes can be checked then) and return a plain
    ndarray.

    Typical thresholds (Key & Benson 2006): < 0.1 unburned, 0.27–0.44
    low severity, 0.44–0.66 moderate, > 0.66 high severity.

    Nodata pixels (non-finite, or equal to that raster's
    ``fill_value``, in either input) hold ``NaN`` in the output,
    which declares ``fill_value=NaN``.

    Examples:
        >>> from geoops.indices import NBR, dNBR
        >>> nbr_op = NBR(nir=7, swir2=11)
        >>> pre_nbr = nbr_op(pre_fire_geoarray)
        >>> post_nbr = nbr_op(post_fire_geoarray)
        >>> severity = dNBR()(pre_nbr, post_nbr)

    References:
        Key, C. H., & Benson, N. C. (2006). "Landscape assessment
        (LA): sampling and analysis methods." USDA Forest Service
        General Technical Report RMRS-GTR-164-CD.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def _apply(
        self, pre: GeoArray | np.ndarray, post: GeoArray | np.ndarray
    ) -> GeoArray | np.ndarray:
        if not grid_matches(pre, post):
            raise ValueError("dNBR inputs must share shape, transform, and CRS.")
        out = np.asarray(pre) - np.asarray(post)
        valid = ~(invalid_values(pre) | invalid_values(post))
        out = restore_fill(out.astype(np.result_type(out, np.float32)), valid, np.nan)
        return wrap_like(pre, out, fill_value=np.nan)


class BSI(Operator):
    r"""Bare Soil Index — Rikimaru et al. 2002.

    .. math::

        \mathrm{BSI} \;=\;
        \frac{(\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{Red}}) -
              (\rho_{\mathrm{NIR}}  + \rho_{\mathrm{Blue}})}
             {(\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{Red}}) +
              (\rho_{\mathrm{NIR}}  + \rho_{\mathrm{Blue}}) + \varepsilon}

    Highlights exposed soil by combining the soil-bright (SWIR1+Red)
    and soil-dark (NIR+Blue) shoulders. Positive over dry soil, near
    zero over mixed vegetation/soil, negative over dense canopy. The
    Rikimaru variant is the most widely cited; other "BSI" variants
    in the literature exist — pick deliberately if comparing studies.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        blue: Blue band: an integer position or a band name. Default ``0``.
        red: Red band: an integer position or a band name. Default ``2``.
        nir: NIR band: an integer position or a band name. Default ``3``.
        swir: SWIR-1 band: an integer position or a band name. Default ``5``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import BSI
        >>> # Sentinel-2: B2=Blue, B4=Red, B8=NIR, B11=SWIR1.
        >>> bsi_op = BSI(blue=1, red=3, nir=7, swir=10)
        >>> v = bsi_op(reflectance_geoarray)

    References:
        Rikimaru, A., Roy, P. S., & Miyatake, S. (2002). "Tropical
        forest cover density mapping." *Tropical Ecology*, 43(1),
        39–47.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        blue: BandRef = 0,
        red: BandRef = 2,
        nir: BandRef = 3,
        swir: BandRef = 5,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.blue = blue
        self.red = red
        self.nir = nir
        self.swir = swir
        self.dim = band_dim(dim, axis, "BSI")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="BSI")
        out = bsi(
            np.asarray(gt),
            _resolve_band(gt, self.blue),
            _resolve_band(gt, self.red),
            _resolve_band(gt, self.nir),
            _resolve_band(gt, self.swir),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(
            gt,
            out,
            axis=axis,
            bands=(self.blue, self.red, self.nir, self.swir),
        )


class IronOxide(Operator):
    r"""Iron Oxide ratio — Segal 1982 / Sabins 1999.

    .. math::

        \mathrm{IronOxide} \;=\; \frac{\rho_{\mathrm{Red}}}
                                       {\rho_{\mathrm{Blue}} + \varepsilon}

    A simple ferric-iron mineral discriminator: hematite, goethite,
    and other Fe(III) oxides absorb strongly in the blue and reflect
    in the red, giving Red/Blue ratios well above unity over
    iron-oxide-rich soils, weathered surfaces, and lateritic crusts.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        red: Red band: an integer position or a band name. Default ``2``.
        blue: Blue band: an integer position or a band name. Default ``0``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import IronOxide
        >>> # Sentinel-2: B2=Blue (1), B4=Red (3).
        >>> iron_op = IronOxide(red=3, blue=1)
        >>> v = iron_op(reflectance_geoarray)

    References:
        Sabins, F. F. (1999). "Remote sensing for mineral
        exploration." *Ore Geology Reviews*, 14(3-4), 157–183.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        red: BandRef = 2,
        blue: BandRef = 0,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.red = red
        self.blue = blue
        self.dim = band_dim(dim, axis, "IronOxide")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="IronOxide")
        out = iron_oxide(
            np.asarray(gt),
            _resolve_band(gt, self.red),
            _resolve_band(gt, self.blue),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.red, self.blue))


class ClayMinerals(Operator):
    r"""Clay Minerals ratio — Sabins 1999 / Crowley et al. 1989.

    .. math::

        \mathrm{ClayMinerals} \;=\; \frac{\rho_{\mathrm{SWIR1}}}
                                          {\rho_{\mathrm{SWIR2}} + \varepsilon}

    OH-bearing minerals (kaolinite, montmorillonite, illite, alunite)
    have a diagnostic 2.2 µm absorption feature that sits inside
    SWIR-2 while SWIR-1 (~1.6 µm) lies in a relative reflectance
    high. Their SWIR1/SWIR2 ratio is therefore well above unity over
    clay-rich exposures and near unity elsewhere.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        swir1: SWIR-1 band: an integer position or a band name. Default ``5``.
        swir2: SWIR-2 band: an integer position or a band name. Default ``6``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.
        eps: Denominator stabiliser. Default ``1e-10``.

    Examples:
        >>> from geoops.indices import ClayMinerals
        >>> clay_op = ClayMinerals(swir1=10, swir2=11)  # S2 B11/B12
        >>> v = clay_op(reflectance_geoarray)

    References:
        Crowley, J. K., Brickey, D. W., & Rowan, L. C. (1989).
        "Airborne imaging spectrometer data of the Ruby Mountains,
        Montana: mineral discrimination using relative absorption
        band-depth images." *Remote Sensing of Environment*, 29(2),
        121–134.
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        swir1: BandRef = 5,
        swir2: BandRef = 6,
        dim: str = "band",
        eps: float = 1e-10,
        axis: int | None = None,
    ) -> None:
        self.swir1 = swir1
        self.swir2 = swir2
        self.dim = band_dim(dim, axis, "ClayMinerals")
        self.eps = eps

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="ClayMinerals")
        out = clay_minerals(
            np.asarray(gt),
            _resolve_band(gt, self.swir1),
            _resolve_band(gt, self.swir2),
            axis=axis,
            eps=self.eps,
        )
        return _wrap_index(gt, out, axis=axis, bands=(self.swir1, self.swir2))


class CIRI(Operator):
    r"""Cirrus Reflectance Index — Sentinel-2 B10 passthrough.

    .. math::

        \mathrm{CIRI} \;=\; \rho_{\mathrm{cirrus}}

    The Sentinel-2 B10 cirrus channel (1.36–1.39 µm) sits inside a
    strong water-vapour absorption window: at sea level virtually no
    surface signal reaches the sensor, so any non-trivial reflectance
    is high-altitude (cirrus) cloud. A simple threshold on B10
    therefore yields an effective cirrus mask — popular for the
    `s2cloudless` and Fmask cirrus screens.

    The default ``cirrus=9`` matches Sentinel-2 B10 in a stack
    ordered by band number as
    ``B01, B02, B03, B04, B05, B06, B07, B08, B8A, B10, B11, B12``.

    Nodata pixels (any band it reads non-finite or equal to the input's
    ``fill_value``) hold ``NaN`` in the output, which declares
    ``fill_value=NaN``.

    Args:
        cirrus: cirrus band: an integer position or a band name. Default ``9``.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.

    Examples:
        >>> from geoops.indices import CIRI
        >>> ciri_op = CIRI()  # defaults to S2 B10
        >>> cirrus_score = ciri_op(reflectance_geoarray)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self,
        *,
        cirrus: BandRef = 9,
        dim: str = "band",
        axis: int | None = None,
    ) -> None:
        self.cirrus = cirrus
        self.dim = band_dim(dim, axis, "CIRI")

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="CIRI")
        out = ciri(np.asarray(gt), _resolve_band(gt, self.cirrus), axis=axis)
        return _wrap_index(gt, out, axis=axis, bands=(self.cirrus,))


class AppendIndex(Operator):
    """Run an index operator and concatenate its output back as a new band.

    Most index operators *collapse* the band axis: NDVI of a ``(C, H, W)``
    carrier returns ``(H, W)``. `AppendIndex` runs the wrapped operator,
    expands the result to ``(1, H, W)``, and concatenates it back onto
    the original carrier along the band axis — so the output has shape
    ``(C+1, H, W)`` with the index sitting as the last channel.

    Nodata pixels come out of ``index_op`` holding the fill value; the
    original bands are passed through unchanged.

    Useful when the index is a feature for a downstream model that
    expects a fixed multi-channel input.

    Args:
        index_op: An index operator (e.g. ``NDVI()``, ``EVI()``). Must
            return a `GeoArray` whose band-axis is collapsed.
        dim: Name of the band dim (default ``"band"``); its position is read
            from the carrier's dims.
        axis: Deprecated: the band axis position. ``-3`` maps to
            ``dim="band"``; removed in the next minor release.

    Examples:
        >>> from geoops.indices import AppendIndex, NDVI
        >>> bands_plus_ndvi = AppendIndex(index_op=NDVI(nir=3, red=2))
        >>> stacked = bands_plus_ndvi(reflectance_geoarray)  # (C+1, H, W)
    """

    backends: ClassVar[tuple[str, ...]] = ("numpy",)

    def __init__(
        self, *, index_op: Operator, dim: str = "band", axis: int | None = None
    ) -> None:
        self.index_op = index_op
        self.dim = band_dim(dim, axis, "AppendIndex")

    def _apply(self, gt: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        axis = dim_axis(gt, self.dim, name="AppendIndex")
        index = self.index_op(gt)
        index_arr = np.asarray(index)
        # Expand back to (..., 1, ..., H, W) along the configured axis so
        # concatenation lines up. np.expand_dims handles negative axes
        # correctly.
        arr = np.asarray(gt)
        index_3d = (
            index_arr
            if index_arr.ndim == arr.ndim
            else np.expand_dims(index_arr, axis=axis)
        )
        stacked = np.concatenate([arr, index_3d], axis=axis)
        # Band-name lists gain the index operator's name; per-band values
        # with no meaning for an index (wavelengths) are dropped.
        src_attrs = getattr(gt, "attrs", None)
        attrs = strip_band_attrs(src_attrs)
        name = type(self.index_op).__name__
        for key in BAND_NAME_KEYS:
            names = per_band_values(src_attrs, key, arr.shape[axis])
            if names is not None:
                attrs[key] = [*names, name]
        # The stacked bands keep the input's values, so the fill follows
        # the input's (NaN once integer bands are promoted by the float index).
        return wrap_filled(
            gt,
            stacked,
            fill_value=carried_fill(gt, stacked.dtype),
            attrs=attrs,
        )

    def get_config(self) -> dict[str, Any]:
        return {"index_op": nested_config(self.index_op), "dim": self.dim}
