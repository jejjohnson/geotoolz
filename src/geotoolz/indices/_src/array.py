"""Tier-A primitives — pure-numpy spectral-index math.

Every function here is a one-liner over ``numpy.ndarray`` that takes raw
band indices and returns the index value. No `GeoTensor` knowledge, no
metadata, no operator state — the carrier-aware wrappers in
`geotoolz.indices._src.operators` provide all that.

The convention for the spatial axes (``H, W``) is the trailing two axes
of the input. The *band* axis is configurable (``axis`` kwarg, default
``0``) so the same primitive works on ``(C, H, W)`` rasters and
``(T, C, H, W)`` time-cubes.

All indices return a single-channel ndarray with the band axis
*collapsed* — for example, ``ndvi`` on a ``(C, H, W)`` input returns
``(H, W)``. Use `geotoolz.indices.AppendIndex` if you'd rather keep the
output as a new channel of the original carrier.

Numerical safety: every band-ratio primitive accepts an ``eps`` term
added to the denominator. Default ``1e-10`` matches the order of
magnitude of TOA reflectance noise (~1e-4) divided by a typical
denominator (~1) — small enough not to bias the index, large enough to
shadow division by zero for genuinely-zero pixels (water in NIR over a
black background, etc.).
"""

from __future__ import annotations

import numpy as np


def normalized_difference(
    arr: np.ndarray,
    a_idx: int,
    b_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Generic normalized-difference band ratio.

    Computes

    .. math::

        \mathrm{ND}(a, b) \;=\; \frac{a - b}{a + b + \varepsilon}

    where :math:`a` and :math:`b` are slices of ``arr`` along ``axis``.
    The denominator's :math:`\varepsilon` shadows division by zero when
    both bands are exactly zero (no-data fill, sensor saturation
    bottoming out, etc.) and biases the result toward zero in that
    degenerate case rather than NaN.

    The normalized-difference family (NDVI, NDWI, NDBI, NBR, …) shares
    this exact form; concrete indices are named convenience aliases that
    pin which two bands play the role of :math:`a` and :math:`b`.

    Args:
        arr: Input ndarray, with the band axis at position ``axis``.
            Any leading batch / time dimensions are preserved untouched.
        a_idx: Index of the "high" band (numerator-positive term). For
            NDVI this is NIR; for NDWI (McFeeters) this is Green; for
            NDBI this is SWIR.
        b_idx: Index of the "low" band. NDVI: Red; NDWI: NIR; NDBI: NIR.
        axis: Position of the band axis. Default ``0`` (band-first
            convention; matches ``rasterio.read()`` output).
        eps: Small constant added to the denominator. Default ``1e-10``.
            Pass ``0.0`` if you'd rather see ``inf``/``nan`` on zero
            pixels (useful when debugging masking issues).

    Returns:
        ndarray with the band axis collapsed. Output range is
        :math:`(-1, +1)` for any non-negative input; values outside that
        range only occur when one of ``a`` or ``b`` is negative
        (atmospheric-correction artifacts).

    References:
        Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W.
        (1974). "Monitoring vegetation systems in the Great Plains with
        ERTS." *Third ERTS Symposium*, NASA SP-351, 309–317. (The
        original normalized-difference vegetation index paper — the
        form generalises to any two bands.)
    """
    a = np.take(arr, a_idx, axis=axis)
    b = np.take(arr, b_idx, axis=axis)
    return (a - b) / (a + b + eps)


def ndvi(
    arr: np.ndarray,
    nir_idx: int,
    red_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Normalized Difference Vegetation Index (Rouse et al. 1974).

    .. math::

        \mathrm{NDVI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
                                {\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}} + \varepsilon}

    Healthy vegetation reflects strongly in the near-infrared (the
    "red-edge plateau" caused by spongy mesophyll cell-wall scattering)
    and absorbs strongly in the red (chlorophyll-a/b absorption peak at
    ~680 nm). The ratio amplifies that contrast and normalises out
    illumination differences — a pixel with twice the brightness but
    the same NIR/Red ratio has the same NDVI.

    Typical values: bare soil 0.0–0.2, sparse vegetation 0.2–0.5, dense
    canopy 0.6–0.9, water and clouds < 0 (clouds reflect roughly equally
    in both bands, water absorbs NIR strongly).

    Saturates above LAI ≈ 3 — for high-biomass canopies prefer EVI or
    SAVI (see :func:`evi`, :func:`savi`).

    Args:
        arr: Input ndarray. Should be reflectance (0–1) for the formula
            to be physically meaningful; DN values work but the result
            is sensor-specific and shouldn't be compared across scenes.
        nir_idx: Index of the NIR band (Sentinel-2 B8, Landsat-8 B5).
        red_idx: Index of the Red band (Sentinel-2 B4, Landsat-8 B4).
        axis: Position of the band axis. Default ``0``.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        ndarray with the band axis collapsed; values in ``[-1, +1]``.
    """
    return normalized_difference(arr, nir_idx, red_idx, axis=axis, eps=eps)


def ndwi_mcfeeters(
    arr: np.ndarray,
    green_idx: int,
    nir_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Normalized Difference Water Index (McFeeters 1996).

    .. math::

        \mathrm{NDWI} \;=\; \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{NIR}}}
                                 {\rho_{\mathrm{Green}} + \rho_{\mathrm{NIR}} + \varepsilon}

    Open water reflects modestly in green (~550 nm) and absorbs nearly
    everything in the NIR. Healthy vegetation does the opposite. The
    index is therefore strongly positive over water and negative over
    vegetation / soil.

    *McFeeters' NDWI is for surface-water delineation*, not leaf-water
    content. The leaf-water-content index (sometimes also called NDWI)
    uses SWIR and NIR — Gao (1996). Pick deliberately.

    Args:
        arr: Input ndarray.
        green_idx: Index of the Green band (S2 B3, L8 B3).
        nir_idx: Index of the NIR band (S2 B8, L8 B5).
        axis: Band axis. Default ``0``.
        eps: Denominator stabiliser.

    Returns:
        ndarray with band axis collapsed; values in ``[-1, +1]``,
        typically > 0 over water, < 0 elsewhere.

    References:
        McFeeters, S. K. (1996). "The use of the Normalized Difference
        Water Index (NDWI) in the delineation of open water features."
        *International Journal of Remote Sensing*, 17(7), 1425–1432.
    """
    return normalized_difference(arr, green_idx, nir_idx, axis=axis, eps=eps)


def ndbi(
    arr: np.ndarray,
    swir_idx: int,
    nir_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Normalized Difference Built-up Index (Zha et al. 2003).

    .. math::

        \mathrm{NDBI} \;=\; \frac{\rho_{\mathrm{SWIR}} - \rho_{\mathrm{NIR}}}
                                 {\rho_{\mathrm{SWIR}} + \rho_{\mathrm{NIR}} + \varepsilon}

    Built-up surfaces (concrete, asphalt, rooftops) tend to be brighter
    in SWIR (~1600 nm) than in NIR, while vegetation is the inverse
    (see NDVI). The ratio is therefore positive over urban material and
    negative over vegetation; bare soil sits near zero.

    Often paired with NDVI in urban-mapping pipelines as
    ``built_up = NDBI - NDVI`` (Zha's original recipe) to suppress
    soil's confounder.

    Args:
        arr: Input ndarray.
        swir_idx: Index of the SWIR-1 band (S2 B11, L8 B6).
        nir_idx: Index of the NIR band (S2 B8, L8 B5).
        axis: Band axis. Default ``0``.
        eps: Denominator stabiliser.

    Returns:
        ndarray with band axis collapsed.

    References:
        Zha, Y., Gao, J., & Ni, S. (2003). "Use of normalized difference
        built-up index in automatically mapping urban areas from TM
        imagery." *Int. J. Remote Sens.*, 24(3), 583–594.
    """
    return normalized_difference(arr, swir_idx, nir_idx, axis=axis, eps=eps)


def nbr(
    arr: np.ndarray,
    nir_idx: int,
    swir2_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Normalized Burn Ratio (Key & Benson 2006).

    .. math::

        \mathrm{NBR} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{SWIR2}}}
                                {\rho_{\mathrm{NIR}} + \rho_{\mathrm{SWIR2}} + \varepsilon}

    Live vegetation reflects strongly in NIR and weakly in SWIR-2
    (~2200 nm); fire-affected surfaces have the opposite signature
    (collapsed cell structure → lower NIR; exposed soil/char → higher
    SWIR). Pre/post-fire differences (``dNBR = NBR_pre − NBR_post``)
    are the standard quantitative measure of burn severity in the
    USGS / EROS Landsat-based burn-severity products.

    Args:
        arr: Input ndarray.
        nir_idx: Index of the NIR band (S2 B8, L8 B5).
        swir2_idx: Index of the SWIR-2 band (S2 B12, L8 B7).
        axis: Band axis. Default ``0``.
        eps: Denominator stabiliser.

    Returns:
        ndarray with band axis collapsed; high over healthy vegetation,
        low (or negative) over recently-burned surfaces.

    References:
        Key, C. H., & Benson, N. C. (2006). "Landscape assessment (LA):
        sampling and analysis methods." USDA Forest Service General
        Technical Report RMRS-GTR-164-CD, LA-1 to LA-55.
    """
    return normalized_difference(arr, nir_idx, swir2_idx, axis=axis, eps=eps)


def savi(
    arr: np.ndarray,
    nir_idx: int,
    red_idx: int,
    *,
    L: float = 0.5,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Soil-Adjusted Vegetation Index (Huete 1988).

    .. math::

        \mathrm{SAVI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
                                 {\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}} + L}
                            \,(1 + L)

    NDVI overweights soil background brightness in sparsely-vegetated
    scenes (drylands, agricultural fields between rows). SAVI's
    ``L``-parameter shifts the "soil line" in NIR/Red space; the
    ``(1 + L)`` factor restores the index range to ``[-1, +1]``.

    ``L = 0`` recovers NDVI exactly. ``L = 0.5`` (Huete's default) is
    appropriate for intermediate vegetation cover; ``L = 1`` for very
    sparse cover; ``L = 0.25`` for dense cover. For unknown cover, the
    self-adjusting variant MSAVI2 (deferred to v0.2) replaces the
    constant with a closed-form scene-adaptive term.

    Note this primitive does **not** add ``eps`` to the denominator —
    when ``L > 0`` the denominator is already strictly positive for
    non-negative reflectance, so the stabiliser is unnecessary. Pass
    reflectance, not DN.

    Args:
        arr: Input ndarray of reflectance.
        nir_idx: NIR band index.
        red_idx: Red band index.
        L: Soil-adjustment factor in ``[0, 1]``. Default ``0.5``.
        axis: Band axis. Default ``0``.

    Returns:
        ndarray with band axis collapsed.

    References:
        Huete, A. R. (1988). "A soil-adjusted vegetation index (SAVI)."
        *Remote Sensing of Environment*, 25(3), 295–309.
    """
    nir = np.take(arr, nir_idx, axis=axis)
    red = np.take(arr, red_idx, axis=axis)
    return (nir - red) / (nir + red + L + eps) * (1.0 + L)


def evi(
    arr: np.ndarray,
    nir_idx: int,
    red_idx: int,
    blue_idx: int,
    *,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Enhanced Vegetation Index (Huete et al. 2002).

    .. math::

        \mathrm{EVI} \;=\; G \cdot \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
            {\rho_{\mathrm{NIR}} + C_1 \rho_{\mathrm{Red}} - C_2 \rho_{\mathrm{Blue}} + L}

    EVI was designed for MODIS to address two NDVI limitations: NDVI
    saturates over dense canopies, and atmospheric aerosols inflate
    red-band reflectance. The Blue-band correction term
    (:math:`-C_2 \rho_{\mathrm{Blue}}`) cancels Rayleigh-scattering
    bias, and the L-term decouples canopy background. The result is
    far more sensitive in high-LAI regimes (rainforest, plantations).

    MODIS / Landsat / Sentinel-2 defaults: ``G=2.5, C1=6, C2=7.5, L=1``.

    Args:
        arr: Input ndarray of TOA or surface reflectance. EVI is
            sensitive to atmospheric correction — apply BOA first for
            cross-scene comparability.
        nir_idx: NIR band index.
        red_idx: Red band index.
        blue_idx: Blue band index.
        G: Gain factor. Default ``2.5`` (MODIS convention).
        C1: Red aerosol-resistance coefficient. Default ``6``.
        C2: Blue aerosol-resistance coefficient. Default ``7.5``.
        L: Canopy-background correction. Default ``1``.
        axis: Band axis. Default ``0``.

    Returns:
        ndarray with band axis collapsed; values in ``[-1, +1]`` for
        physically-reasonable reflectance inputs.

    References:
        Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., &
        Ferreira, L. G. (2002). "Overview of the radiometric and
        biophysical performance of the MODIS vegetation indices."
        *Remote Sensing of Environment*, 83(1-2), 195–213.
    """
    nir = np.take(arr, nir_idx, axis=axis)
    red = np.take(arr, red_idx, axis=axis)
    blue = np.take(arr, blue_idx, axis=axis)
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + eps)


def evi2(
    arr: np.ndarray,
    nir_idx: int,
    red_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Two-band Enhanced Vegetation Index (Jiang et al. 2008).

    .. math::

        \mathrm{EVI2} \;=\; 2.5 \cdot
            \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
                 {\rho_{\mathrm{NIR}} + 2.4\,\rho_{\mathrm{Red}} + 1 + \varepsilon}

    A blue-band-free approximation to EVI tuned to track it within a
    few percent over most cover types.

    References:
        Jiang, Z., Huete, A. R., Didan, K., & Miura, T. (2008).
        *Remote Sensing of Environment*, 112(10), 3833–3845.
    """
    nir = np.take(arr, nir_idx, axis=axis)
    red = np.take(arr, red_idx, axis=axis)
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + eps)


def arvi(
    arr: np.ndarray,
    nir_idx: int,
    red_idx: int,
    blue_idx: int,
    *,
    gamma: float = 1.0,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Atmospherically Resistant Vegetation Index (Kaufman & Tanre 1992).

    .. math::

        \rho_{rb} = \rho_{\mathrm{Red}}
                  - \gamma\,(\rho_{\mathrm{Blue}} - \rho_{\mathrm{Red}}),
        \quad
        \mathrm{ARVI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{rb}}
                                 {\rho_{\mathrm{NIR}} + \rho_{rb} + \varepsilon}

    The Blue-band correction cancels aerosol-driven red inflation.
    Pass ``gamma=1`` for the original MODIS-derived value.
    """
    nir = np.take(arr, nir_idx, axis=axis)
    red = np.take(arr, red_idx, axis=axis)
    blue = np.take(arr, blue_idx, axis=axis)
    rb = red - gamma * (blue - red)
    return (nir - rb) / (nir + rb + eps)


def gci(
    arr: np.ndarray,
    nir_idx: int,
    green_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Green Chlorophyll Index (Gitelson et al. 2003).

    .. math::

        \mathrm{GCI} \;=\; \frac{\rho_{\mathrm{NIR}}}
                                {\rho_{\mathrm{Green}} + \varepsilon} - 1

    Roughly linear in canopy chlorophyll content; saturates much
    later than NDVI.
    """
    nir = np.take(arr, nir_idx, axis=axis)
    green = np.take(arr, green_idx, axis=axis)
    return nir / (green + eps) - 1.0


def kndvi(
    arr: np.ndarray,
    nir_idx: int,
    red_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Kernel NDVI (Camps-Valls et al. 2021).

    .. math::

        \mathrm{kNDVI} \;=\; \tanh\!\bigl(\mathrm{NDVI}^2\bigr)

    Non-linear transform of NDVI with reduced saturation and a
    closer-to-linear relationship to GPP.
    """
    return np.tanh(ndvi(arr, nir_idx, red_idx, axis=axis, eps=eps) ** 2)


def mndwi(
    arr: np.ndarray,
    green_idx: int,
    swir_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Modified Normalized Difference Water Index (Xu 2006).

    .. math::

        \mathrm{MNDWI} \;=\; \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{SWIR1}}}
                                  {\rho_{\mathrm{Green}} + \rho_{\mathrm{SWIR1}}
                                   + \varepsilon}

    Sharper water/non-water separation than McFeeters NDWI; reduces
    urban-surface confusion.
    """
    return normalized_difference(arr, green_idx, swir_idx, axis=axis, eps=eps)


def ndmi(
    arr: np.ndarray,
    nir_idx: int,
    swir1_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Normalized Difference Moisture Index (Gao 1996).

    .. math::

        \mathrm{NDMI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{SWIR1}}}
                                 {\rho_{\mathrm{NIR}} + \rho_{\mathrm{SWIR1}}
                                  + \varepsilon}

    Tracks vegetation *liquid-water content* (not surface water).
    """
    return normalized_difference(arr, nir_idx, swir1_idx, axis=axis, eps=eps)


def ndsi(
    arr: np.ndarray,
    green_idx: int,
    swir_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Normalized Difference Snow Index (Hall et al. 1995).

    .. math::

        \mathrm{NDSI} \;=\; \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{SWIR1}}}
                                 {\rho_{\mathrm{Green}} + \rho_{\mathrm{SWIR1}}
                                  + \varepsilon}

    Snow / ice mapping (NDSI > 0.4 is the standard MODIS threshold).
    Shares its arithmetic form with MNDWI; the two indices are
    interpreted differently.
    """
    return normalized_difference(arr, green_idx, swir_idx, axis=axis, eps=eps)


def nbr2(
    arr: np.ndarray,
    swir1_idx: int,
    swir2_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Normalized Burn Ratio 2 (USGS Landsat product).

    .. math::

        \mathrm{NBR2} \;=\; \frac{\rho_{\mathrm{SWIR1}} - \rho_{\mathrm{SWIR2}}}
                                 {\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{SWIR2}}
                                  + \varepsilon}
    """
    return normalized_difference(arr, swir1_idx, swir2_idx, axis=axis, eps=eps)


def bais2(
    arr: np.ndarray,
    red_idx: int,
    red_edge1_idx: int,
    red_edge2_idx: int,
    nir_idx: int,
    swir2_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Burned Area Index for Sentinel-2 (Filipponi 2018).

    .. math::

        \mathrm{BAIS2} \;=\;
        \Bigl(1 - \sqrt{\tfrac{\rho_{6}\,\rho_{7}\,\rho_{8\mathrm{A}}}
                              {\rho_{4} + \varepsilon}}\Bigr)
        \cdot
        \Bigl(\tfrac{\rho_{12} - \rho_{8\mathrm{A}}}
                    {\sqrt{\rho_{12} + \rho_{8\mathrm{A}} + \varepsilon}} + 1\Bigr)

    Args:
        arr: Reflectance ndarray; non-negative everywhere or the
            square roots will produce NaN.
        red_idx: Red (B04) index.
        red_edge1_idx: First red-edge (B06) index.
        red_edge2_idx: Second red-edge (B07) index.
        nir_idx: Narrow-NIR (B8A) index.
        swir2_idx: SWIR-2 (B12) index.

    References:
        Filipponi, F. (2018). *Proceedings*, 2(7), 364.
    """
    red = np.take(arr, red_idx, axis=axis)
    red_edge1 = np.take(arr, red_edge1_idx, axis=axis)
    red_edge2 = np.take(arr, red_edge2_idx, axis=axis)
    nir = np.take(arr, nir_idx, axis=axis)
    swir2 = np.take(arr, swir2_idx, axis=axis)
    return (1.0 - np.sqrt((red_edge1 * red_edge2 * nir) / (red + eps))) * (
        (swir2 - nir) / np.sqrt(swir2 + nir + eps) + 1.0
    )


def bsi(
    arr: np.ndarray,
    blue_idx: int,
    red_idx: int,
    nir_idx: int,
    swir_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Bare Soil Index (Rikimaru et al. 2002).

    .. math::

        \mathrm{BSI} \;=\;
        \frac{(\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{Red}}) -
              (\rho_{\mathrm{NIR}}  + \rho_{\mathrm{Blue}})}
             {(\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{Red}}) +
              (\rho_{\mathrm{NIR}}  + \rho_{\mathrm{Blue}}) + \varepsilon}
    """
    blue = np.take(arr, blue_idx, axis=axis)
    red = np.take(arr, red_idx, axis=axis)
    nir = np.take(arr, nir_idx, axis=axis)
    swir = np.take(arr, swir_idx, axis=axis)
    return ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + eps)


def iron_oxide(
    arr: np.ndarray,
    red_idx: int,
    blue_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Iron Oxide ratio (Sabins 1999).

    .. math::

        \mathrm{IronOxide} \;=\; \frac{\rho_{\mathrm{Red}}}
                                       {\rho_{\mathrm{Blue}} + \varepsilon}

    Hematite/goethite and related Fe(III) oxides absorb blue and
    reflect red.
    """
    red = np.take(arr, red_idx, axis=axis)
    blue = np.take(arr, blue_idx, axis=axis)
    return red / (blue + eps)


def clay_minerals(
    arr: np.ndarray,
    swir1_idx: int,
    swir2_idx: int,
    *,
    axis: int = 0,
    eps: float = 1e-10,
) -> np.ndarray:
    r"""Clay Minerals ratio (Sabins 1999; Crowley et al. 1989).

    .. math::

        \mathrm{ClayMinerals} \;=\; \frac{\rho_{\mathrm{SWIR1}}}
                                          {\rho_{\mathrm{SWIR2}} + \varepsilon}

    OH-bearing minerals (kaolinite, illite, montmorillonite, alunite)
    have a diagnostic 2.2 µm absorption inside SWIR-2.
    """
    swir1 = np.take(arr, swir1_idx, axis=axis)
    swir2 = np.take(arr, swir2_idx, axis=axis)
    return swir1 / (swir2 + eps)


def ciri(
    arr: np.ndarray,
    cirrus_idx: int,
    *,
    axis: int = 0,
) -> np.ndarray:
    r"""Cirrus Reflectance Index (Sentinel-2 B10 passthrough).

    .. math::

        \mathrm{CIRI} \;=\; \rho_{\mathrm{cirrus}}

    The 1.36–1.39 µm cirrus channel is opaque to surface signal due
    to water-vapour absorption; any non-trivial reflectance comes
    from high-altitude cloud.
    """
    return np.take(arr, cirrus_idx, axis=axis)
