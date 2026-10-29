"""Tier-A kernels: element-wise spectral-index maths.

Every function here takes one array per band it reads, all of one shape,
and returns the index on that shape (signature ``"(),...,()->()"``). There
is no band axis, no `GeoArray` and no metadata: the operators in
`geoops.indices._src.operators` select the bands by dim name, judge
nodata and rewrap the result, through `geoops._src.kernels.band_kernel`
and pipekit-array's `KernelOp`.

The kernels are written against the array namespace of their input, so
the same function runs on NumPy, on JAX (under ``jax.jit`` too) and on
the NumPy blocks of a dask-backed ``xr.DataArray``.

Numerical safety: every band-ratio kernel accepts an ``eps`` term
added to the denominator. Default ``1e-10`` matches the order of
magnitude of TOA reflectance noise (~1e-4) divided by a typical
denominator (~1) — small enough not to bias the index, large enough to
shadow division by zero for genuinely-zero pixels (water in NIR over a
black background, etc.). It is the package-wide default: the
spectral-family ratio (:func:`geoops.spectral.band_ratio` /
:class:`geoops.spectral.BandRatio`) uses the same value, so the same
maths gives the same numbers in every family.

Integer inputs: every kernel that does arithmetic promotes its bands
to floating point *before* touching them, so raw ``uint16`` /
``int16`` DN never wrap around or overflow in ``a - b`` / ``a + b``.
Integers of up to 16 bits are promoted to ``float32``, wider integers
to ``float64``; float inputs keep their own precision.

Example:
    >>> import numpy as np
    >>> nir, red = np.array([0.5, 0.4]), np.array([0.1, 0.4])
    >>> ndvi(nir, red, eps=0.0)
    array([0.66666667, 0.        ])
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Num
from pipekit_array import array_namespace

from geoops._src.dtype import as_float


def normalized_difference(
    a: Num[np.ndarray, "*s"],
    b: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Generic normalized-difference band ratio.

    Computes

    .. math::

        \mathrm{ND}(a, b) \;=\; \frac{a - b}{a + b + \varepsilon}

    where :math:`a` and :math:`b` are two bands on one shape.
    The denominator's :math:`\varepsilon` shadows division by zero when
    both bands are exactly zero (no-data fill, sensor saturation
    bottoming out, etc.) and biases the result toward zero in that
    degenerate case rather than NaN.

    The normalized-difference family (NDVI, NDWI, NDBI, NBR, …) shares
    this exact form; concrete indices are named convenience aliases that
    pin which two bands play the role of :math:`a` and :math:`b`.

    Args:
        a: The "high" band (numerator-positive term). For
            NDVI this is NIR; for NDWI (McFeeters) this is Green; for
            NDBI this is SWIR.
        b: The "low" band. NDVI: Red; NDWI: NIR; NDBI: NIR.
        eps: Small constant added to the denominator. Default ``1e-10``.
            Pass ``0.0`` if you'd rather see ``inf``/``nan`` on zero
            pixels (useful when debugging masking issues).

    Returns:
        An array of the inputs' shape. Output range is
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
    a = as_float(a)
    b = as_float(b)
    return (a - b) / (a + b + eps)


def ndvi(
    nir: Num[np.ndarray, "*s"],
    red: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
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
        nir: The NIR band (Sentinel-2 B8, Landsat-8 B5). Reflectance
            (0–1) for the formula to be physically meaningful; DN values
            work but the result is sensor-specific.
        red: The Red band (Sentinel-2 B4, Landsat-8 B4).
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape; values in ``[-1, +1]``.
    """
    return normalized_difference(nir, red, eps=eps)


def ndwi_mcfeeters(
    green: Num[np.ndarray, "*s"],
    nir: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
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
        green: The Green band (S2 B3, L8 B3).
        nir: The NIR band (S2 B8, L8 B5).
        eps: Denominator stabiliser.

    Returns:
        An array of the inputs' shape; values in ``[-1, +1]``,
        typically > 0 over water, < 0 elsewhere.

    References:
        McFeeters, S. K. (1996). "The use of the Normalized Difference
        Water Index (NDWI) in the delineation of open water features."
        *International Journal of Remote Sensing*, 17(7), 1425–1432.
    """
    return normalized_difference(green, nir, eps=eps)


def ndbi(
    swir: Num[np.ndarray, "*s"],
    nir: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
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
        swir: The SWIR-1 band (S2 B11, L8 B6).
        nir: The NIR band (S2 B8, L8 B5).
        eps: Denominator stabiliser.

    Returns:
        An array of the inputs' shape.

    References:
        Zha, Y., Gao, J., & Ni, S. (2003). "Use of normalized difference
        built-up index in automatically mapping urban areas from TM
        imagery." *Int. J. Remote Sens.*, 24(3), 583–594.
    """
    return normalized_difference(swir, nir, eps=eps)


def nbr(
    nir: Num[np.ndarray, "*s"],
    swir2: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
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
        nir: The NIR band (S2 B8, L8 B5).
        swir2: The SWIR-2 band (S2 B12, L8 B7).
        eps: Denominator stabiliser.

    Returns:
        An array of the inputs' shape; high over healthy vegetation,
        low (or negative) over recently-burned surfaces.

    References:
        Key, C. H., & Benson, N. C. (2006). "Landscape assessment (LA):
        sampling and analysis methods." USDA Forest Service General
        Technical Report RMRS-GTR-164-CD, LA-1 to LA-55.
    """
    return normalized_difference(nir, swir2, eps=eps)


def savi(
    nir: Num[np.ndarray, "*s"],
    red: Num[np.ndarray, "*s"],
    *,
    L: float = 0.5,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
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
    self-adjusting variant MSAVI2 (not provided here) replaces the
    constant with a closed-form scene-adaptive term.

    ``eps`` is added to the denominator like the other ratio indices,
    so ``L = 0`` with ``NIR + Red = 0`` stays finite; for ``L > 0`` and
    non-negative reflectance it is negligible. Pass reflectance, not DN.

    Args:
        nir: NIR band.
        red: Red band.
        L: Soil-adjustment factor in ``[0, 1]``. Default ``0.5``.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape.

    References:
        Huete, A. R. (1988). "A soil-adjusted vegetation index (SAVI)."
        *Remote Sensing of Environment*, 25(3), 295–309.
    """
    nir = as_float(nir)
    red = as_float(red)
    return (nir - red) / (nir + red + L + eps) * (1.0 + L)


def evi(
    nir: Num[np.ndarray, "*s"],
    red: Num[np.ndarray, "*s"],
    blue: Num[np.ndarray, "*s"],
    *,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
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
        nir: NIR band.
        red: Red band.
        blue: Blue band.
        G: Gain factor. Default ``2.5`` (MODIS convention).
        C1: Red aerosol-resistance coefficient. Default ``6``.
        C2: Blue aerosol-resistance coefficient. Default ``7.5``.
        L: Canopy-background correction. Default ``1``.

    Returns:
        An array of the inputs' shape; values in ``[-1, +1]`` for
        physically-reasonable reflectance inputs.

    References:
        Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X., &
        Ferreira, L. G. (2002). "Overview of the radiometric and
        biophysical performance of the MODIS vegetation indices."
        *Remote Sensing of Environment*, 83(1-2), 195–213.
    """
    nir = as_float(nir)
    red = as_float(red)
    blue = as_float(blue)
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + eps)


def evi2(
    nir: Num[np.ndarray, "*s"],
    red: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Two-band Enhanced Vegetation Index (Jiang et al. 2008).

    .. math::

        \mathrm{EVI2} \;=\; 2.5 \cdot
            \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}
                 {\rho_{\mathrm{NIR}} + 2.4\,\rho_{\mathrm{Red}} + 1 + \varepsilon}

    A blue-band-free approximation to EVI tuned to track it within a
    few percent over most cover types.

    Args:
        nir: NIR band.
        red: Red band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape.

    References:
        Jiang, Z., Huete, A. R., Didan, K., & Miura, T. (2008).
        *Remote Sensing of Environment*, 112(10), 3833–3845.
    """
    nir = as_float(nir)
    red = as_float(red)
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + eps)


def arvi(
    nir: Num[np.ndarray, "*s"],
    red: Num[np.ndarray, "*s"],
    blue: Num[np.ndarray, "*s"],
    *,
    gamma: float = 1.0,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Atmospherically Resistant Vegetation Index (Kaufman & Tanre 1992).

    .. math::

        \rho_{rb} = \rho_{\mathrm{Red}}
                  - \gamma\,(\rho_{\mathrm{Blue}} - \rho_{\mathrm{Red}}),
        \quad
        \mathrm{ARVI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{rb}}
                                 {\rho_{\mathrm{NIR}} + \rho_{rb} + \varepsilon}

    The Blue-band correction cancels aerosol-driven red inflation.
    Pass ``gamma=1`` for the original MODIS-derived value.

    Args:
        nir: NIR band.
        red: Red band.
        blue: Blue band.
        gamma: Aerosol-correction strength. Default ``1.0``.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape.
    """
    nir = as_float(nir)
    red = as_float(red)
    blue = as_float(blue)
    rb = red - gamma * (blue - red)
    return (nir - rb) / (nir + rb + eps)


def gci(
    nir: Num[np.ndarray, "*s"],
    green: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Green Chlorophyll Index (Gitelson et al. 2003).

    .. math::

        \mathrm{GCI} \;=\; \frac{\rho_{\mathrm{NIR}}}
                                {\rho_{\mathrm{Green}} + \varepsilon} - 1

    Roughly linear in canopy chlorophyll content; saturates much
    later than NDVI.

    Args:
        nir: NIR band.
        green: Green band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape.
    """
    nir = as_float(nir)
    green = as_float(green)
    return nir / (green + eps) - 1.0


def kndvi(
    nir: Num[np.ndarray, "*s"],
    red: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Kernel NDVI (Camps-Valls et al. 2021).

    .. math::

        \mathrm{kNDVI} \;=\; \tanh\!\bigl(\mathrm{NDVI}^2\bigr)

    Non-linear transform of NDVI with reduced saturation and a
    closer-to-linear relationship to GPP.

    Args:
        nir: NIR band.
        red: Red band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape; values in ``[0, 1)``.
    """
    value = ndvi(nir, red, eps=eps)
    return array_namespace(value).tanh(value**2)


def mndwi(
    green: Num[np.ndarray, "*s"],
    swir: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Modified Normalized Difference Water Index (Xu 2006).

    .. math::

        \mathrm{MNDWI} \;=\; \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{SWIR1}}}
                                  {\rho_{\mathrm{Green}} + \rho_{\mathrm{SWIR1}}
                                   + \varepsilon}

    Sharper water/non-water separation than McFeeters NDWI; reduces
    urban-surface confusion.

    Args:
        green: Green band.
        swir: SWIR-1 band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape; values in ``[-1, +1]``.
    """
    return normalized_difference(green, swir, eps=eps)


def ndmi(
    nir: Num[np.ndarray, "*s"],
    swir1: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Normalized Difference Moisture Index (Gao 1996).

    .. math::

        \mathrm{NDMI} \;=\; \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{SWIR1}}}
                                 {\rho_{\mathrm{NIR}} + \rho_{\mathrm{SWIR1}}
                                  + \varepsilon}

    Tracks vegetation *liquid-water content* (not surface water).

    Args:
        nir: NIR band.
        swir1: SWIR-1 band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape; values in ``[-1, +1]``.
    """
    return normalized_difference(nir, swir1, eps=eps)


def ndsi(
    green: Num[np.ndarray, "*s"],
    swir: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Normalized Difference Snow Index (Hall et al. 1995).

    .. math::

        \mathrm{NDSI} \;=\; \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{SWIR1}}}
                                 {\rho_{\mathrm{Green}} + \rho_{\mathrm{SWIR1}}
                                  + \varepsilon}

    Snow / ice mapping (NDSI > 0.4 is the standard MODIS threshold).
    Shares its arithmetic form with MNDWI; the two indices are
    interpreted differently.

    Args:
        green: Green band.
        swir: SWIR-1 band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape; values in ``[-1, +1]``.
    """
    return normalized_difference(green, swir, eps=eps)


def nbr2(
    swir1: Num[np.ndarray, "*s"],
    swir2: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Normalized Burn Ratio 2 (USGS Landsat product).

    .. math::

        \mathrm{NBR2} \;=\; \frac{\rho_{\mathrm{SWIR1}} - \rho_{\mathrm{SWIR2}}}
                                 {\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{SWIR2}}
                                  + \varepsilon}

    Args:
        swir1: SWIR-1 band.
        swir2: SWIR-2 band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape; values in ``[-1, +1]``.
    """
    return normalized_difference(swir1, swir2, eps=eps)


def bais2(
    red: Num[np.ndarray, "*s"],
    red_edge1: Num[np.ndarray, "*s"],
    red_edge2: Num[np.ndarray, "*s"],
    nir: Num[np.ndarray, "*s"],
    swir2: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Burned Area Index for Sentinel-2 (Filipponi 2018).

    .. math::

        \mathrm{BAIS2} \;=\;
        \Bigl(1 - \sqrt{\tfrac{\rho_{6}\,\rho_{7}\,\rho_{8\mathrm{A}}}
                              {\rho_{4} + \varepsilon}}\Bigr)
        \cdot
        \Bigl(\tfrac{\rho_{12} - \rho_{8\mathrm{A}}}
                    {\sqrt{\rho_{12} + \rho_{8\mathrm{A}} + \varepsilon}} + 1\Bigr)

    Args:
        red: Red (B04) reflectance; every band non-negative, or the
            square roots produce NaN.
        red_edge1: First red-edge (B06).
        red_edge2: Second red-edge (B07).
        nir: Narrow-NIR (B8A).
        swir2: SWIR-2 (B12).
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape.

    References:
        Filipponi, F. (2018). *Proceedings*, 2(7), 364.
    """
    xp = array_namespace(red)
    red = as_float(red)
    red_edge1 = as_float(red_edge1)
    red_edge2 = as_float(red_edge2)
    nir = as_float(nir)
    swir2 = as_float(swir2)
    return (1.0 - xp.sqrt((red_edge1 * red_edge2 * nir) / (red + eps))) * (
        (swir2 - nir) / xp.sqrt(swir2 + nir + eps) + 1.0
    )


def bsi(
    blue: Num[np.ndarray, "*s"],
    red: Num[np.ndarray, "*s"],
    nir: Num[np.ndarray, "*s"],
    swir: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Bare Soil Index (Rikimaru et al. 2002).

    .. math::

        \mathrm{BSI} \;=\;
        \frac{(\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{Red}}) -
              (\rho_{\mathrm{NIR}}  + \rho_{\mathrm{Blue}})}
             {(\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{Red}}) +
              (\rho_{\mathrm{NIR}}  + \rho_{\mathrm{Blue}}) + \varepsilon}

    Args:
        blue: Blue band.
        red: Red band.
        nir: NIR band.
        swir: SWIR-1 band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape; values in ``[-1, +1]``.
    """
    blue = as_float(blue)
    red = as_float(red)
    nir = as_float(nir)
    swir = as_float(swir)
    return ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + eps)


def iron_oxide(
    red: Num[np.ndarray, "*s"],
    blue: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Iron Oxide ratio (Sabins 1999).

    .. math::

        \mathrm{IronOxide} \;=\; \frac{\rho_{\mathrm{Red}}}
                                       {\rho_{\mathrm{Blue}} + \varepsilon}

    Hematite/goethite and related Fe(III) oxides absorb blue and
    reflect red.

    Args:
        red: Red band.
        blue: Blue band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape; a Red/Blue band ratio.
    """
    red = as_float(red)
    blue = as_float(blue)
    return red / (blue + eps)


def clay_minerals(
    swir1: Num[np.ndarray, "*s"],
    swir2: Num[np.ndarray, "*s"],
    *,
    eps: float = 1e-10,
) -> Float[np.ndarray, "*s"]:
    r"""Clay Minerals ratio (Sabins 1999; Crowley et al. 1989).

    .. math::

        \mathrm{ClayMinerals} \;=\; \frac{\rho_{\mathrm{SWIR1}}}
                                          {\rho_{\mathrm{SWIR2}} + \varepsilon}

    OH-bearing minerals (kaolinite, illite, montmorillonite, alunite)
    have a diagnostic 2.2 µm absorption inside SWIR-2.

    Args:
        swir1: SWIR-1 band.
        swir2: SWIR-2 band.
        eps: Denominator stabiliser. Default ``1e-10``.

    Returns:
        An array of the inputs' shape; a SWIR1/SWIR2 band ratio.
    """
    swir1 = as_float(swir1)
    swir2 = as_float(swir2)
    return swir1 / (swir2 + eps)


def ciri(
    cirrus: Num[np.ndarray, "*s"],
) -> Num[np.ndarray, "*s"]:
    r"""Cirrus Reflectance Index (Sentinel-2 B10 passthrough).

    .. math::

        \mathrm{CIRI} \;=\; \rho_{\mathrm{cirrus}}

    The 1.36–1.39 µm cirrus channel is opaque to surface signal due
    to water-vapour absorption; any non-trivial reflectance comes
    from high-altitude cloud.

    Args:
        cirrus: Cirrus band (Sentinel-2 B10).

    Returns:
        An array of the inputs' shape — the cirrus band itself,
        dtype preserved.
    """
    return cirrus


def dnbr(
    nbr_pre: Float[np.ndarray, "*s"],
    nbr_post: Float[np.ndarray, "*s"],
) -> Float[np.ndarray, "*s"]:
    r"""Differenced Normalized Burn Ratio (Key & Benson 2006).

    .. math::

        \mathrm{dNBR} \;=\; \mathrm{NBR}_{\mathrm{pre}} - \mathrm{NBR}_{\mathrm{post}}

    Args:
        nbr_pre: NBR before the fire (see :func:`nbr`).
        nbr_post: NBR after the fire, on the same shape.

    Returns:
        An array of the inputs' shape, floating point; positive where the
        fire lowered NBR.

    Example:
        >>> dnbr(np.array([0.6, 0.2]), np.array([0.1, 0.2]))
        array([0.5, 0. ])
    """
    return as_float(nbr_pre) - as_float(nbr_post)
