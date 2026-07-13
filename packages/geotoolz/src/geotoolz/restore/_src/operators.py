"""Tier-B restoration Operators — carrier-aware wrappers.

Each Operator wraps a primitive in :mod:`geotoolz.restore._src.array`
and rewraps the result to match the input carrier via
:func:`geotoolz._src.wrap.wrap_like`: a ``GeoTensor`` input round-trips
its ``transform`` / ``crs`` / ``fill_value_default``, while a plain
``np.ndarray`` input returns a plain ``np.ndarray``. All restoration
primitives are metadata-independent per-pixel/window math, so every
operator here supports both carriers transparently. All constructor
parameters are keyword-only and JSON-safe for hydra-zen ``builds()``
round-trips, except for :class:`InverseMNF` which holds a runtime
reference to a fitted :class:`MNF` and is therefore marked
``forbid_in_yaml = True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pipekit import Operator

from geotoolz._src.wrap import wrap_like
from geotoolz.restore._src.array import (
    bilateral_denoise,
    despeckle_frost,
    despeckle_lee,
    despeckle_refined_lee,
    destripe_column,
    fit_pca,
    gap_fill_biharmonic,
    gap_fill_idw,
    gap_fill_laplacian,
    gap_fill_nearest,
    gaussian_denoise,
    inverse_pca,
    median_denoise,
    nl_means,
    outlier_mask,
    pca_denoise,
    replace_outliers,
    saturation_flag,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


class DespeckleLee(Operator):
    """Lee local-statistics speckle filter.

    Carrier-aware wrapper around
    :func:`~geotoolz.restore._src.array.despeckle_lee`. Best suited to
    multiplicative speckle (single-look or multi-look SAR amplitude
    imagery). For multi-look intensity, halve ``cu``.

    Args:
        window: Side length of the local window in pixels.
        cu: Noise coefficient of variation. ``0.523`` is the canonical
            single-look value.

    Examples:
        >>> import geotoolz as gz
        >>> pipe = gz.restore.DespeckleLee(window=7) | gz.restore.MedianDenoise(size=3)
        >>> clean = pipe(sar_geotensor)
    """

    def __init__(self, *, window: int = 7, cu: float = 0.523) -> None:
        self.window = window
        self.cu = cu

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(
            gt, despeckle_lee(np.asarray(gt), window=self.window, cu=self.cu)
        )

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window, "cu": self.cu}


class DespeckleFrost(Operator):
    """Frost-style adaptive speckle filter.

    Wraps :func:`~geotoolz.restore._src.array.despeckle_frost`. Uses an
    edge-aware exponential weight on the local mean. Faster than the
    Lee filter and tunable via ``damping``: larger values keep more
    edge contrast, smaller values smooth more aggressively.

    Args:
        window: Side length of the local window in pixels.
        damping: Edge-sensitivity exponent.

    Examples:
        >>> gz.restore.DespeckleFrost(window=7, damping=2.0)(sar_geotensor)
    """

    def __init__(self, *, window: int = 7, damping: float = 2.0) -> None:
        self.window = window
        self.damping = damping

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(
            gt,
            despeckle_frost(np.asarray(gt), window=self.window, damping=self.damping),
        )

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window, "damping": self.damping}


class DespeckleRefinedLee(Operator):
    """Refined-Lee speckle filter approximation.

    Wraps :func:`~geotoolz.restore._src.array.despeckle_refined_lee`.
    This is currently a dependency-light alias for :class:`DespeckleLee`
    with default ``cu``; the eight-direction sub-window selection of the
    canonical Refined-Lee is not yet implemented.

    Args:
        window: Side length of the local window in pixels.

    Examples:
        >>> gz.restore.DespeckleRefinedLee(window=7)(sar_geotensor)
    """

    def __init__(self, *, window: int = 7) -> None:
        self.window = window

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(gt, despeckle_refined_lee(np.asarray(gt), window=self.window))

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window}


class DestripeColumn(Operator):
    """Subtract per-column (or per-row) offsets from striped imagery.

    Wraps :func:`~geotoolz.restore._src.array.destripe_column`. Use
    ``method="mean"`` for additive stripes, ``"median"`` for stripes
    with outlier contamination, and ``"moment_matching"`` to also apply
    a local smoothing pass (the smoothing kernel size is set by
    ``window``).

    Args:
        method: Reducer used to estimate per-column offsets.
        axis: Striping direction. ``"column"`` removes vertical
            stripes, ``"row"`` removes horizontal stripes.
        window: Smoothing window for ``method="moment_matching"``.
            Ignored for the other methods but accepted for hydra-zen
            round-trip uniformity.

    Examples:
        >>> gz.restore.DestripeColumn(method="median", axis="column")(scene)
    """

    def __init__(
        self,
        *,
        method: Literal["mean", "median", "moment_matching"] = "mean",
        axis: Literal["column", "row"] = "column",
        window: int = 21,
    ) -> None:
        self.method = method
        self.axis = axis
        self.window = window

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(
            gt,
            destripe_column(
                np.asarray(gt),
                method=self.method,
                axis=self.axis,
                window=self.window,
            ),
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "axis": self.axis, "window": self.window}


class MomentMatching(Operator):
    """Match per-column moments via a local smoothing pass.

    Convenience operator equivalent to
    ``DestripeColumn(method="moment_matching", axis="column", window=...)``.

    Args:
        window: Side length of the local smoothing window in pixels.

    Examples:
        >>> gz.restore.MomentMatching(window=21)(scene)
    """

    def __init__(self, *, window: int = 21) -> None:
        self.window = window

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(
            gt,
            destripe_column(
                np.asarray(gt),
                method="moment_matching",
                axis="column",
                window=self.window,
            ),
        )

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window}


class DenoisePCA(Operator):
    """Reconstruct a multi-band raster from its top PCA components.

    Projects the carrier onto its leading ``n_components`` principal
    directions and reconstructs in the original space. Effective at
    suppressing band-uncorrelated noise; can soften sharp
    band-localised features.

    Args:
        n_components: Number of principal components to keep.
            ``n_components < bands`` performs noise reduction.
        axis: Position of the band axis. Defaults to ``0``
            (``(bands, H, W)``).

    Examples:
        >>> gz.restore.DenoisePCA(n_components=10)(hyperspectral_scene)
    """

    def __init__(self, *, n_components: int, axis: int = 0) -> None:
        self.n_components = n_components
        self.axis = axis

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = pca_denoise(
            np.asarray(gt), n_components=self.n_components, axis=self.axis
        )
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"n_components": self.n_components, "axis": self.axis}


class MNF(Operator):
    """Forward MNF / PCA transform that captures reconstruction state.

    Wraps :func:`~geotoolz.restore._src.array.fit_pca`. The first call
    fits the principal components on the input and stores them on the
    instance; the output carrier holds the projected scores. The
    fitted state is then consumable by :class:`InverseMNF`.

    Note: this operator is *stateful*. Calling it on a second image
    will refit the components and discard the previous state — the
    forward/inverse pair must be applied to the same image.

    Args:
        n_components: Number of components to keep. ``None`` keeps all.
        axis: Position of the band axis.

    Attributes:
        snr_: Per-component variance (proxy for signal-to-noise ratio),
            populated after the first call. Sorted descending.

    Examples:
        >>> forward = gz.restore.MNF(n_components=3)
        >>> scores = forward(scene)
        >>> reconstructed = gz.restore.InverseMNF(forward=forward)(scores)
    """

    def __init__(self, *, n_components: int | None = None, axis: int = 0) -> None:
        self.n_components = n_components
        self.axis = axis
        self._state: dict[str, np.ndarray | int | tuple[int, ...]] | None = None
        self.snr_: np.ndarray | None = None

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        self._state = fit_pca(
            np.asarray(gt), n_components=self.n_components, axis=self.axis
        )
        self.snr_ = np.asarray(self._state["snr"])
        return wrap_like(gt, np.asarray(self._state["scores"]))

    def get_config(self) -> dict[str, Any]:
        return {"n_components": self.n_components, "axis": self.axis}


class InverseMNF(Operator):
    """Reconstruct a raster from a prior :class:`MNF` transform.

    Holds a runtime reference to a fitted :class:`MNF` so it can reuse
    the principal components and mean. Because the reference points at
    a live, stateful object, this operator cannot be faithfully
    serialised — ``forbid_in_yaml = True`` flags that to future YAML
    loaders, and ``get_config`` returns an empty config rather than a
    spurious payload.

    Args:
        forward: A fitted :class:`MNF` whose ``_apply`` has already
            been invoked. The forward must outlive the inverse.

    Examples:
        >>> forward = gz.restore.MNF(n_components=3)
        >>> _ = forward(scene)
        >>> reconstructed = gz.restore.InverseMNF(forward=forward)(forward(scene))
    """

    # Holds a stateful forward reference that cannot be serialized faithfully.
    forbid_in_yaml = True

    def __init__(self, *, forward: MNF) -> None:
        self.forward = forward

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        if self.forward._state is None:
            raise ValueError("InverseMNF requires a forward MNF that has been applied")
        out = inverse_pca(np.asarray(gt), self.forward._state)
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        # The fitted ``forward`` reference is not JSON-safe; report an
        # empty config and rely on ``forbid_in_yaml`` to block YAML
        # serialisation.
        return {}


class GaussianDenoise(Operator):
    """Gaussian smoother over the trailing two spatial axes.

    NaN-aware: missing pixels are excluded from both the numerator and
    the normalising weight. Non-spatial axes (e.g. bands) are not
    filtered.

    Args:
        sigma: Gaussian standard deviation in pixels.

    Examples:
        >>> gz.restore.GaussianDenoise(sigma=1.0)(scene)
    """

    def __init__(self, *, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(gt, gaussian_denoise(np.asarray(gt), sigma=self.sigma))

    def get_config(self) -> dict[str, Any]:
        return {"sigma": self.sigma}


class MedianDenoise(Operator):
    """Median filter over the trailing two spatial axes.

    Robust to impulse noise (salt-and-pepper, hot pixels). Larger
    ``size`` blurs sharper features.

    Args:
        size: Side length of the median window in pixels.

    Examples:
        >>> gz.restore.MedianDenoise(size=3)(scene)
    """

    def __init__(self, *, size: int = 3) -> None:
        self.size = size

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(gt, median_denoise(np.asarray(gt), size=self.size))

    def get_config(self) -> dict[str, Any]:
        return {"size": self.size}


class BilateralDenoise(Operator):
    """Edge-aware denoise using a range-weighted Gaussian.

    Wraps :func:`~geotoolz.restore._src.array.bilateral_denoise`. Use
    ``sigma_color`` to control how aggressively edges are preserved
    (smaller = stronger preservation) and ``sigma_space`` for the
    spatial smoothing scale.

    Caveat: this is a single-pass approximation of a full bilateral
    filter, not the canonical per-pixel-neighbourhood form.

    Examples:
        >>> gz.restore.BilateralDenoise(sigma_color=0.1, sigma_space=5.0)(scene)
    """

    def __init__(self, *, sigma_color: float = 0.1, sigma_space: float = 5.0) -> None:
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(
            gt,
            bilateral_denoise(
                np.asarray(gt),
                sigma_color=self.sigma_color,
                sigma_space=self.sigma_space,
            ),
        )

    def get_config(self) -> dict[str, Any]:
        return {"sigma_color": self.sigma_color, "sigma_space": self.sigma_space}


class NLMeans(Operator):
    """Lightweight non-local-means-style denoiser.

    Wraps :func:`~geotoolz.restore._src.array.nl_means`. This is a
    dependency-light approximation of the canonical NL-means; for
    production use prefer ``skimage.restoration.denoise_nl_means``.

    Args:
        patch_size: Nominal patch side length in pixels.
        patch_distance: Nominal search-window radius in pixels.
        h: Range bandwidth in data units.

    Examples:
        >>> gz.restore.NLMeans(patch_size=5, patch_distance=6, h=0.1)(scene)
    """

    def __init__(
        self, *, patch_size: int = 5, patch_distance: int = 6, h: float = 0.1
    ) -> None:
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.h = h

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(
            gt,
            nl_means(
                np.asarray(gt),
                patch_size=self.patch_size,
                patch_distance=self.patch_distance,
                h=self.h,
            ),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "patch_size": self.patch_size,
            "patch_distance": self.patch_distance,
            "h": self.h,
        }


class GapFillIDW(Operator):
    """Fill NaNs with inverse-distance weighted finite neighbours.

    Wraps :func:`~geotoolz.restore._src.array.gap_fill_idw`. NaNs whose
    neighbourhood within ``radius`` contains no finite pixels are left
    as NaN — chain with :class:`GapFillNearest` for unconditional fill.

    Args:
        power: IDW exponent. ``2.0`` is the standard choice.
        radius: Search radius in pixels.

    Examples:
        >>> gz.restore.GapFillIDW(power=2.0, radius=5)(scene)
    """

    def __init__(self, *, power: float = 2.0, radius: int = 5) -> None:
        self.power = power
        self.radius = radius

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(
            gt, gap_fill_idw(np.asarray(gt), power=self.power, radius=self.radius)
        )

    def get_config(self) -> dict[str, Any]:
        return {"power": self.power, "radius": self.radius}


class GapFillInpaintBiharmonic(Operator):
    """Smoothly inpaint NaN gaps with a biharmonic-style fill.

    Wraps :func:`~geotoolz.restore._src.array.gap_fill_biharmonic`,
    which already preserves the original finite pixels — the operator
    just round-trips carrier metadata.

    Caveat: this is a dependency-light surrogate, not the canonical
    scikit-image biharmonic inpainting; it can be slow for very large
    masks because it iterates the Laplace solver internally.

    Examples:
        >>> gz.restore.GapFillInpaintBiharmonic()(scene)
    """

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(gt, gap_fill_biharmonic(np.asarray(gt)))


class GapFillLaplacian(Operator):
    """Fill NaNs by solving a discrete Laplace equation.

    Wraps :func:`~geotoolz.restore._src.array.gap_fill_laplacian`. The
    harmonic interpolant — smooth, but linearly biased toward the mean
    of the boundary. Pair with :class:`GaussianDenoise` for a smoother
    transition near the mask edge.

    Examples:
        >>> gz.restore.GapFillLaplacian()(scene)
    """

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(gt, gap_fill_laplacian(np.asarray(gt)))


class GapFillNearest(Operator):
    """Fill NaNs from the nearest finite neighbour.

    Wraps :func:`~geotoolz.restore._src.array.gap_fill_nearest`. Fast
    and unconditional unless ``max_distance`` is set.

    Args:
        max_distance: Maximum Euclidean fill radius in pixels. ``None``
            (default) fills every NaN with a finite neighbour anywhere
            in its 2-D plane.

    Examples:
        >>> gz.restore.GapFillNearest(max_distance=3)(scene)
    """

    def __init__(self, *, max_distance: int | None = None) -> None:
        self.max_distance = max_distance

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(
            gt, gap_fill_nearest(np.asarray(gt), max_distance=self.max_distance)
        )

    def get_config(self) -> dict[str, Any]:
        return {"max_distance": self.max_distance}


class OutlierMask(Operator):
    """Flag global outliers with a MAD or z-score threshold.

    Wraps :func:`~geotoolz.restore._src.array.outlier_mask` and returns
    a boolean carrier (``True`` marks outliers). Outputs the result
    as ``bool`` to keep the mask explicit.

    Args:
        method: ``"mad"`` (robust, default) or ``"zscore"``.
        k: Threshold in scaled units.

    Examples:
        >>> gz.restore.OutlierMask(method="mad", k=3.0)(scene)
    """

    def __init__(
        self, *, method: Literal["mad", "zscore"] = "mad", k: float = 3.0
    ) -> None:
        self.method = method
        self.k = k

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        mask = outlier_mask(np.asarray(gt), method=self.method, k=self.k)
        return wrap_like(gt, mask.astype(bool))

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "k": self.k}


class ReplaceOutliers(Operator):
    """Replace outliers with median, NaN, or nearest-neighbour interpolation.

    Wraps :func:`~geotoolz.restore._src.array.replace_outliers`.

    Args:
        method: Outlier detector. See :class:`OutlierMask`.
        k: Threshold in scaled units.
        fill: ``"median"`` (inlier median), ``"nan"`` (mark as missing),
            or ``"interp"`` (NaN then nearest-neighbour fill).

    Examples:
        >>> gz.restore.ReplaceOutliers(method="mad", k=3.0, fill="median")(scene)
    """

    def __init__(
        self,
        *,
        method: Literal["mad", "zscore"] = "mad",
        k: float = 3.0,
        fill: Literal["median", "nan", "interp"] = "median",
    ) -> None:
        self.method = method
        self.k = k
        self.fill = fill

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        out = replace_outliers(
            np.asarray(gt), method=self.method, k=self.k, fill=self.fill
        )
        return wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "k": self.k, "fill": self.fill}


class SaturationFlag(Operator):
    """Flag saturated pixels as a boolean carrier.

    Wraps :func:`~geotoolz.restore._src.array.saturation_flag`. When
    ``threshold`` is ``None`` the default is ``np.iinfo(dtype).max`` for
    integer carriers and ``1.0`` for floats.

    Examples:
        >>> gz.restore.SaturationFlag()(uint16_scene)
        >>> gz.restore.SaturationFlag(threshold=0.95)(reflectance_scene)
    """

    def __init__(self, *, threshold: float | None = None) -> None:
        self.threshold = threshold

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        return wrap_like(
            gt, saturation_flag(np.asarray(gt), threshold=self.threshold).astype(bool)
        )

    def get_config(self) -> dict[str, Any]:
        return {"threshold": self.threshold}
