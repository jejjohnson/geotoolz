"""Tier-B restoration Operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from geotoolz.core import Operator
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
    """Lee speckle filter."""

    def __init__(self, *, window: int = 7, cu: float = 0.523) -> None:
        self.window = window
        self.cu = cu

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            despeckle_lee(np.asarray(gt), window=self.window, cu=self.cu)
        )

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window, "cu": self.cu}


class DespeckleFrost(Operator):
    """Frost-style adaptive speckle filter."""

    def __init__(self, *, window: int = 7, damping: float = 2.0) -> None:
        self.window = window
        self.damping = damping

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            despeckle_frost(np.asarray(gt), window=self.window, damping=self.damping)
        )

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window, "damping": self.damping}


class DespeckleRefinedLee(Operator):
    """Refined-Lee speckle filter approximation."""

    def __init__(self, *, window: int = 7) -> None:
        self.window = window

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            despeckle_refined_lee(np.asarray(gt), window=self.window)
        )

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window}


class DestripeColumn(Operator):
    """Estimate per-column or per-row offsets and subtract them."""

    def __init__(
        self,
        *,
        method: Literal["mean", "median", "moment_matching"] = "mean",
        axis: Literal["column", "row"] = "column",
    ) -> None:
        self.method = method
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            destripe_column(np.asarray(gt), method=self.method, axis=self.axis)
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "axis": self.axis}


class MomentMatching(Operator):
    """Match per-column moments to neighbouring columns."""

    def __init__(self, *, window: int = 21) -> None:
        self.window = window

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            destripe_column(
                np.asarray(gt),
                method="moment_matching",
                axis="column",
                window=self.window,
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {"window": self.window}


class DenoisePCA(Operator):
    """Reconstruct a multi-band raster from the top PCA components."""

    def __init__(self, *, n_components: int, axis: int = 0) -> None:
        self.n_components = n_components
        self.axis = axis

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = pca_denoise(
            np.asarray(gt), n_components=self.n_components, axis=self.axis
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"n_components": self.n_components, "axis": self.axis}


class MNF(Operator):
    """Minimum Noise Fraction-style forward transform."""

    def __init__(self, *, n_components: int | None = None, axis: int = 0) -> None:
        self.n_components = n_components
        self.axis = axis
        self._state: dict[str, np.ndarray | int | tuple[int, ...]] | None = None
        self.snr_: np.ndarray | None = None

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        self._state = fit_pca(
            np.asarray(gt), n_components=self.n_components, axis=self.axis
        )
        self.snr_ = np.asarray(self._state["snr"])
        return gt.array_as_geotensor(np.asarray(self._state["scores"]))

    def get_config(self) -> dict[str, Any]:
        return {"n_components": self.n_components, "axis": self.axis}


class InverseMNF(Operator):
    """Reconstruct a raster from a prior :class:`MNF` transform."""

    # Holds a stateful forward reference that cannot be serialized faithfully.
    forbid_in_yaml = True

    def __init__(self, *, forward: MNF) -> None:
        self.forward = forward

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        if self.forward._state is None:
            raise ValueError("InverseMNF requires a forward MNF that has been applied")
        out = inverse_pca(np.asarray(gt), self.forward._state)
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"forward": repr(self.forward)}


class GaussianDenoise(Operator):
    """Gaussian denoiser."""

    def __init__(self, *, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(gaussian_denoise(np.asarray(gt), sigma=self.sigma))

    def get_config(self) -> dict[str, Any]:
        return {"sigma": self.sigma}


class MedianDenoise(Operator):
    """Median denoiser."""

    def __init__(self, *, size: int = 3) -> None:
        self.size = size

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(median_denoise(np.asarray(gt), size=self.size))

    def get_config(self) -> dict[str, Any]:
        return {"size": self.size}


class BilateralDenoise(Operator):
    """Bilateral-style edge-aware denoiser."""

    def __init__(self, *, sigma_color: float = 0.1, sigma_space: float = 5.0) -> None:
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            bilateral_denoise(
                np.asarray(gt),
                sigma_color=self.sigma_color,
                sigma_space=self.sigma_space,
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {"sigma_color": self.sigma_color, "sigma_space": self.sigma_space}


class NLMeans(Operator):
    """Non-local-means-style denoiser."""

    def __init__(
        self, *, patch_size: int = 5, patch_distance: int = 6, h: float = 0.1
    ) -> None:
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.h = h

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            nl_means(
                np.asarray(gt),
                patch_size=self.patch_size,
                patch_distance=self.patch_distance,
                h=self.h,
            )
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "patch_size": self.patch_size,
            "patch_distance": self.patch_distance,
            "h": self.h,
        }


class GapFillIDW(Operator):
    """Fill NaNs with inverse-distance weighted neighbours."""

    def __init__(self, *, power: float = 2.0, radius: int = 5) -> None:
        self.power = power
        self.radius = radius

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            gap_fill_idw(np.asarray(gt), power=self.power, radius=self.radius)
        )

    def get_config(self) -> dict[str, Any]:
        return {"power": self.power, "radius": self.radius}


class GapFillInpaintBiharmonic(Operator):
    """Smoothly inpaint NaN gaps."""

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        values = np.asarray(gt)
        out = gap_fill_biharmonic(values)
        return gt.array_as_geotensor(np.where(np.isfinite(values), values, out))


class GapFillLaplacian(Operator):
    """Fill NaNs with a Laplacian smoother."""

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(gap_fill_laplacian(np.asarray(gt)))


class GapFillNearest(Operator):
    """Fill NaNs from the nearest finite neighbour."""

    def __init__(self, *, max_distance: int | None = None) -> None:
        self.max_distance = max_distance

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            gap_fill_nearest(np.asarray(gt), max_distance=self.max_distance)
        )

    def get_config(self) -> dict[str, Any]:
        return {"max_distance": self.max_distance}


class OutlierMask(Operator):
    """Flag outliers with MAD or z-score thresholds."""

    def __init__(
        self, *, method: Literal["mad", "zscore"] = "mad", k: float = 3.0
    ) -> None:
        self.method = method
        self.k = k

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            outlier_mask(np.asarray(gt), method=self.method, k=self.k)
        )

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "k": self.k}


class ReplaceOutliers(Operator):
    """Replace outliers with median, NaN, or nearest-neighbour interpolation."""

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

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        out = replace_outliers(
            np.asarray(gt), method=self.method, k=self.k, fill=self.fill
        )
        return gt.array_as_geotensor(out)

    def get_config(self) -> dict[str, Any]:
        return {"method": self.method, "k": self.k, "fill": self.fill}


class SaturationFlag(Operator):
    """Flag saturated pixels."""

    def __init__(self, *, threshold: float | None = None) -> None:
        self.threshold = threshold

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        return gt.array_as_geotensor(
            saturation_flag(np.asarray(gt), threshold=self.threshold)
        )

    def get_config(self) -> dict[str, Any]:
        return {"threshold": self.threshold}
