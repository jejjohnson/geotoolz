"""RS-safe augmentation operators."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from affine import Affine
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

from pipekit import Operator


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


Range = tuple[float, float]
ScalarOrRange = float | Range
DEFAULT_MIN_WAVELENGTH_NM = 450.0
DEFAULT_MAX_WAVELENGTH_NM = 850.0
BRIGHT_CLOUD_PERCENTILE = 98.0
CLOUD_ALPHA_EPSILON = 1e-12


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _seed(default: int | None, override: int | None) -> int | None:
    return default if override is None else override


def _check_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


def _validate_range(value: ScalarOrRange, name: str) -> None:
    """Validate a scalar-or-range parameter at construction time.

    For tuples, requires ``lo <= hi``. For scalars, no ordering check.
    """
    if isinstance(value, tuple):
        lo, hi = value
        if hi < lo:
            raise ValueError(f"{name} range must satisfy lo <= hi; got ({lo}, {hi})")


def _validate_probability_range(value: ScalarOrRange, name: str) -> None:
    """Validate a scalar-or-range that must lie inside ``[0, 1]``.

    Tuples must satisfy ``0 <= lo <= hi <= 1``; scalars must satisfy
    ``0 <= value <= 1``.
    """
    if isinstance(value, tuple):
        lo, hi = value
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"{name} range must satisfy 0 <= lo <= hi <= 1; got ({lo}, {hi})"
            )
    else:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {value}")


def _accepts_seed_kwarg(op: Operator) -> bool:
    """Return True if ``type(op).__init__`` accepts a ``seed`` parameter."""
    try:
        sig = inspect.signature(type(op).__init__)
    except (TypeError, ValueError):
        return False
    return "seed" in sig.parameters


def _sample_uniform(rng: np.random.Generator, value: ScalarOrRange, name: str) -> float:
    if isinstance(value, tuple):
        lo, hi = value
        if hi < lo:
            raise ValueError(f"{name} range must be ordered as (min, max).")
        return float(rng.uniform(lo, hi))
    return float(value)


def _sample_nonnegative(
    rng: np.random.Generator, value: ScalarOrRange, name: str
) -> float:
    sampled = _sample_uniform(rng, value, name)
    if sampled < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return sampled


def _band_count(arr: np.ndarray) -> int:
    return int(arr.shape[0]) if arr.ndim >= 3 else 1


def _band_shape(arr: np.ndarray) -> tuple[int, ...]:
    shape = [1] * arr.ndim
    if arr.ndim >= 3:
        shape[0] = arr.shape[0]
    return tuple(shape)


def _cast_like(out: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.bool_):
        # Treat boolean arrays as masks: positive augmented values remain True.
        return (out > 0).astype(dtype, copy=False)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype.name)
        out = np.clip(out, info.min, info.max)
    return out.astype(dtype, copy=False)


def _wrap_like(gt: GeoTensor, out: np.ndarray) -> GeoTensor:
    return gt.array_as_geotensor(_cast_like(out, np.asarray(gt).dtype))


def _new_geotensor(gt: GeoTensor, out: np.ndarray, transform: Affine) -> GeoTensor:
    from georeader.geotensor import GeoTensor

    return GeoTensor(
        _cast_like(out, np.asarray(gt).dtype),
        transform=transform,
        crs=gt.crs,
        fill_value_default=gt.fill_value_default,
        attrs=gt.attrs,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(float(v) for v in value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


class Compose(Operator):
    """Apply augmentations sequentially with an optional pipeline probability.

    Per-call ``seed`` deterministically derives a fresh child seed for each
    augmentation so the same top-level seed always produces the same chain
    of inner draws. When ``seed`` is omitted, ``Compose`` uses ``self.seed``
    (set at construction) before falling back to non-deterministic entropy.

    ``get_config`` emits a JSON-safe nested description of each child via
    its ``get_config()``; ``forbid_in_yaml`` is set because the constructor
    accepts arbitrary ``Operator`` instances which a YAML loader cannot
    re-instantiate without an explicit registry.

    Examples:
        >>> import geotoolz as gz
        >>> pipe = gz.augment.Compose(
        ...     [gz.augment.RandomFlip(), gz.augment.GaussianNoise(sigma=0.01)],
        ...     p=1.0,
        ...     seed=0,
        ... )
        >>> out = pipe(patch)  # doctest: +SKIP
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        augmentations: list[Operator],
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        _check_probability(p, "p")
        self.augmentations = list(augmentations)
        self.p = p
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        if rng.random() >= self.p:
            return gt

        out = gt
        child_seeds = rng.integers(0, np.iinfo(np.int64).max, len(self.augmentations))
        for op, child_seed in zip(self.augmentations, child_seeds, strict=True):
            out = op(out, seed=int(child_seed)) if _accepts_seed_kwarg(op) else op(out)
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "augmentations": [
                {"class": type(op).__name__, "config": op.get_config()}
                for op in self.augmentations
            ],
            "p": self.p,
            "seed": self.seed,
        }


class RandomFlip(Operator):
    """Randomly flip a GeoTensor horizontally and/or vertically.

    Preserves the CRS and updates the affine ``transform`` so the output
    pixel grid still maps to the same physical extent. Each axis flips
    independently with probability ``p_horizontal`` / ``p_vertical``.

    Examples:
        >>> import geotoolz as gz
        >>> op = gz.augment.RandomFlip(p_horizontal=1.0, p_vertical=0.0, seed=0)
        >>> out = op(patch)  # doctest: +SKIP
    """

    def __init__(
        self,
        p_horizontal: float = 0.5,
        p_vertical: float = 0.5,
        seed: int | None = None,
    ) -> None:
        _check_probability(p_horizontal, "p_horizontal")
        _check_probability(p_vertical, "p_vertical")
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        arr = np.asarray(gt)
        out = arr
        transform = gt.transform

        if rng.random() < self.p_horizontal:
            out = np.flip(out, axis=-1)
            transform = (
                transform * Affine.translation(gt.width - 1, 0) * Affine.scale(-1, 1)
            )

        if rng.random() < self.p_vertical:
            out = np.flip(out, axis=-2)
            transform = (
                transform * Affine.translation(0, gt.height - 1) * Affine.scale(1, -1)
            )

        if out is arr:
            return gt
        return _new_geotensor(gt, out, transform)

    def get_config(self) -> dict[str, Any]:
        return {
            "p_horizontal": self.p_horizontal,
            "p_vertical": self.p_vertical,
            "seed": self.seed,
        }


class RandomRotate90(Operator):
    """Randomly rotate by 90, 180, or 270 degrees.

    Uses ``np.rot90`` over the trailing two axes and composes the input
    ``transform`` with the matching rigid rotation so the upper-left output
    pixel still maps to the correct world coordinate.

    Examples:
        >>> import geotoolz as gz
        >>> op = gz.augment.RandomRotate90(p=1.0, seed=0)
        >>> out = op(patch)  # doctest: +SKIP
    """

    def __init__(self, p: float = 0.75, seed: int | None = None) -> None:
        _check_probability(p, "p")
        self.p = p
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        if rng.random() >= self.p:
            return gt

        k = int(rng.integers(1, 4))
        arr = np.asarray(gt)
        out = np.rot90(arr, k=k, axes=(-2, -1))
        transform = _rot90_transform(gt.transform, gt.height, gt.width, k)
        return _new_geotensor(gt, out, transform)

    def get_config(self) -> dict[str, Any]:
        return {"p": self.p, "seed": self.seed}


def _rot90_transform(transform: Affine, height: int, width: int, k: int) -> Affine:
    k %= 4
    if k == 1:
        return transform * Affine(0, -1, width - 1, 1, 0, 0)
    if k == 2:
        return (
            transform * Affine.translation(width - 1, height - 1) * Affine.scale(-1, -1)
        )
    if k == 3:
        return transform * Affine(0, 1, 0, -1, 0, height - 1)
    return transform


class RandomCrop(Operator):
    """Randomly crop a spatial window and update the transform.

    Delegates to ``gt.isel`` so the output transform's translation reflects
    the crop origin (i.e. ``transform * (left, top)`` equals the new
    upper-left world coordinate). CRS, dtype and band metadata are
    preserved.

    Examples:
        >>> import geotoolz as gz
        >>> op = gz.augment.RandomCrop(size=(3, 4), seed=0)
        >>> out = op(patch)  # doctest: +SKIP
    """

    def __init__(self, size: tuple[int, int], seed: int | None = None) -> None:
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("size entries must be positive.")
        self.size = size
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        crop_h, crop_w = self.size
        if crop_h > gt.height or crop_w > gt.width:
            raise ValueError("size must fit within the GeoTensor spatial shape.")

        rng = _rng(_seed(self.seed, seed))
        top = int(rng.integers(0, gt.height - crop_h + 1))
        left = int(rng.integers(0, gt.width - crop_w + 1))
        return gt.isel({"y": slice(top, top + crop_h), "x": slice(left, left + crop_w)})

    def get_config(self) -> dict[str, Any]:
        return {"size": _jsonable(self.size), "seed": self.seed}


class RandomShift(Operator):
    """Randomly shift a fixed-size spatial window in pixel units."""

    def __init__(self, max_shift: tuple[int, int], seed: int | None = None) -> None:
        if max_shift[0] < 0 or max_shift[1] < 0:
            raise ValueError("max_shift entries must be non-negative.")
        self.max_shift = max_shift
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        max_y, max_x = self.max_shift
        rng = _rng(_seed(self.seed, seed))
        dy = int(rng.integers(-max_y, max_y + 1)) if max_y else 0
        dx = int(rng.integers(-max_x, max_x + 1)) if max_x else 0
        if dx == 0 and dy == 0:
            return gt
        window = Window(col_off=dx, row_off=dy, width=gt.width, height=gt.height)
        return gt.read_from_window(window, boundless=True)

    def get_config(self) -> dict[str, Any]:
        return {"max_shift": _jsonable(self.max_shift), "seed": self.seed}


class BrightnessJitter(Operator):
    """Scale reflectance values by a sampled scalar or per-band factor."""

    def __init__(
        self,
        factor: Range = (0.9, 1.1),
        per_band: bool = True,
        seed: int | None = None,
    ) -> None:
        _validate_range(factor, "factor")
        self.factor = factor
        self.per_band = per_band
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        arr = np.asarray(gt)
        if self.per_band:
            factors = rng.uniform(self.factor[0], self.factor[1], _band_count(arr))
            factors = factors.reshape(_band_shape(arr))
        else:
            factors = _sample_uniform(rng, self.factor, "factor")
        return _wrap_like(gt, arr.astype(np.float64, copy=False) * factors)

    def get_config(self) -> dict[str, Any]:
        return {
            "factor": _jsonable(self.factor),
            "per_band": self.per_band,
            "seed": self.seed,
        }


class ContrastJitter(Operator):
    """Scale deviations from the per-band spatial mean."""

    def __init__(
        self,
        factor: Range = (0.9, 1.1),
        per_band: bool = True,
        seed: int | None = None,
    ) -> None:
        _validate_range(factor, "factor")
        self.factor = factor
        self.per_band = per_band
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        arr = np.asarray(gt)
        data = arr.astype(np.float64, copy=False)
        mean = np.mean(data, axis=(-2, -1), keepdims=True)
        if self.per_band:
            factors = rng.uniform(self.factor[0], self.factor[1], _band_count(arr))
            factors = factors.reshape(_band_shape(arr))
        else:
            factors = _sample_uniform(rng, self.factor, "factor")
        return _wrap_like(gt, (data - mean) * factors + mean)

    def get_config(self) -> dict[str, Any]:
        return {
            "factor": _jsonable(self.factor),
            "per_band": self.per_band,
            "seed": self.seed,
        }


class GaussianNoise(Operator):
    """Add zero-mean Gaussian sensor noise."""

    def __init__(
        self,
        sigma: ScalarOrRange = 0.01,
        per_band: bool = True,
        seed: int | None = None,
    ) -> None:
        _validate_range(sigma, "sigma")
        self.sigma = sigma
        self.per_band = per_band
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        arr = np.asarray(gt)
        if self.per_band:
            sigmas = np.array(
                [
                    _sample_nonnegative(rng, self.sigma, "sigma")
                    for _ in range(_band_count(arr))
                ]
            ).reshape(_band_shape(arr))
        else:
            sigmas = _sample_nonnegative(rng, self.sigma, "sigma")
        noise = rng.normal(0.0, sigmas, size=arr.shape)
        return _wrap_like(gt, arr.astype(np.float64, copy=False) + noise)

    def get_config(self) -> dict[str, Any]:
        return {
            "sigma": _jsonable(self.sigma),
            "per_band": self.per_band,
            "seed": self.seed,
        }


class SpeckleNoise(Operator):
    """Apply multiplicative Gaussian speckle noise, useful for SAR."""

    def __init__(self, sigma: ScalarOrRange = 0.05, seed: int | None = None) -> None:
        _validate_range(sigma, "sigma")
        self.sigma = sigma
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        sigma = _sample_nonnegative(rng, self.sigma, "sigma")
        arr = np.asarray(gt)
        noise = rng.normal(0.0, sigma, size=arr.shape)
        return _wrap_like(gt, arr.astype(np.float64, copy=False) * (1.0 + noise))

    def get_config(self) -> dict[str, Any]:
        return {"sigma": _jsonable(self.sigma), "seed": self.seed}


class BandDropout(Operator):
    """Fill each band independently with probability ``p``."""

    def __init__(
        self, p: float = 0.1, fill: float = 0.0, seed: int | None = None
    ) -> None:
        _check_probability(p, "p")
        self.p = p
        self.fill = fill
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        arr = np.asarray(gt)
        out = np.array(arr, copy=True)
        if arr.ndim < 3:
            if rng.random() < self.p:
                out[...] = self.fill
            return _wrap_like(gt, out)

        mask = rng.random(arr.shape[0]) < self.p
        out[mask, ...] = self.fill
        return _wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"p": self.p, "fill": self.fill, "seed": self.seed}


class BandJitter(Operator):
    """Permute bands within explicitly configured groups."""

    def __init__(
        self, groups: dict[str, list[str]] | None = None, seed: int | None = None
    ) -> None:
        self.groups = groups
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        if not self.groups:
            return gt

        arr = np.asarray(gt)
        if arr.ndim < 3:
            return gt

        band_names = _band_names(gt)
        rng = _rng(_seed(self.seed, seed))
        out = np.array(arr, copy=True)
        for group in self.groups.values():
            indices = [_band_index(name, band_names, arr.shape[0]) for name in group]
            if len(indices) > 1:
                out[indices, ...] = arr[rng.permutation(indices), ...]
        return _wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"groups": _jsonable(self.groups), "seed": self.seed}


def _band_names(gt: GeoTensor) -> Sequence[Any]:
    for key in ("band_names", "bands", "band_descriptions"):
        names = gt.attrs.get(key)
        if names is not None:
            return list(names)
    raise ValueError("BandJitter requires band names in GeoTensor attrs.")


def _band_index(name: str, names: Sequence[Any], n_bands: int) -> int:
    if (isinstance(name, str) and name.isdigit()) or isinstance(name, Real):
        index = int(name)
    else:
        index = list(names).index(name)
    if not 0 <= index < n_bands:
        raise ValueError(f"Band index {index} is outside [0, {n_bands}).")
    return index


class SunAngleJitter(Operator):
    """Rescale TOA reflectance for a simulated solar-zenith change."""

    def __init__(
        self,
        delta_sza_deg: ScalarOrRange = (-5.0, 5.0),
        seed: int | None = None,
    ) -> None:
        self.delta_sza_deg = delta_sza_deg
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        delta = _sample_uniform(rng, self.delta_sza_deg, "delta_sza_deg")
        sza_value = gt.attrs.get("solar_zenith_angle", gt.attrs.get("sza_deg"))
        if sza_value is None:
            raise ValueError(
                "SunAngleJitter requires `solar_zenith_angle` or `sza_deg` in "
                "gt.attrs; got neither."
            )
        base_sza = float(sza_value)
        denom = np.cos(np.deg2rad(base_sza))
        if np.isclose(denom, 0.0):
            raise ValueError(
                f"solar zenith angle ({base_sza:.2f} deg) is too close to 90 degrees."
            )
        scale = np.cos(np.deg2rad(base_sza + delta)) / denom
        return _wrap_like(gt, np.asarray(gt).astype(np.float64, copy=False) * scale)

    def get_config(self) -> dict[str, Any]:
        return {"delta_sza_deg": _jsonable(self.delta_sza_deg), "seed": self.seed}


class AtmosphericHaze(Operator):
    """Add a sampled haze term following an inverse fourth-power spectrum.

    Wavelength metadata is interpreted as nanometers, except values below
    ``10`` are treated as micrometers and converted to nanometers.
    """

    def __init__(
        self, intensity: ScalarOrRange = (0.0, 0.05), seed: int | None = None
    ) -> None:
        self.intensity = intensity
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        intensity = _sample_nonnegative(rng, self.intensity, "intensity")
        if intensity == 0.0:
            return gt

        arr = np.asarray(gt)
        weights = _spectral_weights(gt, _band_count(arr)).reshape(_band_shape(arr))
        return _wrap_like(gt, arr.astype(np.float64, copy=False) + intensity * weights)

    def get_config(self) -> dict[str, Any]:
        return {"intensity": _jsonable(self.intensity), "seed": self.seed}


def _spectral_weights(gt: GeoTensor, n_bands: int) -> np.ndarray:
    wavelengths = gt.attrs.get("wavelengths_nm", gt.attrs.get("wavelengths"))
    if wavelengths is None:
        wavelengths = np.linspace(
            DEFAULT_MIN_WAVELENGTH_NM, DEFAULT_MAX_WAVELENGTH_NM, n_bands
        )
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if wavelengths.size != n_bands:
        raise ValueError("wavelength metadata must have one value per band.")
    if np.nanmax(wavelengths) < 10.0:
        # Convert micrometers to nanometers; RS visible/NIR values are never <10 nm.
        wavelengths = wavelengths * 1000.0
    weights = 1.0 / np.power(wavelengths, 4)
    return weights / np.nanmax(weights)


class SimulatedClouds(Operator):
    """Overlay a smooth synthetic cloud field onto reflectance imagery."""

    def __init__(
        self,
        coverage: ScalarOrRange = (0.0, 0.3),
        feather: int = 5,
        seed: int | None = None,
    ) -> None:
        if feather < 0:
            raise ValueError("feather must be non-negative.")
        _validate_probability_range(coverage, "coverage")
        self.coverage = coverage
        self.feather = feather
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        coverage = _sample_uniform(rng, self.coverage, "coverage")
        if coverage == 0.0:
            return gt

        arr = np.asarray(gt)
        field = rng.normal(size=arr.shape[-2:])
        if self.feather:
            field = gaussian_filter(field, sigma=self.feather, mode="reflect")
        field = (field - field.min()) / (np.ptp(field) + np.finfo(np.float64).eps)
        threshold = np.quantile(field, 1.0 - coverage)
        alpha = np.clip(
            (field - threshold) / (field.max() - threshold + CLOUD_ALPHA_EPSILON),
            0,
            1,
        )
        alpha = alpha.reshape((1,) * (arr.ndim - 2) + alpha.shape)
        # Assume [0, 1] reflectance if max <= 1; otherwise approximate bright clouds.
        cloud_value = (
            1.0
            if np.nanmax(arr) <= 1.0
            else np.nanpercentile(arr, BRIGHT_CLOUD_PERCENTILE)
        )
        out = arr.astype(np.float64, copy=False) * (1.0 - alpha) + cloud_value * alpha
        return _wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "coverage": _jsonable(self.coverage),
            "feather": self.feather,
            "seed": self.seed,
        }


class CutMix(Operator):
    """Paste a random rectangle from a pool donor that shares geo metadata.

    Donors must match the input ``shape``, ``crs`` and pixel resolution
    (i.e. the absolute components of ``transform.a``/``transform.e``). This
    guards against blending samples drawn at different scales or in
    different reference frames — which would produce a geographically
    incoherent result even though the pixel grids align.

    ``forbid_in_yaml`` is set because ``pool`` holds live ``GeoTensor``
    objects that cannot be round-tripped through YAML. ``get_config`` emits
    only the pool length for debug visibility.

    Examples:
        >>> import geotoolz as gz
        >>> op = gz.augment.CutMix(pool=[donor], p=1.0, seed=0)
        >>> out = op(patch)  # doctest: +SKIP
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self, pool: list[GeoTensor], p: float = 0.5, seed: int | None = None
    ) -> None:
        _check_probability(p, "p")
        self.pool = list(pool)
        self.p = p
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> GeoTensor:
        rng = _rng(_seed(self.seed, seed))
        if not self.pool or rng.random() >= self.p:
            return gt

        donor = self.pool[int(rng.integers(0, len(self.pool)))]
        arr = np.asarray(gt)
        donor_arr = np.asarray(donor)
        if donor_arr.shape != arr.shape:
            raise ValueError("CutMix pool GeoTensors must match the input shape.")
        if donor.crs != gt.crs:
            raise ValueError("CutMix donor CRS must match the input CRS.")
        if not np.allclose(
            (abs(donor.transform.a), abs(donor.transform.e)),
            (abs(gt.transform.a), abs(gt.transform.e)),
        ):
            raise ValueError(
                "CutMix donor pixel resolution must match the input resolution."
            )

        cut_h = int(rng.integers(1, gt.height + 1))
        cut_w = int(rng.integers(1, gt.width + 1))
        top = int(rng.integers(0, gt.height - cut_h + 1))
        left = int(rng.integers(0, gt.width - cut_w + 1))

        out = np.array(arr, copy=True)
        out[..., top : top + cut_h, left : left + cut_w] = donor_arr[
            ..., top : top + cut_h, left : left + cut_w
        ]
        return _wrap_like(gt, out)

    def get_config(self) -> dict[str, Any]:
        return {"pool_size": len(self.pool), "p": self.p, "seed": self.seed}
