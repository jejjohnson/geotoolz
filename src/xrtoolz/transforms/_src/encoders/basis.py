"""Basis-function encoders — cyclical, Fourier, random Fourier, positional.

Per the xarray-native primitive contract
(``docs/design/xarray-native-primitives.md``), the public Layer-0
functions here take a positional :class:`xarray.DataArray` and return a
:class:`xarray.DataArray` (or a :class:`xarray.Dataset` for the
sin/cos multi-output :func:`cyclical_encode`). The feature-adding
encoders introduce a new trailing ``feature_dim`` dimension. The numpy
math is preserved verbatim as underscore-prefixed module-level kernels
so internal callers (e.g.
:func:`xrtoolz.transforms._src.encoders.coord_time.encode_time_cyclical`)
can reuse it without round-tripping through xarray.
"""

from __future__ import annotations

from collections.abc import Hashable

import einx
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray


# ---------- numpy kernels --------------------------------------------------


def _sin_cos(values: ArrayLike, period: float) -> tuple[NDArray, NDArray]:
    """Sin/cos embedding of a periodic numpy array."""
    x = 2.0 * np.pi * np.asarray(values) / period
    return np.sin(x), np.cos(x)


def _fourier_features_array(values: NDArray, num_freqs: int, scale: float) -> NDArray:
    """Deterministic Fourier features; appends a trailing feature axis."""
    freqs = (2.0 ** np.arange(num_freqs)) * scale
    angles = values[..., None] * freqs
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def _random_fourier_features_array(
    values: NDArray,
    num_features: int,
    sigma: float,
    seed: int | None,
    *,
    projected: bool,
) -> NDArray:
    """Random Fourier features (Rahimi & Recht, 2007).

    When ``projected`` is ``True`` the trailing axis of ``values`` is the
    ``d``-channel feature axis that is projected away; otherwise each
    element is treated as a scalar feature (``d = 1``).
    """
    values = np.asarray(values)
    if projected:
        d = values.shape[-1]
        x = values
    else:
        d = 1
        x = values[..., None]
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((d, num_features // 2)) / sigma
    # Contract the channel axis ``d`` into the random-feature axis ``f``.
    projection = einx.dot("... d, d f -> ... f", x, omega)  # (..., num_features // 2)
    return np.concatenate([np.sin(projection), np.cos(projection)], axis=-1)


def _positional_encoding_array(
    values: NDArray, num_freqs: int, include_input: bool
) -> NDArray:
    """NeRF-style positional encoding; appends a trailing feature axis."""
    encoded = _fourier_features_array(values, num_freqs=num_freqs, scale=np.pi)
    if include_input:
        values = np.asarray(values)[..., None]
        return np.concatenate([values, encoded], axis=-1)
    return encoded


# ---------- xarray-native primitives ---------------------------------------


def cyclical_encode(da: xr.DataArray, *, period: float) -> xr.Dataset:
    """Sin/cos embedding of a periodic variable.

    Args:
        da: Input values with period ``period``.
        period: Period length (e.g. 365.25 for day-of-year, 24 for hour
            of day, ``2 * np.pi`` for radians).

    Returns:
        Dataset with ``sin`` and ``cos`` data variables, each the same
        shape as ``da``.
    """
    x = 2.0 * np.pi * da / period
    return xr.Dataset({"sin": np.sin(x), "cos": np.cos(x)})


def fourier_features(
    da: xr.DataArray,
    *,
    num_freqs: int,
    scale: float = 1.0,
    feature_dim: str = "feature",
) -> xr.DataArray:
    """Deterministic Fourier-feature encoding.

    Adds a trailing ``feature_dim`` of length ``2 * num_freqs`` whose
    entries alternate ``sin(2^k * scale * x)`` and ``cos(2^k * scale * x)``.

    Args:
        da: Input DataArray.
        num_freqs: Number of frequency octaves to use.
        scale: Base frequency.
        feature_dim: Name of the appended feature dimension.
    """
    return xr.apply_ufunc(
        _fourier_features_array,
        da,
        input_core_dims=[[]],
        output_core_dims=[[feature_dim]],
        kwargs={"num_freqs": num_freqs, "scale": scale},
        keep_attrs=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {feature_dim: 2 * num_freqs}},
        output_dtypes=[np.promote_types(da.dtype, np.float32)],
    )


def random_fourier_features(
    da: xr.DataArray,
    *,
    num_features: int,
    sigma: float = 1.0,
    seed: int | None = None,
    input_dim: Hashable | None = None,
    feature_dim: str = "feature",
) -> xr.DataArray:
    """Random Fourier features in the style of Rahimi & Recht, 2007.

    Approximates the RBF kernel feature map. Adds a trailing
    ``feature_dim`` of length ``num_features``.

    Args:
        da: Input DataArray.
        num_features: Output feature dimension (must be even).
        sigma: Bandwidth of the underlying RBF kernel.
        seed: Seed for the RNG used to draw the random frequencies.
        input_dim: Optional existing dimension of ``da`` to treat as the
            channel/feature axis to project from. When ``None`` each
            element is a scalar feature (``d = 1``); when given, that
            dimension is replaced by ``feature_dim``.
        feature_dim: Name of the appended feature dimension.
    """
    if num_features % 2 != 0:
        raise ValueError("num_features must be even.")
    core_in = [[input_dim]] if input_dim is not None else [[]]
    return xr.apply_ufunc(
        _random_fourier_features_array,
        da,
        input_core_dims=core_in,
        output_core_dims=[[feature_dim]],
        kwargs={
            "num_features": num_features,
            "sigma": sigma,
            "seed": seed,
            "projected": input_dim is not None,
        },
        keep_attrs=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {feature_dim: num_features}},
        output_dtypes=[np.promote_types(da.dtype, np.float32)],
    )


def positional_encoding(
    da: xr.DataArray,
    *,
    num_freqs: int,
    include_input: bool = True,
    feature_dim: str = "feature",
) -> xr.DataArray:
    """NeRF-style positional encoding.

    Adds a trailing ``feature_dim`` of length
    ``(2 * num_freqs) + include_input``.

    Args:
        da: Input DataArray.
        num_freqs: Number of octave frequencies.
        include_input: If ``True``, prepend the raw input as an
            additional feature column.
        feature_dim: Name of the appended feature dimension.
    """
    out_size = 2 * num_freqs + int(include_input)
    return xr.apply_ufunc(
        _positional_encoding_array,
        da,
        input_core_dims=[[]],
        output_core_dims=[[feature_dim]],
        kwargs={"num_freqs": num_freqs, "include_input": include_input},
        keep_attrs=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {feature_dim: out_size}},
        output_dtypes=[np.promote_types(da.dtype, np.float32)],
    )


__all__ = [
    "cyclical_encode",
    "fourier_features",
    "positional_encoding",
    "random_fourier_features",
]
