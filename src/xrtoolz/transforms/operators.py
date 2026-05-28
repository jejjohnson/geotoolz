"""Layer-1 ``Operator`` wrappers around :mod:`xrtoolz.transforms._src`.

The stateless transforms (Fourier / DCT / wavelet) compose cleanly into
``Sequential`` / ``Graph`` pipelines. The stateful estimators
(PCA / EOF / …) are intentionally **not** wrapped as ``Operator``
subclasses — they need ``.fit()`` before ``.transform()``, which the
stateless ``Operator.__call__`` contract cannot express. Use the
factory functions from :mod:`xrtoolz.transforms` directly for those:
``pca(...)``, ``eof(...)``, etc.

The Operators here all accept an :class:`xr.Dataset`, pull the named
variable, run the underlying ``DataArray``-first function, and
re-wrap the result in a Dataset under the new spectral name.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from xrtoolz._operator import Operator
from xrtoolz.transforms._src import (
    dct as _dct,
    fourier as _fourier,
    wavelet as _wavelet,
)
from xrtoolz.transforms._src.encoders import (
    basis as _basis,
    coord_time as _coord_time,
)


def _datetime_to_jsonable(value: str | np.datetime64 | None) -> str | None:
    """Stringify ``np.datetime64`` for JSON-serializable ``get_config``."""
    if value is None or isinstance(value, str):
        return value
    return str(value)


# ---------- Fourier --------------------------------------------------------


class PowerSpectrum(Operator):
    """Power spectrum of ``ds[variable]``. Set ``isotropic=True`` for the
    radial (2-D) variant."""

    def __init__(
        self,
        variable: str,
        dim: str | Sequence[str],
        *,
        isotropic: bool = False,
        **kwargs: Any,
    ) -> None:
        self.variable = variable
        self.dim = dim
        self.isotropic = isotropic
        self.kwargs = dict(kwargs)

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        out = _fourier.power_spectrum(
            ds[self.variable], self.dim, isotropic=self.isotropic, **self.kwargs
        )
        return out.to_dataset()

    def get_config(self) -> dict[str, Any]:
        dim = list(self.dim) if not isinstance(self.dim, str) else self.dim
        return {
            "variable": self.variable,
            "dim": dim,
            "isotropic": self.isotropic,
            **self.kwargs,
        }


class CrossSpectrum(Operator):
    """Cross-power spectrum of ``ds[var_a]`` and ``ds[var_b]``."""

    def __init__(
        self,
        var_a: str,
        var_b: str,
        dim: str | Sequence[str],
        **kwargs: Any,
    ) -> None:
        self.var_a = var_a
        self.var_b = var_b
        self.dim = dim
        self.kwargs = dict(kwargs)

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        out = _fourier.cross_spectrum(
            ds[self.var_a], ds[self.var_b], self.dim, **self.kwargs
        )
        return out.to_dataset()

    def get_config(self) -> dict[str, Any]:
        dim = list(self.dim) if not isinstance(self.dim, str) else self.dim
        return {
            "var_a": self.var_a,
            "var_b": self.var_b,
            "dim": dim,
            **self.kwargs,
        }


class Coherence(Operator):
    """Magnitude-squared coherence of two variables."""

    def __init__(
        self,
        var_a: str,
        var_b: str,
        dim: str | Sequence[str],
        **kwargs: Any,
    ) -> None:
        self.var_a = var_a
        self.var_b = var_b
        self.dim = dim
        self.kwargs = dict(kwargs)

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        out = _fourier.coherence(
            ds[self.var_a], ds[self.var_b], self.dim, **self.kwargs
        )
        return out.to_dataset()

    def get_config(self) -> dict[str, Any]:
        dim = list(self.dim) if not isinstance(self.dim, str) else self.dim
        return {
            "var_a": self.var_a,
            "var_b": self.var_b,
            "dim": dim,
            **self.kwargs,
        }


class KESpectralFlux(Operator):
    """Kinetic-energy spectral flux from ``ds[u]`` and ``ds[v]``.

    Args:
        u: Name of the zonal velocity variable in the input Dataset.
        v: Name of the meridional velocity variable in the input Dataset.
        dim: Two spatial dimensions to Fourier transform.
        window: Optional Fourier window.
        detrend: Optional Fourier detrending mode.
        avg_dims: Optional non-spectral dimensions to average.
        return_2d: If ``True``, include ``transfer_2d`` in the output Dataset.

    Returns:
        Dataset with ``transfer`` and ``flux`` variables, plus ``transfer_2d``
        when requested.

    Examples:
        >>> op = KESpectralFlux("u", "v", ("x", "y"), avg_dims="time")
        >>> flux_ds = op(ds)
    """

    def __init__(
        self,
        u: str,
        v: str,
        dim: Sequence[str],
        *,
        window: str | None = "tukey",
        detrend: str | None = "linear",
        avg_dims: str | Sequence[str] | None = None,
        return_2d: bool = False,
    ) -> None:
        self.u = u
        self.v = v
        self.dim = dim
        self.window = window
        self.detrend = detrend
        self.avg_dims = avg_dims
        self.return_2d = return_2d

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return _fourier.ke_spectral_flux(
            ds[self.u],
            ds[self.v],
            dim=self.dim,
            window=self.window,
            detrend=self.detrend,
            avg_dims=self.avg_dims,
            return_2d=self.return_2d,
        )

    def get_config(self) -> dict[str, Any]:
        dim = list(self.dim)
        avg_dims = (
            list(self.avg_dims)
            if self.avg_dims is not None and not isinstance(self.avg_dims, str)
            else self.avg_dims
        )
        return {
            "u": self.u,
            "v": self.v,
            "dim": dim,
            "window": self.window,
            "detrend": self.detrend,
            "avg_dims": avg_dims,
            "return_2d": self.return_2d,
        }


class EnstrophySpectralFlux(Operator):
    """Enstrophy spectral flux from ``ds[u]`` and ``ds[v]``.

    Args:
        u: Name of the zonal velocity variable in the input Dataset.
        v: Name of the meridional velocity variable in the input Dataset.
        dim: Two spatial dimensions to Fourier transform.
        window: Optional Fourier window.
        detrend: Optional Fourier detrending mode.
        avg_dims: Optional non-spectral dimensions to average.
        return_2d: If ``True``, include ``transfer_2d`` in the output Dataset.

    Returns:
        Dataset with vorticity-based enstrophy ``transfer`` and ``flux``
        variables, plus ``transfer_2d`` when requested.

    Examples:
        >>> op = EnstrophySpectralFlux("u", "v", ("x", "y"))
        >>> flux_ds = op(ds)
    """

    def __init__(
        self,
        u: str,
        v: str,
        dim: Sequence[str],
        *,
        window: str | None = "tukey",
        detrend: str | None = "linear",
        avg_dims: str | Sequence[str] | None = None,
        return_2d: bool = False,
    ) -> None:
        self.u = u
        self.v = v
        self.dim = dim
        self.window = window
        self.detrend = detrend
        self.avg_dims = avg_dims
        self.return_2d = return_2d

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return _fourier.enstrophy_spectral_flux(
            ds[self.u],
            ds[self.v],
            dim=self.dim,
            window=self.window,
            detrend=self.detrend,
            avg_dims=self.avg_dims,
            return_2d=self.return_2d,
        )

    def get_config(self) -> dict[str, Any]:
        dim = list(self.dim)
        avg_dims = (
            list(self.avg_dims)
            if self.avg_dims is not None and not isinstance(self.avg_dims, str)
            else self.avg_dims
        )
        return {
            "u": self.u,
            "v": self.v,
            "dim": dim,
            "window": self.window,
            "detrend": self.detrend,
            "avg_dims": avg_dims,
            "return_2d": self.return_2d,
        }


class RotarySpectrum(Operator):
    """Rotary power spectrum from ``ds[u]`` / ``ds[v]`` along ``dim``.

    The operator selects the named velocity variables from the input
    Dataset and calls :func:`rotary_spectrum` on the resulting
    DataArrays.

    Args:
        u: Name of the zonal velocity variable in the input Dataset.
        v: Name of the meridional velocity variable in the input Dataset.
        dim: Dimension to Fourier transform.
        avg_dims: Optional dimension or dimensions to average in each
            output.
    """

    def __init__(
        self,
        u: str,
        v: str,
        dim: str,
        *,
        avg_dims: str | Sequence[str] | None = None,
    ) -> None:
        self.u = u
        self.v = v
        self.dim = dim
        self.avg_dims = avg_dims

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return _fourier.rotary_spectrum(
            ds[self.u],
            ds[self.v],
            dim=self.dim,
            avg_dims=self.avg_dims,
        )

    def get_config(self) -> dict[str, Any]:
        avg_dims = (
            list(self.avg_dims)
            if self.avg_dims is not None and not isinstance(self.avg_dims, str)
            else self.avg_dims
        )
        return {
            "u": self.u,
            "v": self.v,
            "dim": self.dim,
            "avg_dims": avg_dims,
        }


class STFT(Operator):
    """Short-time Fourier transform of ``ds[variable]`` along ``dim``."""

    def __init__(
        self,
        variable: str,
        dim: str,
        *,
        window_size: int,
        hop: int | None = None,
        window: str = "tukey",
        detrend: str | None = "linear",
    ) -> None:
        self.variable = variable
        self.dim = dim
        self.window_size = window_size
        self.hop = hop
        self.window = window
        self.detrend = detrend

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        out = _fourier.stft(
            ds[self.variable],
            self.dim,
            window_size=self.window_size,
            hop=self.hop,
            window=self.window,
            detrend=self.detrend,
        )
        return out.to_dataset()

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dim": self.dim,
            "window_size": self.window_size,
            "hop": self.hop,
            "window": self.window,
            "detrend": self.detrend,
        }


# ---------- DCT ------------------------------------------------------------


class DCT(Operator):
    """Discrete Cosine Transform of ``ds[variable]`` along ``dim``."""

    def __init__(
        self,
        variable: str,
        dim: str,
        *,
        type: int = 2,
        norm: str | None = "ortho",
    ) -> None:
        self.variable = variable
        self.dim = dim
        self.type = type
        self.norm = norm

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        out = _dct.dct(ds[self.variable], self.dim, type=self.type, norm=self.norm)
        return out.to_dataset()

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dim": self.dim,
            "type": self.type,
            "norm": self.norm,
        }


class DST(Operator):
    """Discrete Sine Transform of ``ds[variable]`` along ``dim``."""

    def __init__(
        self,
        variable: str,
        dim: str,
        *,
        type: int = 2,
        norm: str | None = "ortho",
    ) -> None:
        self.variable = variable
        self.dim = dim
        self.type = type
        self.norm = norm

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        out = _dct.dst(ds[self.variable], self.dim, type=self.type, norm=self.norm)
        return out.to_dataset()

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dim": self.dim,
            "type": self.type,
            "norm": self.norm,
        }


# ---------- Wavelet --------------------------------------------------------


class CWT(Operator):
    """Continuous Wavelet Transform of ``ds[variable]`` along ``dim``."""

    def __init__(
        self,
        variable: str,
        dim: str,
        *,
        scales: Sequence[float],
        wavelet: str = "morl",
        sampling_period: float = 1.0,
    ) -> None:
        self.variable = variable
        self.dim = dim
        self.scales = list(scales)
        self.wavelet = wavelet
        self.sampling_period = sampling_period

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        out = _wavelet.cwt(
            ds[self.variable],
            self.dim,
            scales=self.scales,
            wavelet=self.wavelet,
            sampling_period=self.sampling_period,
        )
        return out.to_dataset()

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "dim": self.dim,
            "scales": list(self.scales),
            "wavelet": self.wavelet,
            "sampling_period": self.sampling_period,
        }


# ---------- Encoders -------------------------------------------------------


class CyclicalEncode(Operator):
    """Sin/cos embedding of a periodic variable.

    Reads ``ds[variable]``, calls :func:`cyclical_encode`, and attaches
    two new data variables ``{variable}_sin`` and ``{variable}_cos`` to
    the dataset (preserving all other variables).
    """

    def __init__(self, variable: str, period: float) -> None:
        self.variable = variable
        self.period = float(period)

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        encoded = _basis.cyclical_encode(ds[self.variable], period=self.period)
        return ds.assign(
            {
                f"{self.variable}_sin": encoded["sin"],
                f"{self.variable}_cos": encoded["cos"],
            }
        )

    def get_config(self) -> dict[str, Any]:
        return {"variable": self.variable, "period": self.period}


class FourierFeatures(Operator):
    """Deterministic Fourier-feature encoding of ``ds[variable]``.

    Output is a new variable with one extra trailing ``feature_dim``
    axis of length ``2 * num_freqs``.
    """

    def __init__(
        self,
        variable: str,
        num_freqs: int,
        scale: float = 1.0,
        *,
        output_name: str | None = None,
        feature_dim: str = "feature",
    ) -> None:
        self.variable = variable
        self.num_freqs = int(num_freqs)
        self.scale = float(scale)
        self.output_name = output_name
        self.feature_dim = feature_dim

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        encoded = _basis.fourier_features(
            ds[self.variable],
            num_freqs=self.num_freqs,
            scale=self.scale,
            feature_dim=self.feature_dim,
        )
        name = self.output_name or f"{self.variable}_fourier"
        return ds.assign({name: encoded})

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "num_freqs": self.num_freqs,
            "scale": self.scale,
            "output_name": self.output_name,
            "feature_dim": self.feature_dim,
        }


class RandomFourierFeatures(Operator):
    """Random Fourier features (Rahimi & Recht, 2007) of ``ds[variable]``."""

    def __init__(
        self,
        variable: str,
        num_features: int,
        sigma: float = 1.0,
        seed: int | None = None,
        *,
        output_name: str | None = None,
        feature_dim: str = "feature",
    ) -> None:
        self.variable = variable
        self.num_features = int(num_features)
        self.sigma = float(sigma)
        self.seed = seed
        self.output_name = output_name
        self.feature_dim = feature_dim

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        da = ds[self.variable]
        if da.ndim == 0:
            raise ValueError(
                "RandomFourierFeatures requires a non-scalar input variable; "
                f"ds[{self.variable!r}] has ndim=0."
            )
        # 1-D inputs append a feature axis; ≥ 2-D vector inputs project the
        # trailing axis away via a (d, num_features/2) matrix, replacing it
        # with the feature axis.
        input_dim = None if da.ndim == 1 else da.dims[-1]
        encoded = _basis.random_fourier_features(
            da,
            num_features=self.num_features,
            sigma=self.sigma,
            seed=self.seed,
            input_dim=input_dim,
            feature_dim=self.feature_dim,
        )
        name = self.output_name or f"{self.variable}_rff"
        return ds.assign({name: encoded})

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "num_features": self.num_features,
            "sigma": self.sigma,
            "seed": self.seed,
            "output_name": self.output_name,
            "feature_dim": self.feature_dim,
        }


class PositionalEncoding(Operator):
    """NeRF-style positional encoding of ``ds[variable]``."""

    def __init__(
        self,
        variable: str,
        num_freqs: int,
        include_input: bool = True,
        *,
        output_name: str | None = None,
        feature_dim: str = "feature",
    ) -> None:
        self.variable = variable
        self.num_freqs = int(num_freqs)
        self.include_input = bool(include_input)
        self.output_name = output_name
        self.feature_dim = feature_dim

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        encoded = _basis.positional_encoding(
            ds[self.variable],
            num_freqs=self.num_freqs,
            include_input=self.include_input,
            feature_dim=self.feature_dim,
        )
        name = self.output_name or f"{self.variable}_posenc"
        return ds.assign({name: encoded})

    def get_config(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "num_freqs": self.num_freqs,
            "include_input": self.include_input,
            "output_name": self.output_name,
            "feature_dim": self.feature_dim,
        }


class EncodeTimeCyclical(Operator):
    """Wrap :func:`xrtoolz.transforms.encoders.encode_time_cyclical`."""

    def __init__(
        self,
        components: Sequence[str] = ("dayofyear", "hour"),
        time: str = "time",
    ) -> None:
        self.components = tuple(components)
        self.time = time

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        encoded = _coord_time.encode_time_cyclical(
            ds[self.time], components=self.components
        )
        # The primitive returns a Dataset of sin/cos DataArrays carrying
        # ``ds[self.time].dims``; pass them through as DataArrays so
        # ``assign_coords`` reads their dims directly (this handles
        # multi-dim / auxiliary time coords too).
        return ds.assign_coords({name: encoded[name] for name in encoded.data_vars})

    def get_config(self) -> dict[str, Any]:
        return {"components": list(self.components), "time": self.time}


class EncodeTimeOrdinal(Operator):
    """Wrap :func:`xrtoolz.transforms.encoders.encode_time_ordinal`."""

    def __init__(
        self,
        reference_date: str | np.datetime64 | None = None,
        time: str = "time",
        unit: str = "D",
    ) -> None:
        self.reference_date = reference_date
        self.time = time
        self.unit = unit

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        ordinal = _coord_time.encode_time_ordinal(
            ds[self.time],
            reference_date=self.reference_date,
            unit=self.unit,
        )
        # Pass the DataArray directly so its dims (which may be
        # multi-dim or auxiliary) are honoured rather than hardcoding
        # ``(self.time,)``.
        return ds.assign_coords({f"{self.time}_ordinal": ordinal})

    def get_config(self) -> dict[str, Any]:
        return {
            "reference_date": _datetime_to_jsonable(self.reference_date),
            "time": self.time,
            "unit": self.unit,
        }


class TimeRescale(Operator):
    """Wrap :func:`xrtoolz.transforms.encoders.time_rescale`."""

    def __init__(
        self,
        freq_dt: float = 1.0,
        freq_unit: str = "s",
        t0: str | np.datetime64 | None = None,
        time: str = "time",
    ) -> None:
        self.freq_dt = float(freq_dt)
        self.freq_unit = freq_unit
        self.t0 = t0
        self.time = time

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        rescaled = _coord_time.time_rescale(
            ds[self.time],
            freq_dt=self.freq_dt,
            freq_unit=self.freq_unit,
            t0=self.t0,
        )
        # Replace the time coord values while keeping its existing dims —
        # ``ds[self.time].dims`` covers the 1-D dimension-coord case as
        # well as multi-dim / auxiliary time variables.
        time_dims = ds[self.time].dims
        ds = ds.assign_coords({self.time: (time_dims, rescaled.values)})
        ds[self.time].attrs.update(dict(rescaled.attrs))
        return ds

    def get_config(self) -> dict[str, Any]:
        return {
            "freq_dt": self.freq_dt,
            "freq_unit": self.freq_unit,
            "t0": _datetime_to_jsonable(self.t0),
            "time": self.time,
        }


class TimeUnrescale(Operator):
    """Wrap :func:`xrtoolz.transforms.encoders.time_unrescale`."""

    def __init__(self, time: str = "time") -> None:
        self.time = time

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        restored = _coord_time.time_unrescale(ds[self.time])
        # Preserve the original time variable's dimensionality (handles
        # multi-dim / auxiliary time coords) rather than hardcoding
        # ``(self.time,)``.
        time_dims = ds[self.time].dims
        ds = ds.assign_coords({self.time: (time_dims, restored.values)})
        ds[self.time].attrs = {}
        return ds

    def get_config(self) -> dict[str, Any]:
        return {"time": self.time}


__all__ = [
    "CWT",
    "DCT",
    "DST",
    "STFT",
    "Coherence",
    "CrossSpectrum",
    "CyclicalEncode",
    "EncodeTimeCyclical",
    "EncodeTimeOrdinal",
    "EnstrophySpectralFlux",
    "FourierFeatures",
    "KESpectralFlux",
    "PositionalEncoding",
    "PowerSpectrum",
    "RandomFourierFeatures",
    "RotarySpectrum",
    "TimeRescale",
    "TimeUnrescale",
]
