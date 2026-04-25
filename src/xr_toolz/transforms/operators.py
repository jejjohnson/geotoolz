"""Layer-1 ``Operator`` wrappers around :mod:`xr_toolz.transforms._src`.

The stateless transforms (Fourier / DCT / wavelet) compose cleanly into
``Sequential`` / ``Graph`` pipelines. The stateful estimators
(PCA / EOF / …) are intentionally **not** wrapped as ``Operator``
subclasses — they need ``.fit()`` before ``.transform()``, which the
stateless ``Operator.__call__`` contract cannot express. Use the
factory functions from :mod:`xr_toolz.transforms` directly for those:
``pca(...)``, ``eof(...)``, etc.

The Operators here all accept an :class:`xr.Dataset`, pull the named
variable, run the underlying ``DataArray``-first function, and
re-wrap the result in a Dataset under the new spectral name.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import xarray as xr

from xr_toolz.core import Operator
from xr_toolz.transforms._src import (
    dct as _dct,
    fourier as _fourier,
    wavelet as _wavelet,
)


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


__all__ = [
    "CWT",
    "DCT",
    "DST",
    "STFT",
    "Coherence",
    "CrossSpectrum",
    "PowerSpectrum",
]
