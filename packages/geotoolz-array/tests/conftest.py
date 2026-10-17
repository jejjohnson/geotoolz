"""Shared fixtures: a small Sentinel-2-like chip and JAX's 64-bit switch."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import numpy as np
import pytest

from geoarray import GeoArray


#: A 20 m north-up grid in UTM 13N (EPSG:32613), the Delaware Basin chip.
TRANSFORM = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
CRS = "EPSG:32613"


@pytest.fixture
def chip() -> GeoArray:
    """A (2, 8, 8) float32 NumPy carrier with a NaN fill and band names."""
    rng = np.random.default_rng(0)
    data = rng.random((2, 8, 8)).astype(np.float32)
    return GeoArray(
        data,
        TRANSFORM,
        CRS,
        fill_value=np.nan,
        attrs={"band_names": ["B11", "B12"], "wavelengths": np.array([1610.0, 2190.0])},
    )


@pytest.fixture
def x64() -> Iterator[ModuleType]:
    """JAX with ``jax_enable_x64`` on for one test, restored afterwards."""
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def no_x64() -> Iterator[ModuleType]:
    """JAX with ``jax_enable_x64`` off for one test, restored afterwards."""
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield jax
    jax.config.update("jax_enable_x64", previous)
