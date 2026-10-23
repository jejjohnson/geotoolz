"""Spike fixtures: 64-bit JAX for every test, and a Sentinel-2-like scene."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest


#: 20 m north-up grid in UTM 13N, the Delaware Basin chip of the design pages.
T0 = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
CRS = "EPSG:32613"


@pytest.fixture(autouse=True)
def x64() -> Iterator[None]:
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
