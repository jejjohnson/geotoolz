"""Shared fixtures for the geopatcher test suite."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest
from _helpers import make_raster_field

from geopatcher import RasterField


class RasterFieldFactory(Protocol):
    def __call__(
        self,
        size: int = ...,
        *,
        dtype: type = ...,
        crs: str = ...,
    ) -> RasterField: ...


@pytest.fixture
def raster_field_factory() -> RasterFieldFactory:
    """Factory for the standard NxN arange field over an identity transform.

    ``raster_field_factory(32)`` reproduces bit-for-bit the local fixture
    previously duplicated per module (float32 arange ramp, identity
    transform, EPSG:32630).
    """
    return make_raster_field


@pytest.fixture
def field() -> RasterField:
    """The most common shared field: 64x64 float32 arange ramp."""
    return make_raster_field(64, dtype=np.float32)
