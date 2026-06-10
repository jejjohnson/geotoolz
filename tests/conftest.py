"""Shared pytest fixtures, including the numpy/dask backend matrix.

The ``array_backend`` / ``maybe_chunk`` pair lets a single test body run
against both an eager numpy-backed xarray object and a lazy dask-backed one,
so dask compatibility is exercised without duplicating test logic. See
``tests/test_dask_compat.py`` for the repo-wide operator parity sweep.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, cast

import pytest
import xarray as xr


ArrayBackend = Literal["numpy", "dask"]
MaybeChunk = Callable[
    [xr.DataArray | xr.Dataset, ArrayBackend, Mapping[str, int] | None],
    xr.DataArray | xr.Dataset,
]


@pytest.fixture(params=["numpy", "dask"])
def array_backend(request: pytest.FixtureRequest) -> ArrayBackend:
    """Parametrize a test over the ``"numpy"`` and ``"dask"`` backends."""
    return cast(ArrayBackend, request.param)


def _maybe_chunk(
    obj: xr.DataArray | xr.Dataset,
    backend: ArrayBackend,
    chunks: Mapping[str, int] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Chunk ``obj`` for the ``"dask"`` backend; return it as-is otherwise."""
    if backend == "dask":
        pytest.importorskip("dask.array")
        return obj.chunk(chunks or {})
    return obj


@pytest.fixture
def maybe_chunk() -> MaybeChunk:
    """Return a helper that chunks an xarray object for the dask backend."""
    return _maybe_chunk
