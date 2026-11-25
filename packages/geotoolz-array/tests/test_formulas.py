"""Formulas of ``pipekit_array.arrays`` on carriers: nodata stays nodata.

``arrays`` needs array-api-compat, which geoarray does not.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pytest
from pipekit_ops import to_backend
from test_interop import CARRIERS

from geoarray import GeoArray


pytest.importorskip("pipekit_array.arrays", exc_type=ImportError)
from pipekit_array.arrays import floating_inputs
from pipekit_array.kernels import elementwise


@elementwise()
def _half(x: Any) -> Any:
    _, (x,) = floating_inputs(x)
    return x / 2


@pytest.mark.parametrize("make", list(CARRIERS.values()), ids=list(CARRIERS))
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_a_formula_keeps_nodata(make: Any, backend: str, x64: ModuleType) -> None:
    given = make()
    values = np.array(given.data, np.float32)
    values.flat[0], values.flat[1] = given.fill_value, np.nan
    g = to_backend(given.with_data(values), backend)
    out = _half(g)
    assert out.fill_value == given.fill_value and out.backend == backend
    missing = (values == given.fill_value) | np.isnan(values)
    assert missing.sum() >= 2
    want = np.where(missing, given.fill_value, values / 2)
    np.testing.assert_array_equal(np.asarray(out), want)


def test_a_formula_of_two_carriers_keeps_the_nodata_of_both() -> None:
    @elementwise()
    def density(p: Any, t: Any) -> Any:
        _, (p, t) = floating_inputs(p, t)
        return p / (287.0 * t)

    grid = {"transform": (20, 0, 0, 0, -20, 0), "crs": 32613}
    pressure = np.full((2, 3, 3), 1e5, "f4")
    p = GeoArray(pressure, dims="band y x", fill_value=-1.0, **grid)
    p.data[0, 0, 0] = -1.0
    t = GeoArray(np.full((3, 3), 300.0, "f4"), dims="y x", fill_value=-1.0, **grid)
    t.data[2, 2] = -1.0
    out = density(p, t)
    want = np.zeros((2, 3, 3), bool)
    want[0, 0, 0] = want[:, 2, 2] = True
    np.testing.assert_array_equal(out.data == -1.0, want)


def test_a_result_that_cannot_hold_the_fill_is_marked_with_nan() -> None:
    g = GeoArray(
        np.array([[1, 2], [3, 255]], np.int32),
        (20, 0, 0, 0, -20, 0),
        32613,
        dims="y x",
        fill_value=255,
    )
    out = _half(g)
    marked = out.data == 255 if out.fill_value == 255 else np.isnan(out.data)
    np.testing.assert_array_equal(marked, [[False, False], [False, True]])
