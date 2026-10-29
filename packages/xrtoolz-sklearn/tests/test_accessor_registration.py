"""Regression guard for the ``.sklearn`` accessor registration side effect.

The accessors must resolve after a bare ``import xrsklearn``. The check
runs in a fresh interpreter so the registration can't leak in from
another test's imports. (``xrops`` no longer imports this package for
the side effect: import :mod:`xrsklearn` to register the accessors.)
"""

from __future__ import annotations

import subprocess
import sys

import pytest


# Every test here launches a fresh interpreter; keep them out of the fast tier.
pytestmark = pytest.mark.slow


def _run(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_import_xrsklearn_registers_accessors() -> None:
    _run(
        "import numpy as np, xarray as xr, xrsklearn; "
        "da = xr.DataArray(np.arange(4.0), dims='time'); "
        "assert hasattr(da, 'sklearn'); "
        "assert hasattr(xr.Dataset({'v': da}), 'sklearn')"
    )
