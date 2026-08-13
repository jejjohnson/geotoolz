"""Guard: ``import xreinx`` must not eagerly import einx.

einx is a hard dependency of this package, but the bridge import stays
light — einx is only pulled on the first pattern call. This keeps
``import xreinx`` fast for callers that build pipelines up front.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


# Every test here launches a fresh interpreter; keep them out of the fast tier.
pytestmark = pytest.mark.slow


def test_import_xreinx_does_not_import_einx() -> None:
    code = (
        "import sys, xreinx; "
        "assert 'einx' not in sys.modules, "
        "'import xreinx pulled in einx; keep the bridge lazy'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_using_bridge_pulls_einx() -> None:
    code = (
        "import sys, numpy as np, xarray as xr, xreinx as xnx; "
        "assert 'einx' not in sys.modules, 'einx imported at module load'; "
        "xnx.rearrange('a -> a', xr.DataArray(np.arange(3.0), dims='a')); "
        "assert 'einx' in sys.modules, 'einx not imported after a call'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
