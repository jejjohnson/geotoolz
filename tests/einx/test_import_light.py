"""Guard: ``import xrtoolz`` must not eagerly import einx.

einx is a core dependency, but the base package import stays light —
einx is only pulled when ``xrtoolz.einx`` is imported / used. This keeps
``import xrtoolz`` fast for callers that never touch the bridge.
"""

from __future__ import annotations

import subprocess
import sys


def test_import_xrtoolz_does_not_import_einx() -> None:
    code = (
        "import sys, xrtoolz; "
        "assert 'einx' not in sys.modules, "
        "'import xrtoolz pulled in einx; keep the bridge lazy'"
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
        "import sys, numpy as np, xarray as xr, xrtoolz.einx as xnx; "
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
