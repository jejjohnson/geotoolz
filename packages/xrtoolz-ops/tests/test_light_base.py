"""The base install stays without einx and scikit-learn (geostack gate B1)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest


def test_importing_xrops_does_not_import_einx_or_sklearn() -> None:
    code = (
        "import sys, xrops, xrops.interpolate, xrops.metrics, xrops.transforms, "
        "xrops.learn, xrops.utils; "
        "print(sorted(m for m in ('einx', 'sklearn') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize("module", ["sklearn.neighbors", "sklearn.decomposition"])
def test_a_missing_sklearn_names_the_extra(monkeypatch, module) -> None:
    from xrops.interpolate import Grid, kde_to_grid
    from xrops.transforms import pca

    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=r"xrtoolz-ops\[sklearn\]"):
        if module == "sklearn.neighbors":
            grid = Grid.from_bounds((-1.0, 1.0), (-1.0, 1.0), resolution=0.5)
            kde_to_grid(np.zeros(2), np.zeros(2), grid, bandwidth=0.5)
        else:
            pca("time", 2)
