"""cartopy is an optional ``[maps]`` extra: viz must import and run without it."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from xrtoolz.viz import make_axes
from xrtoolz.viz.validation import SpatialMapPanel


@pytest.fixture
def no_cartopy(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ``None`` entry in sys.modules makes ``import cartopy...`` raise ImportError.
    for name in [m for m in sys.modules if m.split(".")[0] == "cartopy"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "cartopy", None)
    monkeypatch.setitem(sys.modules, "cartopy.crs", None)


def _snapshot() -> xr.DataArray:
    lat = np.linspace(30, 40, 5)
    lon = np.linspace(-70, -60, 6)
    return xr.DataArray(
        np.zeros((5, 6)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    )


def test_viz_imports_without_cartopy() -> None:
    # Fresh interpreter so already-imported modules can't mask a top-level import.
    code = textwrap.dedent(
        """
        import sys

        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "cartopy":
                    raise ImportError("blocked")

        sys.meta_path.insert(0, _Block())
        import xrtoolz.viz, xrtoolz.viz.validation, xrtoolz.metrics, xrtoolz.ocn
        assert not any(m.split(".")[0] == "cartopy" for m in sys.modules)
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_plain_axes_need_no_cartopy(no_cartopy: None) -> None:
    fig, _ = make_axes(None)
    plt.close(fig)
    fig = SpatialMapPanel()(_snapshot())
    plt.close(fig)


def test_make_axes_projection_raises_without_cartopy(no_cartopy: None) -> None:
    with pytest.raises(ImportError, match=r"xrtoolz\[maps\]"):
        make_axes("gulf_stream")
    plt.close("all")


def test_spatial_panel_projection_raises_without_cartopy(no_cartopy: None) -> None:
    with pytest.raises(ImportError, match=r"xrtoolz\[maps\]"):
        SpatialMapPanel(projection="PlateCarree")(_snapshot())
    plt.close("all")
