"""The slim base install (#538): `import geoops` needs none of the heavy extras.

GDAL (rasterio), the vector stack (geopandas, shapely), scikit-image and
georeader sit behind ``[raster]``, ``[vector]``, ``[morph]`` and ``[sensors]``;
an operator imports them when it runs, and says which extra is missing.

Each check runs in a fresh interpreter: modules other tests imported cannot be
un-imported in this one.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


HEAVY = {
    "raster": ("rasterio",),
    "vector": ("geopandas", "shapely"),
    "morph": ("skimage",),
    "sensors": ("georeader",),
}

_BLOCK = """
import sys

BLOCKED = {blocked!r}


class Block:  # an install without these packages: importing them fails
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)


sys.meta_path.insert(0, Block())
"""

_SCENE = """
import numpy as np
from geoarray import GeoArray

T = (10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)
scene = GeoArray(
    np.random.default_rng(0).random((4, 16, 16)).astype("float32"),
    T, 32630, dims="band y x", fill_value=np.nan,
)
band = GeoArray(
    np.random.default_rng(1).random((16, 16)).astype("float32"),
    T, 32630, dims="y x", fill_value=np.nan,
)
"""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_base_install_imports_geoops_and_runs_a_kernel() -> None:
    """The slim-base gate: no heavy extra installed, `import geoops` and NDVI work."""
    blocked = sorted(name for names in HEAVY.values() for name in names)
    result = _run(
        _BLOCK.format(blocked=blocked)
        + _SCENE
        + """
import geoops as gz

ndvi = gz.indices.NDVI(red=2, nir=3)(scene)
assert ndvi.shape[-2:] == (16, 16) and np.isfinite(np.asarray(ndvi)).all(), ndvi
loaded = sorted(set(BLOCKED) & {m.split(".")[0] for m in sys.modules})
assert not loaded, loaded
print("ok")
"""
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_importing_geoops_loads_no_heavy_extra_even_when_installed() -> None:
    heavy = sorted(name for names in HEAVY.values() for name in names)
    result = _run(
        f"""
import sys
import geoops, geoops.io, geoops.geom, geoops.mask, geoops.feature, geoops.plume
loaded = sorted(set({heavy!r}) & {{m.split(".")[0] for m in sys.modules}})
print(loaded)
"""
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# One operator per extra whose run imports that extra first.
CALLS = {
    "raster": "gz.io.ReadWindow(src='scene.tif', window=(0, 0, 4, 4))()",
    "vector": "gz.feature.PeakLocalMax()(band)",
    "morph": "gz.segment.SLIC()(scene)",
    "sensors": "gz.geom.Reproject(dst_crs='EPSG:4326')(scene)",
}


@pytest.mark.parametrize("extra", sorted(HEAVY))
def test_an_operator_names_the_extra_it_is_missing(extra: str) -> None:
    result = _run(
        _BLOCK.format(blocked=list(HEAVY[extra]))
        + _SCENE
        + f"""
import geoops as gz

try:
    {CALLS[extra]}
except ImportError as exc:
    print(exc)
else:
    raise SystemExit("no ImportError")
"""
    )
    assert result.returncode == 0, result.stderr
    assert f"pip install 'geotoolz-ops[{extra}]'" in result.stdout, result.stdout
