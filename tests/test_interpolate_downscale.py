"""Tests for ``xr_toolz.interpolate.downscale`` (F3.4, D12).

Per F3.5 resolution of D12 Q1, ``Downscale`` and ``Upscale`` are pure
callable wrappers — patch tiling is delegated to ``xrpatcher`` upstream.
The smoke tests use a trivial bilinear-resize as the wrapped model.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.interpolate.operators import Downscale, Upscale


def _bilinear_to(target_lon: np.ndarray, target_lat: np.ndarray):
    """Return a callable that bilinearly resizes a Dataset to the given grid."""

    def _resize(ds: xr.Dataset) -> xr.Dataset:
        return ds.interp(lon=target_lon, lat=target_lat)

    return _resize


@pytest.fixture
def coarse_ds():
    lon = np.linspace(0.0, 10.0, 5)
    lat = np.linspace(-5.0, 5.0, 5)
    field = np.outer(lat, lon)
    return xr.Dataset(
        {"f": (("lat", "lon"), field)},
        coords={"lat": lat, "lon": lon},
    )


@pytest.fixture
def fine_ds():
    lon = np.linspace(0.0, 10.0, 21)
    lat = np.linspace(-5.0, 5.0, 21)
    field = np.outer(lat, lon)
    return xr.Dataset(
        {"f": (("lat", "lon"), field)},
        coords={"lat": lat, "lon": lon},
    )


def test_downscale_smoke_wrapping_bilinear_resize(coarse_ds):
    """Downscale plumbing: coarse 5x5 → fine 21x21 via bilinear-resize callable."""
    fine_lon = np.linspace(0.0, 10.0, 21)
    fine_lat = np.linspace(-5.0, 5.0, 21)
    op = Downscale(_bilinear_to(fine_lon, fine_lat))
    out = op(coarse_ds)

    assert out.sizes["lon"] == 21
    assert out.sizes["lat"] == 21
    # Bilinear on a smooth f(lat, lon) = lat * lon recovers the analytic answer.
    expected = np.outer(fine_lat, fine_lon)
    np.testing.assert_allclose(out["f"].values, expected, atol=1e-10)


def test_upscale_smoke_wrapping_bilinear_resize(fine_ds):
    """Upscale plumbing: fine 21x21 → coarse 5x5 via bilinear-resize callable."""
    coarse_lon = np.linspace(0.0, 10.0, 5)
    coarse_lat = np.linspace(-5.0, 5.0, 5)
    op = Upscale(_bilinear_to(coarse_lon, coarse_lat))
    out = op(fine_ds)

    assert out.sizes["lon"] == 5
    assert out.sizes["lat"] == 5
    expected = np.outer(coarse_lat, coarse_lon)
    np.testing.assert_allclose(out["f"].values, expected, atol=1e-10)


def test_downscale_rejects_non_callable():
    with pytest.raises(TypeError):
        Downscale(model="not_a_callable")  # type: ignore[arg-type]


def test_target_grid_attribute_is_carried(coarse_ds):
    """The optional target_grid is stored as metadata, not used in _apply."""
    fine_lon = np.linspace(0.0, 10.0, 21)
    fine_lat = np.linspace(-5.0, 5.0, 21)
    grid = xr.Dataset(coords={"lat": fine_lat, "lon": fine_lon})

    op = Downscale(_bilinear_to(fine_lon, fine_lat), target_grid=grid)
    assert op.target_grid is grid


def test_get_config_round_trips_through_json(coarse_ds):
    """Config must survive ``json.dumps``/``loads`` so it stays serializable."""
    import json

    fine_lon = np.linspace(0.0, 10.0, 21)
    fine_lat = np.linspace(-5.0, 5.0, 21)
    op = Downscale(
        _bilinear_to(fine_lon, fine_lat),
        target_grid=xr.Dataset(coords={"lat": fine_lat, "lon": fine_lon}),
    )
    cfg = op.get_config()
    round_tripped = json.loads(json.dumps(cfg))
    assert round_tripped == cfg
    assert round_tripped["target_grid"] == "<grid>"
    assert isinstance(round_tripped["model"], str)


def test_no_top_level_jax_or_torch_import():
    """Per D4, the downscale module must not import JAX or torch.

    Run the import in a fresh subprocess so the result is independent of
    test ordering — other tests (e.g. ``test_inference_jax``) may have
    already pulled JAX into ``sys.modules`` of the parent process.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import xr_toolz.interpolate._src.downscale  # noqa: F401\n"
                "for name in ('jax', 'jaxlib', 'torch'):\n"
                "    if name in sys.modules:\n"
                "        print(name)\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    leaked = [n for n in result.stdout.strip().splitlines() if n]
    assert leaked == [], f"backend leaked into sys.modules: {leaked}"


def test_works_with_arbitrary_operator(coarse_ds):
    """Downscale composes with any Operator (duck-typed callable).

    The wrapped object doesn't have to be a ``ModelOp`` — any Operator
    or plain callable works.
    """
    from xr_toolz.core import Operator

    class TinyResize(Operator):
        def __init__(self, lon, lat):
            self.lon = lon
            self.lat = lat

        def _apply(self, ds):
            return ds.interp(lon=self.lon, lat=self.lat)

        def get_config(self):
            return {"lon": list(self.lon), "lat": list(self.lat)}

    fine_lon = np.linspace(0.0, 10.0, 11)
    fine_lat = np.linspace(-5.0, 5.0, 11)
    op = Downscale(TinyResize(fine_lon, fine_lat))
    out = op(coarse_ds)
    assert out.sizes["lon"] == 11
    assert out.sizes["lat"] == 11
