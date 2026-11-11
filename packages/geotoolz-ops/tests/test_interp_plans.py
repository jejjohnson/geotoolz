"""InterpAt's plan and adjoint, and Reproject(engine="interp") on functax (#644)."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from geoarray import GeoArray, GeoPoints, Grid
from pyproj import Transformer

import geoops as gz


pytest.importorskip("functax.interp")

T = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
CRS = "EPSG:32613"
H, W = 32, 40


@pytest.fixture
def x64() -> Iterator[ModuleType]:
    """JAX with ``jax_enable_x64`` on for one test, restored afterwards."""
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)


def _field(scale: float = 4000.0) -> np.ndarray:
    """A smooth analytic field on the source pixel centres, ``(H, W)``."""
    rows, cols = np.mgrid[0:H, 0:W]
    x = T[2] + T[0] * (cols + 0.5)
    y = T[5] + T[4] * (rows + 0.5)
    return np.sin(x / scale) * np.cos(y / (0.75 * scale))


def _points(n: int = 50, seed: int = 0) -> GeoPoints:
    rng = np.random.default_rng(seed)
    rc = rng.uniform([-1.0, -1.0], [H, W], size=(n, 2))
    xy = np.stack([T[2] + T[0] * (rc[:, 1] + 0.5), T[5] + T[4] * (rc[:, 0] + 0.5)], -1)
    valid = np.ones(n, dtype=bool)
    valid[-3:] = False
    return GeoPoints(np.zeros(n), xy=xy, valid=valid, dims="obs", crs=CRS)


def _lonlat_target(n_rows: int = 20, n_cols: int = 24) -> Grid:
    """A lon/lat grid two source pixels inside the source's footprint."""
    to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon0, lat0 = to_lonlat.transform(T[2] + 3 * T[0], T[5] + 3 * T[4])
    lon1, lat1 = to_lonlat.transform(T[2] + (W - 3) * T[0], T[5] + (H - 3) * T[4])
    res = ((lon1 - lon0) / n_cols, (lat1 - lat0) / n_rows)
    return Grid((res[0], 0.0, lon0, 0.0, res[1], lat0), "EPSG:4326", (n_rows, n_cols))


# -- InterpAt.plan ------------------------------------------------------------------


@pytest.mark.parametrize("method", ["bilinear", "nearest"])
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_the_plan_reads_what_the_operator_reads(
    method: str, backend: str, x64: ModuleType
) -> None:
    grid = GeoArray(_field(), T, CRS, dims="y x")
    points = _points()
    if backend == "jax":
        grid, points = grid.to_jax(), points.to_jax()
    op = gz.InterpAt(method=method)
    plan = op.plan(grid, points)
    # f̂ = H y, with y the grid flattened to (N,)
    got = np.asarray(plan(x64.numpy.asarray(grid.data).reshape(-1)))
    out = op(grid, points)
    valid = np.asarray(out.valid)
    assert valid.sum() > 30
    np.testing.assert_allclose(
        got[valid], np.asarray(out.data)[valid], rtol=0, atol=1e-12
    )


def test_the_plan_is_built_once_per_grid_and_points(x64: ModuleType) -> None:
    grid = GeoArray(_field(), T, CRS, dims="y x")
    op = gz.InterpAt()
    first = op.plan(grid, _points())
    assert op.plan(grid.with_data(grid.data * 2), _points()) is first
    assert op.plan(grid, _points(seed=1)) is not first
    assert op.plan(grid.shift_origin((20.0, 0.0)), _points()) is not first


def test_a_traced_validity_mask_is_not_read_on_the_host(x64: ModuleType) -> None:
    jax = x64
    grid = GeoArray(_field(), T, CRS, dims="y x").to_jax()
    points = _points().to_jax()
    op = gz.InterpAt()

    def read(valid: Any) -> Any:
        moved = GeoPoints(points.data, xy=points.xy, valid=valid, dims="obs", crs=CRS)
        return op(grid, moved).data

    traced = jax.jit(read)(points.valid)
    np.testing.assert_allclose(
        np.asarray(traced), np.asarray(op(grid, points).data), atol=1e-12
    )


def test_the_adjoint_passes_the_dot_product_test(x64: ModuleType) -> None:
    """⟨H y, r⟩ = ⟨y, Hᵀ r⟩ with H = plan.operator() and Hᵀ its transpose."""
    jnp = x64.numpy
    grid = GeoArray(_field(), T, CRS, dims="y x").to_jax()
    operator = gz.InterpAt().plan(grid, _points().to_jax()).operator()
    adjoint = operator.transpose()
    rng = np.random.default_rng(3)
    y = jnp.asarray(rng.standard_normal(H * W))
    r = jnp.asarray(rng.standard_normal(50))
    forward = float(jnp.dot(operator.mv(y), r))
    backward = float(jnp.dot(y, adjoint.mv(r)))
    assert forward == pytest.approx(backward, rel=0, abs=1e-12)


# -- Reproject(engine="interp") ------------------------------------------------------


def _scene(frames: int = 3, bands: int = 2) -> GeoArray:
    base = _field()
    data = np.stack(
        [np.stack([(b + 1) * base + t for b in range(bands)]) for t in range(frames)]
    )
    return GeoArray(data, T, CRS, dims="time band y x", fill_value=np.nan)


def test_the_default_engine_is_gdal_and_unchanged() -> None:
    pytest.importorskip("georeader")
    from georeader import read

    from geoops._src.engine import to_engine
    from geoops.geom._src.array import resolve_resampling

    scene = _scene(1, 1)[0]
    op = gz.geom.Reproject(dst_crs="EPSG:4326")
    assert op.engine == "gdal" and op.target is None
    # what Reproject ran before #644
    before = read.read_to_crs(
        to_engine(scene), "EPSG:4326", resampling=resolve_resampling("bilinear")
    )
    after = op(scene)
    np.testing.assert_array_equal(np.asarray(after.data), np.asarray(before))
    np.testing.assert_array_equal(
        np.asarray(after.transform), np.asarray(tuple(before.transform)[:6])
    )


@pytest.mark.parametrize(
    ("resampling", "tolerance"),
    [("bilinear", 1e-3), ("nearest", 0.0), ("cubic", 1e-3), ("cubic_spline", 2e-3)],
)
def test_interp_agrees_with_gdal_on_a_smooth_field(
    resampling: str, tolerance: float
) -> None:
    pytest.importorskip("georeader")
    scene = _scene(1, 1)[0]
    target = _lonlat_target()
    gdal = gz.geom.Reproject(target=target, resampling=resampling)(scene)
    interp = gz.geom.Reproject(target=target, resampling=resampling, engine="interp")(
        scene
    )
    assert interp.crs == gdal.crs == target.crs
    np.testing.assert_allclose(
        np.asarray(interp.transform), np.asarray(gdal.transform), rtol=0, atol=1e-12
    )
    inner = (..., slice(2, -2), slice(2, -2))
    difference = np.abs(np.asarray(interp.data) - np.asarray(gdal.data))[inner]
    assert np.isfinite(difference).all()
    assert difference.max() <= tolerance


def test_interp_on_jax_under_filter_jit_matches_numpy(x64: ModuleType) -> None:
    eqx = pytest.importorskip("equinox")
    scene = _scene()
    op = gz.geom.Reproject(target=_lonlat_target(), engine="interp")
    on_numpy = op(scene)
    on_jax = eqx.filter_jit(op)(scene.to_jax())
    assert type(on_jax.data).__module__.startswith("jax")
    assert on_jax.dims == ("time", "band", "y", "x")
    assert on_jax.shape == (3, 2, 20, 24) and on_jax.crs == on_numpy.crs
    np.testing.assert_allclose(
        np.asarray(on_jax.data), np.asarray(on_numpy.data), rtol=0, atol=1e-12
    )


def test_one_locate_serves_every_frame_and_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geoops.geom._src import operators

    calls = 0
    original = operators.target_coordinates

    def counting(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(operators, "target_coordinates", counting)
    op = gz.geom.Reproject(target=_lonlat_target(), engine="interp")
    first = op(_scene(4, 3))
    second = op(_scene(4, 3))
    assert calls == 1
    np.testing.assert_array_equal(np.asarray(first.data), np.asarray(second.data))
    # each frame and band is the plan applied alone
    alone = op(_scene(4, 3)[2, 1])
    np.testing.assert_allclose(
        np.asarray(first.data)[2, 1], np.asarray(alone.data), rtol=0, atol=1e-12
    )


def test_nodata_and_off_grid_pixels_hold_the_fill() -> None:
    data = _field()
    data[10, 10] = -9999.0
    scene = GeoArray(data, T, CRS, dims="y x", fill_value=-9999.0)
    # a target that runs off the source on its right
    target = Grid((20.0, 0.0, T[2] + 30.0, 0.0, -20.0, T[5] - 30.0), CRS, (H, W))
    out = gz.geom.Reproject(target=target, engine="interp")(scene)
    assert out.fill_value == -9999.0
    values = np.asarray(out.data)
    # target pixel (r, c) has its centre at source position (r + 1.5, c + 1.5)
    assert values[8, 8] == -9999.0 and values[9, 9] == -9999.0
    assert (values[:, -1] == -9999.0).all()
    assert np.isfinite(values[:20, :20][values[:20, :20] != -9999.0]).all()


def test_a_jax_carrier_needs_the_interp_engine(x64: ModuleType) -> None:
    from pipekit_ops import BackendError

    scene = _scene(1, 1)[0].to_jax()
    with pytest.raises(BackendError, match="Reproject"):
        gz.geom.Reproject(dst_crs="EPSG:4326")(scene)
    with pytest.raises(BackendError, match="Reproject"):
        gz.geom.Reproject(target=_lonlat_target())(scene)
    out = gz.geom.Reproject(target=_lonlat_target(), engine="interp")(scene)
    assert type(out.data).__module__.startswith("jax")


def test_the_constructor_refuses_what_the_engines_cannot_do() -> None:
    target = _lonlat_target()
    with pytest.raises(ValueError, match="not both or neither"):
        gz.geom.Reproject()
    with pytest.raises(ValueError, match="not both or neither"):
        gz.geom.Reproject(dst_crs="EPSG:4326", target=target)
    with pytest.raises(ValueError, match="needs target"):
        gz.geom.Reproject(dst_crs="EPSG:4326", engine="interp")
    with pytest.raises(ValueError, match="engine='gdal'"):
        gz.geom.Reproject(target=target, engine="interp", resampling="average")
    with pytest.raises(ValueError, match=r"Reproject\.engine"):
        gz.geom.Reproject(target=target, engine="numba")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("engine", ["gdal", "interp"])
def test_the_config_round_trips(engine: str) -> None:
    import json

    from pipekit_ops import Operator, dumps, loads

    op = gz.geom.Reproject(target=_lonlat_target(), engine=engine)  # ty: ignore[invalid-argument-type]
    config = op.get_config()
    json.dumps(config, allow_nan=False)
    assert config["engine"] == engine and config["target"]["crs"] == "EPSG:4326"
    again = Operator.from_state(op.state)
    assert isinstance(again, gz.geom.Reproject)
    assert again.target == op.target and again.get_config() == config
    assert loads(dumps(op)).get_config() == config


@pytest.mark.parametrize("engine", ["gdal", "interp"])
def test_the_config_round_trips_through_hydra_zen_yaml(engine: str) -> None:
    hydra_zen = pytest.importorskip("hydra_zen")
    from omegaconf import OmegaConf

    op = gz.geom.Reproject(target=_lonlat_target(), engine=engine)  # ty: ignore[invalid-argument-type]
    cfg = hydra_zen.builds(type(op), **op.get_config())
    text = hydra_zen.to_yaml(cfg)
    restored = hydra_zen.instantiate(OmegaConf.create(text))
    assert isinstance(restored, gz.geom.Reproject)
    assert restored.get_config() == op.get_config()


def test_a_pixel_with_one_nodata_band_is_nodata_in_every_band() -> None:
    """valid_mask's rule: any band at nodata makes the pixel nodata; frames apart."""
    data = np.stack([np.stack([_field(), _field() + 5.0]) for _ in range(2)])
    data[0, 0, 7, 9] = -9999.0  # frame 0, band 0 only
    scene = GeoArray(data, T, CRS, dims="time band y x", fill_value=-9999.0)
    target = Grid(T, CRS, (H, W))  # the source grid itself
    out = np.asarray(
        gz.geom.Reproject(target=target, resampling="nearest", engine="interp")(
            scene
        ).data
    )
    assert out[0, :, 7, 9].tolist() == [-9999.0, -9999.0]
    np.testing.assert_array_equal(out[1, :, 7, 9], data[1, :, 7, 9])
    np.testing.assert_array_equal(out[0, :, 7, 10], data[0, :, 7, 10])
