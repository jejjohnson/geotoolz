"""interp_at on a functax plan equals interp_at as it was at 1e7308b (#643).

The outputs in ``data/interp_at_1e7308b.npz`` were recorded by running
`cases` through the hand-written bilinear and nearest samplers of
``geoops.geom.interp_at`` at commit 1e7308b, on NumPy. The positions avoid
exact nearest-neighbour ties (a fractional pixel position of exactly k + ½),
where functax rounds down and the old sampler rounded half to even; that
change is pinned on its own below.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from geoarray import GeoArray, GeoPoints

from geoops.geom import interp_at


pytest.importorskip("functax.interp")

RECORDED = Path(__file__).parent / "data" / "interp_at_1e7308b.npz"
CRS = "EPSG:32613"
NORTH_UP = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
ROTATED = (20.0, 3.0, 600000.0, -2.0, -20.0, 3550000.0)
HEIGHT, WIDTH = 7, 6


@pytest.fixture
def x64() -> Iterator[ModuleType]:
    """JAX with ``jax_enable_x64`` on for one test, restored afterwards."""
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)


def _xy(transform: tuple[float, ...], rc: np.ndarray) -> np.ndarray:
    """Map coordinates of fractional (row, col) pixel positions, centres at integers."""
    a, b, c, d, e, f = transform
    rows, cols = rc[:, 0] + 0.5, rc[:, 1] + 0.5
    return np.stack([a * cols + b * rows + c, d * cols + e * rows + f], axis=-1)


def _positions(seed: int) -> np.ndarray:
    """Random positions inside, on the half-pixel border, just past it, and far off."""
    rng = np.random.default_rng(seed)
    inside = rng.uniform([-0.5, -0.5], [HEIGHT - 0.5, WIDTH - 0.5], size=(40, 2))
    edges = np.array(
        [
            [-0.5, 2.3],
            [HEIGHT - 0.5, 1.7],
            [3.2, -0.5],
            [4.1, WIDTH - 0.5],
            [-0.5, -0.5],
            [HEIGHT - 0.5, WIDTH - 0.5],
            [-0.2, 3.3],
            [HEIGHT - 0.8, 4.4],
            [0.0, 0.0],
            [HEIGHT - 1.0, WIDTH - 1.0],
            [2.0, 3.0],
            [-0.5000001, 1.2],
            [HEIGHT - 0.4999999, 1.2],
            [-3.0, 2.0],
            [2.0, WIDTH + 4.0],
        ]
    )
    return np.concatenate([inside, edges])


def _points(transform: tuple[float, ...], seed: int, *, pad: bool) -> GeoPoints:
    xy = _xy(transform, _positions(seed))
    valid = None
    if pad:
        # a missing position and two padded slots, one at a NaN position
        xy = np.concatenate([xy, [[np.nan, 3549900.0], [0.0, 0.0], [np.nan, np.nan]]])
        valid = np.ones(len(xy), dtype=bool)
        valid[-2:] = False
    return GeoPoints(np.zeros(len(xy)), xy=xy, valid=valid, dims="obs", crs=CRS)


def _field(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 10.0, size=shape)


def cases() -> dict[str, tuple[GeoArray, GeoPoints, dict[str, Any]]]:
    """Every (grid, points, keyword arguments) the recording covers, by name."""
    out: dict[str, tuple[GeoArray, GeoPoints, dict[str, Any]]] = {}
    nodata = _field((2, HEIGHT, WIDTH), 3)
    nodata[0, 2, 2] = -9999.0
    nodata[1, 5, 1] = -9999.0
    nodata[0, 4, 4] = np.nan
    for method in ("bilinear", "nearest"):
        for dtype in ("float64", "float32"):
            for name, transform in (("north_up", NORTH_UP), ("rotated", ROTATED)):
                grid = GeoArray(
                    _field((HEIGHT, WIDTH), 1).astype(dtype),
                    transform,
                    CRS,
                    dims="y x",
                )
                points = _points(transform, 0, pad=True)
                out[f"{method}-{dtype}-{name}"] = (grid, points, {"method": method})
            bands = GeoArray(
                nodata.astype(dtype),
                NORTH_UP,
                CRS,
                fill_value=-9999.0,
                dims="band y x",
            )
            out[f"{method}-{dtype}-nodata"] = (
                bands,
                _points(NORTH_UP, 1, pad=True),
                {"method": method},
            )
            frames = GeoArray(
                _field((3, HEIGHT, WIDTH), 2).astype(dtype),
                NORTH_UP,
                CRS,
                dims="time y x",
            )
            frame_points = _points(NORTH_UP, 2, pad=False)
            frame = np.arange(frame_points.size) % 3
            out[f"{method}-{dtype}-frame"] = (
                frames,
                frame_points,
                {"method": method, "frame": frame},
            )
        integers = GeoArray(
            np.arange(HEIGHT * WIDTH, dtype=np.int16).reshape(HEIGHT, WIDTH),
            NORTH_UP,
            CRS,
            fill_value=7,
            dims="y x",
        )
        out[f"{method}-int16"] = (
            integers,
            _points(NORTH_UP, 4, pad=True),
            {"method": method},
        )
    return out


CASES = cases()


@pytest.fixture(scope="module")
def recorded() -> Any:
    return np.load(RECORDED)


def _atol(name: str) -> float:
    return 1e-6 if "float32" in name else 1e-12


def _check(name: str, got: GeoPoints, recorded: Any) -> None:
    want, want_valid = recorded[f"{name}/data"], recorded[f"{name}/valid"]
    data = np.asarray(got.data)
    assert str(data.dtype) == str(recorded[f"{name}/dtype"])
    np.testing.assert_array_equal(np.asarray(got.valid), want_valid)
    np.testing.assert_allclose(data, want, rtol=0, atol=_atol(name), equal_nan=True)


@pytest.mark.parametrize("name", sorted(CASES))
def test_numpy_matches_the_recording(name: str, recorded: Any) -> None:
    grid, points, kwargs = CASES[name]
    got = interp_at(grid, points, **kwargs)
    assert isinstance(got.data, np.ndarray)
    _check(name, got, recorded)


@pytest.mark.parametrize("name", sorted(CASES))
def test_jax_matches_the_recording(name: str, recorded: Any, x64: ModuleType) -> None:
    grid, points, kwargs = CASES[name]
    frame = kwargs.get("frame")
    if frame is not None:
        kwargs = {**kwargs, "frame": x64.numpy.asarray(frame)}
    got = interp_at(grid.to_jax(), points.to_jax(), **kwargs)
    assert type(got.data).__module__.startswith("jax")
    _check(name, got, recorded)


def test_jax_matches_the_recording_under_jit(recorded: Any, x64: ModuleType) -> None:
    name = "bilinear-float64-nodata"
    grid, points, kwargs = CASES[name]
    got = x64.jit(lambda g, p: interp_at(g, p, **kwargs))(
        grid.to_jax(), points.to_jax()
    )
    _check(name, got, recorded)


# -- the plan behind interp_at -----------------------------------------------------


@pytest.mark.parametrize("method", ["bilinear", "nearest"])
def test_the_plan_passes_the_dot_product_test(method: str, x64: ModuleType) -> None:
    """⟨H y, r⟩ = ⟨y, Hᵀ r⟩ for random y and r, on a rotated grid."""
    import functax.interp

    from geoops.geom._src.sampling import _plan

    jnp = x64.numpy
    grid = GeoArray(_field((HEIGHT, WIDTH), 5), ROTATED, CRS, dims="y x").to_jax()
    xy = jnp.asarray(_xy(ROTATED, _positions(5)))
    located = jnp.ones(xy.shape[0], dtype=bool)
    stencil = _plan(functax.interp, grid, xy, located, method).read
    rng = np.random.default_rng(6)
    y = jnp.asarray(rng.standard_normal(HEIGHT * WIDTH))
    r = jnp.asarray(rng.standard_normal(xy.shape[0]))
    forward = float(jnp.dot(stencil.apply(y), r))
    backward = float(jnp.dot(y, stencil.apply_adjoint(r)))
    assert forward == pytest.approx(backward, rel=0, abs=1e-12)


def test_the_half_pixel_border_reads_the_edge_and_beyond_it_is_invalid() -> None:
    data = _field((HEIGHT, WIDTH), 7)
    grid = GeoArray(data, NORTH_UP, CRS, dims="y x")
    rc = np.array([[-0.5, 2.0], [HEIGHT - 0.5, 3.0], [1.0, WIDTH - 0.5], [-0.51, 2.0]])
    xy = _xy(NORTH_UP, rc)
    out = interp_at(grid, GeoPoints(np.zeros(4), xy=xy, dims="obs", crs=CRS))
    assert out.valid.tolist() == [True, True, True, False]
    want = [data[0, 2], data[HEIGHT - 1, 3], data[1, WIDTH - 1]]
    np.testing.assert_allclose(out.data[:3], want, rtol=0, atol=1e-12)
    assert np.isnan(out.data[3])


def test_a_nearest_tie_goes_to_the_lower_pixel() -> None:
    """Today (1e7308b): half to even, so row 1.5 read row 2. After: row 1."""
    rows = np.arange(HEIGHT, dtype=float)[:, None] * np.ones((1, WIDTH))
    grid = GeoArray(rows, NORTH_UP, CRS, dims="y x")
    xy = _xy(NORTH_UP, np.array([[0.5, 1.0], [1.5, 1.0], [2.5, 1.0]]))
    out = interp_at(
        grid, GeoPoints(np.zeros(3), xy=xy, dims="obs", crs=CRS), method="nearest"
    )
    assert out.data.tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("with_frame", [False, True])
def test_the_nodata_check_scales_with_the_points_not_the_grid(
    with_frame: bool, x64: ModuleType
) -> None:
    """Only the M·S gathered taps are checked for nodata, never all N pixels."""
    import re

    n_rows, n_cols = 40, 50
    data = np.random.default_rng(8).uniform(size=(3, n_rows, n_cols))
    grid = GeoArray(data, NORTH_UP, CRS, fill_value=-9999.0, dims="time y x")
    rc = np.array([[1.2, 3.4], [10.5, 20.5], [30.1, 2.9], [5.0, 44.4], [0.0, 0.0]])
    points = GeoPoints(np.zeros(5), xy=_xy(NORTH_UP, rc), dims="obs", crs=CRS)
    frame = np.array([0, 2, 1, 1, 0]) if with_frame else None

    def read(values: Any) -> Any:
        return interp_at(grid.to_jax().with_data(values), points, frame=frame).data

    jaxpr = str(x64.make_jaxpr(read)(x64.numpy.asarray(data)))
    checks = [
        line
        for line in jaxpr.splitlines()
        if re.search(r"\b(is_finite|ne)\b", line)
        and re.search(rf"\[[\d,]*\b{n_rows * n_cols}\]", line)
    ]
    assert checks == []
