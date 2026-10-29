"""`geoops.geom.central_gradient` and `geoops.geom.Gradient` (#539, OP-O7).

The gates from the issue:

- a linear field's gradient is exact, interior and edges, on NumPy and JAX;
- second-order convergence in the interior;
- on a geographic grid the metric matches pyproj's geodesic to 1e-6;
- under ``jax.jit`` two same-shape fields with different transforms trace once.
"""

from __future__ import annotations

import numpy as np
import pytest
from geoarray import GeoArray

import geoops as gz
from geoops.geom import Gradient, central_gradient


jax = pytest.importorskip("jax")

UTM = (10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)


@pytest.fixture(autouse=True)
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _centres(transform: tuple[float, ...], shape: tuple[int, int]):
    a, b, c, d, e, f = transform
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]] + 0.5
    return a * cols + b * rows + c, d * cols + e * rows + f


def _linear(transform: tuple[float, ...], crs: str = "EPSG:32613") -> GeoArray:
    x, y = _centres(transform, (7, 9))
    # f = a x + b y with a = 3, b = 2
    return GeoArray(3.0 * x + 2.0 * y, transform, crs, dims=("y", "x"))


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_kernel_is_exact_on_a_linear_field(backend: str) -> None:
    rows, cols = np.mgrid[0:5, 0:6].astype(float)
    f = 2.0 * rows - 0.5 * cols
    if backend == "jax":
        f = jax.numpy.asarray(f)
    d_y, d_x = central_gradient(f, 4.0, -2.0)
    np.testing.assert_allclose(np.asarray(d_y), 0.5)  # every row, edges too
    np.testing.assert_allclose(np.asarray(d_x), 0.25)


def test_kernel_converges_at_second_order_inside() -> None:
    """f = sin(kx) cos(ly): halving the step quarters the interior error."""
    k, ell = 1.3, 0.7

    def interior_error(n: int) -> float:
        h = 2.0 / n
        y, x = np.mgrid[0 : n + 1, 0 : n + 1] * h
        f = np.sin(k * x) * np.cos(ell * y)
        d_y, d_x = central_gradient(f, h, h)
        want_x = k * np.cos(k * x) * np.cos(ell * y)
        want_y = -ell * np.sin(k * x) * np.sin(ell * y)
        inner = (slice(1, -1), slice(1, -1))
        return max(np.abs(d_x - want_x)[inner].max(), np.abs(d_y - want_y)[inner].max())

    ratio = interior_error(32) / interior_error(64)
    assert 3.8 < ratio < 4.2


def test_kernel_propagates_nan_to_neighbours() -> None:
    f = np.zeros((5, 5))
    f[2, 2] = np.nan
    d_y, d_x = central_gradient(f, 1.0, 1.0)
    assert np.isnan(d_y[[1, 3], 2]).all() and np.isfinite(d_y[2, 2] * 0 + d_y[0, 0])
    assert np.isnan(d_x[2, [1, 3]]).all()


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(UTM, id="north-up"),
        pytest.param((10.0, 0.0, 500_000.0, 0.0, 10.0, 3_000_000.0), id="south-up"),
        pytest.param(
            (
                10 * np.cos(0.3),
                -10 * np.sin(0.3),
                500_000.0,
                10 * np.sin(0.3),
                10 * np.cos(0.3),
                4_000_000.0,
            ),
            id="rotated",
        ),
    ],
)
def test_operator_is_exact_in_map_axes(transform: tuple[float, ...]) -> None:
    out = Gradient()(_linear(transform))
    assert out.dims == ("component", "y", "x")
    assert np.isnan(out.fill_value)
    d_y, d_x = np.asarray(out)
    np.testing.assert_allclose(d_y, 2.0, rtol=1e-8)
    np.testing.assert_allclose(d_x, 3.0, rtol=1e-8)
    jitted = jax.jit(Gradient())(_linear(transform).to_jax())
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(out), rtol=1e-12)


def test_geographic_metric_matches_pyproj_geodesic() -> None:
    """∂f/∂east and ∂f/∂north of f = λ and f = φ (degrees) on WGS 84."""
    from pyproj import Geod

    transform = (0.25, 0.0, -10.0, 0.0, -0.25, 60.0)
    lon, lat = _centres(transform, (40, 8))
    geod = Geod(ellps="WGS84")
    step = 1e-6
    # metres per degree, from the geodesic between two nearby points
    _, _, east = geod.inv(lon, lat, lon + step, lat)
    _, _, north = geod.inv(lon, lat, lon, lat + step)

    d_lon = Gradient()(GeoArray(lon, transform, "EPSG:4326", dims=("y", "x")))
    d_lat = Gradient()(GeoArray(lat, transform, "EPSG:4326", dims=("y", "x")))
    np.testing.assert_allclose(np.asarray(d_lon)[1], step / east, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(d_lat)[0], step / north, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(d_lon)[0], 0.0, atol=1e-18)


def test_jit_traces_once_across_transforms() -> None:
    traces = []

    @jax.jit
    def step(g: GeoArray) -> GeoArray:
        traces.append(1)
        return Gradient()(g)

    first = _linear(UTM).to_jax()
    moved = (20.0, 0.0, 600_000.0, 0.0, -20.0, 4_100_000.0)
    second = _linear(moved).to_jax()
    step(first)
    out = step(second)
    assert len(traces) == 1
    np.testing.assert_allclose(np.asarray(out)[1], 3.0, rtol=1e-8)


def test_nodata_and_stacks() -> None:
    values = np.arange(2 * 3 * 6 * 5, dtype=np.float32).reshape(2, 3, 6, 5)
    values[0, 1, 3, 2] = -9999.0
    stack = GeoArray(
        values, UTM, "EPSG:32613", fill_value=-9999.0, dims=("time", "band", "y", "x")
    )
    out = Gradient()(stack)
    assert out.dims == ("time", "band", "component", "y", "x")
    grad = np.asarray(out)
    assert np.isnan(grad[0, 1, 0, [2, 4], 2]).all()  # rows next to the fill
    assert np.isfinite(grad[1]).all()


def test_plain_arrays_are_differentiated_in_pixel_units() -> None:
    rows, cols = np.mgrid[0:4, 0:5].astype(float)
    out = Gradient()(rows + 2.0 * cols)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 4, 5)
    np.testing.assert_allclose(out[0], 1.0)
    np.testing.assert_allclose(out[1], 2.0)


def test_a_dataarray_is_refused_with_the_lift_hint() -> None:
    da = _linear(UTM).to_xarray()
    with pytest.raises(TypeError, match=r"xrops\.lift"):
        Gradient()(da)


def test_exported_at_the_top_level() -> None:
    assert gz.Gradient is Gradient
    assert Gradient().get_config() == {}
