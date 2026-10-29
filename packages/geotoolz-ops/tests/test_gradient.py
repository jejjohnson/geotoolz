"""The gradient stack (#539, OP-O7): stencils, array primitives, `Gradient`.

Three layers, each usable on its own:

- L0 `difference_offsets` / `difference_coeffs`: finitediffx's stencils;
- L1 `difference` / `gradient` (NaN propagates) and the mask-aware
  `adaptive_difference` / `adaptive_gradient`, on any backend;
- L2 `Gradient`: geometry from the carrier, ``nan_policy`` and a mask input.

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
from geoops.geom import (
    Gradient,
    adaptive_difference,
    adaptive_gradient,
    difference,
    difference_coeffs,
    difference_offsets,
    gradient,
    nonuniform_coeffs,
)


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
    d_y, d_x = gradient(f, 4.0, -2.0)
    np.testing.assert_allclose(np.asarray(d_y), 0.5)  # every row, edges too
    np.testing.assert_allclose(np.asarray(d_x), 0.25)


def test_kernel_converges_at_second_order_inside() -> None:
    """f = sin(kx) cos(ly): halving the step quarters the interior error."""
    k, ell = 1.3, 0.7

    def interior_error(n: int) -> float:
        h = 2.0 / n
        y, x = np.mgrid[0 : n + 1, 0 : n + 1] * h
        f = np.sin(k * x) * np.cos(ell * y)
        d_y, d_x = gradient(f, h, h)
        want_x = k * np.cos(k * x) * np.cos(ell * y)
        want_y = -ell * np.sin(k * x) * np.sin(ell * y)
        inner = (slice(1, -1), slice(1, -1))
        return max(np.abs(d_x - want_x)[inner].max(), np.abs(d_y - want_y)[inner].max())

    ratio = interior_error(32) / interior_error(64)
    assert 3.8 < ratio < 4.2


def test_kernel_propagates_nan_to_neighbours() -> None:
    f = np.zeros((5, 5))
    f[2, 2] = np.nan
    d_y, d_x = gradient(f, 1.0, 1.0)
    assert np.isnan(d_y[[1, 3], 2]).all() and np.isfinite(d_y[0, 0])
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
    assert Gradient().get_config() == {
        "accuracy": 1,
        "nan_policy": "propagate",
        "periodic_x": False,
    }
    with pytest.raises(ValueError, match="nan_policy"):
        Gradient(nan_policy="fill")


# ---------------------------------------------------------------------------
# L0: stencils equal finitediffx's
# ---------------------------------------------------------------------------

CASES = [
    (derivative, accuracy, method)
    for derivative in (1, 2)
    for accuracy in (1, 2, 3, 4)
    for method in ("central", "forward", "backward")
]


@pytest.mark.parametrize(("derivative", "accuracy", "method"), CASES)
def test_difference_equals_finitediffx(
    derivative: int, accuracy: int, method: str
) -> None:
    fdx = pytest.importorskip("finitediffx")
    rng = np.random.default_rng(derivative * 10 + accuracy)
    f = rng.normal(size=(17, 13))
    for axis, step in ((0, 0.3), (1, -2.0)):
        want = fdx.difference(
            jax.numpy.asarray(f),
            axis=axis,
            step_size=step,
            derivative=derivative,
            accuracy=accuracy,
            method=method,
        )
        got = difference(
            f,
            axis=axis,
            step=step,
            derivative=derivative,
            accuracy=accuracy,
            method=method,
        )
        np.testing.assert_allclose(got, np.asarray(want), rtol=1e-9, atol=1e-9)


def test_stencils_are_the_textbook_ones() -> None:
    assert difference_offsets(derivative=1, accuracy=1, method="central") == (-1, 0, 1)
    assert difference_coeffs((-1, 0, 1), 1) == (-0.5, 0.0, 0.5)
    assert difference_coeffs((-2, -1, 0, 1, 2), 1) == pytest.approx(
        (1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12)
    )
    assert difference_coeffs((-1, 0, 1), 2) == (1.0, -2.0, 1.0)
    with pytest.raises(ValueError, match="cannot resolve"):
        difference_coeffs((0, 1), 2)


def test_periodic_difference_reads_across_the_seam() -> None:
    lon = np.arange(36) * 10.0
    f = np.sin(np.deg2rad(lon))
    got = difference(f, step=np.deg2rad(10.0), accuracy=3, periodic=True)
    np.testing.assert_allclose(got, np.cos(np.deg2rad(lon)), atol=1e-4)
    assert abs(got[0] - 1.0) < 1e-4  # the seam is a centred stencil too


# ---------------------------------------------------------------------------
# L1: mask-aware differences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("accuracy", [1, 2, 3])
def test_adaptive_is_exact_on_a_linear_field_up_to_the_mask(accuracy: int) -> None:
    rows, cols = np.mgrid[0:12, 0:15].astype(float)
    f = 2.0 * rows - 0.5 * cols
    land = (rows - 6) ** 2 + (cols - 7) ** 2 < 6  # an island
    land[:, 12] = True  # and a wall; every valid run is at least two cells
    for backend in ("numpy", "jax"):
        field = jax.numpy.asarray(f) if backend == "jax" else f
        d_y, d_x = adaptive_gradient(field, mask=land, accuracy=accuracy)
        for got, want in ((d_y, 2.0), (d_x, -0.5)):
            got = np.asarray(got)
            assert np.isnan(got[land]).all()
            np.testing.assert_allclose(got[~land], want, rtol=1e-9)


def test_propagate_spreads_one_halo_and_adaptive_none() -> None:
    f = np.arange(9.0)
    f[4] = np.nan
    plain = difference(f, accuracy=2)
    assert np.isnan(plain).sum() == 3  # the cell and both neighbours
    recovered = adaptive_difference(f, accuracy=2)
    assert np.isnan(recovered).sum() == 1
    np.testing.assert_allclose(recovered[[3, 5]], 1.0)


def test_adaptive_descends_accuracy_for_a_narrow_channel() -> None:
    """A two-cell channel supports a first-order stencil and nothing wider."""
    f = np.array([np.nan, np.nan, 3.0, 4.0, np.nan, np.nan])
    got = adaptive_difference(f, accuracy=3)
    np.testing.assert_allclose(got[2:4], 1.0)
    assert np.isnan(got[[0, 1, 4, 5]]).all()


def test_adaptive_jits_and_traces_once() -> None:
    traces = []

    @jax.jit
    def step(f, mask):
        traces.append(1)
        return adaptive_difference(f, mask=mask, accuracy=2)

    f = jax.numpy.arange(10.0)
    step(f, jax.numpy.zeros(10, bool))
    out = step(f, jax.numpy.arange(10) == 4)
    assert len(traces) == 1
    np.testing.assert_allclose(np.asarray(out)[[3, 5]], 1.0)


def test_coastline_measurement() -> None:
    """xrgrad's comparison: sin(2λ)cos(φ) on 121 x 91 with a synthetic coast.

    Propagate loses the coastal strip; adaptive loses none of it and stays
    within a small multiple of the interior error there.
    """
    lam = np.deg2rad(np.linspace(0.0, 120.0, 121))
    phi = np.deg2rad(np.linspace(-45.0, 45.0, 91))
    lam2, phi2 = np.meshgrid(lam, phi)
    f = np.sin(2 * lam2) * np.cos(phi2)
    want = 2 * np.cos(2 * lam2) * np.cos(phi2)
    cols = np.arange(121)[None, :]
    rows = np.arange(91)[:, None]
    land = (cols < 10) | ((cols - 70) ** 2 + (rows - 45) ** 2 < 15**2)
    step = lam[1] - lam[0]
    padded = np.pad(land, ((0, 0), (1, 1)))
    coast = ~land & (padded[:, :-2] | padded[:, 2:])  # water next to land along x
    plain = difference(np.where(land, np.nan, f), step=step, accuracy=2)
    adaptive = adaptive_difference(f, mask=land, step=step, accuracy=2)
    assert np.isnan(plain[coast]).all()
    assert np.isfinite(adaptive[~land]).all()
    scale = np.abs(want[~land]).mean()
    interior = ~land & ~coast
    assert np.abs(adaptive - want)[coast].max() / scale < 1e-2
    assert np.abs(adaptive - want)[interior].max() / scale < 1e-3


# ---------------------------------------------------------------------------
# L2: Gradient's mask input and policy
# ---------------------------------------------------------------------------


def test_gradient_operator_with_a_land_mask() -> None:
    field = _linear(UTM)
    land_values = np.zeros(field.shape, bool)
    land_values[3, 2:5] = True
    land = GeoArray(land_values, UTM, "EPSG:32613", dims=("y", "x"))

    plain = np.asarray(Gradient()(field, land))
    adaptive = np.asarray(Gradient(nan_policy="adaptive")(field, land))
    assert np.isnan(plain[:, 2:5, 2:5]).any()  # the halo is lost
    assert np.isnan(adaptive[:, land_values]).all()
    np.testing.assert_allclose(adaptive[0][~land_values], 2.0, rtol=1e-8)
    np.testing.assert_allclose(adaptive[1][~land_values], 3.0, rtol=1e-8)

    jitted = jax.jit(lambda f, m: Gradient(nan_policy="adaptive")(f, m))(
        field.to_jax(), land.to_jax()
    )
    np.testing.assert_allclose(np.asarray(jitted), adaptive, rtol=1e-12)


def test_gradient_operator_refuses_a_mask_on_another_grid() -> None:
    field = _linear(UTM)
    shifted = (10.0, 0.0, 500_010.0, 0.0, -10.0, 4_000_000.0)
    land = GeoArray(np.zeros(field.shape, bool), shifted, "EPSG:32613", dims=("y", "x"))
    with pytest.raises(ValueError):
        Gradient()(field, land)


# ---------------------------------------------------------------------------
# Review fixes (#585)
# ---------------------------------------------------------------------------


def test_an_isolated_nodata_pixel_has_no_derivative() -> None:
    values = np.ones((5, 5))
    values[2, 2] = -9999.0
    field = GeoArray(values, UTM, "EPSG:32613", fill_value=-9999.0, dims=("y", "x"))
    for policy in ("propagate", "adaptive"):
        out = Gradient(nan_policy=policy)(field)
        assert np.isnan(np.asarray(out)[:, 2, 2]).all()
        assert not np.asarray(out.valid_mask())[2, 2]


def test_a_plain_jax_array_stays_on_jax() -> None:
    rows, cols = np.mgrid[0:4, 0:5].astype(float)
    f = jax.numpy.asarray(rows + 2.0 * cols)
    eager = Gradient()(f)
    jitted = jax.jit(Gradient())(f)
    for out in (eager, jitted):
        assert isinstance(out, jax.Array)
        np.testing.assert_allclose(np.asarray(out)[1], 2.0)


def test_a_batched_carrier_is_refused_with_map_over() -> None:
    import geoarray

    a = _linear(UTM)
    b = _linear((10.0, 0.0, 600_000.0, 0.0, -10.0, 4_000_000.0))
    batch = geoarray.stack_items([a, b], dim="patch")
    with pytest.raises(ValueError, match="map_over"):
        Gradient()(batch)


@pytest.mark.parametrize(
    ("size", "accuracy", "method"),
    [(3, 2, "forward"), (3, 2, "backward"), (5, 4, "central"), (2, 1, "central")],
)
def test_a_short_axis_raises_instead_of_reading_outside_it(
    size: int, accuracy: int, method: str
) -> None:
    with pytest.raises(ValueError, match="short"):
        difference(np.arange(float(size)), accuracy=accuracy, method=method)


def test_adaptive_descends_on_a_short_axis() -> None:
    got = adaptive_difference(np.arange(3.0), accuracy=4)
    np.testing.assert_allclose(got, 1.0)


def test_a_plain_jax_mask_under_jit() -> None:
    field = _linear(UTM)
    land = np.zeros(field.shape, bool)
    land[3, 2:5] = True
    want = np.asarray(Gradient(nan_policy="adaptive")(field, land))
    got = jax.jit(lambda f, m: Gradient(nan_policy="adaptive")(f, m))(
        field.to_jax(), jax.numpy.asarray(land)
    )
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12)


def test_nodata_in_one_band_drops_the_pixel_in_every_band() -> None:
    values = np.ones((2, 5, 5))
    values[0, 2, 2] = -9999.0
    field = GeoArray(
        values, UTM, "EPSG:32613", fill_value=-9999.0, dims=("band", "y", "x")
    )
    for policy in ("propagate", "adaptive"):
        out = np.asarray(Gradient(nan_policy=policy)(field))
        assert np.isnan(out[:, :, 2, 2]).all()  # both bands, both components


def test_a_time_mask_is_aligned_by_dim_name() -> None:
    values = np.ones((3, 3, 5, 6))  # T == C, the case positional broadcast gets wrong
    stack = GeoArray(values, UTM, "EPSG:32613", dims=("time", "band", "y", "x"))
    land = np.zeros((3, 5, 6), bool)
    land[1, 2, 3] = True  # one pixel, in frame 1 only
    mask = GeoArray(land, UTM, "EPSG:32613", dims=("time", "y", "x"))
    out = np.asarray(Gradient()(stack, mask))  # (time, band, component, y, x)
    assert np.isnan(out[1, :, :, 2, 3]).all()  # every band of frame 1
    assert np.isfinite(out[[0, 2], :, :, 2, 3]).all()  # other frames untouched
    with pytest.raises(ValueError, match="dims"):
        bad = GeoArray(land[None], UTM, "EPSG:32613", dims=("member", "time", "y", "x"))
        Gradient()(stack, bad)


@pytest.mark.parametrize(
    "call",
    [
        lambda t: difference(t),
        lambda t: adaptive_difference(t),
        lambda t: gradient(t),
        lambda t: adaptive_gradient(t),
        lambda t: adaptive_gradient(np.ones((7, 9)), mask=t),
    ],
    ids=["difference", "adaptive_difference", "gradient", "adaptive_gradient", "mask"],
)
def test_the_primitives_refuse_a_geotensor(call) -> None:
    pytest.importorskip("georeader")
    from geoarray.interop.georeader import to_georeader

    tensor = to_georeader(_linear(UTM))
    with pytest.raises(TypeError, match="from_georeader"):
        call(tensor)


def test_a_geographic_crs_in_grads_uses_its_own_unit() -> None:
    """NTF Paris (EPSG:4807, grads) and NTF (EPSG:4275, degrees) share an ellipsoid."""
    deg = (0.25, 0.0, 2.0, 0.0, -0.25, 50.0)
    grad = tuple(v * 400.0 / 360.0 for v in deg)
    _, lat = _centres(deg, (12, 8))
    # the same field, f = latitude in degrees, on the same points in each unit
    in_degrees = Gradient()(GeoArray(lat, deg, "EPSG:4275", dims=("y", "x")))
    in_grads = Gradient()(GeoArray(lat, grad, "EPSG:4807", dims=("y", "x")))
    np.testing.assert_allclose(np.asarray(in_grads), np.asarray(in_degrees), rtol=1e-9)


# ---------------------------------------------------------------------------
# Unevenly spaced axes: `coords=` and `nonuniform_coeffs`
# ---------------------------------------------------------------------------

LEVELS = np.array([0.0, 5.0, 15.0, 30.0, 50.0, 75.0, 105.0, 140.0, 180.0])


@pytest.mark.parametrize(("derivative", "accuracy", "method"), CASES)
def test_nonuniform_coeffs_reduce_to_the_uniform_stencil(
    derivative: int, accuracy: int, method: str
) -> None:
    offsets = difference_offsets(
        derivative=derivative, accuracy=accuracy, method=method
    )
    got = nonuniform_coeffs(2.5 * np.asarray(offsets, dtype=float), derivative)
    want = np.asarray(difference_coeffs(offsets, derivative)) / 2.5**derivative
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(("derivative", "accuracy", "method"), CASES)
def test_evenly_spaced_coords_equal_the_step(
    derivative: int, accuracy: int, method: str
) -> None:
    f = np.random.default_rng(accuracy).normal(size=(4, 17))
    coords = -1.5 * np.arange(17.0)
    got = difference(
        f, coords=coords, derivative=derivative, accuracy=accuracy, method=method
    )
    want = difference(
        f, step=-1.5, derivative=derivative, accuracy=accuracy, method=method
    )
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("accuracy", [1, 2, 3])
def test_uneven_levels_are_exact_for_the_stencils_degree(accuracy: int) -> None:
    """A K-point stencil on uneven levels is exact for polynomials of degree K - 1."""
    for derivative in (1, 2):
        degree = derivative + accuracy - 1  # the one-sided stencils' degree
        f = LEVELS**degree
        want = np.polyval(np.polyder(np.eye(degree + 1)[0], derivative), LEVELS)
        got = difference(f, coords=LEVELS, derivative=derivative, accuracy=accuracy)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_uneven_levels_run_under_jit_with_traced_coords() -> None:
    want = difference(LEVELS**2, coords=LEVELS)
    jitted = jax.jit(lambda f, z: difference(f, coords=z))
    got = jitted(jax.numpy.asarray(LEVELS**2), jax.numpy.asarray(LEVELS))
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12)


def test_adaptive_difference_on_uneven_levels_keeps_the_cells_next_to_a_gap() -> None:
    f = LEVELS.copy()  # f = z: every stencil is exact
    f[4] = np.nan
    got = adaptive_difference(f, coords=LEVELS)
    assert np.isnan(got[4]) and np.isfinite(np.delete(got, 4)).all()
    np.testing.assert_allclose(np.delete(got, 4), 1.0, rtol=1e-12)


def test_coords_must_fit_the_axis_and_not_wrap() -> None:
    with pytest.raises(ValueError, match="entries"):
        difference(np.zeros((3, 9)), coords=LEVELS[:5])
    with pytest.raises(ValueError, match="periodic"):
        difference(np.zeros((3, 9)), coords=LEVELS, periodic=True)
