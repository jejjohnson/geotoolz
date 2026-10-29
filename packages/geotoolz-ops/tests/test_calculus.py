"""Divergence, curl and the Laplacian on the gradient stack (#588).

- L1 `divergence`, `curl`, `laplacian` and their `adaptive_*` twins equal
  finitediffx's on Cartesian grids and keep the invariants (solid-body
  rotation, a quadratic) up to the edges and, adaptively, up to a mask;
- L2 `Divergence`, `Curl`, `Laplacian` map them to the CRS's axes on
  north-up, flipped and rotated grids, use the CRS ellipsoid on a geographic
  CRS, run on NumPy and JAX and trace once under ``jax.jit``.
"""

from __future__ import annotations

import numpy as np
import pytest
from geoarray import GeoArray, GridMismatchError

import geoops as gz
from geoops.geom import (
    Curl,
    Divergence,
    Laplacian,
    adaptive_curl,
    adaptive_divergence,
    adaptive_laplacian,
    curl,
    divergence,
    laplacian,
)


jax = pytest.importorskip("jax")

UTM = (10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)
GRIDS = [
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
]
EARTH_RADIUS = 6_371_000.0
SPHERE = f"+proj=longlat +R={EARTH_RADIUS} +no_defs"


@pytest.fixture(autouse=True)
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _centres(transform: tuple[float, ...], shape: tuple[int, int]):
    a, b, c, d, e, f = transform
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]] + 0.5
    x, y = a * cols + b * rows + c, d * cols + e * rows + f
    return x - x.mean(), y - y.mean()  # centred, so x² + y² stays well scaled


def _carrier(values: np.ndarray, transform: tuple[float, ...], crs: str) -> GeoArray:
    return GeoArray(values, transform, crs, dims=("y", "x"))


def _rotation(transform: tuple[float, ...], crs: str = "EPSG:32613"):
    """Solid-body rotation u = -y, v = x: curl 2, divergence 0."""
    x, y = _centres(transform, (7, 9))
    return _carrier(-y, transform, crs), _carrier(x, transform, crs)


def _bowl(transform: tuple[float, ...], crs: str = "EPSG:32613") -> GeoArray:
    """f = x² + y²: Laplacian 4."""
    x, y = _centres(transform, (7, 9))
    return _carrier(x**2 + y**2, transform, crs)


# ---------------------------------------------------------------------------
# L1: finitediffx on Cartesian grids, and the invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("accuracy", [1, 2, 3, 4])
def test_primitives_equal_finitediffx(accuracy: int) -> None:
    fdx = pytest.importorskip("finitediffx")
    jnp = jax.numpy
    rng = np.random.default_rng(accuracy)
    u, v = rng.normal(size=(2, 15, 12))
    dy, dx = 0.3, -2.0
    # finitediffx orders components and steps by axis: (rows, columns)
    want_div = fdx.divergence(
        jnp.stack([v, u]), step_size=(dy, dx), accuracy=accuracy, keepdims=False
    )
    # its 2-D curl is right-handed in axis order, so pass the (x, y) view
    want_curl = fdx.curl(
        jnp.stack([u.T, v.T]), step_size=(dx, dy), accuracy=accuracy, keepdims=False
    ).T
    want_lap = fdx.laplacian(jnp.asarray(u), step_size=(dy, dx), accuracy=accuracy)
    kw = {"accuracy": accuracy}
    np.testing.assert_allclose(divergence(u, v, dy, dx, **kw), want_div, atol=1e-9)
    np.testing.assert_allclose(curl(u, v, dy, dx, **kw), want_curl, atol=1e-9)
    np.testing.assert_allclose(laplacian(u, dy, dx, **kw), want_lap, atol=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_primitives_keep_the_invariants_up_to_the_edges(backend: str) -> None:
    rows, cols = np.mgrid[0:6, 0:8].astype(float)
    xp = np if backend == "numpy" else jax.numpy
    # coordinates with steps dy = 0.5, dx = 2
    y, x = xp.asarray(0.5 * rows), xp.asarray(2.0 * cols)
    for fn in (curl, adaptive_curl):
        np.testing.assert_allclose(np.asarray(fn(-y, x, 0.5, 2.0)), 2.0, rtol=1e-12)
    for fn in (divergence, adaptive_divergence):
        np.testing.assert_allclose(np.asarray(fn(-y, x, 0.5, 2.0)), 0.0, atol=1e-12)
    for fn in (laplacian, adaptive_laplacian):
        out = np.asarray(fn(x**2 + y**2, 0.5, 2.0))
        np.testing.assert_allclose(out, 4.0, rtol=1e-12)
    x, y = xp.asarray(cols), xp.asarray(rows)  # index coordinates, scaled by the steps
    jitted = jax.jit(lambda f: laplacian(f, 0.5, 2.0))
    np.testing.assert_allclose(jitted(jax.numpy.asarray(x**2 + y**2)), 2 / 4 + 2 / 0.25)


@pytest.mark.parametrize(
    "mask_kind",
    [pytest.param("island", id="island"), pytest.param("wall", id="one-column-wall")],
)
def test_adaptive_keeps_the_invariants_next_to_a_mask(mask_kind: str) -> None:
    y, x = np.mgrid[0:9, 0:11].astype(float)
    mask = (y == 4) & (x == 5) if mask_kind == "island" else x == 5
    zeta = adaptive_curl(-y, x, mask=mask)
    div = adaptive_divergence(-y, x, mask=mask)
    lap = adaptive_laplacian(x**2 + y**2, mask=mask)
    for out in (zeta, div, lap):
        np.testing.assert_array_equal(np.isnan(out), mask)  # only the mask is lost
    np.testing.assert_allclose(zeta[~mask], 2.0, rtol=1e-12)
    np.testing.assert_allclose(div[~mask], 0.0, atol=1e-12)
    np.testing.assert_allclose(lap[~mask], 4.0, rtol=1e-12)


def test_propagate_spreads_nan_one_stencil_halo() -> None:
    y, x = np.mgrid[0:9, 0:11].astype(float)
    f = x**2 + y**2
    f[4, 5] = np.nan
    lost = np.isnan(laplacian(f))
    want = np.zeros_like(lost)
    want[3:6, 5] = True  # the 3-point stencils along rows
    want[4, 4:7] = True  # and along columns
    np.testing.assert_array_equal(lost, want)


def test_the_primitives_refuse_a_geotensor() -> None:
    pytest.importorskip("georeader")
    from geoarray.interop.georeader import to_georeader

    tensor = to_georeader(_bowl(UTM))
    for call in (
        lambda t: divergence(t, t),
        lambda t: adaptive_curl(t, t),
        lambda t: laplacian(t),
        lambda t: adaptive_laplacian(t),
    ):
        with pytest.raises(TypeError, match="from_georeader"):
            call(tensor)


# ---------------------------------------------------------------------------
# L2: the operators in map axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform", GRIDS)
def test_operators_are_exact_in_map_axes(transform: tuple[float, ...]) -> None:
    u, v = _rotation(transform)
    f = _bowl(transform)
    outs = {
        "curl": (Curl()(u, v), 2.0),
        "divergence": (Divergence()(u, v), 0.0),
        "laplacian": (Laplacian()(f), 4.0),
    }
    for out, want in outs.values():
        assert out.dims == ("y", "x")
        assert np.isnan(out.fill_value)
        np.testing.assert_allclose(np.asarray(out), want, rtol=1e-9, atol=1e-9)
    jitted = {
        "curl": jax.jit(Curl())(u.to_jax(), v.to_jax()),
        "divergence": jax.jit(Divergence())(u.to_jax(), v.to_jax()),
        "laplacian": jax.jit(Laplacian())(f.to_jax()),
    }
    for key, got in jitted.items():
        assert got.backend == "jax"
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(outs[key][0]), rtol=1e-12, atol=1e-12
        )


def test_jit_traces_once_across_transforms_and_masks() -> None:
    traces = []

    @jax.jit
    def step(u: GeoArray, v: GeoArray, mask: GeoArray) -> GeoArray:
        traces.append(1)
        return Curl(nan_policy="adaptive")(u, v, mask)

    moved = (20.0, 0.0, 600_000.0, 0.0, -20.0, 4_100_000.0)
    for transform, cell in ((UTM, (2, 3)), (moved, (4, 5))):
        u, v = _rotation(transform)
        land = np.zeros((7, 9), dtype=bool)
        land[cell] = True
        out = step(u.to_jax(), v.to_jax(), _carrier(land, transform, "EPSG:32613"))
        np.testing.assert_array_equal(np.isnan(np.asarray(out)), land)
    assert len(traces) == 1
    np.testing.assert_allclose(np.nanmax(np.abs(np.asarray(out) - 2.0)), 0.0, atol=1e-9)


def test_adaptive_operator_keeps_the_coast() -> None:
    u, v = _rotation(UTM)
    land = np.zeros((7, 9), dtype=bool)
    land[:, 4] = True  # a one-column wall
    mask = _carrier(land, UTM, "EPSG:32613")
    adaptive = np.asarray(Curl(nan_policy="adaptive")(u, v, mask))
    np.testing.assert_array_equal(np.isnan(adaptive), land)
    np.testing.assert_allclose(adaptive[~land], 2.0, rtol=1e-9)
    propagated = np.asarray(Curl()(u, v, mask))
    assert np.isnan(propagated[:, 3]).all() and np.isnan(propagated[:, 5]).all()


def test_either_components_nodata_drops_the_cell() -> None:
    u, v = _rotation(UTM)
    values = np.asarray(v).copy()
    values[3, 4] = -9999.0
    v = GeoArray(values, UTM, "EPSG:32613", fill_value=-9999.0, dims=("y", "x"))
    out = np.asarray(Divergence(nan_policy="adaptive")(u, v))
    assert np.isnan(out[3, 4])
    assert np.isnan(out).sum() == 1


def test_a_band_stack_is_differentiated_band_by_band() -> None:
    f = _bowl(UTM)
    stack = GeoArray(
        np.stack([np.asarray(f), 2.0 * np.asarray(f)]),
        UTM,
        "EPSG:32613",
        dims=("band", "y", "x"),
    )
    out = Laplacian()(stack)
    assert out.dims == ("band", "y", "x")
    want = np.broadcast_to(np.array([4.0, 8.0])[:, None, None], out.shape)
    np.testing.assert_allclose(np.asarray(out), want, rtol=1e-9)


def test_plain_arrays_are_differentiated_in_pixel_units() -> None:
    y, x = np.mgrid[0:5, 0:6].astype(float)
    np.testing.assert_allclose(Curl()(-y, x), 2.0)
    np.testing.assert_allclose(Laplacian()(x**2 + y**2), 4.0)
    on_jax = Laplacian()(jax.numpy.asarray(x**2))
    assert isinstance(on_jax, jax.Array)


def test_a_sheared_grid_raises_naming_reproject() -> None:
    sheared = (10.0, 3.0, 500_000.0, 0.0, -10.0, 4_000_000.0)
    with pytest.raises(ValueError, match="Reproject"):
        Laplacian()(_bowl(sheared))
    # divergence and curl need first derivatives only: exact on any affine grid
    u, v = _rotation(sheared)
    np.testing.assert_allclose(np.asarray(Curl()(u, v)), 2.0, rtol=1e-9)
    # under jit the transform is traced: the sheared grid gives NaN instead
    out = jax.jit(Laplacian())(_bowl(sheared).to_jax())
    assert np.isnan(np.asarray(out)).all()


def test_a_rotated_geographic_grid_raises_for_the_laplacian() -> None:
    rotated = (0.1, 0.01, 10.0, 0.0, -0.1, 50.0)
    with pytest.raises(ValueError, match="latitude and longitude"):
        Laplacian()(_carrier(np.ones((6, 6)), rotated, "EPSG:4326"))


def test_inputs_on_different_grids_or_shapes_are_refused() -> None:
    u, v = _rotation(UTM)
    moved = (10.0, 0.0, 500_010.0, 0.0, -10.0, 4_000_000.0)
    with pytest.raises(GridMismatchError, match="Curl"):
        Curl()(u, _carrier(np.asarray(v), moved, "EPSG:32613"))
    with pytest.raises(ValueError, match="share a shape"):
        Divergence()(np.zeros((3, 4)), np.zeros((4, 4)))
    with pytest.raises(TypeError, match="all GeoArrays or all plain arrays"):
        Divergence()(u, np.asarray(v))


def test_a_dataarray_is_refused_with_the_lift_hint() -> None:
    pytest.importorskip("xarray")
    da = _bowl(UTM).to_xarray()
    with pytest.raises(TypeError, match=r"xrops\.lift\(Laplacian\(\)"):
        Laplacian()(da)
    with pytest.raises(TypeError, match=r"geoops\.geom\.curl"):
        Curl()(da, da)


def test_bad_arguments_fail_in_the_constructor() -> None:
    with pytest.raises(ValueError, match="nan_policy"):
        Curl(nan_policy="fill")
    with pytest.raises(ValueError, match="accuracy"):
        Laplacian(accuracy=0)


def test_exported_at_the_top_level() -> None:
    assert gz.Divergence is Divergence
    assert gz.Curl is Curl
    assert gz.Laplacian is Laplacian


# ---------------------------------------------------------------------------
# Geographic CRS: the sphere, and the ellipsoid
# ---------------------------------------------------------------------------


def _lonlat(shape: tuple[int, int] = (240, 40)):
    """A 0.5° grid from 60° S to 60° N, rows going north."""
    transform = (0.5, 0.0, 0.0, 0.0, 0.5, -60.0)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]] + 0.5
    return transform, np.deg2rad(0.5 * cols), np.deg2rad(-60.0 + 0.5 * rows)


@pytest.mark.parametrize(
    "harmonic",
    [
        pytest.param(lambda lam, phi: 3 * np.sin(phi) ** 2 - 1, id="Y20"),
        pytest.param(
            lambda lam, phi: np.sin(phi) * np.cos(phi) * np.cos(lam), id="Y21"
        ),
        pytest.param(lambda lam, phi: np.cos(phi) ** 2 * np.cos(2 * lam), id="Y22"),
    ],
)
def test_spherical_harmonics_are_eigenfunctions(harmonic) -> None:
    """∇² Y_2^m = -6 Y_2^m / R² on a sphere, to the grid's truncation error."""
    transform, lam, phi = _lonlat()
    f = harmonic(lam, phi)
    want = -6.0 * f / EARTH_RADIUS**2
    scale = 6.0 / EARTH_RADIUS**2
    first = np.asarray(Laplacian()(_carrier(f, transform, SPHERE)))
    inner = (slice(3, -3), slice(3, -3))
    assert np.abs(first - want)[inner].max() < 1e-4 * scale
    # one order higher, the one-sided edge stencils agree as well
    second = np.asarray(Laplacian(accuracy=2)(_carrier(f, transform, SPHERE)))
    assert np.abs(second - want).max() < 1e-3 * scale


def test_solid_body_rotation_on_the_sphere() -> None:
    """u = cos φ (rotation about the pole): ζ = 2 sin φ / R, divergence 0."""
    transform, _, phi = _lonlat()
    u = _carrier(np.cos(phi), transform, SPHERE)
    v = _carrier(np.zeros_like(phi), transform, SPHERE)
    zeta = np.asarray(Curl(accuracy=2)(u, v))
    # the edges' one-sided stencils set the error: about 1e-4 of 2 / R
    np.testing.assert_allclose(
        zeta, 2 * np.sin(phi) / EARTH_RADIUS, rtol=0, atol=2e-4 * 2 / EARTH_RADIUS
    )
    np.testing.assert_allclose(np.asarray(Divergence()(u, v)), 0.0, atol=1e-20)


def _radii(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WGS 84's meridional M(φ) and prime-vertical N(φ) radii."""
    from pyproj import CRS

    ellipsoid = CRS.from_epsg(4326).ellipsoid
    a = ellipsoid.semi_major_metre
    e2 = 1.0 - (ellipsoid.semi_minor_metre / a) ** 2
    w = np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    return a * (1.0 - e2) / w**3, a / w


def test_the_ellipsoidal_flux_form_conserves_a_uniform_flux() -> None:
    """v N cos φ constant has no divergence, u N cos φ constant no curl."""
    transform, _, phi = _lonlat()
    _, prime = _radii(phi)
    flux = _carrier(1.0 / (prime * np.cos(phi)), transform, "EPSG:4326")
    zero = _carrier(np.zeros_like(phi), transform, "EPSG:4326")
    np.testing.assert_allclose(np.asarray(Divergence()(zero, flux)), 0.0, atol=1e-25)
    np.testing.assert_allclose(np.asarray(Curl()(flux, zero)), 0.0, atol=1e-25)


def test_the_ellipsoidal_laplacian_metric_term() -> None:
    """f = φ: ∇²f is the metric term alone, (1/(MN cos φ)) d/dφ(N cos φ / M)."""
    transform, _, phi = _lonlat()
    meridional, prime = _radii(phi)
    lat_degrees = np.rad2deg(phi)
    got = np.asarray(Laplacian()(_carrier(lat_degrees, transform, "EPSG:4326")))
    h = 1e-6
    ratio = lambda p: _radii(p)[1] * np.cos(p) / _radii(p)[0]
    slope = (ratio(phi + h) - ratio(phi - h)) / (2 * h)
    want = np.rad2deg(1.0) * slope / (meridional * prime * np.cos(phi))
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_the_ellipsoid_differs_from_the_sphere_by_a_few_tenths_of_a_percent() -> None:
    """Recorded for xrops, whose xrgrad predecessor uses a sphere of 6371 km."""
    transform, _, phi = _lonlat()
    u = np.cos(phi)
    v = np.zeros_like(phi)
    on_ellipsoid = np.asarray(
        Curl()(_carrier(u, transform, "EPSG:4326"), _carrier(v, transform, "EPSG:4326"))
    )
    on_sphere = np.asarray(
        Curl()(_carrier(u, transform, SPHERE), _carrier(v, transform, SPHERE))
    )
    difference = np.abs(on_ellipsoid / on_sphere - 1.0)
    assert 1e-3 < np.median(difference) < 3e-3
    assert difference.max() < 3e-3
