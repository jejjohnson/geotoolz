"""The calculus family on the gradient stack: 1-D, 2-D and 3-D (#588).

- L1 `divergence_2d`, `curl_2d`, `laplacian_2d` and their `adaptive_*` twins equal
  finitediffx's on Cartesian grids and keep the invariants (solid-body
  rotation, a quadratic) up to the edges and, adaptively, up to a mask;
- L2 `Divergence2D`, `Curl2D`, `Laplacian2D` map them to the CRS's axes on
  north-up, flipped and rotated grids, use the CRS ellipsoid on a geographic
  CRS, run on NumPy and JAX and trace once under ``jax.jit``.
"""

from __future__ import annotations

import numpy as np
import pytest
from geoarray import GeoArray, GridMismatchError
from geoarray.coords import TimeAxis, Vertical, VerticalAxis

import geoops as gz
from geoops.geom import (
    Curl2D,
    Curl3D,
    Divergence1D,
    Divergence2D,
    Divergence3D,
    Gradient1D,
    Gradient2D,
    Gradient3D,
    Laplacian1D,
    Laplacian2D,
    Laplacian3D,
    adaptive_curl_2d,
    adaptive_curl_3d,
    adaptive_divergence_1d,
    adaptive_divergence_2d,
    adaptive_divergence_3d,
    adaptive_gradient_1d,
    adaptive_gradient_3d,
    adaptive_laplacian_1d,
    adaptive_laplacian_2d,
    adaptive_laplacian_3d,
    curl_2d,
    curl_3d,
    divergence_1d,
    divergence_2d,
    divergence_3d,
    gradient_1d,
    gradient_3d,
    laplacian_1d,
    laplacian_2d,
    laplacian_3d,
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
    np.testing.assert_allclose(divergence_2d(u, v, dy, dx, **kw), want_div, atol=1e-9)
    np.testing.assert_allclose(curl_2d(u, v, dy, dx, **kw), want_curl, atol=1e-9)
    np.testing.assert_allclose(laplacian_2d(u, dy, dx, **kw), want_lap, atol=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_primitives_keep_the_invariants_up_to_the_edges(backend: str) -> None:
    rows, cols = np.mgrid[0:6, 0:8].astype(float)
    xp = np if backend == "numpy" else jax.numpy
    # coordinates with steps dy = 0.5, dx = 2
    y, x = xp.asarray(0.5 * rows), xp.asarray(2.0 * cols)
    for fn in (curl_2d, adaptive_curl_2d):
        np.testing.assert_allclose(np.asarray(fn(-y, x, 0.5, 2.0)), 2.0, rtol=1e-12)
    for fn in (divergence_2d, adaptive_divergence_2d):
        np.testing.assert_allclose(np.asarray(fn(-y, x, 0.5, 2.0)), 0.0, atol=1e-12)
    for fn in (laplacian_2d, adaptive_laplacian_2d):
        out = np.asarray(fn(x**2 + y**2, 0.5, 2.0))
        np.testing.assert_allclose(out, 4.0, rtol=1e-12)
    x, y = xp.asarray(cols), xp.asarray(rows)  # index coordinates, scaled by the steps
    jitted = jax.jit(lambda f: laplacian_2d(f, 0.5, 2.0))
    np.testing.assert_allclose(jitted(jax.numpy.asarray(x**2 + y**2)), 2 / 4 + 2 / 0.25)


@pytest.mark.parametrize(
    "mask_kind",
    [pytest.param("island", id="island"), pytest.param("wall", id="one-column-wall")],
)
def test_adaptive_keeps_the_invariants_next_to_a_mask(mask_kind: str) -> None:
    y, x = np.mgrid[0:9, 0:11].astype(float)
    mask = (y == 4) & (x == 5) if mask_kind == "island" else x == 5
    zeta = adaptive_curl_2d(-y, x, mask=mask)
    div = adaptive_divergence_2d(-y, x, mask=mask)
    lap = adaptive_laplacian_2d(x**2 + y**2, mask=mask)
    for out in (zeta, div, lap):
        np.testing.assert_array_equal(np.isnan(out), mask)  # only the mask is lost
    np.testing.assert_allclose(zeta[~mask], 2.0, rtol=1e-12)
    np.testing.assert_allclose(div[~mask], 0.0, atol=1e-12)
    np.testing.assert_allclose(lap[~mask], 4.0, rtol=1e-12)


def test_propagate_spreads_nan_one_stencil_halo() -> None:
    y, x = np.mgrid[0:9, 0:11].astype(float)
    f = x**2 + y**2
    f[4, 5] = np.nan
    lost = np.isnan(laplacian_2d(f))
    want = np.zeros_like(lost)
    want[3:6, 5] = True  # the 3-point stencils along rows
    want[4, 4:7] = True  # and along columns
    np.testing.assert_array_equal(lost, want)


def test_the_primitives_refuse_a_geotensor() -> None:
    pytest.importorskip("georeader")
    from geoarray.interop.georeader import to_georeader

    tensor = to_georeader(_bowl(UTM))
    for call in (
        lambda t: divergence_2d(t, t),
        lambda t: adaptive_curl_2d(t, t),
        lambda t: laplacian_2d(t),
        lambda t: adaptive_laplacian_2d(t),
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
        "curl_2d": (Curl2D()(u, v), 2.0),
        "divergence_2d": (Divergence2D()(u, v), 0.0),
        "laplacian_2d": (Laplacian2D()(f), 4.0),
    }
    for out, want in outs.values():
        assert out.dims == ("y", "x")
        assert np.isnan(out.fill_value)
        np.testing.assert_allclose(np.asarray(out), want, rtol=1e-9, atol=1e-9)
    jitted = {
        "curl_2d": jax.jit(Curl2D())(u.to_jax(), v.to_jax()),
        "divergence_2d": jax.jit(Divergence2D())(u.to_jax(), v.to_jax()),
        "laplacian_2d": jax.jit(Laplacian2D())(f.to_jax()),
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
        return Curl2D(nan_policy="adaptive")(u, v, mask)

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
    adaptive = np.asarray(Curl2D(nan_policy="adaptive")(u, v, mask))
    np.testing.assert_array_equal(np.isnan(adaptive), land)
    np.testing.assert_allclose(adaptive[~land], 2.0, rtol=1e-9)
    propagated = np.asarray(Curl2D()(u, v, mask))
    assert np.isnan(propagated[:, 3]).all() and np.isnan(propagated[:, 5]).all()


def test_either_components_nodata_drops_the_cell() -> None:
    u, v = _rotation(UTM)
    values = np.asarray(v).copy()
    values[3, 4] = -9999.0
    v = GeoArray(values, UTM, "EPSG:32613", fill_value=-9999.0, dims=("y", "x"))
    out = np.asarray(Divergence2D(nan_policy="adaptive")(u, v))
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
    out = Laplacian2D()(stack)
    assert out.dims == ("band", "y", "x")
    want = np.broadcast_to(np.array([4.0, 8.0])[:, None, None], out.shape)
    np.testing.assert_allclose(np.asarray(out), want, rtol=1e-9)


def test_plain_arrays_are_differentiated_in_pixel_units() -> None:
    y, x = np.mgrid[0:5, 0:6].astype(float)
    np.testing.assert_allclose(Curl2D()(-y, x), 2.0)
    np.testing.assert_allclose(Laplacian2D()(x**2 + y**2), 4.0)
    on_jax = Laplacian2D()(jax.numpy.asarray(x**2))
    assert isinstance(on_jax, jax.Array)


def test_a_sheared_grid_raises_naming_reproject() -> None:
    sheared = (10.0, 3.0, 500_000.0, 0.0, -10.0, 4_000_000.0)
    with pytest.raises(ValueError, match="Reproject"):
        Laplacian2D()(_bowl(sheared))
    # divergence and curl need first derivatives only: exact on any affine grid
    u, v = _rotation(sheared)
    np.testing.assert_allclose(np.asarray(Curl2D()(u, v)), 2.0, rtol=1e-9)
    # under jit the transform is traced: the sheared grid gives NaN instead
    out = jax.jit(Laplacian2D())(_bowl(sheared).to_jax())
    assert np.isnan(np.asarray(out)).all()


def test_a_rotated_geographic_grid_raises_for_the_laplacian() -> None:
    rotated = (0.1, 0.01, 10.0, 0.0, -0.1, 50.0)
    with pytest.raises(ValueError, match="latitude and longitude"):
        Laplacian2D()(_carrier(np.ones((6, 6)), rotated, "EPSG:4326"))


def test_inputs_on_different_grids_or_shapes_are_refused() -> None:
    u, v = _rotation(UTM)
    moved = (10.0, 0.0, 500_010.0, 0.0, -10.0, 4_000_000.0)
    with pytest.raises(GridMismatchError, match="Curl2D"):
        Curl2D()(u, _carrier(np.asarray(v), moved, "EPSG:32613"))
    with pytest.raises(ValueError, match="share a shape"):
        Divergence2D()(np.zeros((3, 4)), np.zeros((4, 4)))
    with pytest.raises(TypeError, match="all GeoArrays or all plain arrays"):
        Divergence2D()(u, np.asarray(v))


def test_a_dataarray_is_refused_with_the_lift_hint() -> None:
    pytest.importorskip("xarray")
    da = _bowl(UTM).to_xarray()
    with pytest.raises(TypeError, match=r"xrops\.lift\(Laplacian2D\(\)"):
        Laplacian2D()(da)
    with pytest.raises(TypeError, match=r"geoops\.geom\.curl_2d"):
        Curl2D()(da, da)


def test_bad_arguments_fail_in_the_constructor() -> None:
    with pytest.raises(ValueError, match="nan_policy"):
        Curl2D(nan_policy="fill")
    with pytest.raises(ValueError, match="accuracy"):
        Laplacian2D(accuracy=0)


def test_exported_at_the_top_level() -> None:
    assert gz.Divergence2D is Divergence2D
    assert gz.Curl2D is Curl2D
    assert gz.Laplacian2D is Laplacian2D


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
    first = np.asarray(Laplacian2D()(_carrier(f, transform, SPHERE)))
    inner = (slice(3, -3), slice(3, -3))
    assert np.abs(first - want)[inner].max() < 1e-4 * scale
    # one order higher, the one-sided edge stencils agree as well
    second = np.asarray(Laplacian2D(accuracy=2)(_carrier(f, transform, SPHERE)))
    assert np.abs(second - want).max() < 1e-3 * scale


def test_solid_body_rotation_on_the_sphere() -> None:
    """u = cos φ (rotation about the pole): ζ = 2 sin φ / R, divergence 0."""
    transform, _, phi = _lonlat()
    u = _carrier(np.cos(phi), transform, SPHERE)
    v = _carrier(np.zeros_like(phi), transform, SPHERE)
    zeta = np.asarray(Curl2D(accuracy=2)(u, v))
    # the edges' one-sided stencils set the error: about 1e-4 of 2 / R
    np.testing.assert_allclose(
        zeta, 2 * np.sin(phi) / EARTH_RADIUS, rtol=0, atol=2e-4 * 2 / EARTH_RADIUS
    )
    np.testing.assert_allclose(np.asarray(Divergence2D()(u, v)), 0.0, atol=1e-20)


def _radii(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WGS 84's meridional M(φ) and prime-vertical N(φ) radii."""
    from pyproj import CRS

    ellipsoid = CRS.from_epsg(4326).ellipsoid
    a = ellipsoid.semi_major_metre
    e2 = 1.0 - (ellipsoid.semi_minor_metre / a) ** 2
    w = np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    return a * (1.0 - e2) / w**3, a / w


def test_the_ellipsoidal_flux_form_conserves_a_uniform_flux() -> None:
    """v N cos φ constant has no divergence_2d, u N cos φ constant no curl."""
    transform, _, phi = _lonlat()
    _, prime = _radii(phi)
    flux = _carrier(1.0 / (prime * np.cos(phi)), transform, "EPSG:4326")
    zero = _carrier(np.zeros_like(phi), transform, "EPSG:4326")
    np.testing.assert_allclose(np.asarray(Divergence2D()(zero, flux)), 0.0, atol=1e-25)
    np.testing.assert_allclose(np.asarray(Curl2D()(flux, zero)), 0.0, atol=1e-25)


def test_the_ellipsoidal_laplacian_metric_term() -> None:
    """f = φ: ∇²f is the metric term alone, (1/(MN cos φ)) d/dφ(N cos φ / M)."""
    transform, _, phi = _lonlat()
    meridional, prime = _radii(phi)
    lat_degrees = np.rad2deg(phi)
    got = np.asarray(Laplacian2D()(_carrier(lat_degrees, transform, "EPSG:4326")))
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
        Curl2D()(
            _carrier(u, transform, "EPSG:4326"), _carrier(v, transform, "EPSG:4326")
        )
    )
    on_sphere = np.asarray(
        Curl2D()(_carrier(u, transform, SPHERE), _carrier(v, transform, SPHERE))
    )
    difference = np.abs(on_ellipsoid / on_sphere - 1.0)
    assert 1e-3 < np.median(difference) < 3e-3
    assert difference.max() < 3e-3


# ---------------------------------------------------------------------------
# 3-D: levels (``axis=-3`` / the ``level`` dim), evenly or unevenly spaced
# ---------------------------------------------------------------------------

DEPTHS = np.array([0.0, 5.0, 15.0, 30.0, 50.0, 80.0])


def _levels_carrier(
    values: np.ndarray,
    transform: tuple[float, ...] = UTM,
    *,
    units: str = "m",
    levels: np.ndarray = DEPTHS,
) -> GeoArray:
    depth = Vertical(kind="depth", units=units, positive="down")
    return GeoArray(
        values,
        transform,
        "EPSG:32613",
        dims=("level", "y", "x"),
        coords=(VerticalAxis(values=tuple(levels), vertical=depth),),
    )


def _xyz(transform: tuple[float, ...] = UTM):
    x, y = _centres(transform, (7, 9))
    z = DEPTHS[:, None, None] + 0.0 * x
    return x + 0.0 * z, y + 0.0 * z, z


@pytest.mark.parametrize("accuracy", [1, 2])
def test_3d_primitives_equal_finitediffx(accuracy: int) -> None:
    fdx = pytest.importorskip("finitediffx")
    jnp = jax.numpy
    rng = np.random.default_rng(accuracy)
    u, v, w = rng.normal(size=(3, 9, 10, 11))
    dz, dy, dx = 0.7, 0.3, -2.0
    kw = {"accuracy": accuracy}
    want_div = fdx.divergence(
        jnp.stack([w, v, u]), step_size=(dz, dy, dx), accuracy=accuracy, keepdims=False
    )
    want_lap = fdx.laplacian(jnp.asarray(u), step_size=(dz, dy, dx), accuracy=accuracy)
    np.testing.assert_allclose(
        divergence_3d(u, v, w, dz, dy, dx, **kw), want_div, atol=1e-9
    )
    np.testing.assert_allclose(laplacian_3d(u, dz, dy, dx, **kw), want_lap, atol=1e-9)
    # finitediffx's curl on (x, y, z) components and axes, as for the 2-D one
    t = (2, 1, 0)
    want_curl = fdx.curl(
        jnp.stack([u.transpose(t), v.transpose(t), w.transpose(t)]),
        step_size=(dx, dy, dz),
        accuracy=accuracy,
    )
    omega_x, omega_y, omega_z = (np.asarray(c).transpose(t) for c in want_curl)
    got = curl_3d(u, v, w, dz, dy, dx, **kw)
    np.testing.assert_allclose(got, np.stack([omega_z, omega_y, omega_x]), atol=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_3d_primitives_are_exact_on_uneven_levels(backend: str) -> None:
    xp = np if backend == "numpy" else jax.numpy
    z, y, x = np.mgrid[0:6, 0:5, 0:7].astype(float)
    z = DEPTHS[:, None, None] + 0.0 * z
    z, y, x, levels = (xp.asarray(a) for a in (z, y, x, DEPTHS))
    lap = laplacian_3d(x**2 + y**2 + z**2, z=levels)
    np.testing.assert_allclose(np.asarray(lap), 6.0, rtol=1e-9)
    grad = np.asarray(gradient_3d(3.0 * z + 2.0 * y + x, z=levels))
    for component, want in zip(grad, (3.0, 2.0, 1.0), strict=True):
        np.testing.assert_allclose(component, want, rtol=1e-9)
    omega = np.asarray(curl_3d(z, x, 0.0 * z, z=levels))  # u = z: shear; v = x: spin
    np.testing.assert_allclose(omega[0], 1.0, rtol=1e-9)  # ω_z = ∂v/∂x
    np.testing.assert_allclose(omega[1], 1.0, rtol=1e-9)  # ω_y = ∂u/∂z
    np.testing.assert_allclose(omega[2], 0.0, atol=1e-12)
    div = divergence_3d(x, y, -2.0 * z, z=levels)
    np.testing.assert_allclose(np.asarray(div), 0.0, atol=1e-9)
    jitted = jax.jit(lambda f, zz: laplacian_3d(f, z=zz))
    np.testing.assert_allclose(jitted(xp.asarray(x**2 + z**2), levels), 4.0, rtol=1e-9)


def test_3d_adaptive_primitives_keep_the_invariants_next_to_a_mask() -> None:
    z, y, x = np.mgrid[0:7, 0:7, 0:8].astype(float)
    rock = (z == 3) & (y == 3) & (x == 4)
    lap = adaptive_laplacian_3d(x**2 + y**2 + z**2, mask=rock)
    div = adaptive_divergence_3d(x, y, z, mask=rock)
    omega = adaptive_curl_3d(-y, x, 0.0 * z, mask=rock)
    grad = adaptive_gradient_3d(z + y + x, mask=rock)
    for out in (lap, div, *omega, *grad):
        np.testing.assert_array_equal(np.isnan(out), rock)
    np.testing.assert_allclose(lap[~rock], 6.0, rtol=1e-12)
    np.testing.assert_allclose(div[~rock], 3.0, rtol=1e-12)
    np.testing.assert_allclose(omega[0][~rock], 2.0, rtol=1e-12)


@pytest.mark.parametrize("transform", GRIDS)
def test_3d_operators_are_exact_on_uneven_levels(transform: tuple[float, ...]) -> None:
    x, y, z = _xyz(transform)
    lap = Laplacian3D()(_levels_carrier(x**2 + y**2 + z**2, transform))
    assert lap.dims == ("level", "y", "x")
    np.testing.assert_allclose(np.asarray(lap), 6.0, rtol=1e-9)
    grad = Gradient3D()(_levels_carrier(3.0 * z + 2.0 * y + x, transform))
    assert grad.dims == ("level", "component", "y", "x")
    for k, want in enumerate((3.0, 2.0, 1.0)):
        np.testing.assert_allclose(np.asarray(grad)[:, k], want, rtol=1e-8)
    u, v, w = (_levels_carrier(a, transform) for a in (z - y, x, 0.0 * z))
    omega = np.asarray(Curl3D()(u, v, w))
    np.testing.assert_allclose(omega[:, 0], 2.0, rtol=1e-8)  # dv/dx - du/dy
    np.testing.assert_allclose(omega[:, 1], 1.0, rtol=1e-8)  # ∂u/∂z
    np.testing.assert_allclose(omega[:, 2], 0.0, atol=1e-9)
    div = Divergence3D()(*(_levels_carrier(a, transform) for a in (x, y, -2.0 * z)))
    np.testing.assert_allclose(np.asarray(div), 0.0, atol=1e-9)


def test_3d_operators_run_under_jit_and_trace_once() -> None:
    traces = []

    @jax.jit
    def step(f: GeoArray) -> GeoArray:
        traces.append(1)
        return Laplacian3D(nan_policy="adaptive")(f)

    moved = (20.0, 0.0, 600_000.0, 0.0, -20.0, 4_100_000.0)
    for transform in (UTM, moved):
        x, y, z = _xyz(transform)
        out = step(_levels_carrier(x**2 + y**2 + z**2, transform).to_jax())
        np.testing.assert_allclose(np.asarray(out), 6.0, rtol=1e-9)
    assert len(traces) == 1


def test_3d_operators_need_physical_levels() -> None:
    x, _, z = _xyz()
    with pytest.raises(ValueError, match="'level' dim"):
        Laplacian3D()(_carrier(x[0], UTM, "EPSG:32613"))
    with pytest.raises(ValueError, match="units '1'"):
        Laplacian3D()(_levels_carrier(z, units="1"))
    bare = GeoArray(np.asarray(z), UTM, "EPSG:32613", dims=("level", "y", "x"))
    with pytest.raises(ValueError, match="no VerticalAxis"):
        Gradient3D()(bare)


def test_3d_operators_on_plain_arrays_use_pixel_units() -> None:
    z, y, x = np.mgrid[0:4, 0:5, 0:6].astype(float)
    np.testing.assert_allclose(Laplacian3D()(x**2 + y**2 + z**2), 6.0)
    assert Gradient3D()(z).shape == (4, 3, 5, 6)  # (level, component, y, x)


# ---------------------------------------------------------------------------
# 1-D: along one named dim, with that dim's spacing
# ---------------------------------------------------------------------------


def test_1d_primitives_on_even_and_uneven_axes() -> None:
    s = np.arange(6.0)
    np.testing.assert_allclose(gradient_1d(3.0 * s, 0.5), 6.0)
    np.testing.assert_allclose(divergence_1d(3.0 * s, coords=2.0 * s), 1.5)
    np.testing.assert_allclose(laplacian_1d(DEPTHS**2, coords=DEPTHS), 2.0, rtol=1e-9)
    gappy = DEPTHS.copy()
    gappy[2] = np.nan
    for fn in (adaptive_gradient_1d, adaptive_divergence_1d):
        got = fn(gappy, coords=DEPTHS)
        np.testing.assert_allclose(np.delete(got, 2), 1.0, rtol=1e-9)
    lap = adaptive_laplacian_1d(gappy**2, coords=DEPTHS)
    assert np.isnan(lap[2])


@pytest.mark.parametrize("transform", GRIDS[:2])
def test_1d_operators_along_x_and_y_match_the_2d_gradient(
    transform: tuple[float, ...],
) -> None:
    x, y = _centres(transform, (7, 9))
    f = _carrier(3.0 * x + 2.0 * y + x**2, transform, "EPSG:32613")
    d_y, d_x = np.asarray(Gradient2D()(f))
    np.testing.assert_allclose(np.asarray(Gradient1D(along="x")(f)), d_x, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(Divergence1D(along="y")(f)), d_y, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(Laplacian1D(along="x")(f)), 2.0, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(Laplacian1D(along="y")(f)), 0.0, atol=1e-9)


def test_1d_operators_along_x_or_y_refuse_a_rotated_grid() -> None:
    rotated = GRIDS[2].values[0]
    x, _ = _centres(rotated, (7, 9))
    with pytest.raises(ValueError, match="Reproject"):
        Gradient1D(along="x")(_carrier(x, rotated, "EPSG:32613"))


def test_1d_operators_along_the_level_dim() -> None:
    _, _, z = _xyz()
    got = np.asarray(Gradient1D(along="level")(_levels_carrier(z**2)))
    np.testing.assert_allclose(got[1:-1], 2.0 * z[1:-1], rtol=1e-9)  # interior exact
    lap = Laplacian1D(along="level")(_levels_carrier(z**2))
    np.testing.assert_allclose(np.asarray(lap), 2.0, rtol=1e-9)


def test_1d_operators_along_time_use_seconds_and_trace() -> None:
    stamps = ["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-08"]
    seconds = np.array([0.0, 1.0, 3.0, 7.0]) * 86_400.0
    x, _ = _centres(UTM, (7, 9))
    series = GeoArray(
        2.0 * seconds[:, None, None] + 0.0 * x,
        UTM,
        "EPSG:32613",
        dims=("time", "y", "x"),
        coords=(TimeAxis(stamps=stamps),),
    )
    np.testing.assert_allclose(np.asarray(Gradient1D(along="time")(series)), 2.0)
    jitted = jax.jit(Gradient1D(along="time"))(series.to_jax())
    np.testing.assert_allclose(np.asarray(jitted), 2.0, rtol=1e-12)


def test_1d_along_a_meridian_carries_its_changing_radius() -> None:
    """f = φ: d²f/ds² along a meridian is -M'/M³ (and 0 on a sphere)."""
    transform, _, phi = _lonlat()
    lat_degrees = np.rad2deg(phi)
    meridional, _ = _radii(phi)
    h = 1e-6
    dm = (_radii(phi + h)[0] - _radii(phi - h)[0]) / (2 * h)
    got = np.asarray(
        Laplacian1D(along="y")(_carrier(lat_degrees, transform, "EPSG:4326"))
    )
    want = -np.rad2deg(1.0) * dm / meridional**3
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-25)
    on_sphere = Laplacian1D(along="y")(_carrier(lat_degrees, transform, SPHERE))
    # roundoff of a second difference of a linear field; the term above is ~1e-14
    np.testing.assert_allclose(np.asarray(on_sphere), 0.0, atol=1e-20)


def test_1d_operators_refuse_an_unknown_dim_and_missing_coords() -> None:
    with pytest.raises(ValueError, match="along must be one of"):
        Gradient1D(along="band")
    with pytest.raises(ValueError, match="TimeAxis"):
        Gradient1D(along="time")(_bowl(UTM))
    with pytest.raises(ValueError, match="pass a GeoArray"):
        Laplacian1D(along="level")(np.zeros((3, 4, 5)))


def test_the_family_is_exported_at_the_top_level() -> None:
    for name in (
        "Gradient1D",
        "Gradient2D",
        "Gradient3D",
        "Divergence1D",
        "Divergence2D",
        "Divergence3D",
        "Curl2D",
        "Curl3D",
        "Laplacian1D",
        "Laplacian2D",
        "Laplacian3D",
    ):
        assert getattr(gz, name) is getattr(gz.geom, name)
    for removed in ("Gradient", "Divergence", "Curl", "Laplacian"):
        assert not hasattr(gz, removed)
    for removed in ("gradient", "divergence", "curl", "laplacian", "adaptive_gradient"):
        assert not hasattr(gz.geom, removed)
