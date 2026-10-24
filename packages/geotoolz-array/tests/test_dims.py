"""Stored dims, coordinates and geoarray's named-array adapter (#517)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from types import ModuleType
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit import get_adapter
from pipekit_array import DimArray, KernelOp
from pipekit_array.testing import check_adapter, check_kernel_op

import geoarray
from geoarray import GeoArray, GeoArrayAdapter, GridMismatchError, LabelError
from geoarray.coords import PixelGrid, SpectralAxis, TimeAxis, VariableAxis


TRANSFORM = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
CRS = "EPSG:32613"
BANDS = SpectralAxis(("B2", "B4", "B8"), (490.0, 665.0, 842.0))


def _scene(
    shape: tuple[int, ...] = (3, 8, 6), dims: Any = "band y x", **kw: Any
) -> GeoArray:
    data = np.random.default_rng(0).random(shape)
    return GeoArray(data, TRANSFORM, CRS, dims=dims, **kw)


# -- stored dims ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims",
    [
        ("y", "x"),
        ("band", "y", "x"),
        ("time", "y", "x"),
        ("time", "band", "y", "x"),
        ("run", "time", "variable", "y", "x"),
    ],
)
def test_dims_round_trip(dims: tuple[str, ...]) -> None:
    shape = (2, 3, 4, 5, 6)[-len(dims) :]
    g = GeoArray(np.zeros(shape), TRANSFORM, CRS, dims=dims)
    assert g.dims == dims
    assert g.sizes == dict(zip(dims, shape, strict=True))
    assert GeoArray(np.zeros(shape), TRANSFORM, CRS, dims=" ".join(dims)).dims == dims


def test_time_y_x_is_not_read_as_bands() -> None:
    """Twelve monthly single-band frames are twelve frames, not twelve bands."""
    g = GeoArray(np.zeros((12, 4, 4)), TRANSFORM, CRS, dims="time y x")
    assert g.count == 1
    assert g.valid_mask(keep_time=True).shape == (12, 4, 4)


@pytest.mark.parametrize(
    ("dims", "error", "match"),
    [
        (("band", "x", "y"), ValueError, "end in"),
        (("band", "band", "y", "x"), ValueError, "repeat"),
        (("y", "x"), ValueError, "name 2 axes"),
        ((1, "y", "x"), TypeError, "strings"),
    ],
)
def test_bad_dims_raise(dims: Any, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        GeoArray(np.zeros((3, 4, 4)), TRANSFORM, CRS, dims=dims)


def test_omitted_dims_warn_and_follow_the_0_2_layout() -> None:
    with pytest.warns(DeprecationWarning, match=r"pass dims=\('band', 'y', 'x'\)"):
        g = GeoArray(np.zeros((3, 4, 4)), TRANSFORM, CRS)
    assert g.dims == ("band", "y", "x")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.isel(band=0)  # carriers built from it carry dims: no more warnings
        g[1:, :2]


def test_coordinates_are_checked() -> None:
    with pytest.raises(ValueError, match="size 2"):
        _scene((2, 4, 4), coords=(BANDS,))
    with pytest.raises(ValueError, match="not a non-spatial dim"):
        _scene((3, 4, 4), dims="variable y x", coords=(BANDS,))
    with pytest.raises(TypeError, match=r"geoarray\.coords"):
        _scene(coords=("B2",))
    with pytest.raises(ValueError, match="two coordinates"):
        _scene(coords=(BANDS, BANDS))


def test_batch_dims_index_transform_rows() -> None:
    transforms = np.stack([np.array(TRANSFORM) + k for k in range(4)])
    patches = GeoArray(np.zeros((4, 3, 8, 8)), transforms, CRS, dims="patch band y x")
    assert patches.batch_dims == ("patch",)
    one = patches.isel(patch=2)
    assert one.dims == ("band", "y", "x")
    np.testing.assert_array_equal(one.transform, transforms[2])
    assert "batch=(4,)" in repr(patches)


# -- selection -----------------------------------------------------------------------


def test_isel_moves_the_origin_and_slices_coordinates() -> None:
    """GA-G9: after a selection along a dim, its coordinate matches the data."""
    g = _scene(coords=(BANDS,))
    chip = g.isel(band=[2, 0], y=slice(2, 8, 2), x=slice(1, 5))
    assert chip.dims == ("band", "y", "x")
    assert chip.labels("band") == ("B8", "B2")
    assert chip.coord("band").wavelength_nm == (842.0, 490.0)
    # c′ = c + a·j₀ + b·i₀, f′ = f + d·j₀ + e·i₀; a′ = a·s_c, e′ = e·s_r
    np.testing.assert_array_equal(
        chip.transform, [20.0, 0.0, 600020.0, 0.0, -40.0, 3549960.0]
    )
    assert g.isel(band=1).coords == ()


@pytest.mark.parametrize("seed", range(2))
def test_named_slices_match_georeader(seed: int) -> None:
    """GA-G5: 500 random named slices per seed, steps 1 to 3, match GeoTensor.isel."""
    gt_module = pytest.importorskip("georeader.geotensor")
    from affine import Affine

    rng = np.random.default_rng(seed)
    g = _scene((2, 40, 30), coords=(SpectralAxis(("a", "b")),))
    gt = gt_module.GeoTensor(np.asarray(g.data), Affine(*TRANSFORM), CRS)
    for _ in range(500):
        r0, c0 = int(rng.integers(0, 39)), int(rng.integers(0, 29))
        r1, c1 = int(rng.integers(r0 + 1, 41)), int(rng.integers(c0 + 1, 31))
        sr, sc = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        ours = g.isel(y=slice(r0, r1, sr), x=slice(c0, c1, sc))
        theirs = gt.isel({"y": slice(r0, r1, sr), "x": slice(c0, c1, sc)})
        np.testing.assert_allclose(
            ours.transform, tuple(theirs.transform)[:6], atol=1e-9
        )
        np.testing.assert_array_equal(ours.data, theirs.values)


def test_sel_takes_exact_labels() -> None:
    g = _scene(coords=(BANDS,))
    assert g.sel(band="B8").dims == ("y", "x")
    np.testing.assert_array_equal(g.sel(band="B8").data, g.data[2])
    picked = g.sel(band=["B8", "B4"])
    assert picked.labels("band") == ("B8", "B4")
    np.testing.assert_array_equal(g.sel(band=np.array(["B2"])).data, g.data[:1])


@pytest.mark.parametrize(
    ("selection", "match"),
    [
        ({"band": slice("B2", "B4")}, "to_xarray"),
        ({"band": "B11"}, "not a label"),
        ({"y": 3}, "no labels"),
    ],
)
def test_sel_refuses_what_it_cannot_do(selection: dict, match: str) -> None:
    g = _scene(coords=(BANDS,))
    with pytest.raises(LabelError, match=match):
        g.sel(selection)
    with pytest.raises(LabelError, match="no labels"):
        _scene().sel(band="B2")


def test_by_name_a_band_index_may_leave_time_y_x() -> None:
    stack = _scene((2, 3, 4, 4), dims="time band y x")
    assert stack.isel(band=1).dims == ("time", "y", "x")
    with pytest.raises(IndexError, match=r"\(T, 1, H, W\)"):
        stack[:, 1]  # positionally, the 0.2 guard stays


def test_from_coords() -> None:
    dates = TimeAxis(["2026-06-01", "2026-06-11"])
    grid = PixelGrid((8, 6), CRS)
    g = GeoArray.from_coords(
        np.zeros((2, 3, 8, 6)), dates, BANDS, grid, transform=TRANSFORM
    )
    assert g.dims == ("time", "band", "y", "x")
    assert g.coords == (dates, BANDS)
    patches = GeoArray.from_coords(
        np.zeros((5, 8, 6)), "run", grid, transform=TRANSFORM
    )
    assert patches.dims == ("run", "y", "x") and patches.coords == ()
    point = PixelGrid((8, 6), CRS, "point")
    shifted = GeoArray.from_coords(np.zeros((8, 6)), point, transform=TRANSFORM)
    # a point transform maps index 0 to the first centre: the edge is ½ pixel out
    assert shifted.transform[2] == TRANSFORM[2] - 10.0
    assert shifted.transform[5] == TRANSFORM[5] + 10.0
    with pytest.raises(TypeError, match="PixelGrid"):
        GeoArray.from_coords(np.zeros((8, 6)), BANDS, transform=TRANSFORM)
    with pytest.raises(ValueError, match="spatial shape"):
        GeoArray.from_coords(np.zeros((4, 6)), grid, transform=TRANSFORM)


# -- joins ---------------------------------------------------------------------------


def test_stack_and_concat_by_name() -> None:
    g = _scene(coords=(BANDS,))
    series = geoarray.stack([g, g], "time")
    assert series.dims == ("time", "band", "y", "x")
    assert series.coords == (BANDS,)
    swir = SpectralAxis(("B11",), (1610.0,))
    b11 = g.with_data(g.data[:1], coords=(swir,))
    joined = geoarray.concat([g, b11], "band")
    assert joined.labels("band") == ("B2", "B4", "B8", "B11")
    assert joined.coord("band").wavelength_nm == (490.0, 665.0, 842.0, 1610.0)
    dates = geoarray.concat(
        [
            series.with_data(
                series.data, coords=(TimeAxis(["2026-01-01", "2026-01-02"]), BANDS)
            )
        ]
        * 2,
        "time",
    )
    assert dates.coord("time").size == 4
    with pytest.raises(ValueError, match="new grid"):
        geoarray.concat([g, g], "x")


def test_concat_along_patches_joins_transforms() -> None:
    t = np.stack([np.array(TRANSFORM), np.array(TRANSFORM) + 1.0])
    p = GeoArray(np.zeros((2, 4, 4)), t, CRS, dims="patch y x")
    q = p.shift_origin((100.0, 0.0))
    out = geoarray.concat([p, q], "patch")
    assert out.batch_shape == (4,)
    np.testing.assert_array_equal(out.transform[2:, 2], t[:, 2] + 100.0)


def test_combine_matches_dims_by_name() -> None:
    a = _scene((3, 8, 6), dims="band y x")
    b = GeoArray(
        np.random.default_rng(1).random((3, 8, 6)), TRANSFORM, CRS, dims="band y x"
    )
    out = geoarray.combine(np.subtract, a, b)
    np.testing.assert_array_equal(out.data, a.data - b.data)
    with pytest.raises(ValueError, match="dims"):
        geoarray.combine(np.add, a, a.with_data(a.data, dims="variable y x"))


# -- the adapter ---------------------------------------------------------------------


def test_registered_and_shared() -> None:
    assert isinstance(get_adapter(_scene()), GeoArrayAdapter)
    for name in (
        "isel",
        "sel",
        "transpose",
        "reduce",
        "argreduce",
        "take_along",
        "map_over",
        "concat",
        "combine",
    ):
        assert getattr(geoarray, name) is getattr(pa, name), name


def _make_carrier(xp: Any = np) -> Any:
    def make(data: np.ndarray, dims: tuple[str, ...], labels: dict) -> GeoArray:
        coords = (SpectralAxis(labels["band"]),)
        return GeoArray(xp.asarray(data), TRANSFORM, CRS, dims=dims, coords=coords)

    return make


def test_check_adapter_numpy() -> None:
    check_adapter(GeoArrayAdapter(), _make_carrier())


def test_check_adapter_jax(x64: ModuleType) -> None:
    check_adapter(GeoArrayAdapter(), _make_carrier(x64.numpy))


def test_reduction_over_the_grid_is_a_dimarray() -> None:
    g = _scene(coords=(BANDS,))
    band_means = KernelOp(lambda x: x.mean(-1), "(p)->()", core={"p": "x"})(g)
    assert isinstance(band_means, DimArray)
    assert band_means.dims == ("band", "y")
    assert band_means.labels == {"band": BANDS.labels}
    whole = pa.reduce(g, "mean", ("y", "x"))
    assert isinstance(whole, DimArray) and whole.dims == ("band",)
    picked = pa.isel(g, y=3)
    assert isinstance(picked, DimArray) and picked.dims == ("band", "x")


def test_reductions_off_the_grid_stay_carriers() -> None:
    g = _scene((4, 3, 8, 6), dims="time band y x", coords=(BANDS,), fill_value=np.nan)
    mean = pa.reduce(g, "mean", "time")
    assert isinstance(mean, GeoArray) and mean.dims == ("band", "y", "x")
    assert mean.coords == (BANDS,) and mean.fill_value is not None
    best = pa.argreduce(pa.sel(g, band="B8"), "argmax", "time")
    assert isinstance(best, GeoArray) and best.fill_value is None  # int data
    composite = pa.take_along(g, best, "time")
    assert composite.dims == ("band", "y", "x")


def test_kernel_op_agrees_across_families(x64: ModuleType) -> None:
    """PK-K1 on geoarray: NumPy and JAX carriers, and a DimArray, agree."""
    nd = KernelOp(
        lambda a, b, eps: (a - b) / (a + b + eps), "(),()->()", core={}, eps=1e-10
    )
    g = _scene(coords=(BANDS,))
    nir, red = g.sel(band="B8"), g.sel(band="B4")
    calls = [
        (nir, red),
        (nir.to_jax(), red.to_jax()),
        (DimArray(nir.data, nir.dims), DimArray(red.data, red.dims)),
    ]
    check_kernel_op(nd, calls)


def test_kernel_op_refuses_misaligned_grids() -> None:
    nd = KernelOp(lambda a, b: a - b, "(),()->()", core={})
    g = _scene()
    with pytest.raises(GridMismatchError):
        nd(g, g.shift_origin((1e-7, 0.0)))


def test_geometry_on_a_batched_carrier_names_map_over() -> None:
    """F4: derived geometry from (P, 6) transforms would differ per patch."""
    op = KernelOp(
        lambda x, dx: x * dx,
        "()->()",
        core={},
        geometry={"dx": lambda c: c.transform[..., 0]},
    )
    t = np.stack([np.array(TRANSFORM)] * 2)
    patches = GeoArray(np.ones((2, 4, 4)), t, CRS, dims="patch y x")
    with pytest.raises(ValueError, match="map_over"):
        op(patches)


def test_summary_has_dims(x64: ModuleType) -> None:
    """PK-K4 on a GeoArray pipeline."""
    from pipekit import Sequential

    g = _scene((4, 3, 8, 6), dims="time band y x", coords=(BANDS,))
    pipe = Sequential([pa.Sel(dim="band", label="B8"), pa.Reduce("mean", dim="time")])
    text = pipe.summary(g)
    assert "?" not in text and "(y=8, x=6)" in text


# -- pytree --------------------------------------------------------------------------


def test_dims_and_coords_are_static(x64: ModuleType) -> None:
    jax = x64
    g = _scene(coords=(BANDS,)).to_jax()
    leaves, treedef = jax.tree_util.tree_flatten(g)
    assert len(leaves) == 2
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.identical(g)
    traces = 0

    def step(c: GeoArray) -> Any:
        nonlocal traces
        traces += 1
        return pa.reduce(c, "sum", "band").data

    f = jax.jit(step)
    for origin in (0.0, 1.0, 2.0):
        f(g.shift_origin((origin, 0.0)))
    assert traces == 1
    f(g.with_data(g.data, coords=(SpectralAxis(("a", "b", "c")),)))
    assert traces == 2


def test_bare_vmap_maps_the_leading_dim(x64: ModuleType) -> None:
    jax = x64
    t = np.stack([np.array(TRANSFORM) + k for k in range(3)])
    patches = GeoArray(
        x64.numpy.ones((3, 2, 4, 4)),
        t,
        CRS,
        dims="patch band y x",
        coords=(SpectralAxis(("a", "b")),),
    ).to_jax()
    seen: list[tuple[str, ...]] = []

    def per_patch(p: GeoArray) -> GeoArray:
        seen.append(p.dims)
        assert p.labels("band") == ("a", "b")
        return pa.reduce(p, "sum", "band")

    out = jax.vmap(per_patch)(patches)
    assert seen == [("band", "y", "x")]
    assert out.dims == ("batch", "y", "x") and out.batch_shape == (3,)


def test_pickle_keeps_dims_and_coords() -> None:
    import pickle

    g = _scene(coords=(BANDS,), fill_value=np.nan)
    assert pickle.loads(pickle.dumps(g)).identical(g)


# -- installs ------------------------------------------------------------------------


def test_import_loads_neither_jax_nor_xarray() -> None:
    """GA-G12: the base install (JAX blocked here) imports and runs; no xarray.

    With JAX installed, geoarray registers its pytree at import, as 0.2 did.
    """
    code = textwrap.dedent(
        """
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("jax", "jaxlib"):
                    raise ImportError(name)

        sys.meta_path.insert(0, Block())
        import geoarray, numpy as np
        t = (1, 0, 0, 0, -1, 0)
        g = geoarray.GeoArray(np.zeros((2, 3, 3)), t, None, dims="band y x")
        geoarray.reduce(g, "mean", "band")
        print([m for m in ("jax", "xarray") if sys.modules.get(m) is not None])
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]"


def test_variable_dim_is_a_peer_of_band() -> None:
    met = VariableAxis(("t2m", "psfc"), ("K", "hPa"))
    g = _scene((2, 4, 4), dims="variable y x", coords=(met,))
    assert g.sel(variable="psfc").dims == ("y", "x")
    assert g.count == 1


def test_bool_carrier_keeps_a_float_fill_the_constructor_accepts() -> None:
    """Codex #546: rewrap and the constructor share one fill rule."""

    t = (20.0, 0.0, 0.0, 0.0, -20.0, 0.0)
    g = GeoArray(
        np.zeros((2, 3, 3), dtype=bool), t, 32613, fill_value=0.0, dims="time y x"
    )
    out = pa.reduce(g, "any", "time")
    assert out.dtype == bool
    assert out.fill_value == 0.0
