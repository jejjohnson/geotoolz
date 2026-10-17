"""Geometry primitives: crop (GB2), windows (GB3), pad, coords, joins, footprint."""

from __future__ import annotations

import math
from types import ModuleType

import numpy as np
import pytest
from conftest import CRS, TRANSFORM

from geoarray import (
    GeoArray,
    Grid,
    GridMismatchError,
    PixelWindow,
    X64Error,
    combine,
    concatenate,
    crop,
    footprint,
    pad,
    pixel_coords,
    read_window,
    stack,
    write_window,
)


def _scene(shape: tuple[int, ...] = (2, 8, 8), **kwargs: object) -> GeoArray:
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return GeoArray(data, TRANSFORM, CRS, **kwargs)  # type: ignore[arg-type]


class TestCrop:
    """GB2: crop equals the slice, traces once under vmap, and clamps."""

    def test_crop_equals_the_slice(self) -> None:
        g = _scene()
        patch = crop(g, (2, 3), (4, 4))
        expected = g[..., 2:6, 3:7]
        assert patch.identical(expected)

    def test_offset_past_the_edge_is_clamped(self) -> None:
        g = _scene()
        patch = crop(g, (6, -3), (4, 4))
        assert patch.identical(g[..., 4:8, 0:4])

    def test_size_must_fit(self) -> None:
        with pytest.raises(ValueError, match="must fit"):
            crop(_scene(), (0, 0), (9, 2))

    def test_jax_crop_matches_numpy(self, x64: ModuleType) -> None:
        g = _scene()
        patch = crop(g.to_jax(), (2, 3), (4, 4))
        np.testing.assert_array_equal(np.asarray(patch.data), g.data[:, 2:6, 3:7])
        np.testing.assert_array_equal(
            np.asarray(patch.transform), np.asarray(g[..., 2:6, 3:7].transform)
        )

    def test_numpy_data_with_traced_offsets_moves_to_jax(self, x64: ModuleType) -> None:
        g = _scene()
        out = x64.jit(lambda o: crop(g, o, (4, 4)).data)(x64.numpy.array([6, 1]))
        np.testing.assert_array_equal(np.asarray(out), g.data[:, 4:8, 1:5])

    def test_traced_offset_pair(self, x64: ModuleType) -> None:
        g = _scene()

        def at(row: object, col: object) -> object:
            return crop(g, (row, col), (2, 2)).transform

        out = x64.jit(at)(3, 1)
        np.testing.assert_array_equal(
            np.asarray(out), np.asarray(g[..., 3:5, 1:3].transform)
        )

    def test_jax_crop_needs_x64(self, no_x64: ModuleType) -> None:
        # A concrete JAX offset on NumPy data stays on NumPy: no x64 needed.
        assert crop(_scene(), no_x64.numpy.array([0, 0]), (2, 2)).backend == "numpy"
        with pytest.raises(X64Error):
            no_x64.jit(lambda o: crop(_scene(), o, (2, 2)).data)(
                no_x64.numpy.array([0, 0])
            )

    def test_vmap_over_offsets_traces_once(self, x64: ModuleType) -> None:
        jnp = x64.numpy
        g = _scene().to_jax()
        traces = 0

        def patch(offset: object) -> GeoArray:
            nonlocal traces
            traces += 1
            return crop(g, offset, (4, 4))

        offsets = jnp.array([[0, 0], [2, 3], [4, 4], [7, 7]])
        patches = x64.jit(x64.vmap(patch))(offsets)
        assert traces == 1
        assert patches.shape == (4, 2, 4, 4)
        assert patches.batch_shape == (4,)
        for i, (r, c) in enumerate([(0, 0), (2, 3), (4, 4), (4, 4)]):
            expected = _scene()[..., r : r + 4, c : c + 4]
            np.testing.assert_array_equal(np.asarray(patches.data[i]), expected.data)
            np.testing.assert_array_equal(
                np.asarray(patches.transform[i]), np.asarray(expected.transform)
            )


class TestWindows:
    """GB3: bounds → window → bounds round trips on pixel edges."""

    grid = Grid(TRANSFORM, CRS, (600, 600))

    def test_window_of_bounds_on_pixel_edges(self) -> None:
        window = self.grid.window((600100, 3549000, 600400, 3549900))
        assert window == PixelWindow(col_off=5, row_off=5, width=15, height=45)

    @pytest.mark.parametrize("seed", range(3))
    def test_round_trip(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(100):
            c0, c1 = sorted(rng.choice(601, size=2, replace=False).tolist())
            r0, r1 = sorted(rng.choice(601, size=2, replace=False).tolist())
            window = PixelWindow(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
            bounds = self.grid.bounds_of(window)
            assert self.grid.window(bounds) == window

    def test_window_rounds_outward(self) -> None:
        window = self.grid.window((600005, 3549985, 600015, 3549995))
        assert window == PixelWindow(col_off=0, row_off=0, width=1, height=1)
        window = self.grid.window((600010, 3549970, 600030, 3549990))
        assert window == PixelWindow(col_off=0, row_off=0, width=2, height=2)

    def test_crop_and_from_bounds(self) -> None:
        window = PixelWindow(col_off=3, row_off=2, width=4, height=5)
        sub = self.grid.crop(window)
        assert sub.shape == (5, 4)
        assert sub.transform == (20.0, 0.0, 600060.0, 0.0, -20.0, 3549960.0)
        rebuilt = Grid.from_bounds(sub.bounds, 20, CRS)
        assert rebuilt == sub
        assert Grid.from_bounds((0, 0, 30, 10), (20, 10), None).shape == (1, 2)
        assert Grid.from_bounds((0, 0, 40, 40), np.int64(20), None).shape == (2, 2)

    def test_from_bounds_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Grid.from_bounds((0, 0, 0, 10), 1, None)

    def test_singular_transform(self) -> None:
        grid = Grid((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), None, (2, 2))
        with pytest.raises(ValueError, match="singular"):
            grid.window((0, 0, 1, 1))

    def test_rotated_grid_window_covers_the_corners(self) -> None:
        grid = Grid((10.0, 10.0, 0.0, 10.0, -10.0, 0.0), None, (10, 10))
        window = grid.window((0, -10, 20, 10))
        assert window.width > 0
        assert window.height > 0


class TestReadWriteWindow:
    def test_inside_window_is_a_slice(self) -> None:
        g = _scene(fill_value=-1.0)
        window = PixelWindow(col_off=1, row_off=2, width=3, height=4)
        assert read_window(g, window).identical(g[..., 2:6, 1:4])

    def test_boundless_read_pads_with_fill(self) -> None:
        g = _scene(fill_value=-1.0)
        window = PixelWindow(col_off=-2, row_off=6, width=4, height=4)
        out = read_window(g, window)
        assert out.shape == (2, 4, 4)
        np.testing.assert_array_equal(out.data[:, :2, 2:], g.data[:, 6:, :2])
        assert (out.data[:, 2:, :] == -1).all()
        assert (out.data[:, :, :2] == -1).all()
        expected = Grid(TRANSFORM, CRS, (8, 8)).crop(window)
        assert out.grid == expected

    def test_window_outside_the_array(self) -> None:
        g = _scene(fill_value=0.0)
        window = PixelWindow(col_off=20, row_off=20, width=2, height=3)
        out = read_window(g, window)
        assert out.shape == (2, 3, 2)
        assert (out.data == 0).all()
        assert out.grid == Grid(TRANSFORM, CRS, (8, 8)).crop(window)
        with pytest.raises(ValueError, match="does not overlap"):
            read_window(g, window, boundless=False)

    def test_bounded_read_clips(self) -> None:
        g = _scene()
        window = PixelWindow(col_off=-2, row_off=6, width=4, height=4)
        assert read_window(g, window, boundless=False).identical(g[..., 6:8, 0:2])

    def test_padding_needs_a_fill(self) -> None:
        window = PixelWindow(col_off=-1, row_off=0, width=2, height=2)
        with pytest.raises(ValueError, match="no fill_value"):
            read_window(_scene(), window)
        assert read_window(_scene(), window, fill_value=5.0).data[0, 0, 0] == 5.0

    def test_write_window_clips_and_copies(self) -> None:
        g = _scene((3, 3), fill_value=0.0)
        before = g.data.copy()
        window = PixelWindow(col_off=2, row_off=-1, width=2, height=2)
        out = write_window(g, np.full((2, 2), 9.0, dtype=np.float32), window)
        np.testing.assert_array_equal(g.data, before)
        assert out.data[0, 2] == 9.0
        assert (out.data[1:] == before[1:]).all()
        assert out.grid == g.grid

    def test_write_window_shapes(self) -> None:
        g = _scene()
        window = PixelWindow(col_off=0, row_off=0, width=2, height=2)
        with pytest.raises(ValueError, match="do not fit"):
            write_window(g, np.zeros((2, 2)), window)
        outside = PixelWindow(col_off=50, row_off=50, width=2, height=2)
        assert write_window(g, np.zeros((2, 2, 2)), outside) is g
        values = g[..., 0:2, 0:2].with_data(np.ones((2, 2, 2), dtype=np.float32))
        assert write_window(g, values, window).data[1, 1, 1] == 1.0

    def test_write_window_refuses_kind_changes(self) -> None:
        ints = GeoArray(np.zeros((3, 3), dtype=np.int16), TRANSFORM, CRS)
        window = PixelWindow(col_off=0, row_off=0, width=2, height=2)
        with pytest.raises(TypeError, match="cast them first"):
            write_window(ints, np.full((2, 2), np.nan), window)
        out = write_window(ints, np.ones((2, 2), dtype=np.int64), window)
        assert out.dtype == np.int16

    def test_write_window_jax_values_promote(self, x64: ModuleType) -> None:
        g = _scene((3, 3))
        window = PixelWindow(col_off=0, row_off=0, width=2, height=2)
        out = x64.jit(lambda v: write_window(g, v, window).data)(
            x64.numpy.zeros((2, 2), dtype=x64.numpy.float32)
        )
        assert float(out[0, 0]) == 0.0
        assert float(out[2, 2]) == float(g.data[2, 2])
        eager = write_window(g, x64.numpy.ones((2, 2), dtype=x64.numpy.float32), window)
        assert eager.backend == "jax"

    def test_write_window_jax(self, x64: ModuleType) -> None:
        g = _scene().to_jax()
        window = PixelWindow(col_off=1, row_off=1, width=2, height=2)
        out = write_window(g, x64.numpy.zeros((2, 2, 2)), window)
        assert out.backend == "jax"
        assert float(out.data[0, 1, 1]) == 0.0
        assert float(g.data[0, 1, 1]) != 0.0


class TestPad:
    def test_pad_moves_the_origin(self) -> None:
        g = _scene(fill_value=-1.0)
        out = pad(g, ((1, 2), (3, 0)))
        assert out.shape == (2, 11, 11)
        assert out.transform[2] == 600000.0 - 3 * 20.0
        assert out.transform[5] == 3550000.0 + 20.0
        assert (out.data[:, 0, :] == -1).all()
        assert out[..., 1:9, 3:11].identical(g.with_data(g.data))

    def test_uniform_pad_and_errors(self) -> None:
        g = _scene(fill_value=0.0)
        assert pad(g, 2).shape == (2, 12, 12)
        with pytest.raises(ValueError, match=">= 0"):
            pad(g, ((-1, 0), (0, 0)))
        with pytest.raises(ValueError, match=">= 0"):
            pad(g, -1)
        nan_ints = GeoArray(
            np.zeros((2, 2), dtype=np.int16), TRANSFORM, CRS, fill_value=np.nan
        )
        with pytest.raises(ValueError, match="NaN fill"):
            pad(nan_ints, 1)
        ints = GeoArray(np.zeros((2, 2), dtype=np.int16), TRANSFORM, CRS)
        assert pad(ints, np.int64(1), fill_value=7).data[0, 0] == 7

    def test_explicit_fill_becomes_the_result_fill(self) -> None:
        g = GeoArray(np.ones((4, 4)), TRANSFORM, CRS)
        out = pad(g, 1, fill_value=-9999)
        assert out.fill_value == -9999
        assert out.valid_mask().sum() == 16
        window = PixelWindow(col_off=-1, row_off=0, width=2, height=2)
        assert read_window(g, window, fill_value=-1).fill_value == -1
        far = PixelWindow(col_off=9, row_off=9, width=2, height=2)
        assert read_window(g, far, fill_value=-1).fill_value == -1
        nan_fill = GeoArray(np.ones((2, 2)), TRANSFORM, CRS, fill_value=np.nan)
        assert pad(nan_fill, 1, fill_value=float("nan")).fill_value is math.nan
        with pytest.raises(ValueError, match="differs from the carrier's"):
            pad(nan_fill, 1, fill_value=0.0)


class TestPixelCoords:
    def test_centres_and_corners(self) -> None:
        g = _scene((3, 4))
        coords = pixel_coords(g)
        assert coords.shape == (2, 3, 4)
        assert coords.dtype == np.float64
        np.testing.assert_array_equal(coords[:, 0, 0], [600010.0, 3549990.0])
        np.testing.assert_array_equal(coords[:, 2, 3], [600070.0, 3549950.0])
        corners = pixel_coords(g, center=False)
        np.testing.assert_array_equal(corners[:, 0, 0], [600000.0, 3550000.0])

    def test_batched_is_rejected(self) -> None:
        g = GeoArray(np.zeros((2, 4, 4)), np.stack([np.array(TRANSFORM)] * 2), CRS)
        with pytest.raises(ValueError, match="vmap"):
            pixel_coords(g)

    def test_jit(self, x64: ModuleType) -> None:
        g = _scene((3, 4)).to_jax()
        coords = x64.jit(pixel_coords)(g)
        np.testing.assert_array_equal(np.asarray(coords), pixel_coords(_scene((3, 4))))


class TestJoins:
    def test_combine(self) -> None:
        g = _scene(attrs={"sensor": "S2A"})
        out = combine(np.subtract, g, g.with_data(np.ones_like(g.data)))
        assert out.grid == g.grid
        assert out.attrs["sensor"] == "S2A"
        np.testing.assert_array_equal(out.data, g.data - 1)

    def test_combine_errors(self) -> None:
        g = _scene()
        with pytest.raises(ValueError, match="at least one"):
            combine(np.add)
        with pytest.raises(TypeError, match="takes GeoArrays"):
            combine(np.add, g, g.data)  # type: ignore[arg-type]
        moved = GeoArray(g.data, (20, 0, 600020, 0, -20, 3550000), CRS)
        with pytest.raises(GridMismatchError, match="transform"):
            combine(np.add, g, moved)
        other_crs = GeoArray(g.data, TRANSFORM, "EPSG:32614")
        with pytest.raises(GridMismatchError, match="CRS"):
            combine(np.add, g, other_crs)
        with pytest.raises(GridMismatchError, match="spatial shape"):
            combine(np.add, g, g[..., :4, :])
        assert issubclass(GridMismatchError, ValueError)

    def test_joins_require_one_fill(self) -> None:
        a = _scene(fill_value=0.0)
        b = _scene(fill_value=-1.0)
        for join in (
            lambda: combine(np.add, a, b),
            lambda: stack([a, b]),
            lambda: concatenate([a, b]),
        ):
            with pytest.raises(ValueError, match="fill_value"):
                join()
        nan_a, nan_b = _scene(fill_value=np.nan), _scene(fill_value=float("nan"))
        assert stack([nan_a, nan_b]).fill_value is math.nan

    def test_batched_joins_keep_the_batch_axes_first(self) -> None:
        transforms = np.stack([np.array(TRANSFORM)] * 2)
        g = GeoArray(np.zeros((2, 3, 4, 4)), transforms, CRS)
        assert concatenate([g, g]).shape == (2, 6, 4, 4)
        for join in (lambda: stack([g, g]), lambda: concatenate([g, g], axis=0)):
            with pytest.raises(ValueError, match="batch axes"):
                join()
        single = GeoArray(np.zeros((2, 4, 4)), transforms, CRS)
        assert stack([single, single], axis=1).shape == (2, 2, 4, 4)
        assert stack([single, single], axis=1).batch_shape == (2,)

    def test_mixed_backends_promote_to_jax(self, x64: ModuleType) -> None:
        g = _scene()
        out = combine(x64.numpy.add, g, g.to_jax())
        assert out.backend == "jax"
        # Output stays where the computation ran: a NumPy function -> NumPy.
        assert combine(np.add, g, g.to_jax()).backend == "numpy"
        assert stack([g, g.to_jax()]).backend == "jax"
        assert concatenate([g.to_jax(), g]).backend == "jax"

    def test_promotion_needs_x64(self, x64: ModuleType) -> None:
        g = _scene()
        on_jax = g.to_jax()
        x64.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(X64Error):
                stack([g, on_jax])
        finally:
            x64.config.update("jax_enable_x64", True)

    def test_combine_under_jit(self, x64: ModuleType) -> None:
        jnp = x64.numpy
        g = _scene().to_jax()
        out = x64.jit(lambda a, b: combine(jnp.add, a, b))(g, g)
        np.testing.assert_array_equal(np.asarray(out.data), 2 * _scene().data)

    def test_stack(self) -> None:
        g = _scene()
        out = stack([g, g, g])
        assert out.shape == (3, 2, 8, 8)
        assert out.dims == ("time", "band", "y", "x")
        assert stack([g, g], axis=1).shape == (2, 2, 8, 8)
        with pytest.raises(ValueError, match="leading"):
            stack([g, g], axis=-1)

    def test_concatenate_joins_band_attrs(self) -> None:
        a = _scene(attrs={"band_names": ["B04", "B08"], "sensor": "S2A"})
        b = a.with_data(a.data[:1], attrs={"band_names": ["B11"]})
        out = concatenate([a, b])
        assert out.shape == (3, 8, 8)
        assert out.attrs == {"band_names": ("B04", "B08", "B11"), "sensor": "S2A"}
        c = a.with_data(a.data[:1], attrs={})
        assert "band_names" not in concatenate([a, c]).attrs

    def test_concatenate_time(self) -> None:
        a = _scene((1, 2, 8, 8), attrs={"band_names": ["B04", "B08"]})
        out = concatenate([a, a], axis=0)
        assert out.shape == (2, 2, 8, 8)
        assert out.attrs["band_names"] == ("B04", "B08")


class TestFootprint:
    def test_native_footprint(self) -> None:
        pytest.importorskip("shapely")
        g = _scene((4, 5))
        poly = footprint(g)
        assert poly.bounds == (600000.0, 3549920.0, 600100.0, 3550000.0)
        assert poly.area == pytest.approx(100.0 * 80.0)

    def test_reprojected_footprint(self) -> None:
        pytest.importorskip("shapely")
        g = _scene((4, 5))
        poly = footprint(g, "EPSG:4326")
        lon, lat = poly.centroid.x, poly.centroid.y
        assert -104.5 < lon < -103.5
        assert 32 < lat < 32.2
        assert len(poly.exterior.coords) == 4 * 20 + 1

    def test_footprint_needs_a_crs_to_reproject(self) -> None:
        pytest.importorskip("shapely")
        g = GeoArray(np.zeros((2, 2)), TRANSFORM, None)
        with pytest.raises(ValueError, match="needs a carrier with a CRS"):
            footprint(g, "EPSG:4326")
