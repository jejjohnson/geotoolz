"""`Grid`, `PixelWindow` and the transform / CRS coercions."""

from __future__ import annotations

import math

import numpy as np
import pyproj
import pytest
from affine import Affine

from geoarray import Grid, PixelWindow, as_crs, as_transform


T = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)


class TestAsTransform:
    def test_accepts_affine_six_nine_and_matrices(self) -> None:
        expected = T
        assert as_transform(Affine(*T)) == expected
        assert as_transform(T) == expected
        assert as_transform(list(T)) == expected
        assert as_transform(np.array(T)) == expected
        assert as_transform((*T, 0.0, 0.0, 1.0)) == expected
        assert (
            as_transform(np.array([[20.0, 0.0, 600000.0], [0.0, -20.0, 3550000.0]]))
            == expected
        )
        matrix = np.array(
            [[20.0, 0.0, 600000.0], [0.0, -20.0, 3550000.0], [0.0, 0.0, 1.0]]
        )
        assert as_transform(matrix) == expected
        assert all(
            type(v) is float for v in as_transform(np.array(T, dtype=np.float32))
        )

    def test_rejects_non_affine(self) -> None:
        with pytest.raises(ValueError, match="6 numbers"):
            as_transform((1.0, 2.0, 3.0))
        with pytest.raises(ValueError, match="last row"):
            as_transform((*T, 1.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="not an affine transform"):
            as_transform(("a", "b"))
        with pytest.raises(ValueError, match="not an affine transform"):
            as_transform(object())


class TestAsCrs:
    def test_parses_common_inputs(self) -> None:
        assert as_crs(None) is None
        crs = pyproj.CRS.from_epsg(32613)
        assert as_crs(crs) is crs
        assert as_crs(32613) == crs
        assert as_crs("EPSG:32613") == crs
        assert as_crs(crs.to_wkt()) == crs

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="not a CRS"):
            as_crs("EPSG:not-a-code")


class TestPixelWindow:
    def test_slices_round_trip(self) -> None:
        window = PixelWindow.from_slices(slice(10, 42), slice(5, 37))
        assert window == PixelWindow(col_off=5, row_off=10, width=32, height=32)
        assert window.shape == (32, 32)
        assert window.to_slices() == (slice(10, 42), slice(5, 37))
        assert PixelWindow.from_slices(slice(5, 2), slice(0, 1)).height == 0

    def test_field_order_matches_rasterio(self) -> None:
        # rasterio.windows.Window(col_off, row_off, width, height)
        assert PixelWindow(1, 2, 3, 4) == PixelWindow(
            col_off=1, row_off=2, width=3, height=4
        )

    def test_slices_need_step_one_and_bounds(self) -> None:
        with pytest.raises(ValueError, match="step 1"):
            PixelWindow.from_slices(slice(0, 10, 2), slice(0, 10))
        with pytest.raises(ValueError, match="explicit start and stop"):
            PixelWindow.from_slices(slice(None, 10), slice(0, 10))

    def test_validation(self) -> None:
        window = PixelWindow(
            col_off=np.int64(-2), row_off=3, width=np.int32(4), height=5
        )
        assert window == PixelWindow(col_off=-2, row_off=3, width=4, height=5)
        assert type(window.col_off) is int
        assert type(window.width) is int
        with pytest.raises(ValueError, match="height must be >= 0"):
            PixelWindow(col_off=0, row_off=0, width=2, height=-1)
        with pytest.raises(TypeError, match="row_off must be an integer"):
            PixelWindow(col_off=0, row_off=0.5, width=1, height=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="width must be an integer"):
            PixelWindow(col_off=0, row_off=0, width=True, height=1)

    def test_intersection(self) -> None:
        def w(col: int, row: int, width: int, height: int) -> PixelWindow:
            return PixelWindow(col_off=col, row_off=row, width=width, height=height)

        assert w(0, 0, 10, 10).intersection(w(5, 5, 10, 10)) == w(5, 5, 5, 5)
        assert w(0, 0, 10, 10).intersection(w(0, 10, 5, 5)) is None
        assert w(-3, -3, 6, 6).intersection(w(0, 0, 10, 10)) == w(0, 0, 3, 3)
        assert w(0, 0, 10, 4).intersection(w(2, 1, 3, 9)) == w(2, 1, 3, 3)


class TestGrid:
    def test_north_up_geometry(self) -> None:
        grid = Grid(T, "EPSG:32613", (600, 600))
        assert grid.height == 600
        assert grid.width == 600
        assert grid.res == (20.0, 20.0)
        assert grid.bounds == (600000.0, 3538000.0, 612000.0, 3550000.0)
        assert grid.is_axis_aligned
        assert grid.affine == Affine(*T)
        assert grid.crs == pyproj.CRS.from_epsg(32613)

    def test_rotated_geometry(self) -> None:
        theta = math.radians(30.0)
        a, b = 10 * math.cos(theta), -10 * math.sin(theta)
        d, e = 10 * math.sin(theta), -10 * math.cos(theta)
        grid = Grid((a, b, 0.0, d, e, 0.0), None, (4, 2))
        assert not grid.is_axis_aligned
        assert grid.res == pytest.approx((10.0, 10.0))
        corners_x = [0.0, a * 2, b * 4, a * 2 + b * 4]
        corners_y = [0.0, d * 2, e * 4, d * 2 + e * 4]
        assert grid.bounds == pytest.approx(
            (min(corners_x), min(corners_y), max(corners_x), max(corners_y))
        )

    def test_from_affine_and_shift(self) -> None:
        grid = Grid.from_affine(Affine(*T), 32613, [600, 600])
        assert grid == Grid(T, "EPSG:32613", (600, 600))
        shifted = grid.shift(1, 2)
        assert shifted.transform == (20.0, 0.0, 600040.0, 0.0, -20.0, 3549980.0)
        assert shifted.shape == grid.shape

    def test_equality_hash_and_isclose(self) -> None:
        grid = Grid(T, "EPSG:32613", (600, 600))
        same = Grid(np.array(T), pyproj.CRS.from_epsg(32613), (np.int64(600), 600))
        assert grid == same
        assert {grid: "a"}[same] == "a"
        nudged = grid.shift(1e-9, 0.0)
        assert grid != nudged
        assert grid.isclose(nudged)
        assert not grid.isclose(grid.shift(1.0, 0.0))
        assert not grid.isclose(Grid(T, "EPSG:32614", (600, 600)))
        assert not grid.isclose(Grid(T, "EPSG:32613", (600, 601)))

    @pytest.mark.parametrize(
        "crs",
        ["EPSG:32613", "+proj=sinu +R=6371007.181 +units=m +no_defs", None],
    )
    def test_config_round_trip(self, crs: str | None) -> None:
        grid = Grid(T, crs, (3, 4))
        config = grid.get_config()
        assert isinstance(config["transform"], list)
        assert isinstance(config["shape"], list)
        assert Grid.from_config(config) == grid

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match=r"\(H, W\)"):
            Grid(T, None, (1, 2, 3))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="H must be >= 0"):
            Grid(T, None, (-1, 2))
        with pytest.raises(TypeError, match="W must be an integer"):
            Grid(T, None, (1, 2.0))  # type: ignore[arg-type]
