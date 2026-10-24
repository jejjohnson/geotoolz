"""Indexing: spatial slices move the transform (gate GB1, georeader parity) and
band selection keeps the per-band attrs consistent (gate GB5)."""

from __future__ import annotations

from types import ModuleType
from typing import ClassVar

import numpy as np
import pytest
from conftest import CRS, TRANSFORM

from geoarray import PER_BAND_KEYS, GeoArray, GeoArrayTraceError, Grid, PixelWindow


def _cube(**attrs: object) -> GeoArray:
    data = np.arange(3 * 12 * 10, dtype=np.float32).reshape(3, 12, 10)
    return GeoArray(data, TRANSFORM, CRS, fill_value=0.0, attrs=attrs)


class TestSpatialSlices:
    def test_origin_moves_with_the_slice(self) -> None:
        g = _cube()
        sub = g[..., 2:6, 3:7]
        assert sub.shape == (3, 4, 4)
        assert sub.grid == Grid(TRANSFORM, CRS, (12, 10)).crop(
            PixelWindow(col_off=3, row_off=2, width=4, height=4)
        )
        assert tuple(sub.transform) == (20.0, 0.0, 600060.0, 0.0, -20.0, 3549960.0)
        np.testing.assert_array_equal(sub.data, g.data[:, 2:6, 3:7])

    def test_steps_scale_the_pixel_size(self) -> None:
        sub = _cube()[:, ::3, 1::2]
        assert sub.res == (40.0, 60.0)
        assert sub.transform[2] == 600020.0
        assert sub.shape == (3, 4, 5)

    def test_negative_and_open_bounds_follow_numpy(self) -> None:
        g = _cube()
        assert g[..., -4:, :-2].shape == (3, 4, 8)
        assert g[..., -4:, :-2].transform[5] == 3550000.0 - 8 * 20.0

    def test_spatial_axes_reject_non_slices_and_negative_steps(self) -> None:
        g = _cube()
        with pytest.raises(IndexError, match="slices only"):
            g[0, 3]
        with pytest.raises(IndexError, match="slices only"):
            g[..., [1, 2], :]
        with pytest.raises(IndexError, match="positive steps"):
            g[..., ::-1, :]

    def test_bad_keys(self) -> None:
        g = _cube()
        with pytest.raises(IndexError, match="cannot add axes"):
            g[None]
        with pytest.raises(IndexError, match="one ellipsis"):
            g[..., 0, ...]
        with pytest.raises(IndexError, match="too many indices"):
            g[0, 0, 0:1, 0:1]
        four_d = GeoArray(np.zeros((2, 3, 4, 4)), TRANSFORM, CRS)
        with pytest.raises(IndexError, match="at most one list"):
            four_d[[0, 1], [0, 1]]

    def test_boolean_scalars_are_rejected(self) -> None:
        with pytest.raises(IndexError, match="cannot add axes"):
            _cube()[True]
        with pytest.raises(IndexError, match="cannot add axes"):
            _cube()[np.bool_(False)]

    def test_traced_indices(self, x64: ModuleType) -> None:
        g = _cube(band_names=["a", "b", "c"]).to_jax()
        with pytest.raises(GeoArrayTraceError, match="band index must be concrete"):
            x64.jit(lambda k: g[k].data)(1)
        with pytest.raises(GeoArrayTraceError, match="crop"):
            x64.jit(lambda r: g[..., r : r + 2, :].data)(1)

    def test_leading_indices_reduce_dims(self) -> None:
        stack = GeoArray(np.zeros((2, 3, 4, 4)), TRANSFORM, CRS)
        assert stack[0].dims == ("band", "y", "x")
        assert stack[0, 1].dims == ("y", "x")
        assert stack[:, [1]].dims == ("time", "band", "y", "x")
        with pytest.raises(IndexError, match=r"\(T, 1, H, W\)"):
            stack[:, 1]

    def test_squeeze_keeps_the_band_axis_of_a_time_stack(self) -> None:
        g = GeoArray(np.zeros((2, 1, 4, 4)), TRANSFORM, CRS)
        assert g.squeeze().shape == (2, 1, 4, 4)
        with pytest.raises(ValueError, match="reads as"):
            g.squeeze(1)
        assert GeoArray(np.zeros((1, 1, 4, 4)), TRANSFORM, CRS).squeeze().shape == (
            4,
            4,
        )


@pytest.mark.parametrize("seed", range(4))
def test_slice_parity_with_georeader(seed: int) -> None:
    """GB1: 250 random slices per seed, steps 1 to 3, match GeoTensor.isel."""
    georeader = pytest.importorskip("georeader.geotensor")
    from affine import Affine

    rng = np.random.default_rng(seed)
    data = np.zeros((2, 64, 48), dtype=np.float32)
    transform = (10.0, 0.0, 431020.0, 0.0, -10.0, 4500000.0)
    ours = GeoArray(data, transform, CRS)
    theirs = georeader.GeoTensor(data, Affine(*transform), CRS)
    for _ in range(250):
        r0, r1 = sorted(rng.integers(0, 64, size=2).tolist())
        c0, c1 = sorted(rng.integers(0, 48, size=2).tolist())
        rs, cs = rng.integers(1, 4, size=2).tolist()
        rows, cols = slice(r0, r1 + 1, rs), slice(c0, c1 + 1, cs)
        mine = ours[..., rows, cols]
        reference = theirs.isel({"y": rows, "x": cols})
        expected = np.array(Affine(*reference.transform)[:6])
        np.testing.assert_allclose(
            np.asarray(mine.transform), expected, rtol=0, atol=1e-9
        )
        assert mine.shape == reference.shape


class TestBandAttrs:
    """GB5: every per-band key has one entry per band after any selection."""

    attrs: ClassVar[dict[str, object]] = {
        "band_names": ["B04", "B08", "B11"],
        "wavelengths": [665.0, 842.0, 1610.0],
        "sensor": "S2A",
    }

    @pytest.mark.parametrize(
        ("index", "names"),
        [
            (slice(None), ("B04", "B08", "B11")),
            (slice(1, None), ("B08", "B11")),
            (slice(None, None, 2), ("B04", "B11")),
            ([2, 0], ("B11", "B04")),
            (np.array([True, False, True]), ("B04", "B11")),
            (1, ("B08",)),
            (-1, ("B11",)),
        ],
    )
    def test_band_selection_selects_band_attrs(
        self, index: object, names: tuple[str, ...]
    ) -> None:
        g = _cube(**self.attrs)
        out = g[index]
        assert out.attrs["band_names"] == names
        assert len(out.attrs["wavelengths"]) == out.count
        assert out.attrs["sensor"] == "S2A"

    def test_random_selections_stay_consistent(self) -> None:
        rng = np.random.default_rng(3)
        g = _cube(**self.attrs)
        for _ in range(50):
            index = rng.choice(3, size=rng.integers(1, 4), replace=False).tolist()
            out = g[index]
            for key in PER_BAND_KEYS:
                if key in out.attrs:
                    assert len(out.attrs[key]) == out.count

    def test_mismatched_band_keys_are_dropped(self) -> None:
        g = _cube(band_names=["only-two", "names"], sensor="S2A")
        assert "band_names" not in g[0:2].attrs
        assert g[0:2].attrs["sensor"] == "S2A"

    def test_rewrap_with_a_new_band_count_drops_band_keys(self) -> None:
        g = _cube(**self.attrs)
        assert "band_names" not in g.with_data(g.data[:1]).attrs
        assert g.with_data(g.data * 2).attrs["band_names"] == ("B04", "B08", "B11")
        assert "band_names" not in g.map(lambda d: d.sum(axis=0, keepdims=True)).attrs


class TestIsel:
    def test_named_selection(self) -> None:
        g = _cube(band_names=["a", "b", "c"])
        out = g.isel(band=[0, 2], y=slice(0, 4))
        assert out.shape == (3 - 1, 4, 10)
        assert out.attrs["band_names"] == ("a", "c")
        assert g.isel({"x": slice(5, None)}).transform[2] == 600100.0

    def test_unknown_names(self) -> None:
        with pytest.raises(KeyError, match="time"):
            _cube().isel(time=0)


class TestBatchedIndexing:
    def test_batch_index_selects_the_transform(self) -> None:
        transforms = np.stack(
            [np.array(TRANSFORM), np.array(TRANSFORM) + np.array([0, 0, 100, 0, 0, 0])]
        )
        g = GeoArray(
            np.zeros((2, 3, 4, 4)),
            transforms,
            CRS,
            attrs={"band_names": ["a", "b", "c"]},
        )
        one = g[1]
        assert one.batch_shape == ()
        assert one.transform[2] == 600100.0
        sub = g[:, 1:, 1:3, :]
        assert sub.batch_shape == (2,)
        assert sub.attrs["band_names"] == ("b", "c")
        np.testing.assert_allclose(sub.transform[:, 5], [3549980.0, 3549980.0])


class TestMapShiftSqueeze:
    def test_map_keeps_the_grid(self) -> None:
        g = _cube()
        assert g.map(np.sqrt).grid == g.grid
        with pytest.raises(ValueError, match="keep the spatial shape"):
            g.map(lambda d: d[..., :2, :2])

    def test_shift_origin(self) -> None:
        shifted = _cube().shift_origin((7.3, -2.0))
        assert shifted.transform[2] == pytest.approx(600007.3)
        assert shifted.transform[5] == pytest.approx(3549998.0)

    def test_squeeze_and_expand_dims(self) -> None:
        g = GeoArray(np.zeros((1, 1, 4, 4)), TRANSFORM, CRS)
        assert g.squeeze().shape == (4, 4)
        assert g.squeeze(0).shape == (1, 4, 4)
        with pytest.raises(ValueError, match="not a leading"):
            g.squeeze(-1)
        h = GeoArray(np.zeros((3, 4, 4)), TRANSFORM, CRS)
        assert h.expand_dims(0).shape == (1, 3, 4, 4)
        with pytest.raises(ValueError, match="before the spatial axes"):
            h.expand_dims(-1)


def test_int_and_list_separated_by_a_slice_keep_dim_order() -> None:
    """Codex #546: NumPy would move the list's axis to the front."""
    data = np.arange(2 * 3 * 4 * 2 * 2, dtype=float).reshape(2, 3, 4, 2, 2)
    g = GeoArray(
        data, (20.0, 0.0, 0.0, 0.0, -20.0, 0.0), 32613, dims="run time band y x"
    )
    out = g.isel(run=0, band=[0, 2])
    assert out.dims == ("time", "band", "y", "x")
    assert out.shape == (3, 2, 2, 2)
    np.testing.assert_array_equal(out.data, data[0][:, [0, 2]])
