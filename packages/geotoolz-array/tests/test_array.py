"""`GeoArray` on NumPy data: construction, properties and rewrapping."""

from __future__ import annotations

import math

import numpy as np
import pyproj
import pytest
from affine import Affine
from conftest import CRS, TRANSFORM

from geoarray import FrozenAttrs, GeoArray, Grid
from geoarray._src.array import INHERIT


class TestConstruction:
    @pytest.mark.parametrize(
        ("shape", "dims"),
        [
            ((4, 5), ("y", "x")),
            ((3, 4, 5), ("band", "y", "x")),
            ((2, 3, 4, 5), ("time", "band", "y", "x")),
        ],
    )
    def test_dims_follow_ndim(
        self, shape: tuple[int, ...], dims: tuple[str, ...]
    ) -> None:
        g = GeoArray(np.zeros(shape), TRANSFORM, CRS)
        assert g.dims == dims
        assert g.ndim == len(shape)
        assert g.shape == shape
        assert g.height == 4
        assert g.width == 5
        assert g.count == (shape[-3] if len(shape) >= 3 else 1)

    @pytest.mark.parametrize("shape", [(5,), (1, 2, 3, 4, 5)])
    def test_rejects_other_ndims(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="2-D to 4-D"):
            GeoArray(np.zeros(shape), TRANSFORM, CRS)

    def test_rejects_non_arrays(self) -> None:
        with pytest.raises(TypeError, match=r"numpy\.ndarray or a jax\.Array"):
            GeoArray([[1.0, 2.0]], TRANSFORM, CRS)

    @pytest.mark.parametrize(
        "transform",
        [
            TRANSFORM,
            list(TRANSFORM),
            Affine(*TRANSFORM),
            np.array(TRANSFORM, dtype=np.float32),
        ],
    )
    def test_transform_is_a_float64_array(self, transform: object) -> None:
        g = GeoArray(np.zeros((2, 2)), transform, CRS)
        assert isinstance(g.transform, np.ndarray)
        assert g.transform.dtype == np.float64
        assert g.transform.shape == (6,)
        assert tuple(g.transform) == TRANSFORM

    def test_transform_is_not_copied_when_already_float64(self) -> None:
        transform = np.array(TRANSFORM)
        assert GeoArray(np.zeros((2, 2)), transform, CRS).transform is transform

    def test_bad_transform_raises(self) -> None:
        with pytest.raises(ValueError, match="6 numbers"):
            GeoArray(np.zeros((2, 2)), (1.0, 2.0), CRS)

    def test_crs_is_parsed(self) -> None:
        assert GeoArray(np.zeros((2, 2)), TRANSFORM, 32613).crs == pyproj.CRS.from_epsg(
            32613
        )
        assert GeoArray(np.zeros((2, 2)), TRANSFORM, None).crs is None
        with pytest.raises(ValueError, match="not a CRS"):
            GeoArray(np.zeros((2, 2)), TRANSFORM, "nonsense")

    def test_fill_value_is_a_hashable_python_scalar(self) -> None:
        assert (
            GeoArray(np.zeros((2, 2)), TRANSFORM, CRS, np.float32(np.nan)).fill_value
            is math.nan
        )
        fill = GeoArray(np.zeros((2, 2)), TRANSFORM, CRS, np.uint16(0)).fill_value
        assert fill == 0
        assert type(fill) is int
        assert GeoArray(np.zeros((2, 2)), TRANSFORM, CRS, False).fill_value is False
        with pytest.raises(TypeError, match="fill_value must be"):
            GeoArray(np.zeros((2, 2)), TRANSFORM, CRS, "zero")

    def test_attrs_are_frozen(self, chip: GeoArray) -> None:
        assert isinstance(chip.attrs, FrozenAttrs)
        assert chip.attrs["band_names"] == ("B11", "B12")
        assert chip.attrs["wavelengths"] == (1610.0, 2190.0)
        assert GeoArray(np.zeros((2, 2)), TRANSFORM, CRS, attrs=None).attrs == {}

    def test_frozen(self, chip: GeoArray) -> None:
        with pytest.raises(AttributeError):
            chip.crs = None  # type: ignore[misc]


class TestGeometry:
    def test_grid_res_bounds(self, chip: GeoArray) -> None:
        assert chip.grid == Grid(TRANSFORM, CRS, (8, 8))
        assert chip.res == (20.0, 20.0)
        assert chip.bounds == (600000.0, 3549840.0, 600160.0, 3550000.0)

    def test_backend_and_namespace(self, chip: GeoArray) -> None:
        assert chip.backend == "numpy"
        assert chip.xp is np
        assert chip.dtype == np.float32


class TestRewrap:
    def test_with_data_keeps_the_grid(self, chip: GeoArray) -> None:
        mask = chip.with_data(np.isnan(chip.data[:1]), fill_value=False, attrs={})
        assert mask.shape == (1, 8, 8)
        assert mask.grid == chip.grid
        assert mask.fill_value is False
        assert mask.attrs == {}

    def test_with_data_inherits_fill_and_attrs(self, chip: GeoArray) -> None:
        out = chip.with_data(chip.data * 2)
        assert out.fill_value is chip.fill_value
        assert out.attrs == chip.attrs
        assert chip.with_data(chip.data, fill_value=INHERIT).fill_value is math.nan
        assert chip.with_data(chip.data, fill_value=None).fill_value is None

    def test_with_data_needs_transform_for_a_new_grid(self, chip: GeoArray) -> None:
        with pytest.raises(ValueError, match="pass transform="):
            chip.with_data(chip.data[:, :4, :4])
        half = chip.with_data(
            chip.data[:, ::2, ::2], transform=(40, 0, 600000, 0, -40, 3550000)
        )
        assert half.res == (40.0, 40.0)

    def test_attrs_helpers_and_replace(self, chip: GeoArray) -> None:
        tagged = chip.with_attrs(sensor="S2A")
        assert tagged.attrs["sensor"] == "S2A"
        assert "sensor" not in chip.attrs
        assert "wavelengths" not in tagged.drop_attrs("wavelengths").attrs
        assert chip.replace(fill_value=0).fill_value == 0
        assert chip.replace(crs=None).crs is None

    def test_astype(self, chip: GeoArray) -> None:
        doubled = chip.astype(np.float64)
        assert doubled.dtype == np.float64
        assert doubled.grid == chip.grid


class TestNumpyInterop:
    def test_asarray_is_the_data(self, chip: GeoArray) -> None:
        assert np.asarray(chip) is chip.data
        assert np.asarray(chip, dtype=np.float64).dtype == np.float64
        copied = np.array(chip, copy=True)
        assert copied is not chip.data
        np.testing.assert_array_equal(copied, chip.data)

    def test_to_numpy_is_identity_on_numpy(self, chip: GeoArray) -> None:
        assert chip.to_numpy() is chip


class TestIdentical:
    def test_nan_aware_and_strict(self, chip: GeoArray) -> None:
        data = chip.data.copy()
        data[0, 0, 0] = np.nan
        a = chip.with_data(data)
        b = chip.with_data(data.copy())
        assert a.identical(b)
        assert not a.identical(chip)
        assert not chip.identical(chip.replace(crs="EPSG:32614"))
        assert not chip.identical(chip.replace(fill_value=0.0))
        assert not chip.identical(chip.with_attrs(x=1))
        assert not chip.identical(chip.astype(np.float64))
        assert not chip.identical(chip.data)

    def test_integer_data(self) -> None:
        g = GeoArray(np.arange(4).reshape(2, 2), TRANSFORM, CRS, 0)
        assert g.identical(g.with_data(np.arange(4).reshape(2, 2)))
        assert not g.identical(g.replace(fill_value=False))


class TestRepr:
    def test_repr_names_backend_crs_and_attrs(self, chip: GeoArray) -> None:
        text = repr(chip)
        assert text.startswith("GeoArray(numpy float32 (2, 8, 8), crs=EPSG:32613")
        assert "res=(20.0, 20.0)" in text
        assert "fill_value=nan" in text
        assert "attrs=('band_names', 'wavelengths')" in text

    def test_repr_without_or_with_custom_crs(self) -> None:
        assert "crs=None" in repr(GeoArray(np.zeros((2, 2)), TRANSFORM, None))
        sinu = GeoArray(
            np.zeros((2, 2)), TRANSFORM, "+proj=sinu +R=6371007.181 +units=m"
        )
        assert "crs='unknown'" in repr(sinu) or "crs='" in repr(sinu)


class TestBatched:
    """A carrier with one transform per element, as jax.vmap returns."""

    def test_batch_shape_and_dims(self) -> None:
        transforms = np.stack([np.array(TRANSFORM), np.array(TRANSFORM) + 20.0])
        g = GeoArray(np.zeros((2, 3, 4, 5)), transforms, CRS)
        assert g.batch_shape == (2,)
        assert g.dims == ("batch", "band", "y", "x")
        assert g.count == 3
        assert GeoArray(np.zeros((2, 4, 5)), transforms, CRS).count == 1

    def test_per_scene_geometry_needs_an_element(self) -> None:
        transforms = np.stack([np.array(TRANSFORM)] * 2)
        g = GeoArray(np.zeros((2, 4, 5)), transforms, CRS)
        with pytest.raises(ValueError, match="batched"):
            _ = g.grid

    def test_batch_shapes_must_agree(self) -> None:
        transforms = np.stack([np.array(TRANSFORM)] * 3)
        with pytest.raises(ValueError, match="leading batch shape"):
            GeoArray(np.zeros((2, 4, 5)), transforms, CRS)
        with pytest.raises(ValueError, match="with batch shape"):
            GeoArray(np.zeros((3, 5)), transforms, CRS)


def test_inherit_sentinel_repr() -> None:
    assert repr(INHERIT) == "INHERIT"


def test_pickle_round_trip_keeps_identity(chip: GeoArray) -> None:
    import math
    import pickle

    back = pickle.loads(pickle.dumps(chip))
    assert back.identical(chip)
    assert back.fill_value is math.nan
    assert back.attrs == chip.attrs
