"""`geoarray.stack_items`: items on their own grids → one batch (pipekit#119)."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pipekit_array as pa
import pytest
from pipekit_array import BackendError

import geoarray
from geoarray import GeoArray, GridMismatchError
from geoarray.coords import SpectralAxis


CRS = "EPSG:32613"


def _patch(col: int, value: float = 0.0, **kwargs: object) -> GeoArray:
    """A (2, 4, 4) patch whose origin is ``col`` pixels east of the scene's."""
    transform = (20.0, 0.0, 600_000.0 + 20.0 * col, 0.0, -20.0, 3_550_000.0)
    return GeoArray(
        np.full((2, 4, 4), value, dtype=np.float32),
        transform,
        CRS,
        dims="band y x",
        coords=(SpectralAxis(("B4", "B8")),),
        **kwargs,  # type: ignore[arg-type]
    )


def test_items_keep_their_transforms_as_rows() -> None:
    items = [_patch(0, 1.0), _patch(4, 2.0), _patch(8, 3.0)]

    batch = geoarray.stack_items(items, "patch")

    assert geoarray.stack_items is pa.stack_items
    assert batch.dims == ("patch", "band", "y", "x") and batch.batch_dims == ("patch",)
    assert batch.transform.shape == (3, 6)
    np.testing.assert_array_equal(
        batch.transform[:, 2], [600_000.0, 600_080.0, 600_160.0]
    )
    assert batch.coord("band") == SpectralAxis(("B4", "B8"))
    np.testing.assert_array_equal(batch.data[:, 0, 0, 0], [1.0, 2.0, 3.0])


def test_it_is_the_inverse_of_map_over_along_the_batch_dim() -> None:
    batch = geoarray.stack_items([_patch(0, 1.0), _patch(4, 2.0)], "patch")

    items = [geoarray.isel(batch, patch=i) for i in range(2)]
    again = geoarray.stack_items(items, "patch")

    for i, item in enumerate(items):
        assert item.grid == _patch(4 * i).grid
    np.testing.assert_array_equal(again.transform, batch.transform)
    np.testing.assert_array_equal(again.data, batch.data)


def test_pipekit_batch_pads_geoarrays_with_their_transforms() -> None:
    """`pipekit_array.batch` on carriers: padding copies the last row too."""
    items = [_patch(c, float(c)) for c in (0, 4, 8)]

    _, last = pa.batch(items, size=2, dim="patch")

    assert last.valid.tolist() == [True, False]
    np.testing.assert_array_equal(last.value.transform[1], last.value.transform[0])
    assert last.value.dims == ("patch", "band", "y", "x")
    mean = pa.reduce(last.value, "mean", ("patch", "y", "x"), where=last.valid)
    np.testing.assert_array_equal(np.asarray(mean.data), [8.0, 8.0])


@pytest.mark.parametrize(
    ("items", "error", "match"),
    [
        ([_patch(0), _patch(4)], ValueError, "already a dim"),
        (
            [
                _patch(0),
                GeoArray(
                    np.zeros((2, 4, 4)), _patch(0).transform, 4326, dims="band y x"
                ),
            ],
            GridMismatchError,
            "CRS",
        ),
        ([_patch(0), _patch(4, fill_value=-1.0)], ValueError, "fill_value"),
        (
            [
                _patch(0),
                _patch(4).with_data(
                    np.zeros((2, 4, 4)), coords=(SpectralAxis(("a", "b")),)
                ),
            ],
            ValueError,
            "coords",
        ),
    ],
    ids=["dim-taken", "crs", "fill", "coords"],
)
def test_items_must_agree_on_everything_but_the_transform(
    items: list[GeoArray], error: type[Exception], match: str
) -> None:
    dim = "band" if match == "already a dim" else "patch"
    with pytest.raises(error, match=match):
        geoarray.stack_items(items, dim)


def test_an_already_batched_item_is_refused() -> None:
    batch = geoarray.stack_items([_patch(0), _patch(4)], "patch")
    with pytest.raises(ValueError, match="already batched"):
        geoarray.stack_items([batch, batch], "scene")


def test_numpy_and_jax_items_never_mix(x64: ModuleType) -> None:
    with pytest.raises(BackendError, match="to_jax"):
        geoarray.stack_items([_patch(0), _patch(4).to_jax()], "patch")


def test_jax_items_stack_on_jax(x64: ModuleType) -> None:
    batch = geoarray.stack_items([_patch(0).to_jax(), _patch(4).to_jax()], "patch")
    assert batch.backend == "jax" and batch.transform.shape == (2, 6)
