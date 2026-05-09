from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xr_toolz.interpolate import refine, refine_2d
from xr_toolz.interpolate.operators import Refine


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("skimage") is None,
    reason="scikit-image is not installed",
)


@pytest.fixture
def plate() -> xr.DataArray:
    lat = np.linspace(-1.0, 1.0, 6)
    lon = np.linspace(10.0, 16.0, 8)
    values = np.sin(lat[:, None]) * np.cos(lon[None, :])
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def test_refine_2d_identity_preserves_values(plate: xr.DataArray) -> None:
    result = refine_2d(plate, factor={"lat": 1, "lon": 1}, order=3)

    xr.testing.assert_allclose(result, plate, atol=1e-10)


def test_refine_2d_updates_shape_and_coords(plate: xr.DataArray) -> None:
    result = refine_2d(plate, factor={"lat": 1.5, "lon": 0.5}, order=1)

    assert result.sizes["lat"] == 9
    assert result.sizes["lon"] == 4
    assert result.lat.values[[0, -1]].tolist() == pytest.approx(
        plate.lat.values[[0, -1]].tolist()
    )
    assert result.lon.values[[0, -1]].tolist() == pytest.approx(
        plate.lon.values[[0, -1]].tolist()
    )


def test_refine_2d_anti_aliasing_attenuates_downsampled_checkerboard() -> None:
    row, col = np.indices((32, 32))
    checkerboard = xr.DataArray(
        (-1.0) ** (row + col),
        dims=("lat", "lon"),
        coords={"lat": np.arange(32), "lon": np.arange(32)},
    )

    filtered = refine_2d(
        checkerboard,
        factor={"lat": 0.5, "lon": 0.5},
        order=0,
        anti_aliasing=True,
    )
    aliased = refine_2d(
        checkerboard,
        factor={"lat": 0.5, "lon": 0.5},
        order=0,
        anti_aliasing=False,
    )

    assert float(np.abs(filtered).max()) < float(np.abs(aliased).max())


def test_refine_2d_broadcasts_over_leading_dims(plate: xr.DataArray) -> None:
    cube = xr.concat([plate, plate + 1.0], dim=xr.IndexVariable("time", [0, 1]))

    result = refine_2d(cube, factor={"lat": 2, "lon": 2}, order=1)
    expected = xr.concat(
        [
            refine_2d(cube.isel(time=0), factor={"lat": 2, "lon": 2}, order=1),
            refine_2d(cube.isel(time=1), factor={"lat": 2, "lon": 2}, order=1),
        ],
        dim=xr.IndexVariable("time", [0, 1]),
    )

    xr.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result.isel(time=1) - result.isel(time=0), 1.0)


def test_refine_2d_dask_time_chunks_match_numpy(plate: xr.DataArray) -> None:
    dask_array = pytest.importorskip("dask.array")
    cube = xr.concat([plate, plate + 1.0], dim=xr.IndexVariable("time", [0, 1]))
    chunked = cube.chunk({"time": 1})

    result = refine_2d(chunked, factor={"lat": 2, "lon": 2}, order=1)
    expected = refine_2d(cube, factor={"lat": 2, "lon": 2}, order=1)

    assert isinstance(result.data, dask_array.Array)
    assert result.chunks[0] == (1, 1)
    xr.testing.assert_allclose(result.compute(), expected)


def test_refine_2d_dask_lat_lon_chunks_raise(plate: xr.DataArray) -> None:
    pytest.importorskip("dask.array")
    chunked = plate.chunk({"lat": 3})

    with pytest.raises(ValueError, match=r"core dimension|rechunk|single chunk"):
        refine_2d(chunked, factor={"lat": 2, "lon": 2}, order=1).compute()


def test_refine_operator_default_path_matches_refine(plate: xr.DataArray) -> None:
    op = Refine(factor={"lon": 2}, method="linear")

    xr.testing.assert_allclose(op(plate), refine(plate, factor={"lon": 2}))


def test_refine_operator_order_path_matches_refine_2d(plate: xr.DataArray) -> None:
    op = Refine(
        factor={"lat": 2, "lon": 2},
        order=3,
        anti_aliasing=False,
    )

    expected = refine_2d(
        plate,
        factor={"lat": 2, "lon": 2},
        order=3,
        anti_aliasing=False,
    )
    xr.testing.assert_allclose(op(plate), expected)


def test_refine_operator_config_round_trips_order_fields() -> None:
    cfg = Refine(
        factor={"lat": 2, "lon": 2},
        order=5,
        anti_aliasing=True,
        mode="constant",
        cval=-1.0,
    ).get_config()

    op = Refine(**cfg)

    assert op.get_config() == cfg


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"factor": {"lat": 2, "lon": 2}, "order": 6}, "order must be in 0..5"),
        ({"factor": {"lat": 2}, "order": 3}, "factor must include both"),
        ({"factor": {"lat": 0, "lon": 2}, "order": 3}, "must be positive"),
        ({"factor": {"lat": -1, "lon": 2}, "order": 3}, "must be positive"),
        (
            {"factor": {"lat": 2, "lon": 2}, "mode": "invalid"},
            "mode must be one of",
        ),
    ],
)
def test_refine_2d_validation_errors(
    plate: xr.DataArray,
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        refine_2d(plate, **kwargs)


@pytest.mark.parametrize("rename", [{"lat": "y"}, {"lon": "x"}])
def test_refine_2d_requires_lat_lon_dims(
    plate: xr.DataArray,
    rename: dict[str, str],
) -> None:
    renamed = plate.rename(rename)

    with pytest.raises(ValueError, match="da must have dims"):
        refine_2d(renamed, factor={"lat": 2, "lon": 2})
