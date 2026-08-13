from __future__ import annotations

import json

import numpy as np
import pytest
import rioxarray  # noqa: F401  — needed so ds.rio is populated
import xarray as xr

from xrtoolz.geo import SpatialMosaic, spatial_mosaic


def _tile(
    x0: float,
    value: float = 1.0,
    *,
    crs: str = "EPSG:4326",
    n: int = 4,
) -> xr.DataArray:
    """A north-up (descending ``y``) 4x4 tile whose x-origin is ``x0``."""
    return xr.DataArray(
        np.full((n, n), value, dtype=float),
        dims=("y", "x"),
        coords={
            "y": np.arange(n - 1.0, -1.0, -1.0),
            "x": x0 + np.arange(float(n)),
        },
    ).rio.write_crs(crs)


def test_adjacent_tiles_merge_to_union_extent() -> None:
    out = spatial_mosaic([_tile(0.0), _tile(4.0)])

    assert out.sizes == {"y": 4, "x": 8}
    assert float(out.x.min()) == 0.0
    assert float(out.x.max()) == 7.0
    assert np.all(np.isfinite(out.values))


@pytest.mark.parametrize(
    ("method", "expected"),
    [("first", 1.0), ("last", 5.0), ("min", 1.0), ("max", 5.0), ("sum", 6.0)],
)
def test_overlap_resolves_per_method(method: str, expected: float) -> None:
    # Tiles span x=[0,3] and x=[2,5] → x in {2, 3} is the overlap region.
    left, right = _tile(0.0, value=1.0), _tile(2.0, value=5.0)

    out = spatial_mosaic([left, right], method=method)

    assert float(out.sel(x=2.0, y=0.0)) == pytest.approx(expected)
    # Outside the overlap each tile keeps its own value under every method.
    assert float(out.sel(x=0.0, y=0.0)) == pytest.approx(1.0)
    assert float(out.sel(x=5.0, y=0.0)) == pytest.approx(5.0)


def test_differing_crs_reprojected_to_first_input() -> None:
    first = _tile(0.0)
    second = _tile(4.0).rio.reproject("EPSG:3857")
    assert second.rio.crs.to_epsg() == 3857

    out = spatial_mosaic([first, second])

    assert out.rio.crs == first.rio.crs
    # Union extent, so the merge saw the reprojected tile rather than
    # dropping it or aligning it in the wrong coordinate space.
    assert out.sizes["x"] > first.sizes["x"]


def test_dataset_inputs_use_merge_datasets() -> None:
    out = spatial_mosaic(
        [_tile(0.0).to_dataset(name="sst"), _tile(4.0).to_dataset(name="sst")]
    )

    assert isinstance(out, xr.Dataset)
    assert out.sizes["x"] == 8
    assert "sst" in out


def test_single_input_returns_unchanged() -> None:
    tile = _tile(0.0)

    out = spatial_mosaic([tile])

    xr.testing.assert_identical(out, tile)


def test_empty_sequence_raises() -> None:
    with pytest.raises(ValueError, match="at least one input"):
        spatial_mosaic([])


def test_missing_crs_raises_naming_the_offending_index() -> None:
    bare = xr.DataArray(
        np.ones((4, 4)),
        dims=("y", "x"),
        coords={"y": np.arange(3.0, -1.0, -1.0), "x": np.arange(4.0)},
    )

    with pytest.raises(ValueError, match="input 1 has no CRS"):
        spatial_mosaic([_tile(0.0), bare])


def test_mixed_dataarray_and_dataset_raises() -> None:
    with pytest.raises(ValueError, match="homogeneous sequence"):
        spatial_mosaic([_tile(0.0), _tile(4.0).to_dataset(name="sst")])


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        spatial_mosaic([_tile(0.0), _tile(4.0)], method="bogus")


def test_unknown_resampling_raises() -> None:
    with pytest.raises(ValueError, match="Unknown resampling"):
        spatial_mosaic([_tile(0.0), _tile(4.0)], resampling="bogus")


def test_sum_promotes_integers_so_overlap_cannot_wrap() -> None:
    # uint8 200 + 200 wraps to 144 if the mosaic is allocated in the input
    # dtype, which is what rioxarray does by default.
    left = _tile(0.0, value=200.0).astype("uint8").rio.write_crs("EPSG:4326")
    right = _tile(2.0, value=200.0).astype("uint8").rio.write_crs("EPSG:4326")

    out = spatial_mosaic([left, right], method="sum")

    assert float(out.sel(x=2.0, y=0.0)) == 400.0
    assert np.issubdtype(out.dtype, np.integer)


def test_sum_leaves_float_inputs_alone() -> None:
    out = spatial_mosaic([_tile(0.0, value=1.0), _tile(2.0, value=5.0)], method="sum")

    assert out.dtype == np.dtype("float64")
    assert float(out.sel(x=2.0, y=0.0)) == pytest.approx(6.0)


def test_mismatched_dataset_variables_raise() -> None:
    first = _tile(0.0).to_dataset(name="sst")
    second = _tile(4.0).to_dataset(name="sst")
    second["chl"] = _tile(4.0)

    with pytest.raises(ValueError, match=r"different variable set"):
        spatial_mosaic([first, second])


def test_matching_dataset_variables_pass() -> None:
    first = _tile(0.0).to_dataset(name="sst")
    first["chl"] = _tile(0.0)
    second = _tile(4.0).to_dataset(name="sst")
    second["chl"] = _tile(4.0)

    out = spatial_mosaic([first, second])

    assert set(out.data_vars) == {"sst", "chl"}


def test_differing_crs_input_is_matched_to_first_tile_resolution() -> None:
    # Coarser tile in another CRS: without an explicit resolution, rioxarray
    # would harmonise it at ~2 deg and leave merge_* to resample it nearest,
    # so `resampling` would not govern the output values.
    fine = _tile(0.0)
    coarse = (
        xr.DataArray(
            np.ones((4, 4)),
            dims=("y", "x"),
            coords={
                "y": 2.0 * np.arange(3.0, -1.0, -1.0),
                "x": 8.0 + 2.0 * np.arange(4.0),
            },
        )
        .rio.write_crs("EPSG:4326")
        .rio.reproject("EPSG:3857")
    )

    out = spatial_mosaic([fine, coarse], resampling="bilinear")

    assert out.rio.resolution() == pytest.approx(fine.rio.resolution())


def test_operator_matches_function_and_config_round_trips() -> None:
    tiles = [_tile(0.0, value=1.0), _tile(2.0, value=5.0)]
    op = SpatialMosaic(method="max")

    out = op(tiles)

    xr.testing.assert_allclose(out, spatial_mosaic(tiles, method="max"))
    config = op.get_config()
    assert config == {"method": "max", "resampling": "nearest", "nodata": None}
    assert json.loads(json.dumps(config)) == config
