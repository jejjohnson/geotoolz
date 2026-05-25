"""Smoke tests for `geotoolz.geom.coregister` and matched compositing.

Locks in the operator surface and YAML-round-trip shape so the
Phase 3 PR that wires up the rasterio / scipy / xvec primitives
can't accidentally rename a public field.
"""

from __future__ import annotations

import pytest
from pipekit import Operator

from geotoolz.compositing import BlendMatched, StackMatched
from geotoolz.geom import coregister as cg
from geotoolz.geom.coregister import (
    GridToSwath,
    PointCloudToRaster,
    PointsToRaster,
    RasterToPointCloud,
    RasterToPoints,
    RasterToRasterLike,
    SwathToGrid,
    VectorToRasterAgg,
)


class TestSubnamespaceWiring:
    def test_subnamespace_attached(self) -> None:
        # `from geotoolz.geom import coregister` is the documented
        # import path; the subnamespace must be discoverable from a
        # bare ``import geotoolz`` + attribute walk.
        import geotoolz.geom

        assert geotoolz.geom.coregister is cg

    def test_all_exports(self) -> None:
        expected = {
            "RasterToRasterLike",
            "SwathToGrid",
            "GridToSwath",
            "RasterToPoints",
            "PointsToRaster",
            "RasterToPointCloud",
            "PointCloudToRaster",
            "VectorToRasterAgg",
        }
        assert set(cg.__all__) == expected


class TestOperatorContract:
    """Every coregister op is a `pipekit.Operator` with a working
    `get_config`."""

    @pytest.mark.parametrize(
        "op",
        [
            RasterToRasterLike(resampling="cubic"),
            SwathToGrid(target_crs="EPSG:32629", target_res=(500.0, 500.0)),
            GridToSwath(dt_max="30min"),
            RasterToPoints(extract="bilinear", out_var="albedo"),
            PointsToRaster(method="binned_stat", stat="median"),
            RasterToPointCloud(k=3, max_radius=50.0, method="idw"),
            PointCloudToRaster(method="idw", power=1.5),
            VectorToRasterAgg(agg="majority", attribute="class_id"),
            StackMatched(order=["modis", "s2"], fill=-9999.0),
            BlendMatched(method="weighted_mean", weights=[1.0, 2.0]),
        ],
    )
    def test_is_operator(self, op) -> None:
        assert isinstance(op, Operator)

    @pytest.mark.parametrize(
        ("op", "expected_subset"),
        [
            (RasterToRasterLike(resampling="lanczos"), {"resampling": "lanczos"}),
            (
                SwathToGrid(target_crs="EPSG:4326", target_res=(0.01, 0.01)),
                {"target_crs": "EPSG:4326"},
            ),
            (GridToSwath(dt_max="45min"), {"dt_max": "45min"}),
            (PointsToRaster(stat="sum"), {"stat": "sum"}),
            (RasterToPointCloud(k=5), {"k": 5}),
            (VectorToRasterAgg(agg="count"), {"agg": "count"}),
            (StackMatched(fill=-1.0), {"fill": -1.0}),
            (BlendMatched(method="ivw"), {"method": "ivw"}),
        ],
    )
    def test_get_config_round_trip(self, op, expected_subset) -> None:
        cfg = op.get_config()
        for k, v in expected_subset.items():
            assert cfg[k] == v


class TestValidation:
    def test_raster_to_point_cloud_k_positive(self) -> None:
        with pytest.raises(ValueError):
            RasterToPointCloud(k=0)
        # k=1 is the default; k>=1 is allowed.
        RasterToPointCloud(k=1)


class TestCallNotImplemented:
    """All operator bodies raise NotImplementedError in the scaffolding
    PR — the Phase 3 PR fills them in."""

    def test_raster_to_raster_like(self) -> None:
        with pytest.raises(NotImplementedError):
            RasterToRasterLike()(object(), object())  # type: ignore[arg-type]

    def test_swath_to_grid(self) -> None:
        with pytest.raises(NotImplementedError):
            SwathToGrid(target_crs="EPSG:32629", target_res=(500.0, 500.0))(object())  # type: ignore[arg-type]

    def test_stack_matched(self) -> None:
        with pytest.raises(NotImplementedError):
            StackMatched()([object()])  # type: ignore[list-item]

    def test_blend_matched(self) -> None:
        with pytest.raises(NotImplementedError):
            BlendMatched()([object()])  # type: ignore[list-item]
