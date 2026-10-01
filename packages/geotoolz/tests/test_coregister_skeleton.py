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
    PointCloudToRaster,
    PointsToRaster,
    RasterToPointCloud,
    RasterToPoints,
    RasterToRasterLike,
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
            RasterToPoints(extract="bilinear", out_var="albedo"),
            PointsToRaster(method="binned_stat", stat="median"),
            RasterToPointCloud(k=3, max_radius=50.0, method="idw"),
            PointCloudToRaster(method="idw", power=1.5),
            VectorToRasterAgg(agg="majority", attribute="class_id"),
            StackMatched(order=["modis", "s2"]),
            BlendMatched(method="weighted_mean", weights=[1.0, 2.0]),
        ],
    )
    def test_is_operator(self, op) -> None:
        assert isinstance(op, Operator)

    @pytest.mark.parametrize(
        ("op", "expected_subset"),
        [
            (RasterToRasterLike(resampling="lanczos"), {"resampling": "lanczos"}),
            (PointsToRaster(stat="sum"), {"stat": "sum"}),
            (RasterToPointCloud(k=5, method="idw"), {"k": 5}),
            (VectorToRasterAgg(agg="count"), {"agg": "count"}),
            (StackMatched(order=["a", "b"]), {"order": ["a", "b"]}),
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


class TestSwathStubsRemoved:
    """The never-implemented swath operators are not public API (#162)."""

    @pytest.mark.parametrize("name", ["SwathToGrid", "GridToSwath"])
    def test_not_exported(self, name: str) -> None:
        assert not hasattr(cg, name)

    def test_no_swath_to_grid_primitive(self) -> None:
        from geotoolz.geom._src.coregister import array

        assert not hasattr(array, "swath_to_grid")
