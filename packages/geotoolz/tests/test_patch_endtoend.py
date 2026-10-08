"""End-to-end test of the chip → predict → stitch pipeline through `Sequential`.

Exercises the `geotoolz.patch_ops` bridge end-to-end; skip cleanly when
the optional ``[patch]`` extra (which pulls in geopatcher) isn't installed.
"""

from __future__ import annotations

import pytest


pytest.importorskip(
    "geopatcher",
    reason="geotoolz.patch_ops bridge requires the [patch] extra (geopatcher)",
)

import numpy as np
import rasterio
from geopatcher import RasterField, SpatialPatcher, spatial
from georeader.geotensor import GeoTensor
from pipekit import Lambda

from geotoolz import Sequential
from geotoolz.patch_ops import (
    ApplyToChips,
    GridSampler,
    MergePatches,
)


def _ones_field() -> RasterField:
    arr = np.ones((32, 32), dtype=np.float32)
    gt = GeoTensor(values=arr, transform=rasterio.Affine.identity(), crs="EPSG:32630")
    return RasterField(gt)


def test_sliding_window_inference_boxcar() -> None:
    """Tile -> double -> stitch with non-overlapping Boxcar windows."""
    field = _ones_field()
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=8),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
    pipe = Sequential(
        [
            GridSampler(patcher=patcher),
            ApplyToChips(operator=double),
            MergePatches(
                aggregation=spatial.aggregation.OverlapAdd(), domain=field.reader
            ),
        ]
    )
    result = pipe(field)
    assert result.shape == (32, 32)
    # Boxcar + non-overlapping stride: full coverage with weight=1.
    np.testing.assert_allclose(result, 2.0)


def test_sliding_window_inference_hann_overlap() -> None:
    """Hann window with stride < patch size to ensure overlap fills boundaries."""
    field = _ones_field()
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=4),
        window=spatial.window.Hann(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
    pipe = Sequential(
        [
            GridSampler(patcher=patcher),
            ApplyToChips(operator=double),
            MergePatches(
                aggregation=spatial.aggregation.OverlapAdd(), domain=field.reader
            ),
        ]
    )
    result = pipe(field)
    # Strict interior should be covered by enough Hann patches to sum to 2.
    np.testing.assert_allclose(result[8:24, 8:24], 2.0, rtol=1e-6)


def test_band_collapsing_operator_merges_onto_the_field_grid() -> None:
    """NDVI per chip, stitched onto ``field.domain``: one band, georeferenced.

    The merge used to size its output from the 4-band domain and broadcast
    the index into every band, and returned a bare array.
    """
    import geotoolz as gz

    transform = rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)
    rng = np.random.default_rng(0)
    scene = GeoTensor(
        values=rng.integers(1, 10_000, (4, 64, 64)).astype(np.uint16),
        transform=transform,
        crs="EPSG:32630",
        fill_value_default=0,
        attrs={"band_names": ["B02", "B03", "B04", "B08"]},
    )
    field = RasterField(scene)
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(32, 32)),
        sampler=spatial.sampler.RegularStride(step=(16, 16)),
        window=spatial.window.Hann(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    ndvi = gz.NDVI(nir="B08", red="B04")
    pipe = Sequential(
        [
            GridSampler(patcher=patcher),
            ApplyToChips(operator=ndvi),
            MergePatches(
                aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain
            ),
        ]
    )

    out = pipe(field)

    assert isinstance(out, GeoTensor)
    assert out.shape == (64, 64)
    assert out.transform == transform
    assert "band_names" not in out.attrs
    assert np.isnan(out.fill_value_default)
    whole = np.asarray(ndvi(scene))
    covered = ~np.isnan(np.asarray(out))
    np.testing.assert_allclose(np.asarray(out)[covered], whole[covered], atol=1e-12)
    assert covered[1:, 1:].all()  # only the periodic Hann's zero edge is uncovered
