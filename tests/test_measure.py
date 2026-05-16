"""Tests for measurement operators."""

from __future__ import annotations

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=0,
    )


def test_label_connected_components_returns_int_geotensor() -> None:
    mask = _gt(np.array([[1, 0, 1], [1, 0, 0]], dtype=bool))

    labels = gz.measure.LabelConnectedComponents(connectivity=1)(mask)

    assert labels.shape == mask.shape
    assert labels.transform == mask.transform
    assert np.asarray(labels).dtype == np.int32
    assert int(np.asarray(labels).max()) == 2


def test_regionprops_returns_geodataframe_with_region_fields() -> None:
    labels = _gt(np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32))

    props = gz.measure.RegionProps()(labels)

    assert list(props["label"]) == [1, 2]
    assert "area" in props.columns
    assert props.geometry.notna().all()
