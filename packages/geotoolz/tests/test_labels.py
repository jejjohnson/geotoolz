"""Tests for the shared connected-component / hole / skeleton primitives.

``geotoolz._src.labels`` is the one implementation behind
``mask.remove_small_*``, ``measure.LabelConnectedComponents`` /
``RegionProps`` / ``SkeletonLength`` and ``plume.label_components`` /
``PlumeContours`` / ``PlumeFootprint`` / ``plume_length("skeleton")``.
The ``*_matches_legacy`` tests pin the refactor to the pre-consolidation
algorithms, kept here as reference implementations.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from _helpers import toy_geotensor
from rasterio import features
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.measure import label as sk_label
from skimage.morphology import (
    remove_small_holes as sk_remove_small_holes,
    remove_small_objects as sk_remove_small_objects,
)

import geotoolz as gz
from geotoolz._src.labels import (
    DEFAULT_REGIONPROPS,
    connectivity_structure,
    label_components,
    remove_small_holes,
    skeleton_length,
)
from geotoolz.plume._src.array import plume_length


_SK_CONNECTIVITY = {4: 1, 8: 2}


def _random_masks(n: int = 8, shape: tuple[int, int] = (24, 31)) -> list[np.ndarray]:
    rng = np.random.default_rng(152)
    return [
        ndimage.binary_dilation(rng.random(shape) > 0.9, iterations=i % 3)
        for i in range(n)
    ]


# --- connectivity semantics ----------------------------------------------


def test_diagonal_neighbours_join_only_under_8_connectivity() -> None:
    mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)

    assert label_components(mask, connectivity=8).max() == 1
    labels4 = label_components(mask, connectivity=4)
    assert labels4.max() == 3
    np.testing.assert_array_equal(labels4[mask], [1, 2, 3])


def test_default_connectivity_is_8() -> None:
    mask = np.eye(4, dtype=bool)
    np.testing.assert_array_equal(
        label_components(mask), label_components(mask, connectivity=8)
    )


@pytest.mark.parametrize("bad", [0, 1, 2, 6, None])
def test_skimage_connectivity_spelling_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
        label_components(np.ones((2, 2), dtype=bool), connectivity=bad)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
        gz.measure.LabelConnectedComponents(connectivity=bad)(  # ty: ignore[invalid-argument-type]
            np.ones((2, 2), dtype=bool)
        )


def test_connectivity_structure_shapes() -> None:
    assert connectivity_structure(4).sum() == 5
    assert connectivity_structure(8).sum() == 9


@pytest.mark.parametrize("connectivity", [4, 8])
def test_label_components_matches_skimage_label(connectivity: int) -> None:
    """Raster-order labels identical to ``skimage.measure.label``."""
    for mask in _random_masks():
        expected = sk_label(mask, connectivity=_SK_CONNECTIVITY[connectivity])
        got = label_components(mask, connectivity=connectivity)  # ty: ignore[invalid-argument-type]
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_label_connected_components_matches_legacy_skimage_path(
    connectivity: int,
) -> None:
    """``LabelConnectedComponents(4|8)`` == the old ``label(connectivity=1|2)``."""
    for mask in _random_masks():
        out = gz.measure.LabelConnectedComponents(connectivity=connectivity)(  # ty: ignore[invalid-argument-type]
            toy_geotensor(mask, fill_value_default=None)
        )
        expected = sk_label(mask, connectivity=_SK_CONNECTIVITY[connectivity])
        np.testing.assert_array_equal(np.asarray(out), expected.astype(np.int32))


# --- small-component filter ------------------------------------------------


def test_min_area_drops_small_and_renumbers_contiguously() -> None:
    mask = np.zeros((5, 9), dtype=bool)
    mask[0, 0] = True  # 1 px
    mask[2, 2:5] = True  # 3 px
    mask[4, 7:9] = True  # 2 px

    labels = label_components(mask, min_area=2)

    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[0, 0] == 0
    assert (labels[2, 2:5] == 1).all()
    assert (labels[4, 7:9] == 2).all()


def test_min_area_zero_and_one_keep_everything() -> None:
    for mask in _random_masks(3):
        np.testing.assert_array_equal(
            label_components(mask, min_area=0), label_components(mask, min_area=1)
        )
        np.testing.assert_array_equal(label_components(mask, min_area=0) > 0, mask)


def test_min_area_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        label_components(np.ones((2, 2), dtype=bool), min_area=-1)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_min_area_matches_skimage_remove_small_objects(connectivity: int) -> None:
    for mask in _random_masks():
        expected = sk_remove_small_objects(
            mask, max_size=4, connectivity=_SK_CONNECTIVITY[connectivity]
        )
        got = label_components(mask, connectivity=connectivity, min_area=5) > 0  # ty: ignore[invalid-argument-type]
        np.testing.assert_array_equal(got, expected)


def _legacy_remove_small_objects_2d(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, num = ndimage.label(mask)
    if num == 0 or min_size == 0:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _legacy_remove_small_holes_2d(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    labels, num = ndimage.label(~mask)
    if num == 0 or area_threshold == 0:
        return mask.copy()
    border = set(
        np.unique(
            np.concatenate(
                [labels[0, :], labels[-1, :], labels[1:-1, 0], labels[1:-1, -1]]
            )
        )
    )
    sizes = np.bincount(labels.ravel())
    fill = np.zeros(num + 1, dtype=bool)
    for lbl in range(1, num + 1):
        fill[lbl] = lbl not in border and sizes[lbl] <= area_threshold
    return mask | fill[labels]


@pytest.mark.parametrize("size", [0, 1, 3, 7])
def test_mask_small_object_and_hole_helpers_match_legacy(size: int) -> None:
    for mask in _random_masks():
        for m in (mask, ~mask):
            np.testing.assert_array_equal(
                gz.mask.remove_small_objects(m, size),
                _legacy_remove_small_objects_2d(m, size),
            )
            np.testing.assert_array_equal(
                gz.mask.remove_small_holes(m, size),
                _legacy_remove_small_holes_2d(m, size),
            )


# --- holes ----------------------------------------------------------------------


def test_remove_small_holes_border_exclusion_flag() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False  # enclosed hole
    mask[0, 0] = False  # border-touching gap

    kept_border = remove_small_holes(mask, max_area=1)
    assert kept_border[2, 2]
    assert not kept_border[0, 0]

    filled_all = remove_small_holes(mask, max_area=1, exclude_border=False)
    assert filled_all.all()


def test_remove_small_holes_max_area_is_inclusive() -> None:
    mask = np.ones((6, 6), dtype=bool)
    mask[2:4, 2] = False  # 2-px hole
    assert not remove_small_holes(mask, max_area=1)[2, 2]
    assert remove_small_holes(mask, max_area=2)[2:4, 2].all()


@pytest.mark.parametrize("connectivity", [4, 8])
def test_remove_small_holes_without_border_exclusion_matches_skimage(
    connectivity: int,
) -> None:
    for mask in _random_masks():
        for m in (mask, ~mask):
            expected = sk_remove_small_holes(
                m, max_size=3, connectivity=_SK_CONNECTIVITY[connectivity]
            )
            got = remove_small_holes(
                m,
                max_area=3,
                connectivity=connectivity,  # ty: ignore[invalid-argument-type]
                exclude_border=False,
            )
            np.testing.assert_array_equal(got, expected)


def test_hole_connectivity_changes_what_is_enclosed() -> None:
    # A background pixel touching the border only diagonally is enclosed
    # under 4-connectivity but border-connected under 8-connectivity.
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    mask[1, 1] = False
    assert remove_small_holes(mask, max_area=1, connectivity=4)[1, 1]
    assert not remove_small_holes(mask, max_area=1, connectivity=8)[1, 1]


# --- skeleton length -------------------------------------------------------------


def test_skeleton_length_straight_line_counts_centre_steps() -> None:
    mask = np.zeros((5, 10), dtype=bool)
    mask[2, :] = True
    assert skeleton_length(mask) == pytest.approx(9.0)


def test_skeleton_length_diagonal_steps_are_euclidean() -> None:
    assert skeleton_length(np.eye(5, dtype=bool)) == pytest.approx(4 * np.sqrt(2))


def test_skeleton_length_4_connectivity_does_not_walk_diagonals() -> None:
    assert skeleton_length(np.eye(5, dtype=bool), connectivity=4) == 0.0


def test_skeleton_length_empty_single_and_nan() -> None:
    assert skeleton_length(np.zeros((4, 4), dtype=bool)) == 0.0
    single = np.zeros((4, 4), dtype=bool)
    single[1, 1] = True
    assert skeleton_length(single) == 0.0
    values = np.full((3, 6), np.nan)
    values[1, :] = 1.0
    assert skeleton_length(values) == pytest.approx(5.0)


def test_skeleton_length_step_pair_and_transform_agree() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 4] = True  # vertical arm
    mask[7, 4:8] = True  # horizontal foot
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -20.0, 0.0)

    by_pair = skeleton_length(mask, step=(20.0, 10.0))
    by_transform = skeleton_length(mask, step=transform)

    # The corner is cut by one diagonal (20 m x 10 m) step.
    assert by_pair == pytest.approx(5 * 20.0 + np.hypot(20.0, 10.0) + 2 * 10.0)
    assert by_transform == pytest.approx(by_pair)


def test_measure_and_plume_share_skeleton_length() -> None:
    """``SkeletonLength`` and ``plume_length("skeleton")`` are one function."""
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 3] = True
    mask[9, 3:10] = True
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 120.0)
    gt = toy_geotensor(mask, transform=transform, fill_value_default=None)

    pixels = gz.measure.SkeletonLength()(gt)
    crs_units = gz.measure.SkeletonLength(scale_to_crs=True)(gt)

    assert pixels == pytest.approx(skeleton_length(mask))
    assert crs_units == pytest.approx(10.0 * pixels)
    assert plume_length(mask, transform, method="skeleton") == pytest.approx(crs_units)


def test_skeleton_length_scale_to_crs_requires_geotensor() -> None:
    with pytest.raises(TypeError, match="georeferenced"):
        gz.measure.SkeletonLength(scale_to_crs=True)(np.eye(4, dtype=bool))


# --- region properties ----------------------------------------------------------


def test_plume_footprint_uses_default_regionprops() -> None:
    assert gz.plume.PlumeFootprint().properties == DEFAULT_REGIONPROPS
    assert gz.measure.RegionProps().properties == DEFAULT_REGIONPROPS
    assert gz.measure.DEFAULT_REGIONPROPS is DEFAULT_REGIONPROPS


def test_region_props_default_is_pixel_units_and_scale_to_crs_converts() -> None:
    labels = np.zeros((6, 8), dtype=np.int32)
    labels[1:4, 1:5] = 1  # 3 x 4 px rectangle
    gt = toy_geotensor(
        labels,
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 60.0),
        fill_value_default=0,
    )

    px = gz.measure.RegionProps()(gt)
    crs = gz.measure.RegionProps(scale_to_crs=True)(gt)

    assert px["area"].iloc[0] == 12
    assert crs["area"].iloc[0] == pytest.approx(1200.0)
    for column in ("perimeter", "major_axis_length", "minor_axis_length"):
        assert crs[column].iloc[0] == pytest.approx(10.0 * px[column].iloc[0])
    assert crs["inertia_tensor_eigvals-0"].iloc[0] == pytest.approx(
        100.0 * px["inertia_tensor_eigvals-0"].iloc[0]
    )
    # Positions, angles and ratios are unchanged; geometry is CRS either way.
    for column in ("centroid-0", "bbox-2", "orientation", "eccentricity", "solidity"):
        assert crs[column].iloc[0] == px[column].iloc[0]
    assert crs.geometry.iloc[0].equals(px.geometry.iloc[0])
    assert gz.measure.RegionProps(scale_to_crs=True).get_config()["scale_to_crs"]


def test_region_props_scale_to_crs_rejects_lengths_on_non_square_pixels() -> None:
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1:3, 1:3] = 1
    gt = toy_geotensor(
        labels,
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -20.0, 80.0),
        fill_value_default=0,
    )
    with pytest.raises(ValueError, match="non-square pixels"):
        gz.measure.RegionProps(scale_to_crs=True)(gt)
    area_only = gz.measure.RegionProps(
        properties=("label", "area", "centroid"), scale_to_crs=True
    )(gt)
    assert area_only["area"].iloc[0] == pytest.approx(4 * 200.0)


def _legacy_footprint_geometry(labels: np.ndarray, label_id: int, transform) -> object:
    component = labels == label_id
    return unary_union(
        [
            shape(geom)
            for geom, _ in features.shapes(
                labels.astype(np.int32), mask=component, transform=transform
            )
        ]
    )


def test_plume_footprint_polygons_match_legacy_rasterio_shapes() -> None:
    transform = rasterio.Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
    for mask in _random_masks(4):
        gt = toy_geotensor(mask, transform=transform, fill_value_default=None)
        gdf = gz.plume.PlumeFootprint(min_area_m2=0.0, simplify_tolerance=None)(gt)
        labels = label_components(mask)
        assert len(gdf) == labels.max()
        for _, row in gdf.iterrows():
            legacy = _legacy_footprint_geometry(labels, row["label_id"], transform)
            assert row.geometry.equals(legacy)
            assert row["area_m2"] == pytest.approx(legacy.area)


# --- nodata ----------------------------------------------------------------------


def test_label_connected_components_nodata_never_bridges_components() -> None:
    values = np.array([[1.0, np.nan, 1.0]])
    out = gz.measure.LabelConnectedComponents()(values)
    np.testing.assert_array_equal(out, [[1, 0, 2]])


def test_label_connected_components_min_area_and_background() -> None:
    mask = np.array([[1, 1, 0, 1], [0, 0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(
        gz.measure.LabelConnectedComponents(min_area=2)(mask),
        [[1, 1, 0, 0], [0, 0, 0, 0]],
    )
    # background=1: the zeros are the foreground (one 4-connected region).
    np.testing.assert_array_equal(
        gz.measure.LabelConnectedComponents(background=1)(mask),
        [[0, 0, 1, 0], [1, 1, 1, 1]],
    )
