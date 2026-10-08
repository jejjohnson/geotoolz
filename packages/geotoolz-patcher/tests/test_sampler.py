"""Tests for the spatial `Sampler` family."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from scipy.spatial.distance import pdist

from geopatcher import spatial
from geopatcher.fields import GridDomain


@pytest.fixture
def raster_domain() -> GeoTensor:
    return GeoTensor(
        values=np.zeros((1, 64, 64), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )


@pytest.fixture
def rect() -> spatial.geometry.Rectangular:
    return spatial.geometry.Rectangular(size=(16, 16))


class TestSpatialRegularStride:
    def test_raster_anchor_count(
        self, raster_domain: GeoTensor, rect: spatial.geometry.Rectangular
    ) -> None:
        s = spatial.sampler.RegularStride(step=16)
        anchors = list(s.anchors(raster_domain, rect))
        # 64x64 raster, 16x16 patches, stride 16 -> 4x4 = 16 anchors
        assert len(anchors) == 16
        # All anchors are valid (row, col) pairs
        assert all(0 <= r <= 48 and 0 <= c <= 48 for r, c in anchors)

    @pytest.mark.parametrize("step", [0, -4, (16, 0), 2.5])
    @pytest.mark.parametrize(
        "cls", [spatial.sampler.RegularStride, spatial.sampler.JitteredStride]
    )
    def test_rejects_non_positive_step(self, cls: type, step: object) -> None:
        # #187: step=0 used to fail late with "range() arg 3 must not be zero".
        with pytest.raises(ValueError, match="step must be a positive integer"):
            cls(step=step)

    def test_rejects_negative_n_samples(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            spatial.sampler.Random(n_samples=-1)


class TestSpatialJitteredStride:
    def test_reproducible_seed(
        self, raster_domain: GeoTensor, rect: spatial.geometry.Rectangular
    ) -> None:
        s1 = spatial.sampler.JitteredStride(step=16, jitter=0.5, seed=0)
        s2 = spatial.sampler.JitteredStride(step=16, jitter=0.5, seed=0)
        assert list(s1.anchors(raster_domain, rect)) == list(
            s2.anchors(raster_domain, rect)
        )

    @pytest.mark.parametrize("grid", [False, True], ids=["raster", "grid"])
    def test_jitter_uniform(self, grid: bool) -> None:
        # #187: int() truncation toward zero gave P(offset == 0) = 0.126
        # and never reached -8. floor() makes the 16 offsets -8..7 uniform.
        size, step, n = 4, 16, 64 * 16 + 4
        if grid:
            domain: object = GridDomain(coords={"a": np.arange(n), "b": np.arange(n)})
        else:
            domain = GeoTensor(
                values=np.zeros((n, n), dtype=np.uint8),
                transform=rasterio.Affine.identity(),
                crs="EPSG:32630",
            )
        geom = spatial.geometry.Rectangular(size=(size, size))
        s = spatial.sampler.JitteredStride(step=step, jitter=0.5, seed=0)
        base = spatial.sampler.RegularStride(step=step).anchors(domain, geom)
        offsets = []
        for b, a in zip(base, s.anchors(domain, geom), strict=True):
            pairs = (
                zip(b.values(), a.values(), strict=True)
                if grid
                else zip(b, a, strict=True)
            )
            offsets += [ai - bi for bi, ai in pairs if 8 <= bi <= n - size - 8]
        values, counts = np.unique(offsets, return_counts=True)
        assert values.tolist() == list(range(-8, 8))
        freq = counts / counts.sum()
        assert np.all(np.abs(freq - 1 / 16) < 0.02)

    def test_rejects_negative_jitter(self) -> None:
        with pytest.raises(ValueError, match="jitter"):
            spatial.sampler.JitteredStride(step=4, jitter=-0.1)


class TestSpatialRandom:
    def test_count_matches_request(
        self, raster_domain: GeoTensor, rect: spatial.geometry.Rectangular
    ) -> None:
        s = spatial.sampler.Random(n_samples=7, seed=42)
        anchors = list(s.anchors(raster_domain, rect))
        assert len(anchors) == 7

    def test_grid_anchors_are_dicts(self, rect: spatial.geometry.Rectangular) -> None:
        gd = GridDomain(
            coords={"a": np.arange(64), "b": np.arange(64)},
        )
        s = spatial.sampler.Random(n_samples=3, seed=0)
        anchors = list(s.anchors(gd, rect))
        assert len(anchors) == 3
        assert all(isinstance(a, dict) for a in anchors)


class TestSpatialPoissonDisk:
    def test_min_distance_invariant(
        self, raster_domain: GeoTensor, rect: spatial.geometry.Rectangular
    ) -> None:
        s = spatial.sampler.PoissonDisk(min_dist=8.0, seed=0)
        anchors = np.asarray(list(s.anchors(raster_domain, rect)), dtype=float)
        # #187: the guarantee holds exactly between the integer anchors
        # (it used to be checked on the float candidates, then cast).
        assert len(anchors) > 1
        assert pdist(anchors).min() >= 8.0

    @pytest.mark.parametrize("min_dist", [1.0, 2.0, 2.5, 8.0])
    @pytest.mark.parametrize("seed", range(10))
    def test_poisson_disk_min_distance_exact(self, min_dist: float, seed: int) -> None:
        domain = GeoTensor(
            values=np.zeros((40, 40), dtype=np.float32),
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
        geom = spatial.geometry.Rectangular(size=(1, 1))
        s = spatial.sampler.PoissonDisk(min_dist=min_dist, seed=seed)
        anchors = np.asarray(list(s.anchors(domain, geom)), dtype=float)
        assert len(anchors) > 1
        assert pdist(anchors).min() >= min_dist
        assert len({tuple(a) for a in anchors}) == len(anchors)

    def test_point_subset_is_maximal_and_spaced(self) -> None:
        from scipy.spatial import cKDTree

        from geopatcher.fields import PointDomain

        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 100, size=(3000, 2))
        domain = PointDomain(coords=coords, kdtree=cKDTree(coords))
        s = spatial.sampler.PoissonDisk(min_dist=5.0, seed=1)
        picked = list(s.anchors(domain, spatial.geometry.Rectangular(size=(1, 1))))
        assert len(set(picked)) == len(picked)
        assert pdist(coords[picked]).min() >= 5.0
        # Maximal: every rejected point is within min_dist of a kept one.
        tree = cKDTree(coords[picked])
        rejected = np.setdiff1d(np.arange(len(coords)), picked)
        d, _ = tree.query(coords[rejected])
        assert np.all(d < 5.0)

    @pytest.mark.parametrize("min_dist", [0.0, -1.0, float("nan")])
    def test_rejects_non_positive_min_dist(self, min_dist: float) -> None:
        # min_dist=0 used to surface as an OverflowError from the grid.
        with pytest.raises(ValueError, match="min_dist"):
            spatial.sampler.PoissonDisk(min_dist=min_dist)

    def test_rejects_zero_max_tries(self) -> None:
        with pytest.raises(ValueError, match="max_tries"):
            spatial.sampler.PoissonDisk(min_dist=1.0, max_tries=0)


class TestSpatialExplicit:
    def test_yields_supplied(
        self, raster_domain: GeoTensor, rect: spatial.geometry.Rectangular
    ) -> None:
        anchors = [(0, 0), (10, 5), (32, 16)]
        s = spatial.sampler.Explicit(anchors_=anchors)
        assert list(s.anchors(raster_domain, rect)) == anchors


class TestCheckFullScan:
    """`spatial.sampler.RegularStride(check_full_scan=True)` raises on partial tiles."""

    def test_raster_exact_tiling_passes(
        self, raster_domain: GeoTensor, rect: spatial.geometry.Rectangular
    ) -> None:
        # 64 / 16 = 4 — exact, both axes.
        s = spatial.sampler.RegularStride(step=16, check_full_scan=True)
        anchors = list(s.anchors(raster_domain, rect))
        assert len(anchors) == 16

    def test_raster_partial_tile_raises(
        self, raster_domain: GeoTensor, rect: spatial.geometry.Rectangular
    ) -> None:
        from geopatcher.spatial.sampler import IncompleteScanConfiguration

        # step=20 against (64, 64): (64 - 16) % 20 = 8 ≠ 0
        s = spatial.sampler.RegularStride(step=20, check_full_scan=True)
        with pytest.raises(IncompleteScanConfiguration, match="row"):
            list(s.anchors(raster_domain, rect))

    def test_grid_exact_tiling_passes(self) -> None:
        grid = GridDomain(
            coords={
                "latitude": np.linspace(-30, 30, 24),
                "longitude": np.linspace(0, 60, 36),
            },
        )
        rect = spatial.geometry.Rectangular(size=(6, 6))
        s = spatial.sampler.RegularStride(step=(6, 6), check_full_scan=True)
        anchors = list(s.anchors(grid, rect))
        # Both axes: (24 - 6) / 6 + 1 = 4; (36 - 6) / 6 + 1 = 6 → 4 * 6 = 24
        assert len(anchors) == 24

    def test_grid_partial_tile_raises(self) -> None:
        from geopatcher.spatial.sampler import IncompleteScanConfiguration

        grid = GridDomain(
            coords={
                "latitude": np.linspace(-30, 30, 25),
                "longitude": np.linspace(0, 60, 36),
            },
        )
        rect = spatial.geometry.Rectangular(size=(6, 6))
        s = spatial.sampler.RegularStride(step=(6, 6), check_full_scan=True)
        # (25 - 6) % 6 = 1 ≠ 0 on latitude
        with pytest.raises(IncompleteScanConfiguration, match="latitude"):
            list(s.anchors(grid, rect))

    def test_default_off_preserves_silent_truncation(
        self, raster_domain: GeoTensor, rect: spatial.geometry.Rectangular
    ) -> None:
        # No flag → same behaviour as before: partial tile silently dropped.
        s = spatial.sampler.RegularStride(step=20)
        anchors = list(s.anchors(raster_domain, rect))
        assert len(anchors) > 0  # no raise

    def test_get_config_round_trip(self) -> None:
        s = spatial.sampler.RegularStride(step=(6, 6), check_full_scan=True)
        cfg = s.get_config()
        assert cfg == {"step": [6, 6], "check_full_scan": True}


class TestSpatialAlongTrack:
    def test_vertices_used_when_no_spacing(self, raster_domain: GeoTensor) -> None:

        track = np.array([[8.5, 8.5], [24.5, 8.5], [40.5, 8.5]])
        s = spatial.sampler.AlongTrack(track=track)
        rect = spatial.geometry.Rectangular(size=(4, 4))
        anchors = list(s.anchors(raster_domain, rect))
        # Identity transform: (x, y) -> pixel (row=y, col=x); anchors are
        # the UL corners that centre the 4x4 patch on that pixel.
        assert anchors == [(6, 6), (6, 22), (6, 38)]

    def test_spacing_resamples_to_monotonic_distances(
        self, raster_domain: GeoTensor
    ) -> None:

        # Straight track of length 40, spacing 10 -> s = 0, 10, 20, 30, 40.
        track = np.array([[10.0, 10.0], [50.0, 10.0]])
        s = spatial.sampler.AlongTrack(track=track, spacing=10.0)
        pts = s._resampled()
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert len(pts) == 5
        np.testing.assert_allclose(d, 10.0)
        np.testing.assert_allclose(np.diff(pts[:, 0]), 10.0)  # monotonic

    def test_out_of_domain_points_skipped(self, raster_domain: GeoTensor) -> None:

        track = np.array([[-5.0, 8.0], [8.0, 8.0], [200.0, 8.0]])
        s = spatial.sampler.AlongTrack(track=track)
        rect = spatial.geometry.Rectangular(size=(4, 4))
        anchors = list(s.anchors(raster_domain, rect))
        assert len(anchors) == 1

    def test_point_domain_yields_xy(self) -> None:
        from scipy.spatial import cKDTree

        from geopatcher.fields import PointDomain

        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        domain = PointDomain(coords=coords, kdtree=cKDTree(coords))
        track = np.array([[0.1, 0.1], [1.9, 1.9]])
        s = spatial.sampler.AlongTrack(track=track)
        anchors = list(s.anchors(domain, spatial.geometry.KNNGraph(k=1)))
        assert anchors == [(0.1, 0.1), (1.9, 1.9)]

    def test_duplicate_vertices_collapsed(self) -> None:

        track = np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
        s = spatial.sampler.AlongTrack(track=track, spacing=2.0)
        pts = s._resampled()
        assert len(pts) == 3

    def test_spacing_requires_two_distinct_vertices(self) -> None:

        s = spatial.sampler.AlongTrack(
            track=np.array([[1.0, 1.0], [1.0, 1.0]]), spacing=1.0
        )
        with pytest.raises(ValueError, match="two distinct"):
            s._resampled()

    def test_linestring_track(self) -> None:
        shapely = pytest.importorskip("shapely")

        line = shapely.LineString([(0, 0), (3, 4)])
        s = spatial.sampler.AlongTrack(track=line, spacing=5.0)
        pts = s._resampled()
        np.testing.assert_allclose(pts, [[0.0, 0.0], [3.0, 4.0]])

    def test_get_config(self) -> None:

        s = spatial.sampler.AlongTrack(track=np.zeros((7, 2)), spacing=2.5)
        assert s.get_config() == {
            "track": [[0.0, 0.0]] * 7,
            "spacing": 2.5,
            "crs": None,
            "polar_guard": "warn",
        }

    def test_non_drop_boundary_preserves_centered_overflow(
        self, raster_domain: GeoTensor
    ) -> None:

        # Track point in pixel (0, 0): centring a 4x4 patch needs anchor
        # (-2, -2). "pad" must keep the overflow (boundless read fills
        # the context); only "drop" clamps the anchor in-domain.
        track = np.array([[0.5, 0.5]])
        s = spatial.sampler.AlongTrack(track=track)
        padded = spatial.geometry.Rectangular(size=(4, 4), boundary="pad")
        assert list(s.anchors(raster_domain, padded)) == [(-2, -2)]
        dropped = spatial.geometry.Rectangular(size=(4, 4), boundary="drop")
        assert list(s.anchors(raster_domain, dropped)) == [(0, 0)]

    def test_spacing_keeps_final_vertex(self) -> None:

        # #187: length 25 at spacing 10 used to stop at s = 20, silently
        # dropping the track's end; the last interval is now shorter.
        s = spatial.sampler.AlongTrack(
            track=np.array([[0.0, 0.0], [25.0, 0.0]]), spacing=10.0
        )
        np.testing.assert_allclose(s._resampled()[:, 0], [0.0, 10.0, 20.0, 25.0])

    def test_xyz_array_accepted_like_3d_linestring(self) -> None:
        shapely = pytest.importorskip("shapely")

        xyz = np.array([[0.0, 0.0, 5.0], [3.0, 4.0, 7.0]])
        from_array = spatial.sampler.AlongTrack(track=xyz)
        from_line = spatial.sampler.AlongTrack(track=shapely.LineString(xyz))
        np.testing.assert_array_equal(from_array.track, xyz[:, :2])
        np.testing.assert_array_equal(from_line.track, xyz[:, :2])


class TestCentredAnchorConvention:
    """#187: the coordinate's pixel lands at chip index ``size // 2``.

    `georeader.read.window_from_center_coords` rounds the continuous UL
    corner instead (banker's rounding), so for even sizes the two can
    differ by one pixel; the docstring no longer claims they agree.
    """

    @pytest.mark.parametrize("size", [3, 4, 5, 16])
    @pytest.mark.parametrize("frac", [0.01, 0.5, 0.99])
    def test_pixel_lands_at_chip_centre_index(
        self, raster_domain: GeoTensor, size: int, frac: float
    ) -> None:

        row, col = 30, 21
        s = spatial.sampler.ExplicitCoords([(col + frac, row + frac)])
        geom = spatial.geometry.Rectangular(size=(size, size))
        ((ar, ac),) = list(s.anchors(raster_domain, geom))
        assert (row - ar, col - ac) == (size // 2, size // 2)
