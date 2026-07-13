"""Tests for the grid-alignment feature (geopatcher#59).

Covers:

- `divide_evenly`: aligned passes, misaligned raises with the
  residual surfaced, custom ``tol`` honoured.
- `GeoSlice.align` modes: ``"off"`` is silent, ``"warn"`` emits a
  `GridAlignmentWarning`, ``"error"`` raises, ``"snap"`` rounds
  outward while preserving the affine origin (``xmin`` and ``ymax``
  for north-up rasters).
- Unknown modes are rejected at construction.
- `warnings.warn` is visible even with `loguru.disable("geocatalog")`
  in effect (regression for the "warn invisible by default" bug).
- The `align` field does not participate in equality or hashing.
- `to_crs` does not self-trip the check even with ``align="error"``
  on the parent.
- `iter_slices` (in-memory backend) emits zero warnings on a
  misaligned catalog.
- `is_grid_aligned`: true / false / CRS-mismatch / ``explain``
  paths; y origin is checked against ``ymax`` (the north-up affine
  origin), not ``ymin``.
- `aligned_shape()` raises on misalignment regardless of mode.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest
from shapely.geometry import box

from geocatalog import (
    Align,
    GeoSlice,
    GridAlignmentWarning,
    InMemoryGeoCatalog,
    divide_evenly,
    is_grid_aligned,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_slice(
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
    resolution: tuple[float, float] = (10.0, 10.0),
    *,
    align: Align = "off",
    crs: str = "EPSG:32629",
) -> GeoSlice:
    return GeoSlice(
        bounds=bounds,
        interval=pd.Interval(
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02"),
            closed="both",
        ),
        resolution=resolution,
        crs=crs,
        align=align,
    )


# ---------------------------------------------------------------------------
# divide_evenly
# ---------------------------------------------------------------------------


class TestDivideEvenly:
    def test_exact_returns_quotient(self) -> None:
        assert divide_evenly(100.0, 10.0) == 10

    def test_subpixel_misalignment_raises(self) -> None:
        with pytest.raises(ValueError, match="residual"):
            divide_evenly(100.5, 10.0, label="x-extent")

    def test_error_message_carries_label_and_step(self) -> None:
        with pytest.raises(ValueError, match=r"x-extent.*step=10\.0"):
            divide_evenly(100.5, 10.0, label="x-extent")

    def test_within_default_tol_passes(self) -> None:
        # PIXEL_PRECISION=3 → default tol = 1e-3, so residual 5e-4 passes.
        assert divide_evenly(100.0005, 10.00005) == 10

    def test_custom_tol_tightens(self) -> None:
        # 100.0005 / 10 = 10.00005 → round = 10; residual = -5e-4.
        # The default tol (1e-3) accepts it; a tight tol must reject.
        assert divide_evenly(100.0005, 10.0) == 10
        with pytest.raises(ValueError):
            divide_evenly(100.0005, 10.0, tol=1e-9)


# ---------------------------------------------------------------------------
# GeoSlice.align modes
# ---------------------------------------------------------------------------


class TestAlignModes:
    def test_off_is_silent_and_keeps_bounds(self) -> None:
        sl = _make_slice((0.0, 0.0, 105.0, 100.0), (10.0, 10.0), align="off")
        assert sl.bounds == (0.0, 0.0, 105.0, 100.0)

    def test_error_raises_on_misalignment(self) -> None:
        with pytest.raises(ValueError, match="x-extent"):
            _make_slice((0.0, 0.0, 105.0, 100.0), (10.0, 10.0), align="error")

    def test_warn_emits_grid_alignment_warning(self) -> None:
        with pytest.warns(GridAlignmentWarning, match="x-extent"):
            sl = _make_slice((0.0, 0.0, 105.0, 100.0), (10.0, 10.0), align="warn")
        # warn leaves bounds untouched.
        assert sl.bounds == (0.0, 0.0, 105.0, 100.0)

    def test_warn_silent_when_aligned(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GridAlignmentWarning)
            # No warning → no exception under the strict filter.
            _make_slice((0.0, 0.0, 100.0, 100.0), (10.0, 10.0), align="warn")

    def test_warn_visible_without_logger_enable(self) -> None:
        """`align="warn"` must not be silenced by `loguru.disable`.

        Regression for the codex bug: the package calls
        `logger.disable("geocatalog")` at import, so a loguru-based
        warn implementation would be invisible by default. Stdlib
        `warnings.warn` is independent of the loguru namespace, so a
        user who opts into `align="warn"` actually sees the notice.
        """
        # Importing geocatalog has already run logger.disable; do NOT
        # call logger.enable here, simulating a default user.
        with pytest.warns(GridAlignmentWarning):
            _make_slice((0.0, 0.0, 105.0, 100.0), (10.0, 10.0), align="warn")

    def test_unknown_mode_rejected(self) -> None:
        # A misspelled mode like "warning" must NOT silently disable
        # validation. Literal[...] is not enforced at runtime.
        with pytest.raises(ValueError, match=r"align must be one of"):
            _make_slice(align="warning")  # type: ignore[arg-type]

    def test_snap_x_extends_xmax(self) -> None:
        # 105 wide at 10m → ceil(10.5) = 11 → xmax 105 → 110;
        # xmin (the affine x origin) stays at 0.
        with pytest.warns(GridAlignmentWarning):
            sl = _make_slice((0.0, 0.0, 105.0, 100.0), (10.0, 10.0), align="snap")
        assert sl.bounds == (0.0, 0.0, 110.0, 100.0)
        assert sl.aligned_shape() == (10, 11)

    def test_snap_y_extends_ymin_downward(self) -> None:
        """snap-y holds `ymax` fixed (north-up affine origin).

        Regression for the codex bug: previously snap-y mutated
        `ymax`, shifting the affine origin. The corrected behaviour
        holds `ymax` and pushes `ymin` downward to round outward.
        """
        # y-extent 100.5 at 10 → ceil = 11 pixels → height 110 m;
        # ymax=100.5 preserved, ymin pushed from 0 to -9.5.
        with pytest.warns(GridAlignmentWarning):
            sl = _make_slice((0.0, 0.0, 100.0, 100.5), (10.0, 10.0), align="snap")
        assert sl.bounds[3] == 100.5  # ymax preserved (affine origin)
        assert sl.bounds[1] == pytest.approx(-9.5)  # ymin extended down
        # And the snapped slice's affine transform still maps pixel
        # (0,0) to the original (xmin, ymax) corner.
        assert sl.transform.f == 100.5
        assert sl.aligned_shape() == (11, 10)

    def test_snap_preserves_affine_origin(self) -> None:
        # The whole point of preserving (xmin, ymax) is that the
        # affine transform's c (x origin) and f (y origin) match the
        # pre-snap slice's nominal origin exactly.
        with pytest.warns(GridAlignmentWarning):
            sl = _make_slice((50.0, 50.0, 155.0, 100.5), (10.0, 10.0), align="snap")
        assert sl.transform.c == 50.0  # xmin
        assert sl.transform.f == 100.5  # ymax


# ---------------------------------------------------------------------------
# align field does not participate in identity
# ---------------------------------------------------------------------------


class TestAlignNotInIdentity:
    def test_equal_across_modes(self) -> None:
        a = _make_slice(align="off")
        b = _make_slice(align="error")
        assert a == b

    def test_hash_equal_across_modes(self) -> None:
        a = _make_slice(align="off")
        b = _make_slice(align="warn")
        assert hash(a) == hash(b)

    def test_works_as_dict_key_after_switching_align(self) -> None:
        a = _make_slice(align="off")
        d = {a: "value"}
        b = _make_slice(align="error")
        assert d[b] == "value"

    def test_align_omitted_from_repr(self) -> None:
        sl = _make_slice(align="error")
        assert "align" not in repr(sl)


# ---------------------------------------------------------------------------
# to_crs interaction
# ---------------------------------------------------------------------------


class TestToCrsDoesNotSelfTrip:
    def test_to_crs_strict_parent_does_not_raise(self) -> None:
        sl = GeoSlice(
            bounds=(-10.0, 40.0, -8.0, 42.0),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                closed="both",
            ),
            resolution=(0.01, 0.01),
            crs="EPSG:4326",
            align="error",
        )
        # Reprojection generically yields non-integer multiples — must
        # not raise on its own output.
        out = sl.to_crs("EPSG:32629")
        # And the reprojected slice carries align="off" so further use
        # is silent too.
        assert out.align == "off"


# ---------------------------------------------------------------------------
# iter_slices emits zero warnings on misaligned catalogs
# ---------------------------------------------------------------------------


class TestIterSlicesQuiet:
    def test_inmemory_iter_slices_silent(self) -> None:
        import geopandas as gpd

        # Arbitrary footprints that don't divide evenly at 30m.
        gdf = gpd.GeoDataFrame(
            {
                "filepath": ["a.tif", "b.tif"],
                "geometry": [
                    box(0.0, 0.0, 12345.6, 9876.5),
                    box(100.0, 100.0, 234.5, 678.9),
                ],
            },
            crs="EPSG:32629",
        )
        gdf.index = pd.IntervalIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
                (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-02")),
            ],
            closed="both",
        )
        cat = InMemoryGeoCatalog(gdf, backend="raster")
        with warnings.catch_warnings():
            # Any GridAlignmentWarning emitted during iteration would
            # be a regression — turn it into an error to assert
            # silence.
            warnings.simplefilter("error", GridAlignmentWarning)
            slices = list(cat.iter_slices(resolution=(30.0, 30.0)))
        assert len(slices) == 2
        # All emitted slices must carry align="off".
        assert all(s.align == "off" for s in slices)


# ---------------------------------------------------------------------------
# is_grid_aligned
# ---------------------------------------------------------------------------


class TestIsGridAligned:
    def test_identical_slices_are_aligned(self) -> None:
        a = _make_slice((0.0, 0.0, 100.0, 100.0), (10.0, 10.0))
        b = _make_slice((0.0, 0.0, 100.0, 100.0), (10.0, 10.0))
        assert is_grid_aligned(a, b) is True

    def test_same_lattice_different_extent(self) -> None:
        # Origins differ by an integer multiple of resolution → aligned.
        # Both xmin (0, 30) and ymax (100, 90) congruent mod 10.
        a = _make_slice((0.0, 0.0, 100.0, 100.0), (10.0, 10.0))
        b = _make_slice((30.0, 20.0, 80.0, 90.0), (10.0, 10.0))
        assert is_grid_aligned(a, b) is True

    def test_x_origin_off_by_subpixel_not_aligned(self) -> None:
        a = _make_slice((0.0, 0.0, 100.0, 100.0), (10.0, 10.0))
        b = _make_slice((0.5, 0.0, 100.5, 100.0), (10.0, 10.0))
        assert is_grid_aligned(a, b) is False

    def test_y_origin_uses_ymax_not_ymin(self) -> None:
        """The north-up affine origin is `ymax`; lattice check uses it.

        Regression for the codex bug: previously this check compared
        `ymin`, so two slices with matching `ymin` but `ymax` off by
        a non-integer multiple of the resolution were wrongly
        reported as aligned.

        Construct two slices that:
        - share `ymin = 0` (so the old buggy check thinks they align),
        - have `ymax` differing by 5 m (half a pixel at 10 m) so the
          true affine origins differ by a subpixel residual.
        """
        a = _make_slice((0.0, 0.0, 100.0, 100.0), (10.0, 10.0))
        b = _make_slice((0.0, 0.0, 100.0, 105.0), (10.0, 10.0))
        # ymax differs by 5 → subpixel residual at 10m resolution.
        assert is_grid_aligned(a, b) is False
        report = is_grid_aligned(a, b, explain=True)
        assert isinstance(report, dict)
        assert abs(abs(report["y_origin_residual"]) - 5.0) < 1e-9

    def test_resolution_mismatch_not_aligned(self) -> None:
        a = _make_slice((0.0, 0.0, 100.0, 100.0), (10.0, 10.0))
        b = _make_slice((0.0, 0.0, 100.0, 100.0), (5.0, 5.0))
        assert is_grid_aligned(a, b) is False

    def test_crs_mismatch_returns_false(self) -> None:
        a = _make_slice(crs="EPSG:32629")
        b = _make_slice(crs="EPSG:32630")
        assert is_grid_aligned(a, b) is False

    def test_explain_returns_diagnostic_dict(self) -> None:
        a = _make_slice((0.0, 0.0, 100.0, 100.0), (10.0, 10.0))
        b = _make_slice((0.5, 0.0, 100.5, 100.0), (10.0, 10.0))
        report = is_grid_aligned(a, b, explain=True)
        assert isinstance(report, dict)
        assert report["aligned"] is False
        assert report["x_res_match"] is True
        assert report["y_res_match"] is True
        # x origin off by a 0.5 subpixel residual (sign depends on
        # which side of the lattice the offset falls on).
        assert abs(abs(report["x_origin_residual"]) - 0.5) < 1e-9
        assert abs(report["y_origin_residual"]) < 1e-9

    def test_explain_flags_crs_mismatch(self) -> None:
        a = _make_slice(crs="EPSG:32629")
        b = _make_slice(crs="EPSG:32630")
        report = is_grid_aligned(a, b, explain=True)
        assert report["crs_match"] is False
        assert report["aligned"] is False


# ---------------------------------------------------------------------------
# aligned_shape
# ---------------------------------------------------------------------------


class TestAlignedShape:
    def test_passes_when_aligned(self) -> None:
        sl = _make_slice((0.0, 0.0, 100.0, 80.0), (10.0, 10.0), align="off")
        # (height, width) = (y/ry, x/rx)
        assert sl.aligned_shape() == (8, 10)

    def test_raises_regardless_of_align_off(self) -> None:
        sl = _make_slice((0.0, 0.0, 105.0, 100.0), (10.0, 10.0), align="off")
        with pytest.raises(ValueError, match="x-extent"):
            sl.aligned_shape()

    def test_matches_round_shape_when_aligned(self) -> None:
        sl = _make_slice((0.0, 0.0, 100.0, 80.0), (10.0, 10.0))
        assert sl.aligned_shape() == sl.shape


# ---------------------------------------------------------------------------
# Hybrid layout: new symbols available via geocatalog.types
# ---------------------------------------------------------------------------


class TestHybridLayoutExports:
    def test_types_subnamespace_reexports(self) -> None:
        import geocatalog
        from geocatalog import types as types_ns

        assert types_ns.divide_evenly is geocatalog.divide_evenly
        assert types_ns.is_grid_aligned is geocatalog.is_grid_aligned
        assert types_ns.Align is geocatalog.Align
        assert types_ns.GridAlignmentWarning is geocatalog.GridAlignmentWarning
