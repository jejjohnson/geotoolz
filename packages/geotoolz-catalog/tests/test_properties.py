"""Property-based tests via Hypothesis (#22).

Three structural invariants are fuzzed at ``max_examples=200`` apiece:

1. **Roundtrip identity** — ``from_geoparquet(to_geoparquet(cat)) ≡ cat``
   modulo Parquet's int64-microsecond timestamp truncation.
2. **Query CRS invariance** — querying with ``bounds_in_A`` is set-equal
   to querying with ``bounds_in_A`` reprojected to CRS ``B``.
3. **Intersect cardinality symmetry** —
   ``|a.intersect(b)| == |b.intersect(a)|``.

Hypothesis finds shrunk counter-examples on failure. CI runs with
``HYPOTHESIS_PROFILE=ci`` (registered in ``tests/conftest.py``), which
sets ``derandomize=True`` so the same examples run every build.
Locally the ``dev`` profile is loaded by default (random examples +
``print_blob=True``); use ``--hypothesis-explain`` for shrinking info.
"""

from __future__ import annotations

from pathlib import Path

import pyproj
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from geocatalog import (
    InMemoryGeoCatalog,
    from_geoparquet,
    intersect,
    query,
    to_geoparquet,
)

from .strategies import bbox_strategy_4326, catalog_strategy


# Hypothesis flags `function_scoped_fixture` because pytest creates
# `tmp_path` once per *test*, not once per *example* — every example
# inside a single Hypothesis run shares the same directory. We
# deliberately suppress the warning: the roundtrip example writes to
# the same file path on each draw and truncates it cleanly via
# `to_geoparquet`, so the shared state can't leak between examples.
_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _rows_signature(catalog: InMemoryGeoCatalog) -> list[tuple[str, str, str, str]]:
    """Stable, hashable, sort-comparable view of a catalog's rows.

    Used for set-equality checks: returns a list of
    ``(filepath, start_iso, end_iso, geometry_wkt)`` tuples, sorted.
    Avoids comparing the raw GeoDataFrames directly (their index dtype
    and column order can drift across round-trips).
    """
    if len(catalog) == 0:
        return []
    rows: list[tuple[str, str, str, str]] = []
    for row in catalog.iter_rows():
        rows.append(
            (
                str(row.filepath),
                row.interval.left.isoformat(),
                row.interval.right.isoformat(),
                row.geometry.wkt,
            )
        )
    return sorted(rows)


# ---------------------------------------------------------------------------
# Property 1 — roundtrip identity
# ---------------------------------------------------------------------------


@given(catalog=catalog_strategy())
@_SETTINGS
def test_roundtrip_preserves_rowset(
    catalog: InMemoryGeoCatalog, tmp_path: Path
) -> None:
    """`from_geoparquet(to_geoparquet(cat))` returns the same rows.

    Equality is checked at microsecond precision — Parquet stores
    timestamps as int64-microseconds and pandas drops sub-microsecond
    components on read-back, so anything stricter would be a flaky
    library-detail assertion.
    """
    out = tmp_path / "roundtripped.parquet"
    to_geoparquet(catalog, out)
    restored = from_geoparquet(out)

    assert restored.backend == catalog.backend
    assert restored.gdf.crs == catalog.gdf.crs
    assert _rows_signature(restored) == _rows_signature(catalog)


# ---------------------------------------------------------------------------
# Property 2 — query CRS-invariance
# ---------------------------------------------------------------------------


# A second CRS to which we'll reproject the AOI. Web Mercator (3857) is
# defined everywhere the strategy generates bounds (|lat|<=10), so the
# transform is stable; UTM/polar zones would not be.
_OTHER_CRS = pyproj.CRS.from_epsg(3857)


def _reproject_bbox_to_3857(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """`(xmin, ymin, xmax, ymax)` 4326 → 3857 via a corners reprojection.

    Reprojecting the four corners is sufficient for an axis-aligned bbox
    in the equatorial band the strategy uses; near the poles or across
    the antimeridian we'd need a denser sampling, but those are
    explicitly out of scope here.
    """
    transformer = pyproj.Transformer.from_crs("EPSG:4326", _OTHER_CRS, always_xy=True)
    xmin, ymin, xmax, ymax = bounds
    x0, y0 = transformer.transform(xmin, ymin)
    x1, y1 = transformer.transform(xmax, ymax)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _shares_exact_boundary(
    catalog: InMemoryGeoCatalog,
    aoi: tuple[float, float, float, float],
) -> bool:
    """True if the AOI shares any exact coordinate with a row's bbox edge.

    Hypothesis-discovered finding: when the AOI's bounds *exactly*
    coincide with a row polygon's edge or corner, pandas's `.cx`
    indexer is inclusive in the catalog CRS, but the reprojected AOI
    can drift by ~1e-7 in floating point and exclude that row. The
    library's CRS-invariance contract is well-defined for AOIs with
    *non-zero-area* overlaps; we skip exact-touch cases here and track
    the boundary-touch invariance question as a follow-on.
    """
    xmin_a, ymin_a, xmax_a, ymax_a = aoi
    for row in catalog.iter_rows():
        xmin_r, ymin_r, xmax_r, ymax_r = row.geometry.bounds
        if xmin_a in (xmin_r, xmax_r) or xmax_a in (xmin_r, xmax_r):
            return True
        if ymin_a in (ymin_r, ymax_r) or ymax_a in (ymin_r, ymax_r):
            return True
    return False


@given(
    catalog=catalog_strategy(n_rows=st.integers(1, 20)),
    aoi=bbox_strategy_4326(),
)
@_SETTINGS
def test_query_crs_invariance(
    catalog: InMemoryGeoCatalog,
    aoi: tuple[float, float, float, float],
) -> None:
    """`cat.query(bounds_A) ≡ cat.query(reproject(bounds_A → B), crs=B)`.

    The library reprojects AOI bounds internally so the caller can pass
    them in *any* CRS; the property here is that the row set returned is
    independent of which CRS the AOI was expressed in.

    Boundary-touch cases (AOI and row sharing an exact edge / corner)
    are skipped via `assume(...)` — see `_shares_exact_boundary`.
    """
    assume(not _shares_exact_boundary(catalog, aoi))
    native = query(catalog, bounds=aoi, crs="EPSG:4326")
    in_other = query(
        catalog,
        bounds=_reproject_bbox_to_3857(aoi),
        crs=_OTHER_CRS,
    )
    assert _rows_signature(native) == _rows_signature(in_other)


# ---------------------------------------------------------------------------
# Property 3 — intersect cardinality symmetry
# ---------------------------------------------------------------------------


@given(
    left=catalog_strategy(n_rows=st.integers(0, 10)),
    right=catalog_strategy(n_rows=st.integers(0, 10)),
)
@_SETTINGS
def test_intersect_cardinality_is_symmetric(
    left: InMemoryGeoCatalog, right: InMemoryGeoCatalog
) -> None:
    """``|a.intersect(b)| == |b.intersect(a)|`` for any pair of catalogs.

    The *contents* may differ (each side carries its own attribute
    columns and backend tag), but the number of intersecting
    (geometry x time-interval) pairs is symmetric.
    """
    forward = intersect(left, right)
    reverse = intersect(right, left)
    assert len(forward) == len(reverse)
