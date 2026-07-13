"""Tests for the hybrid namespace layout.

Three import paths must resolve to the same underlying symbol:

- Flat:                ``from geocatalog import X``
- Types sub-namespace: ``from geocatalog.types import GeoSlice``
- Catalog sub-namespace: ``from geocatalog.catalog import InMemoryGeoCatalog``
"""

from __future__ import annotations

import importlib

import pytest

import geocatalog
from geocatalog import catalog as catalog_ns, types as types_ns


class TestTypesSubNamespace:
    def test_geoslice_identity(self) -> None:
        assert types_ns.GeoSlice is geocatalog.GeoSlice

    def test_slice_helpers_identity(self) -> None:
        assert types_ns.slice_to_window is geocatalog.slice_to_window
        assert types_ns.window_to_slice is geocatalog.window_to_slice

    def test_pixel_precision_identity(self) -> None:
        assert types_ns.PIXEL_PRECISION is geocatalog.PIXEL_PRECISION

    def test_all_lists_named_symbols(self) -> None:
        assert set(types_ns.__all__) == {
            "Align",
            "GeoSlice",
            "GridAlignmentWarning",
            "PIXEL_PRECISION",
            "divide_evenly",
            "is_grid_aligned",
            "slice_to_window",
            "window_to_slice",
        }


class TestCatalogSubNamespace:
    @pytest.mark.parametrize(
        "name",
        [
            "CatalogDomain",
            "CatalogRow",
            "GeoCatalog",
            "InMemoryGeoCatalog",
            "build_raster_catalog",
            "from_geoparquet",
            "intersect",
            "load_raster",
            "load_raster_timeseries",
            "open_catalog",
            "query",
            "to_geoparquet",
            "union",
        ],
    )
    def test_eager_symbol_identity(self, name: str) -> None:
        assert getattr(catalog_ns, name) is getattr(geocatalog, name)

    def test_lazy_attr_defers_to_top_level(self) -> None:
        # DuckDBGeoCatalog is extras-gated; resolving it via the
        # sub-namespace __getattr__ must route through the top-level
        # lazy loader and return the same object.
        try:
            top = geocatalog.DuckDBGeoCatalog
        except ImportError:
            pytest.skip("requires the [duckdb] extra")
        assert catalog_ns.DuckDBGeoCatalog is top

    def test_lazy_attr_unknown_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = catalog_ns.does_not_exist  # type: ignore[attr-defined]

    def test_all_excludes_geoslice(self) -> None:
        # GeoSlice lives in geocatalog.types, not geocatalog.catalog.
        assert "GeoSlice" not in catalog_ns.__all__


class TestFlatNamespace:
    def test_geoslice_resolves_at_top_level(self) -> None:
        assert geocatalog.GeoSlice is types_ns.GeoSlice

    def test_unknown_attr_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            _ = geocatalog.does_not_exist  # type: ignore[attr-defined]


class TestSubmoduleImportable:
    def test_types_module_loads(self) -> None:
        mod = importlib.import_module("geocatalog.types")
        assert mod is types_ns

    def test_catalog_module_loads(self) -> None:
        mod = importlib.import_module("geocatalog.catalog")
        assert mod is catalog_ns
