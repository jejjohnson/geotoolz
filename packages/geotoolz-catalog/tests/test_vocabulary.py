"""One meaning per parameter name, with deprecation shims (#246)."""

from __future__ import annotations

import asyncio
import inspect
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyproj
import pytest
import shapely

import geocatalog
from geocatalog import open_catalog
from geocatalog._src._deprecation import deprecated_alias, renamed_kwargs
from geocatalog.backends import (
    CatalogClosedError,
    CatalogMetadataError,
    CatalogSchemaError,
    GeoCatalogError,
    InMemoryGeoCatalog,
)
from geocatalog.storage import CatalogBundle, from_geoparquet, to_geoparquet


SRC = Path(__file__).resolve().parents[1] / "src" / "geocatalog"


def _catalog(kind: str = "raster") -> InMemoryGeoCatalog:
    gdf = gpd.GeoDataFrame(
        {"filepath": ["a.tif", "b.tif"]},
        geometry=[shapely.box(0, 0, 10, 10), shapely.box(5, 5, 20, 20)],
        crs="EPSG:32629",
    )
    gdf.index = pd.IntervalIndex.from_arrays(
        pd.to_datetime(["2024-01-01", "2024-01-02"]),
        pd.to_datetime(["2024-01-01", "2024-01-02"]),
        closed="both",
        name="datetime",
    )
    return InMemoryGeoCatalog(gdf, kind=kind)  # type: ignore[arg-type]


@pytest.fixture
def parquet(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.parquet"
    to_geoparquet(_catalog(), path)
    return path


# ---------------------------------------------------------------------------
# The shim machinery
# ---------------------------------------------------------------------------


def test_renamed_kwargs_maps_warns_and_rejects_both() -> None:
    @renamed_kwargs(old="new")
    def f(*, new: int) -> int:
        return new

    assert f(new=1) == 1
    with pytest.warns(
        DeprecationWarning, match=r"f\(\): `old` is deprecated, use `new`"
    ):
        assert f(old=2) == 2  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="got both `old`"):
        f(old=1, new=2)  # type: ignore[call-arg]


def test_renamed_kwargs_keeps_coroutine_functions_async() -> None:
    @renamed_kwargs(old="new")
    async def f(*, new: int) -> int:
        return new

    assert inspect.iscoroutinefunction(f)
    with pytest.warns(DeprecationWarning):
        assert asyncio.run(f(old=3)) == 3  # type: ignore[call-arg]


def test_deprecated_alias_reads_and_writes_the_new_attribute() -> None:
    class C:
        old = deprecated_alias("new")

        def __init__(self) -> None:
            self.new = 1

    c = C()
    with pytest.warns(DeprecationWarning, match="C: `old` is deprecated, use `new`"):
        assert c.old == 1
    with pytest.warns(DeprecationWarning):
        c.old = 2
    assert c.new == 2


def test_the_warning_points_at_the_caller() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        InMemoryGeoCatalog(_catalog().gdf, backend="raster")  # type: ignore[call-arg]
    assert [Path(w.filename).name for w in caught] == [Path(__file__).name]


# ---------------------------------------------------------------------------
# Every renamed keyword / attribute still works, with a warning
# ---------------------------------------------------------------------------


def test_kind_shims(parquet: Path) -> None:
    with pytest.warns(DeprecationWarning, match="`backend` is deprecated, use `kind`"):
        mem = InMemoryGeoCatalog(_catalog().gdf, backend="vector")  # type: ignore[call-arg]
    assert mem.kind == "vector"
    with pytest.warns(DeprecationWarning, match="`backend`"):
        assert mem.backend == "vector"  # type: ignore[attr-defined]
    with pytest.warns(DeprecationWarning, match="`backend`"):
        assert from_geoparquet(parquet, backend="xarray").kind == "xarray"  # type: ignore[call-arg]
    with pytest.warns(DeprecationWarning, match="`path`"):
        assert len(from_geoparquet(path=parquet)) == 2  # type: ignore[call-arg]
    with pytest.warns(DeprecationWarning, match="`backend`"):
        assert open_catalog(parquet, backend="vector", engine="memory").kind == "vector"  # type: ignore[call-arg]
    assert mem.get_config()["kind"] == "vector"


def test_kind_shims_on_duckdb(parquet: Path) -> None:
    pytest.importorskip("duckdb")
    from geocatalog.backends import DuckDBGeoCatalog

    with pytest.warns(DeprecationWarning, match="`backend`"):
        duck = DuckDBGeoCatalog.open(parquet, backend="vector")  # type: ignore[call-arg]
    with duck:
        assert duck.kind == "vector"
        with pytest.warns(DeprecationWarning, match="`backend`"):
            assert duck.backend == "vector"  # type: ignore[attr-defined]
        assert duck.get_config()["kind"] == "vector"


def test_to_geoparquet_path_shim(tmp_path: Path) -> None:
    out = tmp_path / "x.parquet"
    with pytest.warns(DeprecationWarning, match="`path` is deprecated, use `out_path`"):
        to_geoparquet(_catalog(), path=out)  # type: ignore[call-arg]
    assert out.exists()


def test_join_shim() -> None:
    cat = _catalog()
    with pytest.warns(DeprecationWarning, match="`engine` is deprecated, use `join`"):
        out = cat.intersect(cat, engine="sjoin")  # type: ignore[call-arg]
    assert len(out) == len(cat.intersect(cat, join="sjoin"))


def test_bundle_shims() -> None:
    with pytest.warns(DeprecationWarning, match="`target_crs`"):
        bundle = CatalogBundle.empty(target_crs="EPSG:3857", backend="vector")  # type: ignore[call-arg]
    assert bundle.crs.equals(pyproj.CRS("EPSG:3857"))
    assert bundle.kind == "vector"
    with pytest.warns(
        DeprecationWarning, match="`target_crs` is deprecated, use `crs`"
    ):
        assert bundle.target_crs.equals(bundle.crs)  # type: ignore[attr-defined]
    with pytest.warns(DeprecationWarning, match="`backend`"):
        assert bundle.backend == "vector"  # type: ignore[attr-defined]


def test_builder_shims(tmp_path: Path) -> None:
    with (
        pytest.warns(DeprecationWarning, match="`backend` is deprecated, use `engine`"),
        pytest.warns(DeprecationWarning, match="`target_crs` is deprecated, use `crs`"),
        pytest.raises(ValueError, match="engine must be 'memory' or 'duckdb'"),
    ):
        geocatalog.build.build_raster_catalog(
            [], backend="bogus", target_crs="EPSG:4326"
        )  # type: ignore[call-arg]


def test_stac_search_shims() -> None:
    pytest.importorskip("pystac_client")
    seen: dict[str, Any] = {}

    class _Search:
        def items(self) -> list[Any]:
            return []

    class _Client:
        def search(self, **kwargs: Any) -> _Search:
            seen.update(kwargs)
            return _Search()

    with pytest.warns(DeprecationWarning) as record:
        cat = geocatalog.sources.from_stac_search(
            _Client(),
            collections=["c"],
            bbox=(0, 0, 1, 1),  # type: ignore[call-arg]
            max_items=5,  # type: ignore[call-arg]
            target_crs="EPSG:3857",  # type: ignore[call-arg]
            backend="memory",  # type: ignore[call-arg]
        )
    assert len(record) == 4
    assert seen["bbox"] == (0, 0, 1, 1) and seen["max_items"] == 5
    assert pyproj.CRS(cat.crs).equals(pyproj.CRS("EPSG:3857"))


def test_aload_raster_concurrency_shim() -> None:
    from geocatalog._src.raster import aload_raster

    assert "max_open_workers" in inspect.signature(aload_raster).parameters
    assert inspect.iscoroutinefunction(aload_raster)


# ---------------------------------------------------------------------------
# One meaning per name
# ---------------------------------------------------------------------------


def _public_callables() -> list[tuple[str, Callable[..., Any]]]:
    out = []
    for name in geocatalog.__all__:
        obj = getattr(geocatalog, name)
        if inspect.isclass(obj):
            for meth_name, meth in vars(obj).items():
                fn = (
                    meth.__func__
                    if isinstance(meth, (classmethod, staticmethod))
                    else meth
                )
                if inspect.isfunction(fn) and not meth_name.startswith("_"):
                    out.append((f"{name}.{meth_name}", fn))
            out.append((name, obj.__init__))
        elif inspect.isfunction(obj):
            out.append((name, obj))
    return out


# `GeoSlice.to_crs(target_crs)` names the destination of a transform
# (geopandas' `to_crs`), not a catalog CRS.
_NOT_RETIRED = {"GeoSlice.to_crs"}


@pytest.mark.parametrize("retired", ["backend", "target_crs", "max_items", "bbox"])
def test_no_public_signature_uses_a_retired_name(retired: str) -> None:
    offenders = [
        name
        for name, fn in _public_callables()
        if name not in _NOT_RETIRED
        and retired in inspect.signature(inspect.unwrap(fn)).parameters
    ]
    assert offenders == []


def test_backend_keyword_has_one_meaning_in_src() -> None:
    """``backend=`` survives only in shim declarations and on-disk keys."""
    allowed = re.compile(r"renamed_kwargs\(|deprecated_alias|\"backend\"|'backend'")
    offenders = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\bbackend=", line) and not allowed.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {line.strip()}")
    assert offenders == []


# ---------------------------------------------------------------------------
# One error hierarchy, the same on both backends
# ---------------------------------------------------------------------------


def test_error_hierarchy_keeps_the_builtin_bases() -> None:
    assert issubclass(CatalogMetadataError, GeoCatalogError)
    assert issubclass(CatalogMetadataError, ValueError)
    assert issubclass(CatalogSchemaError, (GeoCatalogError, ValueError))
    assert issubclass(CatalogClosedError, (GeoCatalogError, RuntimeError))


def _open(engine: str, parquet: Path) -> Any:
    if engine == "duckdb":
        pytest.importorskip("duckdb")
    return open_catalog(parquet, engine=engine)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("bad_crs", pyproj.exceptions.CRSError),
        ("bad_bounds", ValueError),
        ("bad_kind", ValueError),
    ],
)
def test_both_engines_raise_the_same_type(
    parquet: Path, case: str, expected: type[Exception]
) -> None:
    raised: dict[str, type[BaseException]] = {}
    for engine in ("memory", "duckdb"):
        cat = _open(engine, parquet)
        with pytest.raises(expected) as info:
            if case == "bad_crs":
                cat.query(bounds=(0, 0, 1, 1), crs="EPSG:999999")
            elif case == "bad_bounds":
                cat.query(bounds=(0, 0, 1), crs="EPSG:32629")
            else:
                open_catalog(parquet, engine=engine, kind="bogus")  # type: ignore[arg-type]
        raised[engine] = info.type
    assert raised["memory"] is raised["duckdb"]


def test_bad_engine_is_a_value_error(parquet: Path) -> None:
    with pytest.raises(ValueError, match="engine"):
        open_catalog(parquet, engine="bogus")  # type: ignore[arg-type]


def test_closed_duckdb_catalog_raises_catalog_closed_error(parquet: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    from geocatalog.backends import DuckDBGeoCatalog

    duck = DuckDBGeoCatalog.open(parquet)
    derived = duck.query(bounds=(0, 0, 50, 50), crs="EPSG:32629")
    duck.close()
    for cat in (duck, derived):
        with pytest.raises(CatalogClosedError, match="already been closed"):
            len(cat)
    # Still the exception the backend raised before the hierarchy existed.
    with pytest.raises(duckdb.ConnectionException):
        len(duck)
    with pytest.raises(GeoCatalogError):
        len(duck)


def test_argument_errors_are_builtins_not_catalog_errors() -> None:
    """The documented split: bad arguments are `ValueError`, not `GeoCatalogError`."""
    with pytest.raises(ValueError, match="kind must be one of") as info:
        InMemoryGeoCatalog(_catalog().gdf, kind="bogus")  # type: ignore[arg-type]
    assert not isinstance(info.value, GeoCatalogError)


class _OldStyleCatalog:
    """A third-party catalog written against the pre-#246 protocol."""

    def __init__(self, inner: InMemoryGeoCatalog) -> None:
        self._inner = inner
        self.backend = inner.kind
        self.crs = inner.crs

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        return self._inner.gdf

    @property
    def total_bounds(self) -> Any:
        return self._inner.total_bounds

    @property
    def temporal_extent(self) -> Any:
        return self._inner.temporal_extent

    def __len__(self) -> int:
        return len(self._inner)

    def query(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.query(*args, **kwargs)

    def intersect(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.intersect(*args, **kwargs)

    def union(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.union(*args, **kwargs)

    def iter_rows(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.iter_rows(*args, **kwargs)

    def iter_slices(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.iter_slices(*args, **kwargs)

    def get_config(self) -> dict[str, Any]:
        return self._inner.get_config()


class _ExplodingCatalog(_OldStyleCatalog):
    @property
    def gdf(self) -> gpd.GeoDataFrame:
        raise AssertionError("isinstance must not evaluate properties")


def test_protocol_still_accepts_a_backend_only_catalog() -> None:
    from geocatalog import GeoCatalog

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(_catalog(), GeoCatalog)  # no warning for `kind`
    with pytest.warns(DeprecationWarning, match="instead of `kind`"):
        assert isinstance(_OldStyleCatalog(_catalog()), GeoCatalog)
    assert not isinstance(object(), GeoCatalog)


def test_backend_only_instance_check_does_not_run_properties() -> None:
    from geocatalog import GeoCatalog

    with pytest.warns(DeprecationWarning):
        assert isinstance(_ExplodingCatalog(_catalog()), GeoCatalog)


def test_field_for_accepts_a_backend_only_catalog_with_an_asset() -> None:
    from geocatalog._src.staging._field_for import _with_asset_paths

    inner = _catalog()
    inner.gdf["assets"] = ['{"B04": "s3://b/a.tif"}', '{"B04": "s3://b/b.tif"}']
    out = _with_asset_paths(_OldStyleCatalog(inner), asset="B04")  # type: ignore[arg-type]
    assert out.kind == "raster"
    assert list(out.gdf["filepath"]) == ["s3://b/a.tif", "s3://b/b.tif"]


def test_duckdb_open_rejects_a_bad_kind_before_connecting(
    parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("duckdb")
    from geocatalog._src import duckdb_backend
    from geocatalog.backends import DuckDBGeoCatalog

    def no_connect() -> None:
        raise AssertionError("connected before validating `kind`")

    monkeypatch.setattr(duckdb_backend.duckdb, "connect", no_connect)
    with pytest.raises(ValueError, match="kind must be one of"):
        DuckDBGeoCatalog.open(parquet, kind="bogus")  # type: ignore[arg-type]


def test_duckdb_open_closes_the_connection_on_a_bad_stored_kind(
    parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duckdb = pytest.importorskip("duckdb")
    from geocatalog._src import duckdb_backend
    from geocatalog.backends import DuckDBGeoCatalog

    opened: list[Any] = []
    real_connect = duckdb.connect

    def recording_connect(*args: Any, **kwargs: Any) -> Any:
        opened.append(real_connect(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(duckdb_backend.duckdb, "connect", recording_connect)
    monkeypatch.setattr(
        duckdb_backend, "_read_backend_tag", lambda *_a, **_k: "corrupt"
    )
    with pytest.raises(ValueError, match="kind must be one of"):
        DuckDBGeoCatalog.open(parquet)
    assert len(opened) == 1
    with pytest.raises(duckdb.ConnectionException):
        opened[0].execute("SELECT 1")


def test_closed_error_pickles(parquet: Path) -> None:
    import pickle

    pytest.importorskip("duckdb")
    from geocatalog.backends import DuckDBGeoCatalog

    duck = DuckDBGeoCatalog.open(parquet)
    duck.close()
    with pytest.raises(CatalogClosedError) as info:
        len(duck)
    restored = pickle.loads(pickle.dumps(info.value))
    assert type(restored) is type(info.value)
    assert str(restored) == str(info.value)
    # A fresh interpreter can resolve the class too.
    import subprocess
    import sys

    code = (
        "import pickle, sys\n"
        "err = pickle.loads(sys.stdin.buffer.read())\n"
        "from geocatalog.backends import CatalogClosedError\n"
        "assert isinstance(err, CatalogClosedError), type(err)\n"
    )
    subprocess.run(
        [sys.executable, "-c", code], input=pickle.dumps(info.value), check=True
    )
