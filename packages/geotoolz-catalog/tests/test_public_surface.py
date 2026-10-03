"""One public surface: top level = disjoint union of the facades (#248)."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections import Counter

import pytest

import geocatalog


FACADES = ("catalog", "types", "sources", "matchup", "bundle", "staging", "io")
OPTIONAL = (
    "duckdb", "xarray", "rioxarray", "pystac", "pystac_client", "planetary_computer",
    "earthaccess", "ee", "fsspec", "geopatcher", "zarr",
)  # fmt: skip


def _facade(name: str) -> object:
    return importlib.import_module(f"geocatalog.{name}")


@pytest.mark.parametrize("name", ("", *FACADES))
def test_all_is_unique_and_sorted(name: str) -> None:
    module = _facade(name) if name else geocatalog
    exported = list(module.__all__)
    assert len(exported) == len(set(exported))
    # ruff's RUF022 order: SCREAMING_CASE, then CamelCase, then the rest.
    key = [(not n.isupper(), not n[:1].isupper(), n) for n in exported]
    assert key == sorted(key)


def test_top_level_is_the_disjoint_union_of_the_facades() -> None:
    owners = Counter(n for f in FACADES for n in _facade(f).__all__)
    assert {n for n, c in owners.items() if c > 1} == set()
    assert set(owners) == set(geocatalog.__all__)


@pytest.mark.parametrize("name", FACADES)
def test_facade_names_resolve_to_the_top_level_objects(name: str) -> None:
    facade = _facade(name)
    for exported in facade.__all__:
        if exported == "matchup":
            # The top-level name is the callable `geocatalog.matchup` module.
            assert geocatalog.matchup is _facade("matchup")
            continue
        assert getattr(facade, exported) is getattr(geocatalog, exported), exported


def test_helpers_have_public_paths() -> None:
    from geocatalog import StreamingParquetWriter, sort_geoparquet
    from geocatalog.io import parse_uri, retry_transient_io, to_utc_ts

    assert StreamingParquetWriter.__module__ == "geocatalog._src.streaming"
    assert callable(sort_geoparquet) and callable(retry_transient_io)
    assert parse_uri("s3://b/k").is_remote
    assert to_utc_ts("2024-01-01").tzinfo is not None


@pytest.mark.parametrize("name", ("", *FACADES))
def test_unknown_attribute_names_the_facade(name: str) -> None:
    module = _facade(name) if name else geocatalog
    full = "geocatalog" + (f".{name}" if name else "")
    with pytest.raises(AttributeError, match=rf"module '{full}' has no attribute"):
        _ = module.no_such_name  # type: ignore[attr-defined]


def test_star_import_works_with_every_extra_missing() -> None:
    script = (
        "import sys\n"
        f"for m in {OPTIONAL!r}:\n"
        "    sys.modules[m] = None\n"
        "ns = {}\n"
        "exec('from geocatalog import *', ns)\n"
        f"for f in {FACADES!r}:\n"
        "    exec(f'from geocatalog.{f} import *', {})\n"
        "import geocatalog\n"
        "missing = [n for n in geocatalog.__all__ if n not in ns]\n"
        "assert not missing, missing\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_matchup_is_one_callable_namespace_in_any_import_order() -> None:
    script = (
        "import inspect, sys\n"
        "import geocatalog.matchup as ns\n"
        "import geocatalog\n"
        "from geocatalog import matchup\n"
        "from geocatalog._src.matchup import matchup as function\n"
        "assert ns is geocatalog.matchup is matchup\n"
        "assert ns is sys.modules['geocatalog.matchup']\n"
        "assert ns.matchup is function\n"
        "assert ns.MatchupRow is geocatalog.MatchupRow\n"
        "assert 'spatial' in inspect.signature(geocatalog.matchup).parameters\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_calling_the_matchup_namespace_runs_the_engine() -> None:
    import pandas as pd
    import shapely

    from geocatalog import Intersects, SourceRow, Synchronous

    t = pd.Timestamp("2024-06-01", tz="UTC")

    def row(i: str, src: str) -> SourceRow:
        return SourceRow(
            id=i,
            source=src,
            collection="c",
            geometry=shapely.box(0, 0, 1, 1),
            interval=pd.Interval(t, t, closed="both"),
        )

    rows = list(
        geocatalog.matchup(
            [row("p", "a")],
            [row("s", "b")],
            spatial=Intersects(),
            temporal=Synchronous(),
        )
    )
    assert [r.member_ids for r in rows] == [("p", "s")]


def test_matchup_namespace_survives_a_package_reload() -> None:
    code = (
        "import importlib, geocatalog\n"
        "importlib.reload(geocatalog)\n"
        "import geocatalog.matchup as ns\n"
        "assert ns is geocatalog.matchup and callable(ns), type(ns)\n"
        "assert ns.MatchupRow is geocatalog.MatchupRow\n"
        "assert geocatalog.catalog.open_catalog is geocatalog.open_catalog\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_matchup_namespace_pickles_by_reference() -> None:
    import pickle

    restored = pickle.loads(pickle.dumps(geocatalog.matchup))
    assert restored is geocatalog.matchup
    assert callable(restored)
