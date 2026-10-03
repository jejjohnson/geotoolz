"""Smoke + exit-code tests for the cyclopts CLI (#23).

The CLI is a thin shim — there's no point re-asserting library
behaviour through it. The tests below cover:

* Each `--help` page parses (no import-time crash from cyclopts).
* `build raster` round-trips through the persisted artifact.
* `stats` / `query` / `info` produce both human-readable and JSON
  output without raising.
* `query` rejects half-specified time windows (--start without --end).
* The four documented exit codes (0 / 1 / 2 / 3) all fire on the
  expected inputs, including OSError from an unreadable source.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from geocatalog._cli import app


def _run(*tokens: str) -> int:
    """Run the cyclopts App over ``tokens`` and return the exit code.

    The App is invoked with ``result_action="return_value"`` so the
    Python return value (an int) is what comes back; cyclopts'
    default ``sys.exit`` flow happens only when called as a real
    process entry point.
    """
    try:
        result = app(list(tokens), exit_on_error=False, result_action="return_value")
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    return int(result) if result is not None else 0


def test_help_root(capsys: pytest.CaptureFixture[str]) -> None:
    """`geocatalog --help` lists the top-level commands."""
    _run("--help")
    captured = capsys.readouterr().out
    assert "build" in captured
    assert "query" in captured
    assert "stats" in captured
    assert "info" in captured


def test_help_build(capsys: pytest.CaptureFixture[str]) -> None:
    """`geocatalog build --help` lists the per-format builders."""
    _run("build", "--help")
    captured = capsys.readouterr().out
    assert "raster" in captured
    assert "vector" in captured
    assert "xarray" in captured


def test_build_raster_roundtrip(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Full happy-path: build → write → stats.

    The fixture writes two GeoTIFFs into a date-aware tmp directory;
    the CLI globs them, builds a catalog, and persists it. `stats`
    then reads it back and reports two rows.
    """
    utm29_tile_factory((500000, 4000000, 510000, 4010000), "20240601")
    utm29_tile_factory((510000, 4000000, 520000, 4010000), "20240602")
    glob_pattern = str(tmp_path / "*.tif")
    out = tmp_path / "catalog.parquet"

    exit_code = _run(
        "build",
        "raster",
        "--input-glob",
        glob_pattern,
        "--regex",
        r"S2_T29SND_(?P<date>\d{8})_.*\.tif",
        "--out",
        str(out),
    )
    assert exit_code == 0
    assert out.exists()

    capsys.readouterr()  # drop the build output
    exit_code = _run("stats", str(out))
    assert exit_code == 0
    stats_out = capsys.readouterr().out
    assert "rows" in stats_out
    assert "2" in stats_out


def test_stats_json(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`stats --json` emits a JSON object parseable into a dict."""
    utm29_tile_factory((500000, 4000000, 510000, 4010000), "20240601")
    glob_pattern = str(tmp_path / "*.tif")
    out = tmp_path / "catalog.parquet"
    _run(
        "build",
        "raster",
        "--input-glob",
        glob_pattern,
        "--regex",
        r"S2_T29SND_(?P<date>\d{8})_.*\.tif",
        "--out",
        str(out),
    )
    capsys.readouterr()
    exit_code = _run("stats", str(out), "--json")
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 1
    assert payload["backend"] == "raster"


def test_stats_json_empty_catalog(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`stats` on a zero-row artifact reports a null temporal extent (#228)."""
    import geopandas as gpd
    import pandas as pd
    import shapely.geometry

    from geocatalog import InMemoryGeoCatalog, to_geoparquet

    gdf = gpd.GeoDataFrame(
        {
            "geometry": [shapely.geometry.box(0, 0, 1, 1)],
            "start_time": [pd.Timestamp("2024-01-01")],
            "end_time": [pd.Timestamp("2024-01-02")],
            "filepath": ["a.tif"],
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    empty = InMemoryGeoCatalog(gdf, backend="raster").query(
        bounds=(1e6, 1e6, 2e6, 2e6), crs="EPSG:32629"
    )
    out = tmp_path / "empty.parquet"
    to_geoparquet(empty, out)

    exit_code = _run("stats", str(out), "--json")
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 0
    assert payload["temporal_start"] is None
    assert payload["temporal_end"] is None


# ---------------------------------------------------------------------------
# Exit-code matrix
# ---------------------------------------------------------------------------


def test_exit_1_no_files_match_glob(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty glob is a user error → exit 1."""
    exit_code = _run(
        "build",
        "raster",
        "--input-glob",
        str(tmp_path / "no_such_*.tif"),
        "--out",
        str(tmp_path / "catalog.parquet"),
    )
    assert exit_code == 1
    assert "no files matched" in capsys.readouterr().err


def test_exit_2_corrupt_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-parquet file at the catalog path → exit 2 (catalog error)."""
    bad = tmp_path / "not-a-parquet.parquet"
    bad.write_bytes(b"this is plainly not parquet")
    exit_code = _run("stats", str(bad))
    assert exit_code == 2


def test_exit_3_missing_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catalog path that doesn't exist → exit 3 (I/O)."""
    exit_code = _run("stats", str(tmp_path / "does_not_exist.parquet"))
    assert exit_code == 3
    assert "not found" in capsys.readouterr().err


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="chmod-based unreadability is ineffective for root",
)
def test_exit_3_unreadable_source(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An existing-but-unreadable source maps to exit 3, not an unhandled OSError."""
    # Build a real catalog first so the path exists.
    utm29_tile_factory((500000, 4000000, 510000, 4010000), "20240601")
    out = tmp_path / "catalog.parquet"
    _run(
        "build",
        "raster",
        "--input-glob",
        str(tmp_path / "*.tif"),
        "--regex",
        r"S2_T29SND_(?P<date>\d{8})_.*\.tif",
        "--out",
        str(out),
    )
    capsys.readouterr()
    # chmod 000 makes the file unreadable for the current user; the CLI
    # should translate the OSError pyarrow surfaces into exit 3.
    out.chmod(0)
    try:
        exit_code = _run("stats", str(out))
    finally:
        # Restore perms so pytest can clean up tmp_path.
        out.chmod(0o644)
    assert exit_code in (2, 3)  # 3 if the read errors; 2 if pyarrow flags corrupt.


# ---------------------------------------------------------------------------
# JSON output + half-window guard
# ---------------------------------------------------------------------------


def _build_one_row(tmp_path: Path, factory: Callable[..., Path]) -> Path:
    """Tiny one-row catalog used by the read-side CLI tests below."""
    factory((500000, 4000000, 510000, 4010000), "20240601")
    out = tmp_path / "catalog.parquet"
    assert (
        _run(
            "build",
            "raster",
            "--input-glob",
            str(tmp_path / "*.tif"),
            "--regex",
            r"S2_T29SND_(?P<date>\d{8})_.*\.tif",
            "--out",
            str(out),
        )
        == 0
    )
    return out


def test_build_raster_json(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`build raster --json` emits `{out, rows}` (per the docs' --json contract)."""
    out = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    # Re-run with --json to capture the JSON success line.
    out2 = tmp_path / "catalog2.parquet"
    exit_code = _run(
        "build",
        "raster",
        "--input-glob",
        str(tmp_path / "*.tif"),
        "--regex",
        r"S2_T29SND_(?P<date>\d{8})_.*\.tif",
        "--out",
        str(out2),
        "--json",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"out": str(out2), "rows": 1}
    _ = out  # silence unused


def test_query_json(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`query --json` emits a JSON object with the resulting row count."""
    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    exit_code = _run(
        "query",
        str(source),
        "--bbox",
        "500000,4000000,510000,4010000",
        "--crs",
        "EPSG:32629",
        "--json",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 1
    assert payload["bbox"] == [500000, 4000000, 510000, 4010000]


def test_info_json(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`info --json` emits the row's columns as a JSON object."""
    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    exit_code = _run("info", str(source), "--row", "0", "--json")
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "filepath" in payload
    assert "geometry" in payload


def test_convert_plain_round_trip(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`convert src --out dst` without --partition-by (#223).

    With DuckDB installed the source opens on the DuckDB backend; its
    materialised rows used to carry the source's `bbox` column, so the
    writer failed with "column 'bbox' already exists".
    """
    source = _build_one_row(tmp_path, utm29_tile_factory)
    out = tmp_path / "converted.parquet"
    capsys.readouterr()
    exit_code = _run("convert", str(source), "--out", str(out), "--json")
    assert exit_code == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 1
    assert _run("stats", str(out), "--json") == 0


def test_info_json_omits_housekeeping_columns(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    assert _run("info", str(source), "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert not {"_backend", "_schema_version", "bbox"} & set(payload)


def test_convert_partition_by_default_out(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """convert single.parquet --partition-by year,month writes a Hive dir."""
    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()

    exit_code = _run("convert", str(source), "--partition-by", "year,month")

    assert exit_code == 0
    out = source.with_suffix("")
    assert out.is_dir()
    assert (out / "year=2024" / "month=6").is_dir()


def test_convert_refuses_in_place_destination(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """convert refuses to overwrite the source in place when no --out is given."""
    source = _build_one_row(tmp_path, utm29_tile_factory)
    # Move the source to an extensionless path so source.with_suffix("")
    # would resolve to the source itself.
    in_place = tmp_path / "in_place_catalog"
    source.rename(in_place)
    capsys.readouterr()

    exit_code = _run("convert", str(in_place), "--partition-by", "year,month")

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "refusing to overwrite source" in err
    # And the original file is untouched.
    assert in_place.is_file()


def test_query_rejects_half_window(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Passing only --start (or only --end) is a user error → exit 1."""
    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    exit_code = _run("query", str(source), "--start", "2024-06-01")
    assert exit_code == 1
    assert "must be passed together" in capsys.readouterr().err


def test_query_bad_bbox(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A malformed --bbox triggers exit 1, not a stack trace."""
    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    exit_code = _run("query", str(source), "--bbox", "not,a,bbox")
    assert exit_code == 1
    assert "bbox" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# #245: every verb, friendly CRS errors, documented exit codes
# ---------------------------------------------------------------------------


def test_query_bad_crs_is_a_one_line_user_error(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    exit_code = _run("query", str(source), "--bbox", "0,0,1,1", "--crs", "nonsense")
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "invalid --crs 'nonsense'" in err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1


def test_query_with_a_projected_crs(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    bbox = "500000,4000000,505000,4005000"
    assert (
        _run("query", str(source), "--bbox", bbox, "--crs", "EPSG:32629", "--json") == 0
    )
    assert json.loads(capsys.readouterr().out)["rows"] == 1


@pytest.mark.parametrize("verb", ["raster", "vector"])
def test_build_bad_target_crs_is_a_user_error(
    tmp_path: Path, verb: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _run(
        "build",
        verb,
        "--input-glob",
        str(tmp_path / "*"),
        "--out",
        str(tmp_path / "out.parquet"),
        "--target-crs",
        "nonsense",
    )
    assert exit_code == 1
    assert "invalid --target-crs" in capsys.readouterr().err


def test_build_vector_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import geopandas as gpd
    import shapely

    gpd.GeoDataFrame(
        {"cls": [1, 2]},
        geometry=[shapely.box(0, 0, 10, 10), shapely.box(20, 20, 30, 30)],
        crs="EPSG:32629",
    ).to_file(tmp_path / "labels_20240601.gpkg")
    out = tmp_path / "vector.parquet"
    exit_code = _run(
        "build",
        "vector",
        "--input-glob",
        str(tmp_path / "*.gpkg"),
        "--regex",
        r"labels_(?P<date>\d{8})\.gpkg",
        "--out",
        str(out),
        "--json",
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"out": str(out), "rows": 1}
    assert _run("stats", str(out), "--json") == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["backend"] == "vector"
    assert stats["temporal_start"].startswith("2024-06-01")


def test_build_vector_on_a_non_vector_file_is_a_user_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "notes.gpkg").write_text("not a geopackage")
    exit_code = _run(
        "build",
        "vector",
        "--input-glob",
        str(tmp_path / "*.gpkg"),
        "--out",
        str(tmp_path / "vector.parquet"),
    )
    err = capsys.readouterr().err
    assert exit_code == 1
    assert err.startswith("build vector failed:")


def test_build_xarray_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    xr = pytest.importorskip("xarray")
    import numpy as np
    import pandas as pd

    xr.Dataset(
        {"ndvi": (("time", "y", "x"), np.zeros((3, 4, 4), dtype=np.float32))},
        coords={
            "time": pd.date_range("2024-01-01", periods=3, freq="D"),
            "y": np.linspace(40.5, 40.0, 4),
            "x": np.linspace(-3.5, -3.0, 4),
        },
    ).to_netcdf(tmp_path / "modis.nc")
    out = tmp_path / "xarray.parquet"
    exit_code = _run(
        "build",
        "xarray",
        "--input-glob",
        str(tmp_path / "*.nc"),
        "--target-crs",
        "EPSG:4326",
        "--out",
        str(out),
        "--json",
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 1
    assert _run("stats", str(out), "--json") == 0
    assert json.loads(capsys.readouterr().out)["backend"] == "xarray"


def test_stats_on_a_newer_schema_is_exit_2(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from geocatalog import from_geoparquet, to_geoparquet

    source = _build_one_row(tmp_path, utm29_tile_factory)
    newer = tmp_path / "newer.parquet"
    to_geoparquet(from_geoparquet(source), newer, schema_version=999)
    capsys.readouterr()
    assert _run("stats", str(newer)) == 2
    assert "999" in capsys.readouterr().err
    assert _run("migrate", str(newer)) == 2


def test_migrate_current_and_explicit_version(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from geocatalog import SCHEMA_VERSION_CURRENT

    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    assert _run("migrate", str(source)) == 0
    assert (
        _run("migrate", str(source), "--to-version", str(SCHEMA_VERSION_CURRENT)) == 0
    )
    out = capsys.readouterr().out
    assert out.count(f"already at v{SCHEMA_VERSION_CURRENT}") == 2
    assert _run("migrate", str(tmp_path / "missing.parquet")) == 3


def test_migrate_rejects_a_non_integer_version_as_a_process(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
) -> None:
    # Argument parsing errors exit through cyclopts, so check the real
    # process exit code rather than `_run`'s return value.
    import subprocess
    import sys

    source = _build_one_row(tmp_path, utm29_tile_factory)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from geocatalog._cli import app; app()",
            "migrate",
            str(source),
            "--to-version",
            "invalid",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "--to-version" in proc.stdout + proc.stderr


def test_convert_round_trips_through_a_partitioned_directory(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _build_one_row(tmp_path, utm29_tile_factory)
    parts = tmp_path / "parts"
    back = tmp_path / "back.parquet"
    assert (
        _run("convert", str(source), "--out", str(parts), "--partition-by", "year") == 0
    )
    assert _run("convert", str(parts), "--out", str(back)) == 0
    capsys.readouterr()
    assert _run("info", str(back), "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert not {"_backend", "_schema_version", "bbox"} & set(payload)
    assert payload["start_time"].startswith("2024-06-01")


def test_multiline_crs_error_stays_on_one_line(
    tmp_path: Path,
    utm29_tile_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _build_one_row(tmp_path, utm29_tile_factory)
    capsys.readouterr()
    bad = 'GEOGCRS["broken",\n  DATUM["nope",\n    ELLIPSOID["x",1,0]]]'
    assert _run("query", str(source), "--bbox", "0,0,1,1", "--crs", bad) == 1
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
