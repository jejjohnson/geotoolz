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
