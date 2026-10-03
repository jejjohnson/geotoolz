"""Writers never destroy unrelated files or leave partial artifacts (#234)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
import shapely.geometry

from geocatalog import InMemoryGeoCatalog, to_geoparquet
from geocatalog._src import streaming
from geocatalog._src.streaming import StreamingParquetWriter, write_partitioned_rows


def _row(day: int, **extras: Any) -> dict[str, Any]:
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
    return {
        "filepath": f"f{day}.tif",
        "geometry": shapely.geometry.box(day, 0, day + 1, 1),
        "start_time": start,
        "end_time": start + pd.Timedelta(hours=1),
        **extras,
    }


def _catalog(n: int = 3) -> InMemoryGeoCatalog:
    starts = pd.date_range("2024-01-01", periods=n, freq="40D")
    gdf = gpd.GeoDataFrame(
        {"filepath": [f"f{i}.tif" for i in range(n)]},
        geometry=[shapely.geometry.box(i, 0, i + 1, 1) for i in range(n)],
        crs="EPSG:4326",
        index=pd.IntervalIndex.from_arrays(
            starts, starts + pd.Timedelta(hours=1), closed="both"
        ),
    )
    return InMemoryGeoCatalog(gdf, kind="raster")


def _toy_extract(filepath: str | Path) -> dict[str, Any]:
    date = pd.Timestamp(Path(filepath).stem[:10])
    return {
        "filepath": str(filepath),
        "geometry": shapely.geometry.box(date.day, 0, date.day + 1, 1),
        "start_time": date,
        "end_time": date + pd.Timedelta(hours=1),
    }


def _nrows(directory: Path) -> int:
    return sum(
        pq.ParquetFile(f).metadata.num_rows for f in directory.rglob("*.parquet")
    )


def _leftovers(directory: Path) -> list[str]:
    """Temp/staging entries a writer left behind next to its output."""
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.name.startswith(".") or ".partitioned." in p.name
    )


# ---------------------------------------------------------------------------
# Partitioned replace keeps what the writer does not own
# ---------------------------------------------------------------------------


def test_partitioned_to_geoparquet_keeps_unrelated_files(tmp_path: Path) -> None:
    dest = tmp_path / "archive"
    dest.mkdir()
    (dest / "README.md").write_text("keep me")
    (dest / "sidecars").mkdir()
    (dest / "sidecars" / "notes.json").write_text("{}")

    to_geoparquet(_catalog(), dest, partition_by=("year", "month"))

    assert (dest / "README.md").read_text() == "keep me"
    assert (dest / "sidecars" / "notes.json").exists()
    assert _nrows(dest / "year=2024") == 3


def test_partitioned_replace_swaps_out_old_partitions(tmp_path: Path) -> None:
    dest = tmp_path / "archive"
    write_partitioned_rows(
        iter([_row(400)]),
        out_path=dest,
        crs="EPSG:4326",
        kind="raster",
        partition_by=("year",),
    )
    (dest / "stray.parquet").write_bytes(b"")  # a top-level shard is owned too
    (dest / "README.md").write_text("keep me")

    write_partitioned_rows(
        iter([_row(0), _row(1)]),
        out_path=dest,
        crs="EPSG:4326",
        kind="raster",
        partition_by=("year",),
    )

    assert sorted(p.name for p in dest.iterdir()) == ["README.md", "year=2024"]
    assert _leftovers(tmp_path) == []


def test_failed_partition_swap_restores_the_old_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "archive"
    write_partitioned_rows(
        iter([_row(400)]),
        out_path=dest,
        crs="EPSG:4326",
        kind="raster",
        partition_by=("year",),
    )
    before = sorted(p.relative_to(dest) for p in dest.rglob("*"))

    real_replace = os.replace

    def flaky(src: Any, dst: Any) -> None:
        if ".partitioned." in str(src):  # installing a new partition dir
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(streaming.os, "replace", flaky)
    with pytest.raises(OSError, match="disk full"):
        write_partitioned_rows(
            iter([_row(0)]),
            out_path=dest,
            crs="EPSG:4326",
            kind="raster",
            partition_by=("year",),
        )
    monkeypatch.undo()

    assert sorted(p.relative_to(dest) for p in dest.rglob("*")) == before
    assert _nrows(dest / "year=2025") == 1


def test_partitioned_replace_over_a_single_file(tmp_path: Path) -> None:
    dest = tmp_path / "archive"
    dest.write_bytes(b"old single-file artifact")
    write_partitioned_rows(
        iter([_row(0)]),
        out_path=dest,
        crs="EPSG:4326",
        kind="raster",
        partition_by=("year",),
    )
    assert (dest / "year=2024").is_dir()
    assert _leftovers(tmp_path) == []


# ---------------------------------------------------------------------------
# Single-file writers are atomic
# ---------------------------------------------------------------------------


def test_failed_to_geoparquet_keeps_the_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "cat.parquet"
    to_geoparquet(_catalog(2), dest)
    original = dest.read_bytes()

    def crash(self: gpd.GeoDataFrame, path: Any, **kwargs: Any) -> None:
        Path(path).write_bytes(b"PAR1 partial")
        raise OSError("killed mid-write")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", crash)
    with pytest.raises(OSError, match="killed"):
        to_geoparquet(_catalog(3), dest)

    assert dest.read_bytes() == original
    assert _leftovers(tmp_path) == []


def test_streaming_writer_error_leaves_destination_untouched(tmp_path: Path) -> None:
    dest = tmp_path / "cat.parquet"
    dest.write_bytes(b"previous artifact")
    with (
        pytest.raises(RuntimeError, match="extractor"),
        StreamingParquetWriter(dest, crs="EPSG:4326", kind="raster", batch_size=1) as w,
    ):
        w.write_row(_row(0))
        w.write_row(_row(1))
        raise RuntimeError("extractor blew up")

    assert dest.read_bytes() == b"previous artifact"
    assert _leftovers(tmp_path) == []


def test_streaming_writer_failing_final_flush_leaves_no_artifact(
    tmp_path: Path,
) -> None:
    dest = tmp_path / "cat.parquet"
    bad = _row(1)
    bad["geometry"] = None
    w = StreamingParquetWriter(dest, crs="EPSG:4326", kind="raster")
    w.write_row(_row(0))
    w.write_row(bad)
    with pytest.raises(TypeError, match="geometry"):
        w.close()
    assert not dest.exists()
    assert _leftovers(tmp_path) == []


def test_streaming_writer_only_exposes_complete_files(tmp_path: Path) -> None:
    dest = tmp_path / "cat.parquet"
    w = StreamingParquetWriter(dest, crs="EPSG:4326", kind="raster", batch_size=1)
    w.write_row(_row(0))
    w.write_row(_row(1))
    assert not dest.exists()  # rows so far live in a hidden temp file
    assert not list(tmp_path.glob("*.parquet"))
    w.close()
    assert len(gpd.read_parquet(dest)) == 2


# ---------------------------------------------------------------------------
# append_files is idempotent
# ---------------------------------------------------------------------------


def _append(archive: Path, paths: list[Path]) -> Any:
    pytest.importorskip("duckdb")
    from geocatalog import append_files

    return append_files(
        archive,
        paths,
        _toy_extract,
        crs="EPSG:4326",
        kind="raster",
        partition_by=("year", "month"),
    )


def test_append_rerun_does_not_duplicate_rows(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    paths = [tmp_path / "2024-01-01-a.tif", tmp_path / "2024-02-01-b.tif"]
    _append(archive, paths)
    shards = sorted(archive.rglob("*.parquet"))

    catalog = _append(archive, paths)

    assert len(catalog) == 2
    assert sorted(archive.rglob("*.parquet")) == shards


def test_append_after_partial_run_completes_without_duplicates(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    paths = [tmp_path / f"2024-0{m}-01-x.tif" for m in (1, 2, 3)]
    _append(archive, paths[:2])  # an interrupted run that got two in

    catalog = _append(archive, paths)

    fps = sorted(Path(r.filepath).name for r in catalog.iter_rows())
    assert fps == sorted(p.name for p in paths)


def test_append_dedups_repeats_within_one_call(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    path = tmp_path / "2024-01-01-a.tif"
    catalog = _append(archive, [path, path, Path(str(path))])
    assert len(catalog) == 1


def test_append_with_no_rows_and_nothing_indexed_still_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no files yielded a row"):
        _append(tmp_path / "archive", [])
