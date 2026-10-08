"""`StreamingParquetWriter` schema handling (#233).

The extras schema is inferred over the whole first batch (or given via
``schema=``), never sealed from the first row; values that pyarrow can
represent — numpy scalars, `pandas.Timestamp`, dicts — round-trip; and
the time columns follow the catalog's naive-UTC contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely.geometry

from geocatalog._src.storage.streaming import StreamingParquetWriter
from geocatalog.backends import InMemoryGeoCatalog
from geocatalog.storage import to_geoparquet


def _row(i: int = 0, **extras: Any) -> dict[str, Any]:
    return {
        "filepath": f"f{i}.tif",
        "geometry": shapely.geometry.box(i, 0, i + 1, 1),
        "start_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
        "end_time": pd.Timestamp("2024-01-02") + pd.Timedelta(days=i),
        **extras,
    }


def _write(path: Path, rows: list[dict[str, Any]], **kwargs: Any) -> None:
    with StreamingParquetWriter(path, crs="EPSG:4326", kind="raster", **kwargs) as w:
        for row in rows:
            w.write_row(row)


def test_key_first_seen_in_a_later_row_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "a.parquet"
    _write(path, [_row(0, a=1), _row(1, a=2, b=3.0)])
    gdf = gpd.read_parquet(path)
    assert "b" in gdf.columns
    assert pd.isna(gdf["b"].iloc[0])
    assert gdf["b"].iloc[1] == 3.0


def test_none_in_first_row_takes_the_type_of_later_values(tmp_path: Path) -> None:
    path = tmp_path / "a.parquet"
    _write(path, [_row(0, cloud=None), _row(1, cloud=12.5)])
    assert pq.read_schema(path).field("cloud").type == pa.float64()


def test_numpy_timestamp_and_dict_extras_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "a.parquet"
    acquired = pd.Timestamp("2024-01-01T10:00", tz="UTC")
    _write(
        path,
        [
            _row(
                0,
                n=np.int64(7),
                x=np.float32(1.5),
                acquired=acquired,
                meta={"sensor": "S2A", "orbit": 12},
            )
        ],
    )
    gdf = gpd.read_parquet(path)
    assert gdf["n"].iloc[0] == 7
    assert gdf["x"].iloc[0] == 1.5
    assert gdf["acquired"].iloc[0] == acquired
    assert gdf["meta"].iloc[0] == {"sensor": "S2A", "orbit": 12}


def test_key_first_seen_after_the_schema_is_sealed_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\['b'\].*schema="):
        _write(tmp_path / "a.parquet", [_row(0, a=1), _row(1, a=2, b=3)], batch_size=1)


def test_all_null_first_batch_then_values_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"'cloud'.*all-null in the first batch"):
        _write(
            tmp_path / "a.parquet",
            [_row(0, cloud=None), _row(1, cloud=3.5)],
            batch_size=1,
        )


def test_explicit_schema_covers_keys_absent_from_the_first_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.parquet"
    schema = pa.schema([("a", pa.int64()), ("cloud", pa.float64())])
    _write(
        path,
        [_row(0, a=1), _row(1, a=2, cloud=4.0)],
        batch_size=1,
        schema=schema,
    )
    gdf = gpd.read_parquet(path)
    assert pd.isna(gdf["cloud"].iloc[0])
    assert gdf["cloud"].iloc[1] == 4.0
    assert pq.read_schema(path).field("a").type == pa.int64()


def test_empty_writer_with_schema_keeps_extras_columns(tmp_path: Path) -> None:
    path = tmp_path / "a.parquet"
    _write(path, [], schema=pa.schema([("cloud", pa.float64())]))
    assert pq.read_schema(path).field("cloud").type == pa.float64()


def test_unconvertible_value_names_the_column(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="'mixed'"):
        _write(tmp_path / "a.parquet", [_row(0, mixed=1), _row(1, mixed="x")])


def test_times_are_naive_utc_nanoseconds(tmp_path: Path) -> None:
    path = tmp_path / "a.parquet"
    start = pd.Timestamp("2024-01-01T12:00:00.000000001", tz="Europe/Madrid")
    row = _row(0)
    row["start_time"] = start
    row["end_time"] = start + pd.Timedelta(hours=1)
    _write(path, [row])
    schema = pq.read_schema(path)
    assert schema.field("start_time").type == pa.timestamp("ns")
    got = gpd.read_parquet(path)["start_time"].iloc[0]
    assert got.tzinfo is None
    assert got == start.tz_convert("UTC").tz_localize(None)
    assert got.nanosecond == 1


def test_streaming_and_geopandas_writers_agree_on_times(tmp_path: Path) -> None:
    starts = pd.to_datetime(["2024-01-01T03:00", "2024-01-02T04:00"])
    ends = starts + pd.Timedelta(hours=1)
    gdf = gpd.GeoDataFrame(
        {"filepath": ["a.tif", "b.tif"]},
        geometry=[shapely.geometry.box(0, 0, 1, 1)] * 2,
        crs="EPSG:4326",
        index=pd.IntervalIndex.from_arrays(starts, ends, closed="both"),
    )
    to_geoparquet(InMemoryGeoCatalog(gdf, kind="raster"), tmp_path / "gpd.parquet")
    _write(
        tmp_path / "stream.parquet",
        [
            {
                "filepath": fp,
                "geometry": shapely.geometry.box(0, 0, 1, 1),
                "start_time": s.tz_localize("UTC").tz_convert("Asia/Tokyo"),
                "end_time": e,
            }
            for fp, s, e in zip(["a.tif", "b.tif"], starts, ends, strict=True)
        ],
    )
    a = gpd.read_parquet(tmp_path / "gpd.parquet")
    b = gpd.read_parquet(tmp_path / "stream.parquet")
    assert list(a["start_time"]) == list(b["start_time"])
    assert list(a["end_time"]) == list(b["end_time"])
    assert b["start_time"].dt.tz is None


def test_sort_rewrite_carries_the_input_schema(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    from geocatalog._src.storage.streaming import sort_geoparquet

    src = tmp_path / "src.parquet"
    # The row that sorts first has no `cloud`; re-inferring from the first
    # sorted batch would type the column as null/string and fail on 7.5.
    _write(src, [_row(5, cloud=7.5), _row(0, cloud=None)])
    dst = tmp_path / "dst.parquet"
    sort_geoparquet(
        src,
        dst,
        sort_by=["start_time"],
        crs="EPSG:4326",
        kind="raster",
        batch_size=1,
    )
    assert pq.read_schema(dst).field("cloud").type == pa.float64()
    gdf = gpd.read_parquet(dst)
    assert list(gdf["filepath"]) == ["f0.tif", "f5.tif"]
    assert pd.isna(gdf["cloud"].iloc[0])
    assert gdf["cloud"].iloc[1] == 7.5
