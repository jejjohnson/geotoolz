"""Remote reads, retry classification, vector pushdown, async failures (#220)."""

from __future__ import annotations

import http.server
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from rasterio.errors import RasterioIOError
from rasterio.transform import from_bounds

from geocatalog import GeoSlice
from geocatalog._src.utils import paths as catalog_io
from geocatalog._src.utils.retry import _is_transient, retry_transient_io
from geocatalog.build import build_raster_catalog, build_vector_catalog
from geocatalog.load import load_vector


class TestGdalVsiPaths:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("s3://bucket/a/b.tif", "/vsis3/bucket/a/b.tif"),
            ("gs://bucket/b.tif", "/vsigs/bucket/b.tif"),
            ("gcs://bucket/b.tif", "/vsigs/bucket/b.tif"),
            ("az://container/b.tif", "/vsiaz/container/b.tif"),
            ("https://host/x/b.tif", "/vsicurl/https://host/x/b.tif"),
            ("hf://datasets/o/r/b.tif", None),
        ],
    )
    def test_mapping(self, uri: str, expected: str | None) -> None:
        assert catalog_io._gdal_vsi_path(uri) == expected

    def test_prefer_gdal_skips_fsspec(self) -> None:
        assert (
            catalog_io._resolve_uri("s3://bucket/b.tif", prefer_gdal=True)
            == "/vsis3/bucket/b.tif"
        )

    def test_storage_options_keep_the_pool_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        geocloud_fs = pytest.importorskip("geocloud.fs")
        calls: list[dict[str, Any] | None] = []

        class _FS:
            def open(self, uri: str, mode: str) -> str:
                return f"handle:{uri}"

        def fake_filesystem(storage_options: dict[str, Any] | None = None) -> _FS:
            calls.append(storage_options)
            return _FS()

        monkeypatch.setattr(geocloud_fs, "filesystem", fake_filesystem)
        out = catalog_io._resolve_uri(
            "s3://bucket/b.tif",
            storage_options={"skip_signature": True},
            prefer_gdal=True,
        )
        assert out == "handle:s3://bucket/b.tif"
        assert calls == [{"skip_signature": True}]


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with HTTP Range support; counts body bytes sent."""

    sent = 0

    def log_message(self, *args: object) -> None:  # silence
        pass

    def do_GET(self) -> None:
        path = self.translate_path(self.path)
        size = os.path.getsize(path)
        start, end = 0, size - 1
        header = self.headers.get("Range")
        if header and header.startswith("bytes="):
            lo, _, hi = header[6:].partition("-")
            start = int(lo) if lo else size - int(hi)
            end = min(int(hi), size - 1) if (hi and lo) else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        type(self).sent += len(body)
        self.wfile.write(body)


@pytest.fixture
def http_raster(tmp_path: Path) -> Iterator[tuple[str, int]]:
    path = tmp_path / "S2_20240601_big.tif"
    size = 2048
    rng = np.random.default_rng(0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint16",
        crs="EPSG:32629",
        transform=from_bounds(0, 0, size * 10, size * 10, size, size),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(rng.integers(0, 60000, (1, size, size), dtype=np.uint16))
    handler = type("H", (_RangeHandler,), {"sent": 0})
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), lambda *a: handler(*a, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{path.name}", handler
    finally:
        server.shutdown()


def test_remote_raster_metadata_uses_ranged_reads(
    http_raster: tuple[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, handler = http_raster
    monkeypatch.setenv("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    cat = build_raster_catalog([url], filename_regex=r"S2_(?P<date>\d{8})_big\.tif")
    assert len(cat) == 1
    file_size = 2048 * 2048 * 2
    # A header read must not pull the ~8 MB file.
    assert handler.sent < file_size / 10, handler.sent


class TestRetryClassification:
    def test_missing_local_file_is_not_transient(self, tmp_path: Path) -> None:
        with pytest.raises(RasterioIOError) as info:
            rasterio.open(tmp_path / "nope.tif")
        assert not _is_transient(info.value)

    def test_missing_local_file_fails_fast(self, tmp_path: Path) -> None:
        start = time.monotonic()
        with pytest.raises(RasterioIOError):
            retry_transient_io(rasterio.open, tmp_path / "nope.tif", retries=3)
        assert time.monotonic() - start < 1.0

    def test_generic_rasterio_io_error_still_transient(self) -> None:
        assert _is_transient(RasterioIOError("HTTP response code: 503"))

    def test_obstore_network_errors_are_transient(self) -> None:
        # Named like obstore's, without importing obstore (base install).
        generic = type(
            "GenericError", (Exception,), {"__module__": "obstore.exceptions"}
        )
        assert _is_transient(generic("Error performing GET: timed out"))
        other = type("GenericError", (Exception,), {"__module__": "elsewhere"})
        assert not _is_transient(other("not obstore"))
        assert not _is_transient(FileNotFoundError("object not found"))

    def test_async_build_reraises_original_error(self, tmp_path: Path) -> None:
        with pytest.raises(RasterioIOError):
            build_raster_catalog([tmp_path / "missing.tif"], concurrency="async")


def _vector_file(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {"cls": [1, 2]},
        geometry=[shapely.box(0, 0, 10, 10), shapely.box(1000, 1000, 1010, 1010)],
        crs="EPSG:32629",
    )
    path = tmp_path / "labels.gpkg"
    gdf.to_file(path)
    return path


def test_vector_builder_reads_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _vector_file(tmp_path)

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("builder read every feature")

    monkeypatch.setattr(gpd, "read_file", boom)
    cat = build_vector_catalog([path])
    assert cat.gdf.geometry.iloc[0].bounds == (0.0, 0.0, 1010.0, 1010.0)


def test_vector_builder_reprojects_bounds(tmp_path: Path) -> None:
    cat = build_vector_catalog([_vector_file(tmp_path)], crs="EPSG:4326")
    xmin, ymin, xmax, ymax = cat.gdf.geometry.iloc[0].bounds
    assert -13.5 < xmin < xmax < -13.4 and 0 <= ymin < ymax < 0.01


def test_load_vector_pushes_bbox_to_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _vector_file(tmp_path)
    cat = build_vector_catalog([path])
    seen: list[object] = []
    real = gpd.read_file

    def spy(*a: object, **k: object) -> gpd.GeoDataFrame:
        seen.append(k.get("bbox"))
        return real(*a, **k)

    monkeypatch.setattr(gpd, "read_file", spy)
    slc = GeoSlice(
        (0.0, 0.0, 20.0, 20.0),
        pd.Interval(
            pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01"), closed="both"
        ),
        (1.0, 1.0),
        "EPSG:32629",
    )
    out = load_vector(cat, slc, label_field="cls")
    assert seen and seen[0] is not None
    assert int(np.asarray(out).max()) == 1
