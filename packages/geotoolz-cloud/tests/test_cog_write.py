"""Tests for `geocloud.cog.write_cog` — validated COGs, local or in a bucket.

No network: cloud destinations are `MemoryStore`s mounted with
`geocloud.store.mount`, read back through rasterio's ``MemoryFile``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import obstore
import pytest
import rasterio
from affine import Affine
from georeader.geotensor import GeoTensor
from obstore.store import MemoryStore
from rasterio.io import MemoryFile
from rasterio.windows import Window

from geocloud._src import cog_write
from geocloud.cog import write_cog
from geocloud.store import clear_obstore_pool, mount, unmount


TRANSFORM = Affine(10, 0, 500_000, 0, -10, 4_000_000)


def scene(
    values: np.ndarray, *, fill: Any = 0, attrs: dict[str, Any] | None = None
) -> GeoTensor:
    return GeoTensor(
        values,
        transform=TRANSFORM,
        crs="EPSG:32630",
        fill_value_default=fill,
        attrs=attrs,
    )


@pytest.fixture
def bucket() -> Iterator[MemoryStore]:
    store = MemoryStore()
    mount("s3://out", store)
    yield store
    unmount("s3://out")
    clear_obstore_pool()


def layout(src: Any) -> str | None:
    return src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")


# --- local writes -----------------------------------------------------------


def test_float_scene_round_trips(tmp_path: Path) -> None:
    values = np.random.default_rng(0).random((2, 600, 520)).astype("float32")
    values[:, :10, :10] = np.nan
    gt = scene(values, fill=np.nan, attrs={"band_names": ["red", "nir"]})
    out = write_cog(gt, tmp_path / "sub" / "scene.tif", blocksize=256, tags={"run": 7})
    assert out == tmp_path / "sub" / "scene.tif"
    with rasterio.open(out) as src:
        assert layout(src) == "COG"
        assert src.count == 2 and src.shape == (600, 520)
        assert src.crs.to_epsg() == 32630 and src.transform == TRANSFORM
        assert np.isnan(src.nodata)
        assert src.descriptions == ("red", "nir")
        assert src.tags()["run"] == "7"
        assert src.block_shapes[0] == (256, 256)
        assert src.overviews(1) == [2, 4]
        structure = src.tags(ns="IMAGE_STRUCTURE")
        assert structure["COMPRESSION"] == "DEFLATE"
        assert structure["PREDICTOR"] == "3"  # floating-point predictor
        assert structure["OVERVIEW_RESAMPLING"] == "AVERAGE"
        np.testing.assert_array_equal(src.read(), values)
    # Only the finished file is left: the staging directory is gone.
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["scene.tif"]


def test_classes_keep_their_values_in_overviews(tmp_path: Path) -> None:
    classes = np.random.default_rng(1).choice([3, 7, 9], size=(800, 800))
    out = write_cog(scene(classes.astype("uint8")), tmp_path / "lc.tif", blocksize=256)
    with rasterio.open(out) as src:
        assert src.tags(ns="IMAGE_STRUCTURE")["OVERVIEW_RESAMPLING"] == "NEAREST"
        coarse = src.read(1, out_shape=(200, 200))  # served from an overview
        assert set(np.unique(coarse)) <= {3, 7, 9}
        assert src.nodata == 0


def test_two_d_bool_mask(tmp_path: Path) -> None:
    mask = np.zeros((300, 300), dtype=bool)
    mask[100:200, 50:80] = True
    out = write_cog(scene(mask, fill=False), tmp_path / "mask.tif", blocksize=128)
    with rasterio.open(out) as src:
        assert src.count == 1 and src.dtypes[0] == "uint8"
        assert src.nodata is None  # False is data in a mask
        np.testing.assert_array_equal(src.read(1), mask.astype("uint8"))


def test_options(tmp_path: Path) -> None:
    values = np.arange(256 * 256, dtype="uint16").reshape(256, 256)
    out = write_cog(
        scene(values),
        tmp_path / "x.tif",
        compress="zstd",
        level=9,
        predictor=False,
        blocksize=128,
        overviews=False,
        nodata=65535,
        descriptions=["dn"],
        creation_options={"statistics": "YES"},
    )
    with rasterio.open(out) as src:
        structure = src.tags(ns="IMAGE_STRUCTURE")
        assert structure["COMPRESSION"] == "ZSTD"
        assert "PREDICTOR" not in structure
        assert src.overviews(1) == []
        assert src.nodata == 65535
        assert src.descriptions == ("dn",)
        assert "STATISTICS_MAXIMUM" in src.tags(1)


def test_lazy_reader_is_staged_in_strips(tmp_path: Path, monkeypatch) -> None:
    values = np.random.default_rng(2).random((3, 700, 300)).astype("float32")
    gt = scene(values, fill=np.nan)

    class Lazy:
        """A GeoData that only hands out windows, never its whole array."""

        transform, crs, fill_value_default = gt.transform, gt.crs, np.nan
        shape, dtype, attrs = values.shape, values.dtype, {}
        windows: ClassVar[list[Window]] = []

        def read_from_window(self, window: Window, boundless: bool = True) -> Any:
            self.windows.append(window)
            rows = slice(window.row_off, window.row_off + window.height)
            return values[:, rows, :]

    monkeypatch.setattr(cog_write, "_STRIP_BLOCKS", 1)
    lazy = Lazy()
    out = write_cog(lazy, tmp_path / "lazy.tif", blocksize=256)
    assert [(w.row_off, w.height) for w in lazy.windows] == [
        (0, 256),
        (256, 256),
        (512, 188),
    ]
    with rasterio.open(out) as src:
        np.testing.assert_array_equal(src.read(), values)


def test_overwrite_false(tmp_path: Path) -> None:
    dest = tmp_path / "x.tif"
    dest.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        write_cog(scene(np.ones((16, 16), "uint8")), dest, overwrite=False)
    assert dest.read_bytes() == b"keep me"


def test_failed_build_leaves_the_old_file(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "x.tif"
    dest.write_bytes(b"previous")

    def broken(*args: Any, **kwargs: Any) -> None:
        raise rasterio.errors.RasterioIOError("disk full")

    monkeypatch.setattr(rasterio.shutil, "copy", broken)
    with pytest.raises(rasterio.errors.RasterioIOError):
        write_cog(scene(np.ones((64, 64), "uint8")), dest)
    assert dest.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["x.tif"]


def test_validation_catches_a_non_cog(tmp_path: Path, monkeypatch) -> None:
    def plain_tiff(src: Path, dst: Path, **kwargs: Any) -> None:
        with rasterio.open(src) as s:
            profile = {**s.profile, "driver": "GTiff", "tiled": False}
            profile.pop("blockxsize"), profile.pop("blockysize")
            with rasterio.open(dst, "w", **profile) as d:
                d.write(s.read())

    monkeypatch.setattr(rasterio.shutil, "copy", plain_tiff)
    with pytest.raises(RuntimeError, match="not the expected COG"):
        write_cog(scene(np.ones((64, 64), "uint8")), tmp_path / "x.tif")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"compress": "jpeg2000"}, "compress="),
        ({"resampling": "fancy"}, "resampling="),
        ({"blocksize": 300}, "power of two"),
        ({"nodata": -1}, "cannot be stored in uint8"),
        ({"nodata": np.nan}, "cannot be stored in uint8"),
        ({"descriptions": ["a", "b"]}, "2 descriptions for 1 band"),
    ],
)
def test_bad_arguments(tmp_path: Path, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        write_cog(scene(np.ones((32, 32), "uint8")), tmp_path / "x.tif", **kwargs)


def test_bad_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\(T, C, H, W\)"):
        write_cog(scene(np.ones((2, 1, 8, 8), "uint8")), tmp_path / "x.tif")
    no_crs = GeoTensor(np.ones((8, 8), "uint8"), transform=TRANSFORM, crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        write_cog(no_crs, tmp_path / "x.tif")


def test_auto_nodata_drops_a_fill_the_dtype_cannot_hold(tmp_path: Path) -> None:
    out = write_cog(scene(np.ones((32, 32), "int16"), fill=np.nan), tmp_path / "x.tif")
    with rasterio.open(out) as src:
        assert src.nodata is None


# --- cloud writes -------------------------------------------------------------


def test_writes_to_a_bucket(bucket: MemoryStore) -> None:
    values = np.random.default_rng(3).random((1, 300, 300)).astype("float32")
    dest = write_cog(
        scene(values, fill=np.nan), "s3://out/products/x.tif", blocksize=128
    )
    assert dest == "s3://out/products/x.tif"
    blob = bytes(obstore.get(bucket, "products/x.tif").bytes())
    with MemoryFile(blob) as mem, mem.open() as src:
        assert layout(src) == "COG"
        np.testing.assert_array_equal(src.read(), values)


def test_bucket_overwrite_false(bucket: MemoryStore) -> None:
    obstore.put(bucket, "x.tif", b"old")
    with pytest.raises(FileExistsError):
        write_cog(scene(np.ones((8, 8), "uint8")), "s3://out/x.tif", overwrite=False)
    assert bytes(obstore.get(bucket, "x.tif").bytes()) == b"old"


def test_cog_source_reads_what_write_cog_wrote(bucket: MemoryStore) -> None:
    pytest.importorskip("async_geotiff")
    from geocloud.cog import CogSource

    values = np.random.default_rng(4).integers(0, 1000, (2, 520, 520)).astype("int16")
    write_cog(scene(values, fill=-1), "s3://out/x.tif", blocksize=256)
    src = CogSource.open("s3://out/x.tif")
    chip = src.read_window(Window(100, 200, 300, 150))  # type: ignore[call-arg]
    np.testing.assert_array_equal(np.asarray(chip), values[:, 200:350, 100:400])
