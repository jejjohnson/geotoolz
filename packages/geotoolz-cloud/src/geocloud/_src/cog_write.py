"""Write a Cloud-Optimized GeoTIFF to a local path or any object store.

`write_cog` turns a georeader `GeoTensor` (or a lazy `GeoData` reader) into
a validated COG in two GDAL steps:

1. **Stage** the pixels into a tiled GeoTIFF in a private temporary
   directory: in one write for an in-memory `GeoTensor`, strip by strip
   for a lazy reader, so a scene larger than memory never loads whole.
   Nodata, band descriptions and tags are set here.
2. **Translate** it with GDAL's ``COG`` driver (``rasterio.shutil.copy``),
   which lays out the tiles, builds the overviews and puts the header
   first, then **check** the result really is a COG (``LAYOUT=COG``, the
   expected shape, dtype and overviews).

The finished file only then reaches ``dest``: renamed into place for a
local path (so a failed or interrupted write never leaves a truncated
file), or uploaded through `geocloud.files.upload` for a cloud URI (with
the credentials registered in `geocloud.credentials`).

Every choice is an explicit keyword with a safe default — compression,
predictor, block size, overviews and their resampling, nodata, band
descriptions — instead of a free-form rasterio profile;
``creation_options`` passes anything else straight to the GDAL driver.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from geocloud._src.files import _local_path, exists, upload
from geocloud._src.redact import redact


if TYPE_CHECKING:
    from rasterio.io import DatasetWriter


__all__ = ["write_cog"]

#: GDAL ``COG`` driver compression codecs.
COMPRESSIONS = frozenset(
    {
        "deflate",
        "lzw",
        "zstd",
        "lerc",
        "lerc_deflate",
        "lerc_zstd",
        "webp",
        "jpeg",
        "none",
    }
)
#: Codecs a horizontal / floating-point predictor helps.
_PREDICTED = frozenset({"deflate", "lzw", "zstd"})
#: GDAL overview resampling methods.
RESAMPLINGS = frozenset(
    {
        "nearest",
        "average",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "mode",
        "rms",
        "gauss",
    }
)
# Rows of a lazy reader read per step while staging (a multiple of the
# block size, so each strip fills whole tiles).
_STRIP_BLOCKS = 4

Auto = Literal["auto"]


def _values(data: Any) -> np.ndarray | None:
    """The in-memory pixels of a GeoTensor, or ``None`` for a lazy reader."""
    from georeader.geotensor import GeoTensor

    return np.asarray(data) if isinstance(data, GeoTensor) else None


def _bands_hw(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) == 2:
        return 1, int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise ValueError(
        f"write_cog: expected (H, W) or (C, H, W) data, got shape {tuple(shape)}; "
        "write each frame of a (T, C, H, W) stack to its own COG."
    )


def _output_dtype(dtype: np.dtype) -> np.dtype:
    """The dtype the TIFF holds: GDAL has no bool and no float16."""
    if dtype == np.bool_:
        return np.dtype("uint8")
    if dtype == np.float16:
        return np.dtype("float32")
    return dtype


def _resolve_nodata(
    nodata: float | int | None | Auto, fill: Any, dtype: np.dtype
) -> float | int | None:
    """The nodata to write, checked to be representable in ``dtype``.

    ``"auto"`` takes the GeoTensor's ``fill_value_default`` when the
    output dtype can hold it, else writes none (a NaN fill on integer
    data, say). An explicit value that does not fit raises.
    """
    explicit = not (isinstance(nodata, str) and nodata == "auto")
    value = nodata if explicit else fill
    if value is None:
        return None
    if isinstance(value, bool):
        value = int(value)
    if dtype.kind == "f":
        return float(value)
    representable = (
        isinstance(value, int | np.integer | float | np.floating)
        and math.isfinite(float(value))
        and float(value).is_integer()
        and np.iinfo(dtype).min <= value <= np.iinfo(dtype).max
    )
    if representable:
        return int(value)
    if explicit:
        raise ValueError(
            f"write_cog: nodata={nodata!r} cannot be stored in {dtype} data; "
            "pass a value in the dtype's range, or nodata=None."
        )
    return None


def _resolve_descriptions(
    descriptions: Sequence[str] | None, attrs: Mapping[str, Any], count: int
) -> list[str] | None:
    """Explicit band descriptions, else ``band_names`` / ``descriptions`` attrs."""
    if descriptions is not None:
        names = [str(name) for name in descriptions]
        if len(names) != count:
            raise ValueError(
                f"write_cog: {len(names)} descriptions for {count} band(s)."
            )
        return names
    for key in ("band_names", "descriptions"):
        names = attrs.get(key)
        if (
            isinstance(names, Sequence)
            and not isinstance(names, str)
            and len(names) == count
        ):
            return [str(name) for name in names]
    return None


def _strips(height: int, rows: int) -> Iterator[tuple[int, int]]:
    for top in range(0, height, rows):
        yield top, min(rows, height - top)


def _stage(
    data: Any,
    values: np.ndarray | None,
    path: Path,
    *,
    dtype: np.dtype,
    count: int,
    height: int,
    width: int,
    blocksize: int,
    nodata: float | int | None,
    descriptions: list[str] | None,
    tags: Mapping[str, Any] | None,
) -> None:
    """Write the pixels to a tiled, uncompressed GeoTIFF at ``path``."""
    import rasterio
    from rasterio.windows import Window

    profile = {
        "driver": "GTiff",
        "count": count,
        "height": height,
        "width": width,
        "dtype": dtype.name,
        "crs": data.crs,
        "transform": data.transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        _describe(dst, descriptions, tags)
        if values is not None:
            dst.write(values.reshape(count, height, width).astype(dtype, copy=False))
            return
        for top, rows in _strips(height, blocksize * _STRIP_BLOCKS):
            window = Window(col_off=0, row_off=top, width=width, height=rows)
            chunk = np.asarray(data.read_from_window(window, boundless=False))
            dst.write(
                chunk.reshape(count, rows, width).astype(dtype, copy=False),
                window=window,
            )


def _describe(
    dst: DatasetWriter,
    descriptions: list[str] | None,
    tags: Mapping[str, Any] | None,
) -> None:
    if descriptions is not None:
        for index, name in enumerate(descriptions, start=1):
            dst.set_band_description(index, name)
    if tags:
        dst.update_tags(**{str(k): str(v) for k, v in tags.items()})


def _check(
    path: Path,
    *,
    count: int,
    height: int,
    width: int,
    dtype: np.dtype,
    blocksize: int,
    overviews: bool,
) -> None:
    """Raise unless ``path`` is a COG with the expected layout."""
    import rasterio

    with rasterio.open(path) as src:
        layout = src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
        problems = []
        if layout != "COG":
            problems.append(f"LAYOUT is {layout!r}, not 'COG'")
        if (src.count, src.height, src.width) != (count, height, width):
            problems.append(f"shape {(src.count, src.height, src.width)}")
        if np.dtype(src.dtypes[0]) != dtype:
            problems.append(f"dtype {src.dtypes[0]}")
        if src.block_shapes[0] != (blocksize, blocksize):
            problems.append(f"blocks {src.block_shapes[0]}")
        if overviews and max(height, width) > blocksize and not src.overviews(1):
            problems.append("no overviews")
    if problems:
        raise RuntimeError(
            "write_cog: GDAL wrote a file that is not the expected COG: "
            + "; ".join(problems)
        )


def write_cog(
    data: Any,
    dest: str | os.PathLike[str],
    *,
    compress: str = "deflate",
    level: int | None = None,
    predictor: bool = True,
    blocksize: int = 512,
    overviews: bool = True,
    resampling: str | None = None,
    nodata: float | int | None | Auto = "auto",
    descriptions: Sequence[str] | None = None,
    tags: Mapping[str, Any] | None = None,
    creation_options: Mapping[str, Any] | None = None,
    overwrite: bool = True,
    validate: bool = True,
    storage_options: Mapping[str, Any] | None = None,
) -> str | Path:
    """Write ``data`` as a Cloud-Optimized GeoTIFF at ``dest``.

    The COG is built and checked in a private temporary directory, then
    renamed into place (a local ``dest``) or uploaded (a cloud URI), so
    ``dest`` only ever holds a complete, valid file. See the module
    docstring for the steps.

    Args:
        data: A georeader `GeoTensor` of shape ``(H, W)`` or ``(C, H, W)``,
            or a lazy `GeoData` reader (``transform``, ``crs``, ``shape``,
            ``read_from_window``), which is staged strip by strip without
            loading it whole. It must carry a CRS.
        dest: A local path or a URI `geocloud.files` can upload to
            (``s3://``, ``gs://``, ``az://`` / ``abfs[s]://``, Azure
            ``https://``).
        compress: ``"deflate"`` (default), ``"zstd"``, ``"lzw"``,
            ``"lerc"`` / ``"lerc_deflate"`` / ``"lerc_zstd"``, ``"webp"``,
            ``"jpeg"`` (both 8-bit only) or ``"none"``.
        level: Compression level (DEFLATE 1-12, ZSTD 1-22, …); ``None``
            keeps GDAL's default.
        predictor: Apply the predictor GDAL picks for the dtype
            (horizontal for integers, floating-point for floats) with
            DEFLATE / LZW / ZSTD — usually a large size win.
        blocksize: Tile size in pixels (a power of two; 256 or 512).
        overviews: Build overviews down to a single tile.
        resampling: Overview resampling (``"nearest"``, ``"average"``,
            ``"bilinear"``, ``"cubic"``, ``"cubicspline"``,
            ``"lanczos"``, ``"mode"``, ``"rms"``, ``"gauss"``). ``None``
            picks ``"nearest"`` for integer and boolean data (classes and
            masks keep their values) and ``"average"`` for floats.
        nodata: The nodata value. ``"auto"`` (default) takes the data's
            ``fill_value_default`` when the dtype can hold it (never for a
            boolean mask, written as ``uint8`` 0 / 1); ``None`` writes
            none. An explicit value must fit the dtype.
        descriptions: Band descriptions; ``None`` takes the data's
            ``band_names`` (or ``descriptions``) attribute when it has one
            entry per band.
        tags: Dataset metadata tags (values written as strings).
        creation_options: Extra GDAL ``COG`` driver options, passed as is
            (``{"NUM_THREADS": "4", "STATISTICS": "YES"}``); they win over
            the keywords above.
        overwrite: Replace an existing ``dest`` (default); ``False``
            raises `FileExistsError` instead.
        validate: Check the written file is a COG with the expected
            shape, dtype, tiling and overviews before it reaches ``dest``.
        storage_options: Forwarded to `get_obstore` for a cloud ``dest``.

    Returns:
        ``dest`` (a ``Path`` for a local destination).

    Raises:
        ValueError: The data is not 2-D / 3-D or has no CRS, an option is
            unknown, ``blocksize`` is not a power of two, or ``nodata`` /
            ``descriptions`` do not fit the data.
        FileExistsError: ``overwrite=False`` and ``dest`` exists.
        RuntimeError: ``validate`` found the output is not the expected COG.

    Examples:
        >>> import tempfile, pathlib
        >>> import numpy as np, rasterio
        >>> from affine import Affine
        >>> from georeader.geotensor import GeoTensor
        >>> scene = GeoTensor(
        ...     np.arange(2 * 600 * 600, dtype="float32").reshape(2, 600, 600),
        ...     transform=Affine(10, 0, 500_000, 0, -10, 4_000_000),
        ...     crs="EPSG:32630", fill_value_default=np.nan,
        ...     attrs={"band_names": ["red", "nir"]},
        ... )
        >>> path = write_cog(scene, pathlib.Path(tempfile.mkdtemp()) / "scene.tif",
        ...                  blocksize=256)
        >>> with rasterio.open(path) as src:
        ...     layout = src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"]
        ...     layout, src.descriptions, src.overviews(1)
        ('COG', ('red', 'nir'), [2, 4])
    """
    import rasterio
    import rasterio.shutil

    compress = compress.lower()
    if compress not in COMPRESSIONS:
        raise ValueError(
            f"write_cog: compress={compress!r}; choose one of {sorted(COMPRESSIONS)}."
        )
    if blocksize < 16 or blocksize & (blocksize - 1):
        raise ValueError(
            f"write_cog: blocksize must be a power of two >= 16; got {blocksize}."
        )
    if getattr(data, "crs", None) is None:
        raise ValueError(
            "write_cog: the data has no CRS; a COG without georeferencing is "
            "just a TIFF. Set one on the GeoTensor first."
        )
    values = _values(data)
    count, height, width = _bands_hw(values.shape if values is not None else data.shape)
    source_dtype = values.dtype if values is not None else np.dtype(data.dtype)
    dtype = _output_dtype(np.dtype(source_dtype))
    if resampling is None:
        resampling = "average" if dtype.kind == "f" else "nearest"
    resampling = resampling.lower()
    if resampling not in RESAMPLINGS:
        raise ValueError(
            f"write_cog: resampling={resampling!r}; "
            f"choose one of {sorted(RESAMPLINGS)}."
        )
    fill = getattr(data, "fill_value_default", None)
    if np.dtype(source_dtype) == np.bool_:
        fill = None  # a mask's False is data, not nodata
    written_nodata = _resolve_nodata(nodata, fill, dtype)
    names = _resolve_descriptions(
        descriptions, getattr(data, "attrs", None) or {}, count
    )

    local = _local_path(dest)
    if not overwrite and exists(dest, storage_options=storage_options):
        raise FileExistsError(
            f"write_cog: {redact(str(dest))!r} exists (overwrite=False)."
        )

    options: dict[str, Any] = {
        "COMPRESS": compress.upper(),
        "BLOCKSIZE": blocksize,
        "OVERVIEWS": "AUTO" if overviews else "NONE",
        "RESAMPLING": resampling.upper(),
        "BIGTIFF": "IF_SAFER",
    }
    if level is not None:
        options["LEVEL"] = level
    if predictor and compress in _PREDICTED:
        options["PREDICTOR"] = "YES"
    options.update({str(k).upper(): v for k, v in (creation_options or {}).items()})

    # A temporary directory beside a local `dest` keeps the final rename on
    # one filesystem (atomic); a cloud `dest` stages in the system temp dir.
    if local is not None:
        local = Path(os.path.abspath(local.expanduser()))
        local.parent.mkdir(parents=True, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix=f".{local.name}.", dir=local.parent)
    else:
        workdir = tempfile.mkdtemp(prefix="geocloud-cog-")
    try:
        staged = Path(workdir) / "stage.tif"
        built = Path(workdir) / "cog.tif"
        _stage(
            data,
            values,
            staged,
            dtype=dtype,
            count=count,
            height=height,
            width=width,
            blocksize=blocksize,
            nodata=written_nodata,
            descriptions=names,
            tags=tags,
        )
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
            rasterio.shutil.copy(staged, built, driver="COG", **options)
        if validate:
            _check(
                built,
                count=count,
                height=height,
                width=width,
                dtype=dtype,
                blocksize=blocksize,
                overviews=overviews,
            )
        if local is not None:
            os.replace(built, local)
            return local
        upload(built, dest, storage_options=storage_options)
        return str(dest)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
