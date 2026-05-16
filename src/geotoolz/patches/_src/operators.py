"""Patch extraction, sampling, and stitching operators."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
import rasterio
from affine import Affine
from georeader import slices
from georeader.geotensor import GeoTensor
from pyproj import CRS, Transformer
from rasterio.windows import Window

from geotoolz.core import Operator
from geotoolz.geom import Stitch as GeomStitch


class _GeoDataFrameLike(Protocol):
    """Minimal point GeoDataFrame protocol used by sampling operators."""

    geometry: Any
    crs: Any


PointInput = np.ndarray | _GeoDataFrameLike


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _seed(default: int | None, override: int | None) -> int | None:
    return default if override is None else override


def _validate_size(size: tuple[int, int], name: str = "size") -> None:
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"{name} must contain positive integers.")


def _validate_stride(stride: tuple[int, int]) -> None:
    if stride[0] <= 0 or stride[1] <= 0:
        raise ValueError("stride must contain positive integers.")


def _overlap_from_stride(
    size: tuple[int, int], stride: tuple[int, int]
) -> tuple[int, int]:
    overlap = (size[0] - stride[0], size[1] - stride[1])
    if overlap[0] < 0 or overlap[1] < 0:
        raise ValueError("stride must be less than or equal to size.")
    return overlap


def _windows(
    shape: tuple[int, int],
    size: tuple[int, int],
    stride: tuple[int, int],
    *,
    drop_incomplete: bool,
) -> list[Window]:
    return slices.create_windows(
        shape,
        size,
        overlap=_overlap_from_stride(size, stride),
        include_incomplete=not drop_incomplete,
        trim_incomplete=drop_incomplete,
    )


def _safe_cast_to_dtype(out: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype.type)
        out = np.clip(out, info.min, info.max)
    return out.astype(dtype, copy=False)


def _pad_mode(mode: str, source: np.ndarray) -> str:
    if mode not in {"constant", "reflect", "edge"}:
        raise ValueError(
            f"pad_mode must be one of: constant, reflect, edge; got {mode!r}."
        )
    if mode == "reflect" and min(source.shape[-2:]) < 2:
        warnings.warn(
            "pad_mode='reflect' requires at least two source pixels per "
            "spatial axis; falling back to pad_mode='edge'.",
            stacklevel=2,
        )
        return "edge"
    return mode


def _read_window(
    gt: GeoTensor,
    window: Window,
    *,
    pad_mode: str,
    fill: float,
) -> GeoTensor:
    arr = np.asarray(gt)
    height, width = gt.shape[-2:]
    row0 = int(window.row_off)
    col0 = int(window.col_off)
    row1 = row0 + int(window.height)
    col1 = col0 + int(window.width)
    src_row0 = min(max(row0, 0), height)
    src_col0 = min(max(col0, 0), width)
    src_row1 = min(max(row1, 0), height)
    src_col1 = min(max(col1, 0), width)
    source = arr[..., src_row0:src_row1, src_col0:src_col1]
    pad_width_list = [(0, 0)] * source.ndim
    pad_width_list[-2] = (src_row0 - row0, row1 - src_row1)
    pad_width_list[-1] = (src_col0 - col0, col1 - src_col1)
    pad_width = tuple(pad_width_list)
    if any(before or after for before, after in pad_width):
        mode = _pad_mode(pad_mode, source)
        if mode == "constant":
            source = np.pad(source, pad_width, mode="constant", constant_values=fill)
        elif mode == "edge":
            source = np.pad(source, pad_width, mode="edge")
        else:
            source = np.pad(source, pad_width, mode="reflect")
    transform = gt.transform * Affine.translation(col0, row0)
    return GeoTensor(
        _safe_cast_to_dtype(source, arr.dtype),
        transform=transform,
        crs=gt.crs,
        fill_value_default=fill if pad_mode == "constant" else gt.fill_value_default,
        attrs=gt.attrs,
    )


def _nan_fraction(tile: GeoTensor) -> float:
    arr = np.asarray(tile)
    if not np.issubdtype(arr.dtype, np.floating):
        return 0.0
    return float(np.isnan(arr).mean())


def _labels_2d(labels: GeoTensor) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim == 2:
        return arr
    return arr.reshape(-1, *arr.shape[-2:])[0]


def _window_for_origin(row: int, col: int, size: tuple[int, int]) -> Window:
    return Window(col_off=col, row_off=row, width=size[1], height=size[0])


def _with_label(tile: GeoTensor, label: int) -> GeoTensor:
    attrs = dict(tile.attrs or {})
    attrs["class_label"] = int(label)
    return GeoTensor(
        np.asarray(tile),
        transform=tile.transform,
        crs=tile.crs,
        fill_value_default=tile.fill_value_default,
        attrs=attrs,
    )


def _candidate_origins(
    labels: GeoTensor, label: int, size: tuple[int, int]
) -> list[tuple[int, int]]:
    label_arr = _labels_2d(labels)
    height, width = label_arr.shape
    patch_h, patch_w = size
    if patch_h > height or patch_w > width:
        raise ValueError(
            f"Patch size {size} exceeds label dimensions {(height, width)}."
        )
    half_h = patch_h // 2
    half_w = patch_w // 2
    max_row = height - patch_h
    max_col = width - patch_w
    origins: set[tuple[int, int]] = set()
    for row, col in np.argwhere(label_arr == label):
        row0 = int(row) - half_h
        col0 = int(col) - half_w
        if 0 <= row0 <= max_row and 0 <= col0 <= max_col:
            origins.add((row0, col0))
    return sorted(origins)


def _sample_origins(
    candidates: Sequence[tuple[int, int]],
    n: int,
    rng: np.random.Generator,
    label: int,
) -> list[tuple[int, int]]:
    if not candidates:
        warnings.warn(f"No candidate patches for class {label}.", stacklevel=2)
        return []
    if len(candidates) < n:
        message = (
            f"Only {len(candidates)} candidate patches for class {label}; "
            f"requested {n}."
        )
        warnings.warn(message, stacklevel=2)
        n = len(candidates)
    indices = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[int(index)] for index in np.atleast_1d(indices)]


def _points_array(points: PointInput, crs: str | None) -> tuple[np.ndarray, str | None]:
    if hasattr(points, "geometry"):
        point_crs = None if getattr(points, "crs", None) is None else str(points.crs)
        coords = np.column_stack(
            [points.geometry.x.to_numpy(), points.geometry.y.to_numpy()]
        )
        return coords.astype(float), point_crs or crs
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {coords.shape}.")
    return coords, crs


def _reproject_points(
    coords: np.ndarray, source_crs: str | None, target_crs: Any
) -> np.ndarray:
    if source_crs is None or CRS.from_user_input(source_crs) == CRS.from_user_input(
        target_crs
    ):
        return coords
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(coords[:, 0], coords[:, 1])
    return np.column_stack([x, y])


def _sample_nearest(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    height, width = arr.shape[-2:]
    row_index = np.floor(rows).astype(int)
    col_index = np.floor(cols).astype(int)
    valid = (
        (row_index >= 0) & (row_index < height) & (col_index >= 0) & (col_index < width)
    )
    all_valid = bool(np.all(valid))
    out_dtype = arr.dtype if all_valid else np.float32
    fill = 0 if all_valid else np.nan
    out = np.full((*arr.shape[:-2], len(rows)), fill, dtype=out_dtype)
    out[..., valid] = arr[..., row_index[valid], col_index[valid]]
    return np.moveaxis(out, -1, 0)


def _sample_bilinear(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    height, width = arr.shape[-2:]
    # Affine inversion yields pixel-corner coordinates; bilinear weights use
    # distances between neighboring pixel centers.
    rows = rows - 0.5
    cols = cols - 0.5
    row0 = np.floor(rows).astype(int)
    col0 = np.floor(cols).astype(int)
    row1 = row0 + 1
    col1 = col0 + 1
    valid = (row0 >= 0) & (col0 >= 0) & (row1 < height) & (col1 < width)
    out = np.full((*arr.shape[:-2], len(rows)), np.nan, dtype=np.float32)
    wy = rows - row0
    wx = cols - col0
    valid_index = np.nonzero(valid)[0]
    if valid_index.size:
        wx_valid = wx[valid_index]
        wy_valid = wy[valid_index]
        row0_valid = row0[valid_index]
        row1_valid = row1[valid_index]
        col0_valid = col0[valid_index]
        col1_valid = col1[valid_index]
        top = (1.0 - wx_valid) * arr[..., row0_valid, col0_valid] + wx_valid * arr[
            ..., row0_valid, col1_valid
        ]
        bottom = (1.0 - wx_valid) * arr[..., row1_valid, col0_valid] + wx_valid * arr[
            ..., row1_valid, col1_valid
        ]
        out[..., valid_index] = (1.0 - wy_valid) * top + wy_valid * bottom
    return np.moveaxis(out, -1, 0)


def _track_points(track: PointInput, crs: str | None) -> tuple[np.ndarray, str | None]:
    coords, track_crs = _points_array(track, crs)
    if len(coords) < 2:
        raise ValueError(f"track must contain at least 2 points, got {len(coords)}.")
    return coords, track_crs


def _resample_track(
    coords: np.ndarray, spacing: float | None
) -> tuple[np.ndarray, np.ndarray]:
    if spacing is not None and spacing <= 0:
        raise ValueError("spacing must be positive.")
    deltas = np.diff(coords, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if spacing is None:
        return coords, distance
    total = distance[-1]
    samples = np.arange(0.0, total, spacing)
    if len(samples) == 0 or samples[-1] < total:
        samples = np.append(samples, total)
    x = np.interp(samples, distance, coords[:, 0])
    y = np.interp(samples, distance, coords[:, 1])
    return np.column_stack([x, y]), samples


def _allocate_largest_remainder(
    proportions: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    raw = proportions * n_samples
    counts = np.floor(raw).astype(int)
    remainder = n_samples - int(counts.sum())
    if remainder:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


class ExtractPatches(Operator):
    """Extract fixed-size spatial patches from a `GeoTensor`.

    Uses :func:`georeader.slices.create_windows` for window generation and
    preserves per-patch CRS/transform metadata for later stitching.
    When both ``stride`` and ``overlap`` are supplied, the explicit
    ``stride`` takes precedence and ``overlap`` is retained only in the
    serialized config.
    """

    def __init__(
        self,
        *,
        size: tuple[int, int],
        stride: tuple[int, int] | None = None,
        overlap: int = 0,
        pad_mode: str = "reflect",
        fill: float = 0.0,
        nan_cutoff: float = 1.0,
        drop_incomplete: bool = False,
    ) -> None:
        _validate_size(size)
        if stride is None:
            stride = (size[0] - overlap, size[1] - overlap)
        _validate_stride(stride)
        if not 0.0 <= nan_cutoff <= 1.0:
            raise ValueError("nan_cutoff must be in [0, 1].")
        self.size = size
        self.stride = stride
        self.overlap = overlap
        self.pad_mode = pad_mode
        self.fill = fill
        self.nan_cutoff = nan_cutoff
        self.drop_incomplete = drop_incomplete

    def _iter_patches(self, gt: GeoTensor) -> Iterator[GeoTensor]:
        windows = _windows(
            gt.shape[-2:], self.size, self.stride, drop_incomplete=self.drop_incomplete
        )
        for window in windows:
            tile = _read_window(gt, window, pad_mode=self.pad_mode, fill=self.fill)
            if _nan_fraction(tile) <= self.nan_cutoff:
                yield tile

    def _apply(self, gt: GeoTensor) -> list[GeoTensor]:
        return list(self._iter_patches(gt))

    def get_config(self) -> dict[str, Any]:
        return {
            "size": list(self.size),
            "stride": list(self.stride),
            "overlap": self.overlap,
            "pad_mode": self.pad_mode,
            "fill": self.fill,
            "nan_cutoff": self.nan_cutoff,
            "drop_incomplete": self.drop_incomplete,
        }


class SlidingWindow(Operator):
    """Lazy iterator variant of :class:`ExtractPatches`."""

    def __init__(self, *, size: tuple[int, int], stride: tuple[int, int]) -> None:
        _validate_size(size)
        _validate_stride(stride)
        self.size = size
        self.stride = stride

    def _apply(self, gt: GeoTensor) -> Iterator[GeoTensor]:
        windows = _windows(
            gt.shape[-2:],
            self.size,
            self.stride,
            drop_incomplete=False,
        )
        for window in windows:
            yield _read_window(gt, window, pad_mode="reflect", fill=0.0)

    def get_config(self) -> dict[str, Any]:
        return {"size": list(self.size), "stride": list(self.stride)}


class StitchPatches(Operator):
    """Stitch patch predictions onto a target scene grid."""

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        target_shape: tuple[int, int],
        target_transform: Affine,
        target_crs: str,
        blend: str = "average",
        feather_width: int = 16,
    ) -> None:
        self.target_shape = target_shape
        self.target_transform = target_transform
        self.target_crs = target_crs
        self.blend = blend
        self.feather_width = feather_width

    def _apply(self, patches: list[GeoTensor]) -> GeoTensor:
        return GeomStitch(
            blend=self.blend,
            feather_width=self.feather_width,
            target_shape=self.target_shape,
            target_transform=self.target_transform,
            target_crs=self.target_crs,
        )(patches)

    def get_config(self) -> dict[str, Any]:
        return {
            "target_shape": list(self.target_shape),
            "target_transform": list(self.target_transform[:6]),
            "target_crs": self.target_crs,
            "blend": self.blend,
            "feather_width": self.feather_width,
        }


class SamplePoints(Operator):
    """Sample GeoTensor values at point coordinates."""

    def __init__(
        self, *, points: PointInput, crs: str | None = None, interp: str = "nearest"
    ) -> None:
        if interp not in {"nearest", "bilinear"}:
            raise ValueError(f"interp must be nearest or bilinear, got {interp!r}.")
        self.points = points
        self.crs = crs
        self.interp = interp

    def _apply(self, gt: GeoTensor) -> np.ndarray:
        coords, source_crs = _points_array(self.points, self.crs)
        coords = _reproject_points(coords, source_crs, gt.crs)
        cols, rows = (~gt.transform) * (coords[:, 0], coords[:, 1])
        arr = np.asarray(gt)
        if self.interp == "nearest":
            return _sample_nearest(arr, np.asarray(rows), np.asarray(cols))
        return _sample_bilinear(
            arr.astype(np.float32, copy=False), np.asarray(rows), np.asarray(cols)
        )

    def get_config(self) -> dict[str, Any]:
        return {"crs": self.crs, "interp": self.interp}


class SampleAlongTrack(Operator):
    """Sample GeoTensor values along an ordered point track."""

    def __init__(
        self,
        *,
        track: PointInput,
        crs: str | None = None,
        spacing: float | None = None,
        interp: str = "nearest",
    ) -> None:
        self.track = track
        self.crs = crs
        self.spacing = spacing
        self.interp = interp

    def _apply(self, gt: GeoTensor) -> dict[str, np.ndarray]:
        coords, source_crs = _track_points(self.track, self.crs)
        coords = _reproject_points(coords, source_crs, gt.crs)
        points, distance = _resample_track(coords, self.spacing)
        samples = SamplePoints(points=points, crs=str(gt.crs), interp=self.interp)(gt)
        return {"points": points, "distance": distance, "samples": samples}

    def get_config(self) -> dict[str, Any]:
        return {"crs": self.crs, "spacing": self.spacing, "interp": self.interp}


class RandomCrop(Operator):
    """Draw reproducible random fixed-size crops from a `GeoTensor`.

    A per-call ``seed=`` overrides the constructor seed, matching the
    reproducibility semantics used by the augmentation operators.
    """

    def __init__(
        self, *, size: tuple[int, int], n_samples: int = 1, seed: int | None = None
    ) -> None:
        _validate_size(size)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        self.size = size
        self.n_samples = n_samples
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> list[GeoTensor]:
        patch_h, patch_w = self.size
        if patch_h > gt.height or patch_w > gt.width:
            raise ValueError(
                f"Patch size {self.size} exceeds GeoTensor dimensions "
                f"{(gt.height, gt.width)}."
            )
        rng = _rng(_seed(self.seed, seed))
        rows = rng.integers(0, gt.height - patch_h + 1, size=self.n_samples)
        cols = rng.integers(0, gt.width - patch_w + 1, size=self.n_samples)
        return [
            _read_window(
                gt,
                _window_for_origin(int(row), int(col), self.size),
                pad_mode="constant",
                fill=gt.fill_value_default,
            )
            for row, col in zip(rows, cols, strict=True)
        ]

    def get_config(self) -> dict[str, Any]:
        return {"size": list(self.size), "n_samples": self.n_samples, "seed": self.seed}


class StratifiedSample(Operator):
    """Sample patches by class proportions defined on a label GeoTensor.

    A per-call ``seed=`` overrides the constructor seed for reproducible
    resampling without rebuilding the operator.
    """

    def __init__(
        self,
        *,
        labels: GeoTensor,
        target_proportions: dict[int, float],
        n_samples: int,
        size: tuple[int, int],
        seed: int | None = None,
    ) -> None:
        _validate_size(size)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        total = sum(target_proportions.values())
        if total <= 0:
            raise ValueError("target_proportions must sum to a positive value.")
        self.labels = labels
        self.target_proportions = dict(target_proportions)
        self.n_samples = n_samples
        self.size = size
        self.seed = seed

    def _counts(self) -> dict[int, int]:
        labels = list(self.target_proportions)
        proportions = np.asarray(
            [self.target_proportions[label] for label in labels], dtype=float
        )
        proportions = proportions / proportions.sum()
        # Largest-remainder allocation: floor all fractional class counts, then
        # give leftover samples to the classes with the largest fractional parts.
        counts = _allocate_largest_remainder(proportions, self.n_samples)
        return {
            int(label): int(count) for label, count in zip(labels, counts, strict=True)
        }

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> list[GeoTensor]:
        rng = _rng(_seed(self.seed, seed))
        patches: list[GeoTensor] = []
        for label, count in self._counts().items():
            candidates = _candidate_origins(self.labels, label, self.size)
            origins = _sample_origins(candidates, count, rng, label)
            for row, col in origins:
                tile = _read_window(
                    gt,
                    _window_for_origin(row, col, self.size),
                    pad_mode="constant",
                    fill=gt.fill_value_default,
                )
                patches.append(_with_label(tile, label))
        return patches

    def get_config(self) -> dict[str, Any]:
        return {
            "target_proportions": self.target_proportions,
            "n_samples": self.n_samples,
            "size": list(self.size),
            "seed": self.seed,
        }


class BalancedSampler(Operator):
    """Sample up to ``n_per_class`` patches for every label class.

    A per-call ``seed=`` overrides the constructor seed for reproducible
    resampling without rebuilding the operator.
    """

    def __init__(
        self,
        *,
        labels: GeoTensor,
        n_per_class: int,
        size: tuple[int, int],
        seed: int | None = None,
    ) -> None:
        _validate_size(size)
        if n_per_class <= 0:
            raise ValueError("n_per_class must be positive.")
        self.labels = labels
        self.n_per_class = n_per_class
        self.size = size
        self.seed = seed

    def _apply(self, gt: GeoTensor, *, seed: int | None = None) -> list[GeoTensor]:
        rng = _rng(_seed(self.seed, seed))
        patches: list[GeoTensor] = []
        for label in np.unique(_labels_2d(self.labels)):
            candidates = _candidate_origins(self.labels, int(label), self.size)
            origins = _sample_origins(candidates, self.n_per_class, rng, int(label))
            for row, col in origins:
                tile = _read_window(
                    gt,
                    _window_for_origin(row, col, self.size),
                    pad_mode="constant",
                    fill=gt.fill_value_default,
                )
                patches.append(_with_label(tile, int(label)))
        return patches

    def get_config(self) -> dict[str, Any]:
        return {
            "n_per_class": self.n_per_class,
            "size": list(self.size),
            "seed": self.seed,
        }


class TileGrid(Operator):
    """Tile a GeoTensor lazily, optionally writing tiles to GeoTIFF files."""

    def __init__(
        self,
        *,
        size: tuple[int, int],
        stride: tuple[int, int] | None = None,
        out_dir: str | None = None,
    ) -> None:
        _validate_size(size)
        self.size = size
        self.stride = stride or size
        _validate_stride(self.stride)
        self.out_dir = out_dir

    def _apply(self, gt: GeoTensor) -> Iterator[GeoTensor] | list[Path]:
        iterator = ExtractPatches(
            size=self.size, stride=self.stride, drop_incomplete=False
        )._iter_patches(gt)
        if self.out_dir is None:
            return iterator
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, tile in enumerate(iterator):
            path = out_dir / f"tile_{index:05d}.tif"
            self._write_tile(tile, path)
            paths.append(path)
        return paths

    def _write_tile(self, tile: GeoTensor, path: Path) -> None:
        arr = np.asarray(tile)
        # GeoTIFF is band-oriented, so every leading non-spatial dimension is
        # flattened into the raster band axis.
        count = 1 if arr.ndim == 2 else int(np.prod(arr.shape[:-2]))
        data = arr.reshape(count, *arr.shape[-2:])
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=tile.height,
            width=tile.width,
            count=count,
            dtype=data.dtype,
            crs=tile.crs,
            transform=tile.transform,
            nodata=tile.fill_value_default,
        ) as dst:
            dst.write(data)

    def get_config(self) -> dict[str, Any]:
        return {
            "size": list(self.size),
            "stride": list(self.stride),
            "out_dir": self.out_dir,
        }
