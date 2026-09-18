"""Local-file adapter: open satellite L2 products already on disk."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from xrreader._src.base import DatasetInfo, DataSource
from xrreader._src.local.catalog import LOCAL_DATASETS
from xrreader._src.local.openers import (
    open_emit_ch4_l2b,
    open_ghgsat_ch4_l2,
    open_tropomi_ch4_l2,
)
from xrreader.types import (
    BBox,
    DepthRange,
    PressureLevels,
    TimeRange,
    Variable,
    subset_bbox,
)


_OPENERS: dict[str, Callable[..., xr.Dataset]] = {
    "tropomi.ch4": open_tropomi_ch4_l2,
    "emit.ch4": open_emit_ch4_l2b,
    "ghgsat.ch4": open_ghgsat_ch4_l2,
}


class LocalL2Source(DataSource):
    """Open satellite L2 product files from the local filesystem.

    ``dataset_id`` selects a product opener from :data:`LOCAL_DATASETS`
    (``"tropomi.ch4"``, ``"emit.ch4"``, ``"ghgsat.ch4"``); the file path
    comes through ``path=`` (or ``paths=`` for multi-granule concatenation
    along ``time``). ``bbox`` is applied with
    :func:`xrreader.subset_bbox` on the 2-D ``lon``/``lat`` coordinates
    (after matching the box to the dataset's longitude convention) and
    ``time`` as a mask on the ``time`` coordinate (or on the per-scanline
    ``scanline_time`` for swaths). Nothing is fetched: :meth:`download`
    raises.

    The openers read NetCDF, which the core install does not ship a
    backend for: ``pip install 'xrtoolz-reader[local]'``.
    """

    source_id = "local"

    # ---- DataSource API ---------------------------------------------------

    def list_datasets(self) -> list[DatasetInfo]:
        return list(LOCAL_DATASETS.values())

    def describe(self, dataset_id: str) -> DatasetInfo:
        try:
            return LOCAL_DATASETS[dataset_id]
        except KeyError:
            raise KeyError(
                f"Unknown local dataset {dataset_id!r}; known: {sorted(LOCAL_DATASETS)}"
            ) from None

    def download(
        self,
        dataset_id: str,
        output: Path,
        *,
        variables: list[str | Variable] | None = None,
        bbox: BBox | None = None,
        time: TimeRange | None = None,
        depth: DepthRange | None = None,
        levels: PressureLevels | None = None,
        **extras: Any,
    ) -> Path:
        """Not supported: the product files are already on disk.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("local files are already on disk")

    def open(
        self,
        dataset_id: str,
        *,
        path: str | Path | None = None,
        paths: Sequence[str | Path] | None = None,
        glt_paths: Sequence[str | Path] | None = None,
        variables: list[str | Variable] | None = None,
        bbox: BBox | None = None,
        time: TimeRange | None = None,
        depth: DepthRange | None = None,
        levels: PressureLevels | None = None,
        qa_min: float | None = None,
        **extras: Any,
    ) -> xr.Dataset:
        """Open one or more product files as a single flat Dataset.

        Args:
            dataset_id: Catalog short name (``"tropomi.ch4"``, ``"emit.ch4"``
                or ``"ghgsat.ch4"``).
            path: The product file. Exactly one of ``path`` / ``paths``.
            paths: Several granules / scenes, concatenated along ``time``
                in the order given (they must share the swath or raster
                shape).
            glt_paths: One EMIT geometric lookup table per entry of
                ``paths`` (same order, same length). A single scene takes
                ``glt_path=`` through ``extras`` instead.
            variables: Registry names or :class:`Variable` instances to
                keep; ``None`` keeps everything the opener produced.
            bbox: Keep only pixels whose 2-D ``lon`` / ``lat`` fall inside
                the box (``where(..., drop=True)`` — scanlines / rows
                that fall entirely outside are dropped). The box is
                normalised to the dataset's longitude convention first
                (``[-180, 180]`` when any ``lon`` is negative, ``[0, 360]``
                when any exceeds 180), so either convention works.
            time: Keep measurements inside the window. Swaths are masked
                on their per-scanline ``scanline_time``; scenes on their
                ``time`` coordinate (a mask, so ``paths`` need not be in
                chronological order).
            depth: Ignored (no vertical axis in these products).
            levels: Ignored (no vertical axis in these products).
            qa_min: Forwarded to the opener when given (TROPOMI:
                ``qa_value < qa_min`` is masked). ``None`` leaves the
                opener's own default in force (``0.5`` for TROPOMI); pass
                ``0.0`` to keep every pixel.
            **extras: Forwarded to the opener (``keep_groups=``,
                ``glt_path=`` for a single EMIT scene ...).

        Returns:
            The screened Dataset.

        Raises:
            KeyError: If ``dataset_id`` is not a local product, or a
                requested variable is absent from the opened file.
            ValueError: If neither or both of ``path`` / ``paths`` are
                given; if ``glt_paths`` does not line up with ``paths`` or
                a single ``glt_path`` is given for several scenes; or if
                ``bbox`` crosses the antimeridian once expressed in the
                dataset's longitude convention.
        """
        opener = _OPENERS.get(dataset_id)
        if opener is None:
            raise KeyError(
                f"Unknown local dataset {dataset_id!r}; known: {sorted(_OPENERS)}"
            )
        if (path is None) == (paths is None):
            raise ValueError("pass exactly one of path= or paths=")
        if qa_min is not None:
            extras["qa_min"] = qa_min
        files = [path] if path is not None else list(paths or ())
        per_file = _per_file_extras(files, glt_paths, extras)
        parts = [
            opener(Path(p), **extras, **kw)
            for p, kw in zip(files, per_file, strict=True)
        ]
        ds = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")
        if variables:
            keep = {v.name for v in self._resolve_variables(variables)}
            if missing := sorted(keep - set(ds.data_vars)):
                raise KeyError(f"{missing} not in dataset; have {sorted(ds.data_vars)}")
            ds = ds.drop_vars([n for n in ds.data_vars if n not in keep])
        if bbox is not None:
            bbox = _match_lon_convention(bbox, ds["lon"])
            if bbox.crosses_antimeridian:
                raise ValueError(
                    "BBox crosses the antimeridian in the dataset's longitude "
                    "convention; split it at the antimeridian and open each half"
                )
            ds = subset_bbox(
                ds, (bbox.lon_min, bbox.lon_max), (bbox.lat_min, bbox.lat_max)
            )
        if time is not None:
            ds = _subset_time_range(ds, time)
        return ds


def _per_file_extras(
    files: Sequence[str | Path],
    glt_paths: Sequence[str | Path] | None,
    extras: dict[str, Any],
) -> list[dict[str, Path]]:
    """Per-file opener kwargs: one ``glt_path`` per scene from ``glt_paths``."""
    if glt_paths is None:
        if len(files) > 1 and "glt_path" in extras:
            raise ValueError(
                "a single glt_path= cannot serve several scenes; pass "
                "glt_paths= (one lookup table per entry of paths=)"
            )
        return [{} for _ in files]
    if "glt_path" in extras:
        raise ValueError("pass either glt_path= or glt_paths=, not both")
    if len(glt_paths) != len(files):
        raise ValueError(
            f"glt_paths has {len(glt_paths)} entries but paths has {len(files)}"
        )
    return [{"glt_path": Path(g)} for g in glt_paths]


def _match_lon_convention(bbox: BBox, lon: xr.DataArray) -> BBox:
    """Express ``bbox`` in the longitude convention ``lon`` uses.

    Any negative longitude marks a ``[-180, 180]`` dataset, any longitude
    above 180 a ``[0, 360]`` one; a coordinate entirely within ``[0, 180]``
    is compatible with both, so the box is left as given.
    """
    if bool((lon < 0).any()):
        return bbox.to_180()
    if bool((lon > 180).any()):
        return bbox.to_360()
    return bbox


def _subset_time_range(ds: xr.Dataset, time: TimeRange) -> xr.Dataset:
    """Apply a :class:`TimeRange` to a swath (per scanline) or scene."""
    start = _naive(time.start)
    end = _naive(time.end)
    if "scanline_time" in ds.coords:
        st = ds["scanline_time"]
        return ds.where((st >= start) & (st <= end), drop=True)
    # A mask, not a label slice: the ``time`` index is not monotonic when
    # ``paths`` are given out of chronological order.
    t = ds["time"]
    return ds.isel(time=((t >= start) & (t <= end)).values)


def _naive(ts: pd.Timestamp) -> np.datetime64:
    """Drop the UTC tz so comparisons against ``datetime64`` coords work."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_datetime64()
