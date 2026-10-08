"""The ABI fixed grid and the per-file metadata shared by L1b and L2 files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from geoproducts.goes._src.hdf import attr, decoded_coordinate, scalar


def _edges(centres: np.ndarray, height_m: float) -> tuple[float, float]:
    """Outer pixel edges (metres) of an evenly spaced scan-angle axis."""
    half = (centres[-1] - centres[0]) / (len(centres) - 1) / 2.0
    return float((centres[0] - half) * height_m), float((centres[-1] + half) * height_m)


def _parse_time(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


@dataclass(frozen=True)
class FixedGrid:
    """A file's ``+proj=geos`` grid: shape, affine transform and CRS.

    ``x`` / ``y`` in an ABI file are pixel-centre scan angles in radians;
    scaled by the perspective-point height they are metres in the
    geostationary projection, evenly spaced, so the grid is affine.
    """

    height: int
    width: int
    transform: Affine
    crs: CRS

    @classmethod
    def from_file(cls, f: Any) -> FixedGrid:
        if "goes_imager_projection" not in f or "x" not in f or "y" not in f:
            raise ValueError(
                "not a GOES-R ABI file: expected the 'x' / 'y' scan angles and "
                "the 'goes_imager_projection' grid mapping."
            )
        proj = f["goes_imager_projection"]
        height_m = float(attr(proj, "perspective_point_height"))
        x = decoded_coordinate(f, "x")
        y = decoded_coordinate(f, "y")
        left, right = _edges(x, height_m)
        top, bottom = _edges(y, height_m)
        n_rows, n_cols = len(y), len(x)
        transform = Affine(
            (right - left) / n_cols, 0.0, left, 0.0, (bottom - top) / n_rows, top
        )
        # ABI sweeps along x (``sweep=x``); proj's geos default is y, so the
        # axis is always spelled out.
        crs = CRS.from_proj4(
            f"+proj=geos +lon_0={float(attr(proj, 'longitude_of_projection_origin'))}"
            f" +h={height_m} +a={float(attr(proj, 'semi_major_axis'))}"
            f" +b={float(attr(proj, 'semi_minor_axis'))}"
            f" +sweep={attr(proj, 'sweep_angle_axis', 'x')} +units=m +no_defs"
        )
        return cls(height=n_rows, width=n_cols, transform=transform, crs=crs)


@dataclass(frozen=True)
class FileInfo:
    """Global metadata every ABI product file carries."""

    platform: str
    scene: str
    dataset_name: str
    title: str
    start_time: datetime | None
    end_time: datetime | None
    satellite_lon_deg: float
    satellite_height_m: float

    @classmethod
    def from_file(cls, f: Any) -> FileInfo:
        sat_height = scalar(f, "nominal_satellite_height")
        if "nominal_satellite_height" in f and attr(
            f["nominal_satellite_height"], "units"
        ) in {"km", "kilometers"}:
            sat_height *= 1000.0
        return cls(
            platform=str(attr(f, "platform_ID", "")),
            scene=str(attr(f, "scene_id", "")),
            dataset_name=str(attr(f, "dataset_name", "")),
            title=str(attr(f, "title", "")),
            start_time=_parse_time(attr(f, "time_coverage_start")),
            end_time=_parse_time(attr(f, "time_coverage_end")),
            satellite_lon_deg=round(scalar(f, "nominal_satellite_subpoint_lon"), 4),
            satellite_height_m=sat_height,
        )
