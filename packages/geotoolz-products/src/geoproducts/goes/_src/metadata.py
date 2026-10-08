"""The global metadata every ABI product file carries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from geoproducts._src.hdf import attr, scalar, time_attr


# ABI scalar variables use -999 as their fill (PUG Vol. 3), not always
# declared as a ``_FillValue``.
SCALAR_FILL = -999.0


def abi_scalar(f: Any, name: str) -> float:
    """An ABI scalar variable as ``float``; ``NaN`` when absent or ``-999``."""
    return scalar(f, name, fill=SCALAR_FILL)


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
        sat_height = abi_scalar(f, "nominal_satellite_height")
        if "nominal_satellite_height" in f and attr(
            f["nominal_satellite_height"], "units"
        ) in {"km", "kilometers"}:
            sat_height *= 1000.0
        return cls(
            platform=str(attr(f, "platform_ID", "")),
            scene=str(attr(f, "scene_id", "")),
            dataset_name=str(attr(f, "dataset_name", "")),
            title=str(attr(f, "title", "")),
            start_time=time_attr(f, "time_coverage_start"),
            end_time=time_attr(f, "time_coverage_end"),
            satellite_lon_deg=round(abi_scalar(f, "nominal_satellite_subpoint_lon"), 4),
            satellite_height_m=sat_height,
        )
