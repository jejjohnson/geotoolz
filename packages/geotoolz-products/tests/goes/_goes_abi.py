"""Synthetic GOES-R ABI L1b files, built with h5py, for the reader tests.

``write_abi`` lays a file out like a NOAA ABI L1b ``.nc`` (NetCDF-4 is
HDF5): ``Rad`` / ``DQF`` stored as signed ints flagged ``_Unsigned``,
packed ``x`` / ``y`` scan angles, the ``goes_imager_projection`` grid
mapping, scalar calibration variables with a ``-999`` fill, and the
global attributes the reader reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


PERSPECTIVE_HEIGHT_M = 35_786_023.0
X0_RAD = -0.10
Y0_RAD = 0.12
STEP_RAD = 5.6e-05

# Representative C13 Planck coefficients (GOES-19 file metadata).
PLANCK = {"fk1": 10803.3, "fk2": 1392.74, "bc1": 0.0755, "bc2": 0.99975}
KAPPA0 = 0.001537


def _arr(value: Any, dtype: Any) -> np.ndarray:
    return np.array([value], dtype=dtype)


def write_abi(
    path: Path,
    *,
    band_id: int = 13,
    counts: np.ndarray | None = None,
    fill: int = 16383,
    dqf: np.ndarray | None = None,
    scale: float = 0.5,
    offset: float = -1.0,
    platform: str = "G19",
    scene: str = "CONUS",
) -> Path:
    """Write a minimal ABI L1b radiance file; returns ``path``."""
    import h5py

    counts = (
        np.arange(6 * 8, dtype=np.uint16).reshape(6, 8) + 100
        if counts is None
        else np.asarray(counts, dtype=np.uint16)
    )
    n_rows, n_cols = counts.shape
    dqf = np.zeros(counts.shape, dtype=np.uint8) if dqf is None else np.asarray(dqf)
    reflective = band_id <= 6
    with h5py.File(path, "w") as f:
        rad = f.create_dataset("Rad", data=counts.view(np.int16), chunks=True)
        rad.attrs["_FillValue"] = _arr(fill, np.uint16).view(np.int16)
        rad.attrs["_Unsigned"] = np.bytes_(b"true")
        rad.attrs["scale_factor"] = _arr(scale, np.float32)
        rad.attrs["add_offset"] = _arr(offset, np.float32)
        rad.attrs["units"] = np.bytes_(
            b"W m-2 sr-1 um-1" if reflective else b"mW m-2 sr-1 (cm-1)-1"
        )
        q = f.create_dataset("DQF", data=dqf.astype(np.uint8).view(np.int8))
        q.attrs["_FillValue"] = _arr(-1, np.int8)
        for name, n, start, step in (
            ("x", n_cols, X0_RAD, STEP_RAD),
            ("y", n_rows, Y0_RAD, -STEP_RAD),
        ):
            coord = f.create_dataset(name, data=np.arange(n, dtype=np.int16))
            coord.attrs["scale_factor"] = _arr(step, np.float64)
            coord.attrs["add_offset"] = _arr(start, np.float64)
        proj = f.create_dataset("goes_imager_projection", data=np.int32(0))
        proj.attrs["perspective_point_height"] = _arr(PERSPECTIVE_HEIGHT_M, np.float64)
        proj.attrs["semi_major_axis"] = _arr(6378137.0, np.float64)
        proj.attrs["semi_minor_axis"] = _arr(6356752.31414, np.float64)
        proj.attrs["longitude_of_projection_origin"] = _arr(-75.0, np.float64)
        proj.attrs["sweep_angle_axis"] = np.bytes_(b"x")
        f.create_dataset("band_id", data=_arr(band_id, np.int8))
        f.create_dataset(
            "band_wavelength", data=_arr(0.47 if reflective else 10.3, "f4")
        )
        scalars = {
            "kappa0": KAPPA0 if reflective else -999.0,
            "planck_fk1": -999.0 if reflective else PLANCK["fk1"],
            "planck_fk2": -999.0 if reflective else PLANCK["fk2"],
            "planck_bc1": -999.0 if reflective else PLANCK["bc1"],
            "planck_bc2": -999.0 if reflective else PLANCK["bc2"],
            "nominal_satellite_subpoint_lon": -75.2,
        }
        for name, value in scalars.items():
            f.create_dataset(name, data=np.float32(value))
        height = f.create_dataset(
            "nominal_satellite_height", data=np.float32(35786.023)
        )
        height.attrs["units"] = np.bytes_(b"km")
        f.attrs["platform_ID"] = np.bytes_(platform.encode())
        f.attrs["scene_id"] = np.bytes_(scene.encode())
        f.attrs["dataset_name"] = np.bytes_(b"OR_ABI-L1b-RadC-M6C13_G19_test.nc")
        f.attrs["time_coverage_start"] = np.bytes_(b"2026-10-07T12:01:17.8Z")
        f.attrs["time_coverage_end"] = np.bytes_(b"2026-10-07T12:03:55.1Z")
    return path
