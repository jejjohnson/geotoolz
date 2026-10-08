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
        q.attrs["_Unsigned"] = np.bytes_(b"true")
        q.attrs["flag_values"] = np.arange(5, dtype=np.int8)
        q.attrs["flag_meanings"] = np.bytes_(
            b"good_pixel_qf conditionally_usable_pixel_qf out_of_range_pixel_qf "
            b"no_value_pixel_qf focal_plane_temperature_threshold_exceeded_qf"
        )
        _write_grid(f, n_rows, n_cols)
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
        _write_info(f, platform, scene, "OR_ABI-L1b-RadC-M6C13_G19_test.nc", rad=True)
    return path


def _write_grid(
    f: Any,
    n_rows: int,
    n_cols: int,
    step: float = STEP_RAD,
    origin: tuple[float, float] = (X0_RAD, Y0_RAD),
) -> None:
    for name, n, start, delta in (
        ("x", n_cols, origin[0], step),
        ("y", n_rows, origin[1], -step),
    ):
        coord = f.create_dataset(name, data=np.arange(n, dtype=np.int16))
        coord.attrs["scale_factor"] = _arr(delta, np.float64)
        coord.attrs["add_offset"] = _arr(start, np.float64)
    proj = f.create_dataset("goes_imager_projection", data=np.int32(0))
    proj.attrs["perspective_point_height"] = _arr(PERSPECTIVE_HEIGHT_M, np.float64)
    proj.attrs["semi_major_axis"] = _arr(6378137.0, np.float64)
    proj.attrs["semi_minor_axis"] = _arr(6356752.31414, np.float64)
    proj.attrs["longitude_of_projection_origin"] = _arr(-75.0, np.float64)
    proj.attrs["sweep_angle_axis"] = np.bytes_(b"x")


def _write_info(
    f: Any, platform: str, scene: str, name: str, *, rad: bool = False
) -> None:
    if not rad:
        f.create_dataset("nominal_satellite_subpoint_lon", data=np.float32(-75.2))
    height = f.create_dataset("nominal_satellite_height", data=np.float32(35786.023))
    height.attrs["units"] = np.bytes_(b"km")
    f.attrs["platform_ID"] = np.bytes_(platform.encode())
    f.attrs["scene_id"] = np.bytes_(scene.encode())
    f.attrs["dataset_name"] = np.bytes_(name.encode())
    f.attrs["time_coverage_start"] = np.bytes_(b"2026-10-07T12:01:17.8Z")
    f.attrs["time_coverage_end"] = np.bytes_(b"2026-10-07T12:03:55.1Z")


def l2_name(code: str, sector: str = "C") -> str:
    """A NOAA-style L2 file name for product ``code`` (``"ACM"``, ``"MCMIP"``)."""
    return (
        f"OR_ABI-L2-{code}{sector}-M6_G19_s20262801801178_e20262801803551"
        "_c20262801803578.nc"
    )


def write_l2(
    path: Path,
    variables: dict[str, tuple[np.ndarray, dict[str, Any]]],
    *,
    code: str = "ACHA",
    step: float = STEP_RAD,
    origin: tuple[float, float] = (X0_RAD, Y0_RAD),
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write an L2 product file: ``{name: (stored array, attrs)}`` on one grid.

    ``attrs`` are written as given (``_FillValue``, ``scale_factor``,
    ``add_offset``, ``_Unsigned``, ``units``, ``flag_values`` /
    ``flag_meanings``); ``extra`` adds scalar / 1-D variables; ``origin`` is
    the first pixel centre's ``(x, y)`` scan angle.
    """
    import h5py

    shape = next(iter(variables.values()))[0].shape
    with h5py.File(path, "w") as f:
        _write_grid(f, *shape, step=step, origin=origin)
        for name, (data, attrs) in variables.items():
            var = f.create_dataset(name, data=data)
            for key, value in attrs.items():
                var.attrs[key] = value
        for name, value in (extra or {}).items():
            f.create_dataset(name, data=value)
        _write_info(f, "G19", "CONUS", l2_name(code))
    return path
