"""Synthetic ``wrfout`` fixture for the :mod:`xrtoolz.atm` WRF tests.

Writes a small NetCDF file with WRF's variable names, dimensions,
staggering and unit conventions, populated from analytic fields that are
*linear* in every coordinate. Linear fields survive destaggering
(averaging) exactly, so the tests compare the opened Dataset against the
analytic functions to round-off.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from xrtoolz.atm._src.wrf import GRAVITY, KAPPA, P_REFERENCE, THETA_BASE


@dataclass(frozen=True)
class SyntheticWrf:
    """The analytic fields a synthetic ``wrfout`` was written from."""

    path: Path
    dx: float
    dy: float
    n_time: int
    n_lev: int
    n_y: int
    n_x: int
    terrain: float
    z_stag: np.ndarray  # staggered level heights above ground [m]
    start: np.datetime64  # first stamp; steps are hourly

    @property
    def z_mass(self) -> np.ndarray:
        return 0.5 * (self.z_stag[:-1] + self.z_stag[1:])

    @property
    def x_mass(self) -> np.ndarray:
        return self.dx * np.arange(self.n_x, dtype=np.float64)

    @property
    def y_mass(self) -> np.ndarray:
        return self.dy * np.arange(self.n_y, dtype=np.float64)

    @property
    def times(self) -> np.ndarray:
        return self.start + np.arange(self.n_time) * np.timedelta64(3600, "s")

    # Fields linear in every coordinate; ``t`` is the time-step index.
    @staticmethod
    def u(x, y, z, t):
        return 2.0 + 0.01 * x + 0.02 * z + 0.5 * t

    @staticmethod
    def v(x, y, z, t):
        return 1.0 + 0.005 * y - 0.01 * z + 0.1 * t

    @staticmethod
    def w(x, y, z, t):
        return 0.001 * z + 0.0002 * x

    @staticmethod
    def pressure(z):
        return P_REFERENCE - 8.0 * z

    @staticmethod
    def theta(z):
        return 295.0 + 0.003 * z

    @classmethod
    def temperature(cls, z):
        return cls.theta(z) * (cls.pressure(z) / P_REFERENCE) ** KAPPA

    @staticmethod
    def qvapor(z):
        return 0.008 - 1.0e-6 * z

    @staticmethod
    def pbl_height(x, y, t):
        return 500.0 + 0.1 * x + 0.05 * y + 20.0 * t

    def grid(self, x=None, y=None, z=None):
        """``(t, z, y, x)`` coordinate blocks in WRF's storage order."""
        x = self.x_mass if x is None else x
        y = self.y_mass if y is None else y
        z = self.z_mass if z is None else z
        t = np.arange(self.n_time, dtype=np.float64)
        return np.meshgrid(t, z, y, x, indexing="ij")


def write_synthetic_wrfout(
    path: Path,
    *,
    n_time: int = 3,
    n_lev: int = 6,
    n_y: int = 10,
    n_x: int = 12,
    dx: float = 1000.0,
    dy: float = 1000.0,
    dz: float = 100.0,
    terrain: float = 250.0,
    hgt_time_axis: bool = True,
    with_times: bool = True,
    with_xtime: bool = True,
    with_alpha: bool = True,
    alpha: float = 0.0,
    moving_nest: bool = False,
) -> SyntheticWrf:
    """Write a flat-terrain synthetic ``wrfout`` and return its analytic spec.

    ``U`` / ``V`` are written as the *grid-relative* components of the
    analytic wind, i.e. the analytic ``u`` / ``v`` rotated by ``-alpha``
    (radians); with ``with_alpha`` the file also carries ``COSALPHA`` /
    ``SINALPHA`` so :func:`open_wrfout` can rotate them back. ``moving_nest``
    shifts ``XLONG`` / ``XLAT`` / ``HGT`` by one grid step per time step.
    """
    spec = SyntheticWrf(
        path=path,
        dx=dx,
        dy=dy,
        n_time=n_time,
        n_lev=n_lev,
        n_y=n_y,
        n_x=n_x,
        terrain=terrain,
        z_stag=dz * np.arange(n_lev + 1, dtype=np.float64),
        start=np.datetime64("2024-06-01T00:00:00", "s"),
    )
    x_stag = dx * (np.arange(n_x + 1, dtype=np.float64) - 0.5)
    y_stag = dy * (np.arange(n_y + 1, dtype=np.float64) - 0.5)

    tt, zz, yy, xx = spec.grid()
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    # Grid-relative components: the analytic (earth-relative) wind rotated
    # by ``-alpha``, so that ``u_e = u cos a - v sin a`` recovers it.
    u_grid = (
        spec.u(*spec.grid(x=x_stag)[::-1]) * cos_a
        + spec.v(*spec.grid(x=x_stag)[::-1]) * sin_a
    )
    v_grid = (
        spec.v(*spec.grid(y=y_stag)[::-1]) * cos_a
        - spec.u(*spec.grid(y=y_stag)[::-1]) * sin_a
    )
    w = spec.w(*spec.grid(z=spec.z_stag)[::-1])
    ph = 0.05 * GRAVITY * spec.grid(z=spec.z_stag)[1]
    phb = GRAVITY * (terrain + spec.grid(z=spec.z_stag)[1]) - ph
    p_pert = np.full_like(zz, 50.0)
    pb = spec.pressure(zz) - p_pert
    theta_pert = spec.theta(zz) - THETA_BASE
    hgt = np.full((n_time, n_y, n_x), terrain)
    pblh = spec.pbl_height(xx[:, 0], yy[:, 0], tt[:, 0])
    shift = np.arange(n_time)[:, None, None] if moving_nest else 0
    xlong = np.broadcast_to(-3.0 + 0.01 * (np.arange(n_x) + shift), (n_time, n_y, n_x))
    xlat = np.broadcast_to(
        40.0 + 0.009 * (np.arange(n_y)[:, None] + shift), (n_time, n_y, n_x)
    )
    if moving_nest:
        hgt = hgt + 10.0 * shift

    mass = ("Time", "bottom_top", "south_north", "west_east")
    fields = {
        "U": (("Time", "bottom_top", "south_north", "west_east_stag"), u_grid, "X"),
        "V": (("Time", "bottom_top", "south_north_stag", "west_east"), v_grid, "Y"),
        "W": (("Time", "bottom_top_stag", "south_north", "west_east"), w, "Z"),
        "PH": (("Time", "bottom_top_stag", "south_north", "west_east"), ph, "Z"),
        "PHB": (("Time", "bottom_top_stag", "south_north", "west_east"), phb, "Z"),
        "T": (mass, theta_pert, ""),
        "P": (mass, p_pert, ""),
        "PB": (mass, pb, ""),
        "QVAPOR": (mass, spec.qvapor(zz), ""),
        "HGT": (("Time", "south_north", "west_east"), hgt, ""),
        "PBLH": (("Time", "south_north", "west_east"), pblh, ""),
        "XLONG": (("Time", "south_north", "west_east"), xlong, ""),
        "XLAT": (("Time", "south_north", "west_east"), xlat, ""),
    }
    if not hgt_time_axis:
        fields["HGT"] = (("south_north", "west_east"), hgt[0], "")
    if with_alpha:
        shape = (n_time, n_y, n_x)
        fields["COSALPHA"] = (
            ("Time", "south_north", "west_east"),
            np.full(shape, cos_a),
            "",
        )
        fields["SINALPHA"] = (
            ("Time", "south_north", "west_east"),
            np.full(shape, sin_a),
            "",
        )

    # netCDF4 < 1.7.5 sets ``.shape`` on the arrays it writes, which NumPy
    # 2.5 deprecates; the fixture is not the place to surface that.
    with (
        warnings.catch_warnings(),
        netCDF4.Dataset(path, "w", format="NETCDF4") as nc,
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        for name, size in (
            ("bottom_top", n_lev),
            ("bottom_top_stag", n_lev + 1),
            ("south_north", n_y),
            ("south_north_stag", n_y + 1),
            ("west_east", n_x),
            ("west_east_stag", n_x + 1),
        ):
            nc.createDimension(name, size)
        for name, (dims, data, stagger) in fields.items():
            var = nc.createVariable(name, "f8", dims)
            var[...] = data
            var.stagger = stagger
            var.coordinates = "XLONG XLAT XTIME"
        if with_times:
            rows = [str(t).replace("T", "_").encode() for t in spec.times]
            times = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
            times[...] = np.array([np.frombuffer(r, dtype="S1") for r in rows])
        if with_xtime:
            xtime = nc.createVariable("XTIME", "f8", ("Time",))
            xtime[...] = 60.0 * np.arange(n_time)
            xtime.units = "minutes since 2024-06-01 00:00:00"
            xtime.description = "minutes since simulation start"
        nc.DX = dx
        nc.DY = dy
        nc.MAP_PROJ = 1
        nc.CEN_LAT = 40.05
        nc.CEN_LON = -2.945
        nc.TRUELAT1 = 30.0
        nc.TRUELAT2 = 60.0
        nc.STAND_LON = -2.945
        nc.SIMULATION_START_DATE = "2024-06-01_00:00:00"
        nc.TITLE = "synthetic wrfout for xrtoolz tests"
    return spec


@pytest.fixture(scope="session")
def synthetic_wrf(tmp_path_factory: pytest.TempPathFactory) -> SyntheticWrf:
    path = tmp_path_factory.mktemp("wrf") / "wrfout_d01_synthetic.nc"
    return write_synthetic_wrfout(path)
