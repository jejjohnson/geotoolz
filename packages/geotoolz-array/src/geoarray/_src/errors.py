"""The exceptions geoarray raises, each a subclass of a builtin one."""

from __future__ import annotations


__all__ = [
    "GeoArrayError",
    "GeoArrayTraceError",
    "GridMismatchError",
    "LabelError",
    "X64Error",
]


class GeoArrayError(Exception):
    """Base class of every error geoarray raises on its own account."""


class X64Error(GeoArrayError, RuntimeError):
    """A JAX carrier needs ``jax_enable_x64``: its transform is float64.

    In float32 the spacing of representable numbers is 0.25 m at a UTM
    northing of 3.55 × 10⁶ m, so a float32 transform silently misplaces
    pixels. geoarray refuses rather than rounding.
    """


class GeoArrayTraceError(GeoArrayError, TypeError):
    """An operation needs concrete values but got JAX tracers.

    Inside ``jit``, ``grad`` or ``vmap`` a GeoArray's data and transform are
    tracers. Anything that must read their values on the host (a `Grid`, a
    NumPy function, `GeoArray.to_numpy`) raises this instead of JAX's
    ``TracerArrayConversionError``, naming what to do instead.
    """


class GridMismatchError(GeoArrayError, ValueError):
    """Carriers that must share one pixel grid do not.

    `combine`, `stack` and `concatenate` need the same CRS, spatial shape,
    batch shape and (when concrete) an exactly equal transform. Resample
    one carrier onto the other's grid first (``geoops.geom``).
    """


class LabelError(GeoArrayError, KeyError):
    """A selection by label that a carrier cannot do.

    `GeoArray.sel` looks up exact labels of a dim's coordinate (band names,
    level values, time stamps). A dim without labels, a label that is not
    there, or a range or nearest-neighbour lookup raises this; selection by
    value lives in xarray (``g.to_xarray().sel(...)``).
    """

    def __str__(self) -> str:
        # KeyError quotes its message; show it as written
        return str(self.args[0]) if self.args else ""
