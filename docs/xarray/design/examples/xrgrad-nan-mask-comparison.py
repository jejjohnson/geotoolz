"""Quantify NaN/land-mask mitigations for xrgrad's finite differences.

Produces the error-vs-distance-from-coast table in
``docs/design/xrgrad-nan-masks.md``. Run with::

    uv run python docs/design/examples/xrgrad-nan-mask-comparison.py

Three mitigations are compared on an analytic field with a synthetic
coastline, scoring each by RMS relative error binned on distance from
the mask (in cells):

propagate
    Today's behaviour: a stencil touching land yields NaN, so a band of
    ocean ``~accuracy`` cells wide is lost.
fill-then-mask
    Harmonic-fill the land with :func:`xrtoolz.interpolate.fillnan_laplacian`,
    differentiate, then re-impose the mask. Recovers the band but the
    filled values are invented, so the derivative near the coast is
    biased toward the fill's smoothness.
adaptive
    ``nan_policy="adaptive"``: differentiate three times
    (central/forward/backward) and take, per point, the first variant
    whose stencil lies wholly on finite data. NaN itself detects
    support, since any stencil touching land returns NaN — including
    through a zero centre coefficient, because ``0 * nan`` is ``nan``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage

import xrgrad
from xrtoolz.interpolate import fillnan_laplacian


NLON, NLAT = 121, 91


def analytic_field() -> tuple[xr.DataArray, np.ndarray]:
    """Return ``sin(2x)cos(y)`` and its exact ``∂/∂lon`` (per degree)."""
    lon = np.linspace(0.0, 60.0, NLON)
    lat = np.linspace(0.0, 45.0, NLAT)
    grid_lon, grid_lat = np.meshgrid(lon, lat, indexing="xy")
    values = np.sin(2 * np.deg2rad(grid_lon)) * np.cos(np.deg2rad(grid_lat))
    exact = (
        (2 * np.cos(2 * np.deg2rad(grid_lon)) * np.cos(np.deg2rad(grid_lat)))
        * np.pi
        / 180.0
    )
    da = xr.DataArray(
        values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="f"
    )
    return da, exact


def coastline_mask() -> np.ndarray:
    """A land mask: a straight western shelf plus a curved bay."""
    lon_idx = np.arange(NLON)[None, :]
    lat_idx = np.arange(NLAT)[:, None]
    shelf = lon_idx < 12
    bay = (lon_idx - 60) ** 2 / 400.0 + (lat_idx - 45) ** 2 / 120.0 < 1.0
    return np.broadcast_to(shelf, (NLAT, NLON)) | bay


def rms_by_distance(
    error: np.ndarray, distance: np.ndarray, *, ocean: np.ndarray, max_d: int = 6
) -> list[tuple[int, float, float]]:
    """RMS error and NaN fraction per distance-from-coast bin."""
    rows = []
    for d in range(1, max_d + 1):
        sel = ocean & (distance == d)
        if not sel.any():
            continue
        vals = error[sel]
        finite = np.isfinite(vals)
        rms = (
            float(np.sqrt(np.mean(vals[finite] ** 2))) if finite.any() else float("nan")
        )
        rows.append((d, rms, float(1.0 - finite.mean())))
    return rows


def main() -> None:
    da, exact = analytic_field()
    land = coastline_mask()
    ocean = ~land
    # Distance in cells from the nearest land cell.
    distance = ndimage.distance_transform_edt(ocean).astype(int)

    masked = da.where(xr.DataArray(ocean, dims=da.dims, coords=da.coords))
    scale = np.abs(exact[ocean]).mean()

    for accuracy in (1, 2):
        print(
            f"\n{'=' * 68}\naccuracy={accuracy}"
            f"   (RMS relative error; 'lost' = fraction returned as NaN)\n{'=' * 68}"
        )

        propagate = xrgrad.partial(masked, "lon", accuracy=accuracy).values

        filled = fillnan_laplacian(masked, max_iter=2000, tol=1e-8)
        fill_then_mask = np.where(
            ocean, xrgrad.partial(filled, "lon", accuracy=accuracy).values, np.nan
        )

        adaptive = xrgrad.partial(
            masked, "lon", accuracy=accuracy, nan_policy="adaptive"
        ).values

        results = {
            "propagate": propagate,
            "fill-then-mask": fill_then_mask,
            "adaptive": adaptive,
        }
        header = f"{'dist':>5} " + "".join(f"{k:>26}" for k in results)
        print(header)
        print("-" * len(header))
        table = {
            name: dict(
                (d, (rms, lost))
                for d, rms, lost in rms_by_distance(
                    (values - exact) / scale, distance, ocean=ocean
                )
            )
            for name, values in results.items()
        }
        for d in range(1, 7):
            cells = [f"{'dist':>5}".replace("dist", str(d))]
            for name in results:
                rms, lost = table[name].get(d, (float("nan"), float("nan")))
                cells.append(f"{rms:>16.3e}  lost {lost:>4.0%}")
            print("".join(cells))

        interior = ocean & (distance > 4)
        print(
            "\n  interior (dist>4) max |adaptive - propagate| =",
            f"{np.nanmax(np.abs(adaptive[interior] - propagate[interior])):.3e}",
        )


if __name__ == "__main__":
    main()
