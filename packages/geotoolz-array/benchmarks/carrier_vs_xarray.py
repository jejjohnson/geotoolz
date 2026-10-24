"""Per-op cost of the geoarray carrier against xr.DataArray (gate GA-G22, #536).

Replaces the "Indicative costs" table in geostack's docs/design/geoarray.md,
which came from one uncommitted run on a mock carrier. This times the real
carrier (geotoolz-array on this commit) and xarray on the same data:

- normalized difference, ND = (a − b) / (a + b + ε), of the two bands of a
  (2, H, W) scene: two selections and four array ops;
- a 32 × 32 window: carrier slicing, which moves the transform, against
  ``DataArray.isel``;

at (2, 64, 64), (2, 256, 256) and (2, 2000, 2000). It also records the cold
import time of xarray and NumPy, and what each does with two grids 1e-7 m
apart.

Each timing is the median over repeats of the mean of a ``timeit`` loop,
after a warm-up call. Output is JSON with the machine and versions.

Usage, from packages/geotoolz-array:

    uv run python benchmarks/carrier_vs_xarray.py \\
        --out benchmarks/results/carrier_vs_xarray.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import timeit
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

import geoarray


#: 20 m north-up UTM 13N grid: the Delaware Basin chip of the design pages.
TRANSFORM = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
CRS = "EPSG:32613"
SHAPES: tuple[tuple[int, int, int], ...] = ((2, 64, 64), (2, 256, 256), (2, 2000, 2000))
EPS = 1e-10
WINDOW = 32


def scene_carrier(shape: tuple[int, int, int], seed: int = 0) -> geoarray.GeoArray:
    """A float32 (2, H, W) carrier on the chip's grid."""
    rng = np.random.default_rng(seed)
    data = rng.random(shape, dtype=np.float32)
    return geoarray.GeoArray(data, TRANSFORM, CRS, dims=("band", "y", "x"))


def scene_xarray(g: geoarray.GeoArray) -> xr.DataArray:
    """The same data as a DataArray with centre coordinates."""
    a, _, c, _, e, f = (float(v) for v in g.transform)
    height, width = g.shape[-2:]
    # xⱼ = c + a (j + ½),  yᵢ = f + e (i + ½)
    x = c + a * (np.arange(width) + 0.5)
    y = f + e * (np.arange(height) + 0.5)
    coords = {"band": ["B11", "B12"], "y": y, "x": x}
    return xr.DataArray(g.data, dims=("band", "y", "x"), coords=coords)


def nd_carrier(g: geoarray.GeoArray) -> geoarray.GeoArray:
    """ND on the carrier: band selections, then one combine of four ops."""
    first = g[0]
    second = g[1]

    def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # ND = (a − b) / (a + b + ε)
        difference = a - b
        total = a + b
        loaded = total + EPS
        return difference / loaded

    return geoarray.combine(nd, first, second)


def nd_xarray(da: xr.DataArray) -> xr.DataArray:
    """ND on the DataArray: the same selections and four ops."""
    first = da.isel(band=0)
    second = da.isel(band=1)
    # ND = (a − b) / (a + b + ε)
    difference = first - second
    total = first + second
    loaded = total + EPS
    return difference / loaded


def window_carrier(g: geoarray.GeoArray) -> geoarray.GeoArray:
    """A 32 × 32 window by slicing; the transform origin moves."""
    return g[..., 8 : 8 + WINDOW, 8 : 8 + WINDOW]


def window_xarray(da: xr.DataArray) -> xr.DataArray:
    """The same window with isel; the coordinates are sliced."""
    return da.isel(y=slice(8, 8 + WINDOW), x=slice(8, 8 + WINDOW))


def time_us(fn: Callable[[], Any], repeats: int) -> float:
    """Median over repeats of the per-call mean, in µs, after a warm-up."""
    fn()
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    runs = timer.repeat(repeat=repeats, number=number)
    per_call = [run / number for run in runs]
    return statistics.median(per_call) * 1e6


def import_seconds(module: str, repeats: int = 5) -> float:
    """Median cold import time of a module in a fresh interpreter, in s."""
    code = (
        "import time; t = time.perf_counter(); "
        f"import {module}; print(time.perf_counter() - t)"
    )
    times = []
    for _ in range(repeats):
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        times.append(float(out.stdout.strip()))
    return statistics.median(times)


def misaligned() -> dict[str, Any]:
    """Two 1 × 4 × 4 grids whose x differ by 1e-7 m, added on each side."""
    g = geoarray.GeoArray(np.ones((1, 4, 4)), TRANSFORM, CRS, dims="band y x")
    shifted = (20.0, 0.0, 600000.0 + 1e-7, 0.0, -20.0, 3550000.0)
    h = geoarray.GeoArray(np.ones((1, 4, 4)), shifted, CRS, dims="band y x")
    da, db = scene_xarray_1band(g), scene_xarray_1band(h)
    xarray_shape = tuple(int(n) for n in (da + db).shape)
    try:
        geoarray.combine(np.add, g, h)
        carrier = "no error"
    except geoarray.GridMismatchError as exc:
        carrier = f"GridMismatchError: {str(exc)[:60]}…"
    return {"xarray_result_shape": xarray_shape, "carrier": carrier}


def scene_xarray_1band(g: geoarray.GeoArray) -> xr.DataArray:
    """A one-band DataArray with centre coordinates."""
    a, _, c, _, e, f = (float(v) for v in g.transform)
    height, width = g.shape[-2:]
    x = c + a * (np.arange(width) + 0.5)
    y = f + e * (np.arange(height) + 0.5)
    return xr.DataArray(g.data, dims=("band", "y", "x"), coords={"y": y, "x": x})


def run(shapes: tuple[tuple[int, int, int], ...], repeats: int) -> dict[str, Any]:
    """Every timing, the import times and the misaligned case."""
    rows = []
    for shape in shapes:
        g = scene_carrier(shape)
        da = scene_xarray(g)
        np.testing.assert_allclose(nd_carrier(g).data, nd_xarray(da).values)
        rows.append(
            {
                "shape": list(shape),
                "nd_carrier_us": time_us(lambda g=g: nd_carrier(g), repeats),
                "nd_xarray_us": time_us(lambda da=da: nd_xarray(da), repeats),
                "window_carrier_us": time_us(lambda g=g: window_carrier(g), repeats),
                "window_xarray_us": time_us(lambda da=da: window_xarray(da), repeats),
            }
        )
    return {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xarray": xr.__version__,
            "geotoolz_array": geoarray.__version__,
        },
        "method": "median over repeats of the timeit autorange mean, after warm-up",
        "repeats": repeats,
        "rows": rows,
        "import_s": {
            "numpy": import_seconds("numpy", min(repeats, 5)),
            "xarray": import_seconds("xarray", min(repeats, 5)),
        },
        "misaligned_1e-7_m": misaligned(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="JSON output path")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--smallest", action="store_true", help="only the smallest shape (smoke run)"
    )
    args = parser.parse_args(argv)
    shapes = SHAPES[:1] if args.smallest else SHAPES
    result = run(shapes, args.repeats)
    text = json.dumps(result, indent=2) + "\n"
    if args.out is None:
        print(text, end="")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # write beside, then rename: an interrupted run never truncates the result
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(text)
    partial.replace(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
