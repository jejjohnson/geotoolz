"""Shared test factories for the geotoolz suite.

Every test module used to carry its own copy of a toy-GeoTensor
factory (``_gt`` / ``_toy_geotensor``); this is the single shared
implementation. Import it as::

    from _helpers import toy_geotensor

(pytest puts ``tests/`` on ``sys.path`` via rootdir conftest discovery).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor


#: 10 m UTM grid anchored in zone 29N — arbitrary but stable, so tests
#: can assert on exact transform round-trips.
DEFAULT_TRANSFORM = rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)
DEFAULT_CRS = "EPSG:32629"


def toy_geotensor(
    values: np.ndarray,
    *,
    transform: rasterio.Affine | None = None,
    crs: Any = DEFAULT_CRS,
    fill_value_default: Any = -9999,
    attrs: dict[str, Any] | None = None,
) -> GeoTensor:
    """Wrap an array in a GeoTensor with stable toy georeferencing.

    Args:
        values: The pixel array, 2-D ``(H, W)`` up to 4-D ``(T, C, H, W)``.
        transform: Affine geotransform; defaults to a 10 m UTM grid.
        crs: Coordinate reference system. Default ``EPSG:32629``.
        fill_value_default: Fill value stored on the carrier.
        attrs: Optional metadata dict.

    Returns:
        A ``GeoTensor`` viewing ``values``.
    """
    return GeoTensor(
        values,
        transform=DEFAULT_TRANSFORM if transform is None else transform,
        crs=crs,
        fill_value_default=fill_value_default,
        attrs=attrs,
    )


def uint16_dn_cube(n_bands: int = 2, *, size: int = 4, seed: int = 0) -> np.ndarray:
    """A ``uint16`` DN cube whose band arithmetic would wrap in its own dtype.

    Band ``k`` holds DN drawn from ``[1000 + 1000 * k, 2000 + 1000 * k)``,
    so every band is strictly darker than the next: ``band[0] - band[1]``
    is negative everywhere, which in ``uint16`` silently wraps to ~65 000.
    Values stay far below the ``uint16`` limit, so the float64 reference
    ``cube.astype(np.float64)`` is exact.

    Args:
        n_bands: Number of bands (the leading axis). Default ``2``.
        size: Height and width in pixels. Default ``4``.
        seed: RNG seed. Default ``0``.

    Returns:
        A ``(n_bands, size, size)`` ``uint16`` array.
    """
    rng = np.random.default_rng(seed)
    offsets = (1000 + 1000 * np.arange(n_bands)).reshape(-1, 1, 1)
    noise = rng.integers(0, 1000, size=(n_bands, size, size))
    return (offsets + noise).astype(np.uint16)


def all_operator_classes() -> list[type]:
    """Every ``pipekit.Operator`` subclass defined in ``geotoolz``.

    Imports every ``geotoolz`` submodule first (skipping ones whose
    optional extras are missing) so that lazily-registered subclasses
    are visible, then walks ``Operator`` subclasses transitively.

    Returns:
        The classes sorted by qualified name, for stable parametrisation.
    """
    import importlib
    import pkgutil

    from pipekit import Operator

    import geotoolz

    for info in pkgutil.walk_packages(geotoolz.__path__, "geotoolz."):
        try:
            importlib.import_module(info.name)
        except ImportError:
            continue

    seen: set[type] = set()
    stack = [Operator]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return sorted(
        (c for c in seen if c.__module__.startswith("geotoolz")),
        key=lambda c: f"{c.__module__}.{c.__qualname__}",
    )
