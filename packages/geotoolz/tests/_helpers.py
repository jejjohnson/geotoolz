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
    with_fill_pixels: bool = False,
) -> GeoTensor:
    """Wrap an array in a GeoTensor with stable toy georeferencing.

    Args:
        values: The pixel array, 2-D ``(H, W)`` up to 4-D ``(T, C, H, W)``.
        transform: Affine geotransform; defaults to a 10 m UTM grid.
        crs: Coordinate reference system. Default ``EPSG:32629``.
        fill_value_default: Fill value stored on the carrier.
        attrs: Optional metadata dict.
        with_fill_pixels: Write ``fill_value_default`` into every band of
            the pixels marked by :func:`fill_pixel_mask` (the first and
            last pixel of the grid). ``values`` is copied first.

    Returns:
        A ``GeoTensor`` viewing ``values`` (a copy when
        ``with_fill_pixels`` is set).
    """
    if with_fill_pixels:
        if fill_value_default is None:
            raise ValueError("with_fill_pixels needs a fill_value_default")
        values = np.array(values, copy=True)
        values[..., fill_pixel_mask(values.shape)] = fill_value_default
    return GeoTensor(
        values,
        transform=DEFAULT_TRANSFORM if transform is None else transform,
        crs=crs,
        fill_value_default=fill_value_default,
        attrs=attrs,
    )


def fill_pixel_mask(shape: tuple[int, ...]) -> np.ndarray:
    """``(H, W)`` mask of the pixels ``toy_geotensor(with_fill_pixels=True)`` fills.

    The first and the last pixel of the grid -- the ``[0, 0]`` and
    ``[-1, -1]`` corners -- so tests can check both edges.

    Args:
        shape: The carrier's shape; only the trailing two dims are used.

    Returns:
        A boolean ``(H, W)`` array, ``True`` at fill pixels.
    """
    mask = np.zeros(shape[-2:], dtype=bool)
    mask[0, 0] = True
    mask[-1, -1] = True
    return mask


#: Shape of :func:`time_stack`: 2 frames, 3 bands, 4 x 4 pixels. The band
#: axis (3) differs from the time axis (2), so an operator that takes axis
#: 0 as the band axis fails loudly or produces a visibly wrong shape.
TIME_STACK_SHAPE: tuple[int, int, int, int] = (2, 3, 4, 4)


def time_stack(
    shape: tuple[int, int, int, int] = TIME_STACK_SHAPE,
    *,
    seed: int = 0,
    fill_value_default: Any = -9999,
    attrs: dict[str, Any] | None = None,
    with_fill_pixels: bool = False,
) -> GeoTensor:
    """A ``(T, C, H, W)`` GeoTensor time stack (dims ``time, band, y, x``).

    Values are in ``[0.01, 1)`` and differ between frames and bands. The
    default ``attrs`` carry one ``band_names`` / ``wavelengths`` entry per
    *band* (``shape[1]``), as a reader of a time series would write them.

    Args:
        shape: ``(T, C, H, W)``. Default :data:`TIME_STACK_SHAPE`.
        seed: RNG seed. Default ``0``.
        fill_value_default: Fill value stored on the carrier.
        attrs: Metadata; defaults to per-band ``band_names`` and
            ``wavelengths`` (nm).
        with_fill_pixels: Write the fill into every band and frame of the
            pixels marked by :func:`fill_pixel_mask`.

    Returns:
        A float64 ``GeoTensor`` of the given shape.
    """
    n_bands = shape[1]
    if attrs is None:
        attrs = {
            "band_names": [f"b{i}" for i in range(n_bands)],
            "wavelengths": [490.0 + 100.0 * i for i in range(n_bands)],
        }
    values = np.random.default_rng(seed).uniform(0.01, 1.0, shape)
    return toy_geotensor(
        values,
        fill_value_default=fill_value_default,
        attrs=attrs,
        with_fill_pixels=with_fill_pixels,
    )


def frames(stack: Any) -> list[Any]:
    """The ``(C, H, W)`` frames of a 4-D stack (GeoTensors keep their grid).

    Args:
        stack: A ``(T, C, H, W)`` GeoTensor or ndarray.

    Returns:
        One carrier per time step (``stack.isel({"time": t})`` for a
        GeoTensor, with a copy of its attrs).
    """
    if isinstance(stack, GeoTensor):
        out = []
        for t in range(stack.shape[0]):
            frame = stack.isel({"time": t})
            frame.attrs = dict(stack.attrs or {})
            out.append(frame)
        return out
    return [np.asarray(stack)[t] for t in range(np.shape(stack)[0])]


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
