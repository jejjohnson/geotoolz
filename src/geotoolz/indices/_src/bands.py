"""Shared utilities for resolving band references against GeoTensor metadata.

Spectral-index operators accept band references either as integer indices
(``red_idx=3``) or as sensor-style names (``red="B04"``). The helpers
here translate names to integer positions using metadata carried on the
``GeoTensor`` — looked up under a configurable list of attribute keys.

The defaults match conventions used across `rasterio` (``descriptions``),
``xarray`` (``band_names``), and assorted DataArray pipelines (``bands``)
so common upstream readers Just Work without per-key wiring.

The helpers live here so they can be reused by other modules that grow a
similar idiom (e.g. masking, radiometric corrections that target a named
band) without dragging in `geotoolz.indices.operators`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


#: Type alias for band references — either an integer index or a string
#: band name that needs resolution against ``GeoTensor.attrs``.
BandRef = int | str


#: Default lookup order for named-band resolution. The first key that
#: holds a non-``None`` iterable wins; subsequent keys are only consulted
#: when the band name is missing from earlier keys.
DEFAULT_BAND_KEYS: tuple[str, ...] = ("descriptions", "band_names", "bands")


def resolve_band(
    gt: GeoTensor,
    ref: BandRef,
    *,
    keys: tuple[str, ...] = DEFAULT_BAND_KEYS,
) -> int:
    """Resolve a band reference to an integer band-axis index.

    Integer references pass through unchanged. String references are
    looked up against ``gt.attrs[key]`` for each ``key`` in ``keys`` (in
    order). The first key whose iterable contains the requested name
    wins; if a key exists but doesn't contain the name, the search
    continues to the next key. Missing keys, ``None`` values, and
    non-iterable values are skipped silently.

    Args:
        gt: Carrier `GeoTensor`. Its ``attrs`` dict is consulted.
        ref: Either an existing integer index (returned as-is) or a
            string band name to look up.
        keys: Attribute keys to consult, in precedence order. Defaults
            to ``("descriptions", "band_names", "bands")``.

    Returns:
        The integer position of the band along the carrier's band axis.

    Raises:
        ValueError: If ``ref`` is a string and the name is not found
            under any of the configured ``keys``.

    Examples:
        >>> import numpy as np, rasterio
        >>> from georeader.geotensor import GeoTensor
        >>> gt = GeoTensor(
        ...     values=np.zeros((4, 2, 2), dtype=np.float32),
        ...     transform=rasterio.Affine.identity(),
        ...     crs="EPSG:4326",
        ... )
        >>> gt.attrs["descriptions"] = ("B02", "B03", "B04", "B08")
        >>> resolve_band(gt, "B04")
        2
        >>> resolve_band(gt, 7)  # integers pass through untouched
        7
    """
    if not isinstance(ref, str):
        return ref

    for key in keys:
        names = gt.attrs.get(key)
        if names is None:
            continue
        try:
            band_names = tuple(names)
        except TypeError:
            continue
        try:
            return band_names.index(ref)
        except ValueError:
            continue

    raise ValueError(
        f"Band {ref!r} was not found in GeoTensor attrs "
        + ", ".join(f"{k!r}" for k in keys)
        + "."
    )


def configured_ref(value: BandRef | None, fallback: BandRef | None) -> BandRef:
    """Apply the dual ``band=`` / ``band_idx=`` constructor pattern.

    Index operators accept both a named-or-positional ``band`` keyword
    *and* an integer-only ``band_idx`` keyword (with a sensible
    sensor-agnostic default) so that callers can either:

    * leave defaults alone and pass integer ``..._idx`` overrides, or
    * pass named bands via the sensor-style alias keyword
      (``red="B04"``).

    Args:
        value: The named-or-positional keyword's value (e.g. ``red=``).
            Wins when not ``None``.
        fallback: The integer-only keyword's value (e.g. ``red_idx=``).
            Used when ``value`` is ``None``.

    Returns:
        Whichever of the two is non-``None``.

    Raises:
        ValueError: When both arguments are ``None``.
    """
    if value is not None:
        return value
    if fallback is None:
        raise ValueError(
            "A band reference must be provided through the named parameter "
            "or its *_idx fallback."
        )
    return fallback
