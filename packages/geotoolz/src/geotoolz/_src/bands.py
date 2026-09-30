"""Shared utilities for resolving band references against GeoTensor metadata.

Spectral-index operators accept band references either as integer indices
(``red_idx=3``) or as sensor-style names (``red="B04"``). The helpers
here translate names to integer positions using metadata carried on the
``GeoTensor`` — looked up under a configurable list of attribute keys.

The lookup order (:data:`DEFAULT_BAND_KEYS`) is ``band_names`` (what
geotoolz readers and operators write), then ``descriptions`` (rasterio),
then ``bands`` (assorted DataArray pipelines), so common upstream readers
Just Work without per-key wiring.

The helpers live in ``geotoolz._src`` so every operator family (indices,
spectral, qa, augment, viz, compositing, plume and
:func:`geotoolz._src.wrap.wrap_like`) shares one definition of which
``attrs`` keys describe the band axis and in which order they are read.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


#: Type alias for band references — either an integer index or a string
#: band name that needs resolution against ``GeoTensor.attrs``.
BandRef = int | str


#: Package-wide lookup order for named-band resolution: geotoolz's own
#: ``band_names`` first, then rasterio-style ``descriptions``, then the
#: legacy ``bands`` alias. The first key whose names contain the
#: requested band wins; later keys are only consulted when the name is
#: missing from earlier ones. Readers write only ``band_names``.
DEFAULT_BAND_KEYS: tuple[str, ...] = ("band_names", "descriptions", "bands")

#: Sentinel-2 L2A band order (12 bands, no B10) — the layout assumed for
#: a *plain ndarray* when an operator (``plume.SBMP``) is given S2 band
#: names but the carrier has no ``attrs`` to resolve them against. Never
#: used for a GeoTensor: carriers must name their bands.
SENTINEL2_L2A_BANDS: tuple[str, ...] = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
)

#: Canonical ``attrs`` key operators write band names under.
CANONICAL_BAND_KEY: str = "band_names"

#: Every ``attrs`` key that holds one entry per band. Band-name aliases
#: (:data:`DEFAULT_BAND_KEYS` plus ``band_descriptions``) and per-band
#: spectral metadata (``wavelengths`` / ``wavelengths_nm``). A value under
#: any of these keys goes stale as soon as the band axis changes size, so
#: :func:`geotoolz._src.wrap.wrap_like` drops them in that case.
PER_BAND_KEYS: tuple[str, ...] = (
    *DEFAULT_BAND_KEYS,
    "band_descriptions",
    "wavelengths",
    "wavelengths_nm",
)

#: The band-name aliases within :data:`PER_BAND_KEYS` (everything except
#: spectral metadata). Superseded when new band names are written.
BAND_NAME_KEYS: tuple[str, ...] = (*DEFAULT_BAND_KEYS, "band_descriptions")


def band_count(shape: tuple[int, ...]) -> int:
    """Size of the band axis of a carrier with the given ``shape``.

    Carriers are ``(H, W)``, ``(C, H, W)`` or ``(T, C, H, W)``: the band
    axis is ``shape[-3]`` for 3-D and 4-D carriers, and a 2-D carrier is a
    single band.

    Args:
        shape: The carrier's shape (at least 2-D).

    Returns:
        The number of bands.

    Examples:
        >>> band_count((5, 4, 4)), band_count((2, 5, 4, 4)), band_count((4, 4))
        (5, 5, 1)
    """
    return int(shape[-3]) if len(shape) >= 3 else 1


def per_band_values(
    attrs: Mapping[str, Any] | None, key: str, n_bands: int
) -> list[Any] | None:
    """The per-band list stored under ``attrs[key]``, if it fits ``n_bands``.

    Args:
        attrs: A carrier's ``attrs`` (``None`` means empty).
        key: The attrs key to read.
        n_bands: The carrier's band count.

    Returns:
        ``attrs[key]`` as a list of plain Python values (NumPy scalars are
        unwrapped so the result stays JSON-friendly), or ``None`` when the
        key is missing, is not a sequence (strings and mappings are not
        per-band lists), or has a length other than ``n_bands``.
    """
    value = (attrs or {}).get(key)
    if value is None or isinstance(value, str | bytes | Mapping):
        return None
    try:
        values = list(value)
    except TypeError:
        return None
    if len(values) != n_bands:
        return None
    return [v.item() if isinstance(v, np.generic) else v for v in values]


def strip_band_attrs(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Shallow copy of ``attrs`` without any :data:`PER_BAND_KEYS` entry.

    For outputs whose bands no longer correspond to any input band (an
    index or band-math result), so input band metadata would be stale.

    Examples:
        >>> strip_band_attrs({"band_names": ["a"], "wavelengths": [1.0], "id": 7})
        {'id': 7}
    """
    return {k: v for k, v in (attrs or {}).items() if k not in PER_BAND_KEYS}


def take_band_attrs(
    attrs: Mapping[str, Any] | None, indexes: Sequence[int], *, n_bands: int
) -> dict[str, Any]:
    """Copy ``attrs`` for a band selection, subsetting every per-band key.

    Each key in :data:`PER_BAND_KEYS` whose value is a list of
    ``n_bands`` entries is replaced by the entries at ``indexes`` (in that
    order); per-band keys that do not fit ``n_bands`` are dropped rather
    than left stale. Other keys are shallow-copied unchanged.

    Args:
        attrs: The source carrier's ``attrs`` (``None`` means empty).
        indexes: Selected band positions, in output order.
        n_bands: The source carrier's band count.

    Returns:
        A new attrs dict for the selected bands.

    Examples:
        >>> take_band_attrs(
        ...     {"band_names": ["a", "b", "c"], "sensor": "x"}, [2, 0], n_bands=3
        ... )
        {'sensor': 'x', 'band_names': ['c', 'a']}
    """
    out = strip_band_attrs(attrs)
    for key in PER_BAND_KEYS:
        values = per_band_values(attrs, key, n_bands)
        if values is not None:
            out[key] = [values[int(i)] for i in indexes]
    return out


def concat_band_attrs(
    attrs_list: Sequence[Mapping[str, Any] | None], band_counts: Sequence[int]
) -> dict[str, list[Any]]:
    """Concatenate per-band attrs across carriers stacked along the band axis.

    A key from :data:`PER_BAND_KEYS` is kept only when every input carries
    it with one entry per band; otherwise the stacked output could not
    carry a list matching its band count, so the key is omitted.

    Args:
        attrs_list: Each input's ``attrs``, in stacking order.
        band_counts: Each input's band count (1 for a 2-D carrier).

    Returns:
        The concatenated per-band keys only; merge them into the output's
        other attrs.

    Examples:
        >>> concat_band_attrs(
        ...     [{"band_names": ["a"]}, {"band_names": ["b", "c"]}], [1, 2]
        ... )
        {'band_names': ['a', 'b', 'c']}
    """
    out: dict[str, list[Any]] = {}
    for key in PER_BAND_KEYS:
        parts = [
            per_band_values(attrs, key, n)
            for attrs, n in zip(attrs_list, band_counts, strict=True)
        ]
        if parts and all(part is not None for part in parts):
            out[key] = [v for part in parts for v in part or ()]
    return out


def _names_under(
    attrs: Mapping[str, Any], key: str
) -> list[str] | dict[str, int] | None:
    """Band-name metadata stored under ``attrs[key]``, normalised.

    A sequence value becomes a list of ``str`` names (in band order); a
    ``Mapping`` value (the ``{name: index}`` form some QA products use)
    becomes a ``dict[str, int]``. Missing, ``None``, string and
    non-iterable values yield ``None`` so the caller moves on.
    """
    value = attrs.get(key)
    if value is None or isinstance(value, str | bytes):
        return None
    if isinstance(value, Mapping):
        return {str(name): int(idx) for name, idx in value.items()}
    try:
        return [str(name) for name in value]
    except TypeError:
        return None


def band_names(
    gt: GeoTensor | np.ndarray | Any,
    *,
    keys: tuple[str, ...] = DEFAULT_BAND_KEYS,
) -> list[str] | None:
    """The carrier's band names, read from the first usable ``attrs`` key.

    Keys are consulted in ``keys`` order (default
    :data:`DEFAULT_BAND_KEYS`: ``band_names``, then ``descriptions``, then
    ``bands``); the first one holding a sequence or a ``{name: index}``
    mapping wins. A mapping is returned as a list ordered by index.

    Args:
        gt: A GeoTensor-like carrier with ``attrs``, or a plain array.
        keys: Attribute keys to consult, in precedence order.

    Returns:
        The names as ``str`` in band order, or ``None`` when the carrier
        has no ``attrs`` (a plain ndarray) or none of ``keys`` holds
        band names.

    Examples:
        >>> import numpy as np
        >>> class Carrier:
        ...     attrs = {"descriptions": ("red", "nir"), "band_names": ["B04", "B08"]}
        >>> band_names(Carrier())
        ['B04', 'B08']
        >>> band_names(np.zeros((2, 3, 3))) is None
        True
    """
    attrs = getattr(gt, "attrs", None)
    if attrs is None:
        return None
    for key in keys:
        names = _names_under(attrs, key)
        if isinstance(names, dict):
            return [name for name, _ in sorted(names.items(), key=lambda kv: kv[1])]
        if names is not None:
            return names
    return None


def resolve_band(
    gt: GeoTensor | np.ndarray | Any,
    ref: BandRef,
    *,
    keys: tuple[str, ...] = DEFAULT_BAND_KEYS,
    fallback: Sequence[str] | None = None,
) -> int:
    """Resolve a band reference to an integer band-axis index.

    This is the one band-name resolver every operator family uses, so
    ``"B04"`` means the same band in ``indices``, ``spectral``, ``qa``,
    ``augment``, ``viz``, ``compositing`` and ``plume``.

    Integer references (any :class:`numbers.Integral` except ``bool``,
    so ``np.int64`` works) pass through as ``int`` for any carrier —
    plain ndarrays included. String references are looked up against
    ``gt.attrs[key]`` for each ``key`` in ``keys`` (in order). The first
    key whose names contain the requested one wins; a key that exists
    but doesn't contain the name hands over to the next key. Each value
    may be a sequence of names (position = band index) or a
    ``{name: index}`` mapping. Missing keys, ``None`` values and
    non-iterable values are skipped silently.

    Args:
        gt: Carrier ``GeoTensor`` (its ``attrs`` dict is consulted for
            named lookups) or a plain ndarray.
        ref: An integer index (returned as ``int``) or a band name.
        keys: Attribute keys to consult, in precedence order. Defaults
            to :data:`DEFAULT_BAND_KEYS` —
            ``("band_names", "descriptions", "bands")``.
        fallback: Band names in array order, used *only* when ``gt``
            has no ``attrs`` at all (a plain ndarray). ``plume.SBMP``
            passes :data:`SENTINEL2_L2A_BANDS` here so its ``"B11"`` /
            ``"B12"`` defaults work on a bare 12-band L2A array. A
            GeoTensor never uses the fallback.

    Returns:
        The integer position of the band along the carrier's band axis.

    Raises:
        TypeError: If ``ref`` is neither a string nor an integer, or
            ``ref`` is a string, the carrier has no ``attrs`` (e.g. a
            plain ``np.ndarray``) and no ``fallback`` was given.
        ValueError: If ``ref`` is a string and the name is not found
            under any of the configured ``keys`` (or in ``fallback``).

    Examples:
        >>> import numpy as np, rasterio
        >>> from georeader.geotensor import GeoTensor
        >>> gt = GeoTensor(
        ...     values=np.zeros((4, 2, 2), dtype=np.float32),
        ...     transform=rasterio.Affine.identity(),
        ...     crs="EPSG:4326",
        ... )
        >>> gt.attrs["band_names"] = ("B02", "B03", "B04", "B08")
        >>> resolve_band(gt, "B04")
        2
        >>> resolve_band(gt, np.int64(7))  # integers pass through as int
        7
    """
    if isinstance(ref, numbers.Integral) and not isinstance(ref, bool):
        return int(ref)
    if not isinstance(ref, str):
        raise TypeError(
            f"Band reference must be an integer index or a band name; got {ref!r}"
        )

    attrs = getattr(gt, "attrs", None)
    if attrs is None:
        if fallback is not None:
            try:
                return list(fallback).index(ref)
            except ValueError:
                raise ValueError(
                    f"Band {ref!r} is not one of the default band names "
                    f"{tuple(fallback)} assumed for a plain array; pass an "
                    "integer index or a GeoTensor carrying band names."
                ) from None
        raise TypeError(
            f"Named-band resolution ({ref!r}) requires a georeferenced "
            "GeoTensor input carrying band-name metadata in `attrs`; got a "
            "plain array. Pass an integer band index instead."
        )

    for key in keys:
        names = _names_under(attrs, key)
        if names is None:
            continue
        if isinstance(names, dict):
            if ref in names:
                return names[ref]
            continue
        if ref in names:
            return names.index(ref)

    raise ValueError(
        f"Band {ref!r} was not found in the band names under GeoTensor attrs "
        + ", ".join(f"{k!r}" for k in keys)
        + "."
    )


def resolve_bands(
    gt: GeoTensor | np.ndarray | Any,
    refs: Iterable[BandRef],
    *,
    keys: tuple[str, ...] = DEFAULT_BAND_KEYS,
    fallback: Sequence[str] | None = None,
) -> list[int]:
    """Resolve several band references with :func:`resolve_band`.

    Args:
        gt: Carrier whose ``attrs`` name the bands.
        refs: Integer indices and/or band names, in output order.
        keys: Attribute keys to consult, in precedence order.
        fallback: Plain-array fallback names (see :func:`resolve_band`).

    Returns:
        One integer index per reference, in ``refs`` order.

    Examples:
        >>> class Carrier:
        ...     attrs = {"band_names": ["B02", "B03", "B04"]}
        >>> resolve_bands(Carrier(), ["B04", 0, "B03"])
        [2, 0, 1]
    """
    return [resolve_band(gt, ref, keys=keys, fallback=fallback) for ref in refs]


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


def resolve_wavelengths(
    gt: Any,
    wavelengths: Sequence[float] | np.ndarray | None = None,
    *,
    n_bands: int | None = None,
    name: str = "wavelengths",
) -> np.ndarray:
    """Per-band centre wavelengths: an explicit value, else ``gt.attrs``.

    The one lookup every wavelength-dependent operator (SRF convolution,
    spectral binning, continuum removal) uses: the constructor argument
    wins; otherwise ``gt.attrs["wavelengths"]`` is read (plain arrays have
    no attrs, so they need the explicit value).

    Args:
        gt: The carrier.
        wavelengths: Explicit wavelengths, or ``None`` to read attrs.
        n_bands: When given, the number of bands the wavelengths must
            describe.
        name: Argument name used in error messages.

    Returns:
        The wavelengths as a 1-D float64 array.

    Raises:
        ValueError: If no wavelengths are available, or their count does
            not match ``n_bands``.

    Examples:
        >>> resolve_wavelengths(None, [490, 665]).tolist()
        [490.0, 665.0]
    """
    if wavelengths is None:
        wavelengths = (getattr(gt, "attrs", None) or {}).get("wavelengths")
    if wavelengths is None:
        raise ValueError(
            f"{name} must be provided or available as gt.attrs['wavelengths']"
        )
    values = np.asarray(wavelengths, dtype=float).reshape(-1)
    if n_bands is not None and values.size != n_bands:
        raise ValueError(
            f"{name} has {values.size} entries but the input has {n_bands} band(s)"
        )
    return values
