"""Coordinate validation and harmonization.

These helpers normalize coordinate names (``longitude`` → ``lon``,
``latitude`` → ``lat``), wrap values into the standard geographic
ranges, and attach CF-style ``units``, ``standard_name``, and
``long_name`` attributes.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from xrtoolz.transforms._src.encoders.coord_space import (
    lat_180_to_90,
    lon_360_to_180,
)


_LONGITUDE_ALIASES = ("longitude",)
_LATITUDE_ALIASES = ("latitude",)

_LON_ATTRS = {
    "units": "degrees_east",
    "standard_name": "longitude",
    "long_name": "Longitude",
}
_LAT_ATTRS = {
    "units": "degrees_north",
    "standard_name": "latitude",
    "long_name": "Latitude",
}


def validate_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Normalize the longitude coordinate.

    Renames ``longitude`` to ``lon`` if present, wraps values into the
    ``[-180, 180)`` range, and assigns CF ``units``, ``standard_name``,
    and ``long_name`` attributes (preserving any pre-existing attrs).

    Args:
        ds: Input dataset.

    Returns:
        Dataset with a harmonized ``lon`` coordinate.
    """
    new_ds = _rename_first_match(ds, _LONGITUDE_ALIASES, "lon")
    if "lon" not in new_ds.coords and "lon" not in new_ds.variables:
        raise KeyError("No longitude coordinate found (expected 'lon' or 'longitude').")

    existing_attrs = dict(new_ds["lon"].attrs)
    new_ds["lon"] = lon_360_to_180(new_ds["lon"])
    new_ds["lon"] = new_ds["lon"].assign_attrs(**{**existing_attrs, **_LON_ATTRS})
    return new_ds


def validate_latitude(ds: xr.Dataset) -> xr.Dataset:
    """Normalize the latitude coordinate.

    Renames ``latitude`` to ``lat`` if present, wraps values into the
    ``[-90, 90)`` range, and assigns CF ``units``, ``standard_name``,
    and ``long_name`` attributes (preserving any pre-existing attrs).

    Args:
        ds: Input dataset.

    Returns:
        Dataset with a harmonized ``lat`` coordinate.
    """
    new_ds = _rename_first_match(ds, _LATITUDE_ALIASES, "lat")
    if "lat" not in new_ds.coords and "lat" not in new_ds.variables:
        raise KeyError("No latitude coordinate found (expected 'lat' or 'latitude').")

    existing_attrs = dict(new_ds["lat"].attrs)
    new_ds["lat"] = lat_180_to_90(new_ds["lat"])
    new_ds["lat"] = new_ds["lat"].assign_attrs(**{**existing_attrs, **_LAT_ATTRS})
    return new_ds


def rename_coords(ds: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    """Rename any coordinates or variables that match ``mapping``.

    Keys not present in ``ds`` are silently ignored, so this helper is
    safe to use as a first-pass harmonizer without pre-checking names.

    Args:
        ds: Input dataset.
        mapping: ``{old_name: new_name}``. Only names actually present
            are renamed.

    Returns:
        Dataset with matching names renamed.
    """
    present = {old: new for old, new in mapping.items() if old in ds.variables}
    if not present:
        return ds
    return ds.rename(present)


def rename_variables(ds: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    """Rename data variables that match ``mapping``.

    Companion to :func:`rename_coords`: ``rename_variables`` only acts on
    entries in ``ds.data_vars`` so a typo in ``mapping`` doesn't silently
    rename a coordinate. Keys not present in ``ds.data_vars`` are
    ignored.
    """
    present = {old: new for old, new in mapping.items() if old in ds.data_vars}
    if not present:
        return ds
    return ds.rename(present)


def decode_cf_time(
    ds: xr.Dataset,
    *,
    time: str = "time",
    units: str | None = None,
) -> xr.Dataset:
    """Assign a CF ``units`` attribute and decode the time coordinate.

    Useful when raw integer / float time values and their epoch string
    arrive separately (e.g. CMEMS NetCDF or older OPeNDAP feeds).

    Args:
        ds: Input dataset.
        time: Name of the time coordinate.
        units: CF ``units`` string (e.g. ``"days since 1950-01-01"``).
            When ``None`` the attribute is left unchanged; if the
            coordinate is already ``datetime64`` this is effectively a
            no-op.

    Returns:
        Dataset with the time coordinate decoded to ``datetime64``.
    """
    ds = ds.copy()
    if units is not None:
        ds[time] = ds[time].assign_attrs(units=units)
    return xr.decode_cf(ds)


def validate_time(
    ds: xr.Dataset,
    *,
    time: str = "time",
    unit: str | None = None,
    origin: str = "unix",
) -> xr.Dataset:
    """Coerce the time coordinate to ``pandas`` datetime.

    Strings and ``datetime64`` arrays pass through unchanged. For
    numeric inputs, ``unit`` and ``origin`` must be supplied so the
    values can be interpreted unambiguously — otherwise
    :func:`pandas.to_datetime` defaults to nanoseconds since the Unix
    epoch, which silently mangles common "seconds since 1970-01-01"
    encodings.

    Args:
        ds: Input dataset.
        time: Name of the time coordinate.
        unit: Forwarded to :func:`pandas.to_datetime`. For numeric
            coords, set this to e.g. ``"s"`` / ``"ms"`` / ``"D"``.
        origin: Forwarded to :func:`pandas.to_datetime`. Defaults to
            the Unix epoch.

    Returns:
        Dataset with the time coordinate cast to ``datetime64[ns]``,
        preserving the original dims and attrs.
    """
    ds = ds.copy()
    coord = ds[time]
    converted = pd.to_datetime(coord.values, unit=unit, origin=origin)
    ds[time] = xr.DataArray(
        np.asarray(converted, dtype="datetime64[ns]"),
        dims=coord.dims,
        attrs=dict(coord.attrs),
    )
    return ds


_COORD_VALIDATORS: dict[str, Callable[[xr.Dataset], xr.Dataset]] = {
    "lon": validate_longitude,
    "lat": validate_latitude,
    "time": validate_time,
}


def check_dataset_coords(
    ds: xr.Dataset,
    *,
    require: tuple[str, ...] = ("time", "lat", "lon"),
    validate: bool = True,
) -> None:
    """Assert that required coordinates are present and (optionally) valid.

    Args:
        ds: Dataset to check.
        require: Names that must appear in ``ds.coords``. Data
            variables with these names do not satisfy the requirement —
            downstream coord-based ops like ``.sel`` need an actual
            coordinate.
        validate: When ``True``, run a round-trip through the
            appropriate validator for ``"lon"``, ``"lat"``, and
            ``"time"`` and assert the result is identical to the
            input.  Pass ``False`` to skip the round-trip checks.

    Raises:
        AssertionError: If any name in ``require`` is missing from
            ``ds.coords``, or if a round-trip validator produces a
            different result (``validate=True``).
    """
    missing = set(require) - set(ds.coords)
    if missing:
        raise AssertionError(f"Dataset missing required coords: {sorted(missing)}")
    if validate:
        for name in require:
            fn = _COORD_VALIDATORS.get(name)
            if fn is not None:
                xr.testing.assert_identical(ds[[name]], fn(ds)[[name]])


def _rename_first_match(
    ds: xr.Dataset,
    candidates: tuple[str, ...],
    target: str,
) -> xr.Dataset:
    """Rename the first matching alias to ``target`` if ``target`` is missing."""
    if target in ds.variables:
        return ds
    for name in candidates:
        if name in ds.variables:
            return ds.rename({name: target})
    return ds


def rename_to_cf_standard_names(
    ds: xr.Dataset,
    *,
    include_coords: bool = True,
) -> xr.Dataset:
    """Rename variables to their declared CF ``standard_name`` attribute.

    For every variable / coord with a non-empty ``standard_name`` attr,
    rename to that value. Variables without the attr are left unchanged.

    Args:
        ds: Input dataset.
        include_coords: If ``True`` (default), rename coords too. Set
            ``False`` to limit renaming to data variables only.

    Returns:
        Dataset with matching variables renamed to their CF
        ``standard_name``.

    Raises:
        ValueError: If two source variables resolve to the same CF
            ``standard_name`` (rename collision).
    """
    candidates: list[str] = (
        [str(k) for k in ds.variables]
        if include_coords
        else [str(k) for k in ds.data_vars]
    )
    mapping: dict[str, str] = {}
    for name in candidates:
        sn = ds[name].attrs.get("standard_name")
        if sn and sn != name:
            mapping[name] = str(sn)
    # Detect collisions: two source vars mapping to the same target name.
    inverse: dict[str, str] = {}
    for src, tgt in mapping.items():
        if tgt in inverse:
            raise ValueError(
                f"rename_to_cf_standard_names: two source variables "
                f"({inverse[tgt]!r} and {src!r}) both map to "
                f"standard_name={tgt!r}; cannot rename both."
            )
        inverse[tgt] = src
    # Detect collisions with names that already exist in the dataset
    # but aren't being renamed; xarray.rename would error on these,
    # but with a less helpful message.
    existing = set(ds.variables) - set(mapping)
    for src, tgt in mapping.items():
        if tgt in existing:
            raise ValueError(
                f"rename_to_cf_standard_names: source {src!r} maps to "
                f"standard_name={tgt!r}, which already exists in the "
                "dataset; rename or drop the existing variable first."
            )
    return ds.rename(mapping) if mapping else ds


def rename_from_cf_standard_names(
    ds: xr.Dataset,
    *,
    fallback: Literal["passthrough", "raise"] = "passthrough",
    include_coords: bool = True,
) -> xr.Dataset:
    """Rename CF ``standard_name``-shaped variables to xrtoolz canonical names.

    Uses the :mod:`xrreader.types.Variable` registry as the authoritative
    ``standard_name → canonical_name`` mapping. Variables / coords whose
    name is not a registered CF ``standard_name`` pass through unchanged
    (``fallback="passthrough"``, default) or raise ``KeyError``
    (``fallback="raise"``).

    Single-word names (e.g. ``"ssh"``) are never flagged as unknown — they
    are already canonical and pass through silently regardless of
    ``fallback``.

    Args:
        ds: Input dataset.
        fallback: ``"passthrough"`` (default) leaves unrecognized
            CF-shaped names unchanged. ``"raise"`` raises ``KeyError``
            listing all unrecognized names.
        include_coords: If ``True`` (default), rename coords too.

    Returns:
        Dataset with CF-named variables renamed to their canonical
        xrtoolz names.

    Raises:
        ValueError: If ``fallback`` is not ``"passthrough"`` or
            ``"raise"``.
        KeyError: If ``fallback="raise"`` and any variable name looks
            like a CF ``standard_name`` (contains ``"_"``) but is not
            in the registry — and is also not itself a registered
            canonical name (e.g. ``"sst_obs"``, ``"analysed_sst"``).
    """
    if fallback not in ("passthrough", "raise"):
        raise ValueError(
            f"fallback must be 'passthrough' or 'raise'; got {fallback!r}."
        )
    cf_to_canonical = _build_cf_index()
    canonical_names = _canonical_name_set()
    candidates: list[str] = (
        [str(k) for k in ds.variables]
        if include_coords
        else [str(k) for k in ds.data_vars]
    )
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    for name in candidates:
        if name in cf_to_canonical:
            canon = cf_to_canonical[name]
            if canon != name:
                mapping[name] = canon
        elif "_" in name and name not in canonical_names:
            # Only flag as unknown if the name looks like a CF standard_name
            # (snake_case multi-word) AND isn't a registered canonical name.
            # Single-word names ("ssh") and underscored canonicals
            # ("sst_obs", "analysed_sst") pass through silently.
            unknown.append(name)
    if unknown and fallback == "raise":
        raise KeyError(
            f"rename_from_cf_standard_names: unknown CF standard_name(s) "
            f"{unknown!r}. Pass fallback='passthrough' to ignore, or "
            "extend the Variable registry."
        )
    return ds.rename(mapping) if mapping else ds


@functools.cache
def _build_cf_index() -> dict[str, str]:
    """Build a ``standard_name → canonical_name`` index from the Variable registry.

    Cached on first call; subsequent calls reuse the dict.
    Only the first registry entry per ``standard_name`` is kept (collision
    deduplication happens naturally when iterating insertion order).
    """
    from xrreader.types import REGISTRY

    index: dict[str, str] = {}
    for var in REGISTRY.values():
        if var.standard_name and var.standard_name not in index:
            index[var.standard_name] = var.name
    return index


@functools.cache
def _canonical_name_set() -> frozenset[str]:
    """Set of canonical short names from the Variable registry.

    Used to recognize underscored canonicals like ``sst_obs`` so they
    aren't mistakenly flagged as unknown CF standard_names by
    ``rename_from_cf_standard_names(..., fallback="raise")``.
    """
    from xrreader.types import REGISTRY

    return frozenset(var.name for var in REGISTRY.values())
