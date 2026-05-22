"""Tests for CF standard_name rename helpers in :mod:`xrtoolz.geo`."""

from __future__ import annotations

import pytest
import xarray as xr

from xrtoolz.geo import (
    rename_from_cf_standard_names,
    rename_to_cf_standard_names,
)
from xrtoolz.geo.operators import RenameFromCFStandardNames, RenameToCFStandardNames


# ---------- helpers -----------------------------------------------------------


def _ds_with_cf_attrs(**kwargs) -> xr.Dataset:
    """Build a tiny Dataset whose data-vars each carry a ``standard_name`` attr."""
    data_vars = {}
    for name, standard_name in kwargs.items():
        da = xr.DataArray([1.0, 2.0], dims="x")
        if standard_name is not None:
            da.attrs["standard_name"] = standard_name
        data_vars[name] = da
    return xr.Dataset(data_vars)


# ---------- rename_to_cf_standard_names ---------------------------------------


def test_rename_to_cf_renames_annotated_vars():
    ds = _ds_with_cf_attrs(
        ssh="sea_surface_height_above_geoid",
        sst="sea_surface_temperature",
    )
    out = rename_to_cf_standard_names(ds)
    assert "sea_surface_height_above_geoid" in out.data_vars
    assert "sea_surface_temperature" in out.data_vars
    assert "ssh" not in out.data_vars
    assert "sst" not in out.data_vars


def test_rename_to_cf_leaves_unannotated_unchanged():
    ds = _ds_with_cf_attrs(ssh="sea_surface_height_above_geoid", myvar=None)
    out = rename_to_cf_standard_names(ds)
    assert "myvar" in out.data_vars


def test_rename_to_cf_already_cf_named_is_noop():
    """If the variable name already equals its standard_name, nothing changes."""
    ds = _ds_with_cf_attrs(
        sea_surface_height_above_geoid="sea_surface_height_above_geoid"
    )
    out = rename_to_cf_standard_names(ds)
    assert "sea_surface_height_above_geoid" in out.data_vars


def test_rename_to_cf_include_coords_false_skips_coords():
    ds = _ds_with_cf_attrs(ssh="sea_surface_height_above_geoid")
    # Add a coord with a standard_name attr
    ds = ds.assign_coords(
        mycoord=xr.DataArray([0.0, 1.0], dims="x", attrs={"standard_name": "depth"})
    )
    out = rename_to_cf_standard_names(ds, include_coords=False)
    # data var was renamed
    assert "sea_surface_height_above_geoid" in out.data_vars
    # coord was not renamed
    assert "mycoord" in out.coords
    assert "depth" not in out.coords


def test_rename_to_cf_include_coords_true_renames_coords():
    ds = _ds_with_cf_attrs(ssh="sea_surface_height_above_geoid")
    ds = ds.assign_coords(
        mycoord=xr.DataArray([0.0, 1.0], dims="x", attrs={"standard_name": "depth"})
    )
    out = rename_to_cf_standard_names(ds, include_coords=True)
    assert "depth" in out.coords
    assert "mycoord" not in out.coords


def test_rename_to_cf_collision_raises():
    ds = _ds_with_cf_attrs(
        a="sea_surface_height_above_geoid",
        b="sea_surface_height_above_geoid",
    )
    with pytest.raises(ValueError, match="sea_surface_height_above_geoid"):
        rename_to_cf_standard_names(ds)


# ---------- rename_from_cf_standard_names -------------------------------------


def test_rename_from_cf_known_name_is_renamed():
    """A registered CF standard_name is renamed to the canonical name."""
    ds = xr.Dataset({"sea_surface_height_above_geoid": xr.DataArray([1.0], dims="x")})
    out = rename_from_cf_standard_names(ds)
    assert "ssh" in out.data_vars
    assert "sea_surface_height_above_geoid" not in out.data_vars


def test_rename_from_cf_already_canonical_passthrough():
    """A single-word canonical name like 'ssh' passes through silently."""
    ds = xr.Dataset({"ssh": xr.DataArray([1.0], dims="x")})
    out = rename_from_cf_standard_names(ds)
    assert "ssh" in out.data_vars


def test_rename_from_cf_unknown_cf_shaped_passthrough():
    """An unknown multi-word name passes through when fallback='passthrough'."""
    ds = xr.Dataset({"unknown_cf_quantity": xr.DataArray([1.0], dims="x")})
    out = rename_from_cf_standard_names(ds, fallback="passthrough")
    assert "unknown_cf_quantity" in out.data_vars


def test_rename_from_cf_unknown_cf_shaped_raise():
    """An unknown multi-word name raises when fallback='raise'."""
    ds = xr.Dataset({"unknown_cf_quantity": xr.DataArray([1.0], dims="x")})
    with pytest.raises(KeyError, match="unknown_cf_quantity"):
        rename_from_cf_standard_names(ds, fallback="raise")


def test_rename_from_cf_single_word_never_raises():
    """Single-word names (already canonical) never raise even with fallback='raise'."""
    ds = xr.Dataset({"ssh": xr.DataArray([1.0], dims="x")})
    out = rename_from_cf_standard_names(ds, fallback="raise")
    assert "ssh" in out.data_vars


def test_rename_from_cf_include_coords_false_skips_coords():
    ds = xr.Dataset({"sea_surface_height_above_geoid": xr.DataArray([1.0], dims="x")})
    ds = ds.assign_coords(eastward_sea_water_velocity=xr.DataArray([0.0], dims="x"))
    out = rename_from_cf_standard_names(ds, include_coords=False)
    # data var renamed
    assert "ssh" in out.data_vars
    # coord not renamed
    assert "eastward_sea_water_velocity" in out.coords


def test_rename_from_cf_include_coords_true_renames_coords():
    ds = xr.Dataset({"sea_surface_height_above_geoid": xr.DataArray([1.0], dims="x")})
    ds = ds.assign_coords(eastward_sea_water_velocity=xr.DataArray([0.0], dims="x"))
    out = rename_from_cf_standard_names(ds, include_coords=True)
    assert "uo" in out.coords
    assert "eastward_sea_water_velocity" not in out.coords


# ---------- round-trip -------------------------------------------------------


def test_round_trip_to_cf_and_back():
    """rename_from_cf(rename_to_cf(ds)) is identity for registry-known vars."""
    ds = _ds_with_cf_attrs(
        ssh="sea_surface_height_above_geoid",
        sst="sea_surface_temperature",
    )
    out_cf = rename_to_cf_standard_names(ds)
    out_back = rename_from_cf_standard_names(out_cf)
    # Variable names should be restored to their canonical forms
    assert "ssh" in out_back.data_vars
    assert "sst" in out_back.data_vars


# ---------- operators --------------------------------------------------------


def test_rename_to_cf_operator_applies():
    ds = _ds_with_cf_attrs(ssh="sea_surface_height_above_geoid")
    op = RenameToCFStandardNames()
    out = op(ds)
    assert "sea_surface_height_above_geoid" in out.data_vars


def test_rename_from_cf_operator_applies():
    ds = xr.Dataset({"sea_surface_height_above_geoid": xr.DataArray([1.0], dims="x")})
    op = RenameFromCFStandardNames()
    out = op(ds)
    assert "ssh" in out.data_vars


def test_rename_to_cf_operator_get_config_round_trip():
    op = RenameToCFStandardNames(include_coords=False)
    cfg = op.get_config()
    assert cfg == {"include_coords": False}
    op2 = RenameToCFStandardNames(**cfg)
    assert op2.include_coords is False


def test_rename_from_cf_operator_get_config_round_trip():
    op = RenameFromCFStandardNames(fallback="raise", include_coords=False)
    cfg = op.get_config()
    assert cfg == {"fallback": "raise", "include_coords": False}
    op2 = RenameFromCFStandardNames(**cfg)
    assert op2.fallback == "raise"
    assert op2.include_coords is False


def test_rename_from_cf_strict_skips_underscored_canonicals():
    """Registered canonical names that legitimately contain underscores
    (e.g. ``ice_conc``, ``air_temperature_daily_mean``) must not be
    flagged as unknown CF standard_names in strict mode — they're
    already canonical even though they contain underscores."""
    ds = xr.Dataset(
        {
            "ice_conc": xr.DataArray([1.0], dims="x"),
            "air_temperature_daily_mean": xr.DataArray([1.0], dims="x"),
        }
    )
    out = rename_from_cf_standard_names(ds, fallback="raise")
    assert "ice_conc" in out.data_vars
    assert "air_temperature_daily_mean" in out.data_vars


def test_rename_from_cf_invalid_fallback_raises():
    ds = xr.Dataset({"unknown_cf_quantity": xr.DataArray([1.0], dims="x")})
    with pytest.raises(ValueError, match="fallback"):
        rename_from_cf_standard_names(ds, fallback="pass_through")


def test_rename_to_cf_existing_target_raises_with_clear_message():
    """If a non-renamed variable already has the target CF name, surface
    a helpful error before xarray's generic rename-collision message."""
    ds = _ds_with_cf_attrs(ssh="sea_surface_height_above_geoid")
    ds["sea_surface_height_above_geoid"] = xr.DataArray([0.0, 0.0], dims="x")
    with pytest.raises(ValueError, match="already exists"):
        rename_to_cf_standard_names(ds)


def test_rename_to_cf_dim_coord_renames_dimension_too():
    """Renaming a dimension coord (e.g. ``lat`` with
    standard_name='latitude') also renames the corresponding dim. This
    test locks in that behavior so accidental dim-renaming doesn't
    regress silently."""
    ds = xr.Dataset(
        {"ssh": (("lat",), [1.0, 2.0])},
        coords={
            "lat": xr.DataArray(
                [0.0, 1.0], dims="lat", attrs={"standard_name": "latitude"}
            )
        },
    )
    out = rename_to_cf_standard_names(ds, include_coords=True)
    assert "latitude" in out.dims
    assert "lat" not in out.dims
