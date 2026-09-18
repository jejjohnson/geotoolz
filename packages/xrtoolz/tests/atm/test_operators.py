"""Layer-1 wrapper tests for :mod:`xrtoolz.atm` and :mod:`xrtoolz.atm.gas.ch4`.

Every wrapper must map over a two-leaf ``DataTree`` leaf-by-leaf and
return a JSON-serialisable ``get_config()`` whose keys are constructor
arguments.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrtoolz.atm import (
    ColumnIntegral,
    HypsometricHeight,
    PBLHeightBulkRichardson,
    WindComponents,
    WindDirection,
    WindSpeed,
    column_integral,
    hypsometric_height,
    pbl_height_bulk_richardson,
    wind_components,
    wind_direction,
    wind_speed,
)
from xrtoolz.atm.gas.ch4 import (
    ApplyColumnAveragingKernel,
    DryAirColumn,
    MixingRatioToColumn,
    apply_column_averaging_kernel,
    dry_air_column,
    mixing_ratio_to_column,
)


@pytest.fixture
def ds_met() -> xr.Dataset:
    z = np.linspace(0.0, 2000.0, 21)
    p = 1.0e5 * np.exp(-z / 8000.0)
    return xr.Dataset(
        {
            "u": (
                ("level", "x"),
                5.0 * np.sqrt(z / 1000.0)[:, None] + np.arange(2)[None, :],
            ),
            "v": (("level", "x"), np.full((z.size, 2), 1.0)),
            "theta_v": (
                ("level", "x"),
                np.broadcast_to(300.0 + 0.005 * z[:, None], (z.size, 2)).copy(),
            ),
            "temperature": (("level", "x"), np.full((z.size, 2), 270.0)),
            "pressure": (
                ("level", "x"),
                np.broadcast_to(p[:, None], (z.size, 2)).copy(),
            ),
            "sp": ("x", [1.0e5, 1.0e5]),
            "wind_speed": (("level", "x"), np.full((z.size, 2), 4.0)),
            "wind_direction": (("level", "x"), np.full((z.size, 2), 225.0)),
        },
        coords={"level": np.arange(z.size), "z_agl": ("level", z)},
    )


@pytest.fixture
def ds_ch4() -> xr.Dataset:
    p = np.linspace(1.0e5, 0.0, 12)
    return xr.Dataset(
        {
            "x": (("layer", "x"), np.full((12, 2), 1900.0)),
            "xa": (("layer", "x"), np.full((12, 2), 1850.0)),
            "ak": ("layer", np.linspace(0.6, 1.05, 12)),
            "h": ("layer", np.full(12, 1.0 / 12.0)),
            "chi": (("layer", "x"), np.full((12, 2), 1.9e-6)),
            "sp": ("x", [1.0e5, 9.5e4]),
        },
        coords={"layer": np.arange(12), "pressure": ("layer", p)},
    )


CASES = [
    ("met", WindSpeed(), lambda ds: wind_speed(ds), "wind_speed"),
    (
        "met",
        WindDirection(convention="to"),
        lambda ds: wind_direction(ds, convention="to"),
        "wind_direction",
    ),
    (
        "met",
        WindComponents(u="u2", v="v2"),
        lambda ds: wind_components(ds, u="u2", v="v2"),
        "u2",
    ),
    (
        "met",
        ColumnIntegral("theta_v", dim="level", coord="z_agl", method="sum"),
        lambda ds: ds.assign(
            theta_v_column=column_integral(
                ds["theta_v"], dim="level", coord="z_agl", method="sum"
            )
        ),
        "theta_v_column",
    ),
    ("met", HypsometricHeight(), lambda ds: hypsometric_height(ds), "height"),
    (
        "met",
        PBLHeightBulkRichardson("theta_v"),
        lambda ds: pbl_height_bulk_richardson(ds, theta_v="theta_v"),
        "pbl_height",
    ),
    (
        "ch4",
        ApplyColumnAveragingKernel("x", "xa", "ak", "h"),
        lambda ds: apply_column_averaging_kernel(
            ds, profile="x", prior="xa", averaging_kernel="ak", pressure_weights="h"
        ),
        "xch4_smoothed",
    ),
    ("ch4", DryAirColumn(), lambda ds: dry_air_column(ds), "dry_air_column"),
    (
        "ch4",
        MixingRatioToColumn("chi", level="layer"),
        lambda ds: mixing_ratio_to_column(ds, vmr="chi", level="layer"),
        "column",
    ),
]
IDS = [type(case[1]).__name__ for case in CASES]


@pytest.fixture
def datasets(ds_met, ds_ch4) -> dict[str, xr.Dataset]:
    return {"met": ds_met, "ch4": ds_ch4}


@pytest.mark.parametrize(("key", "op", "reference", "output"), CASES, ids=IDS)
def test_operator_matches_primitive_and_maps_over_datatree(
    datasets, key, op, reference, output
):
    ds = datasets[key]
    expected = reference(ds)[output]
    xr.testing.assert_allclose(op(ds)[output], expected)

    tree = xr.DataTree.from_dict({"a": ds, "b": ds * 1.0})
    out = op(tree)
    assert isinstance(out, xr.DataTree)
    assert set(out.children) == {"a", "b"}
    for leaf in ("a", "b"):
        xr.testing.assert_allclose(out[leaf].dataset[output], expected)


@pytest.mark.parametrize(("key", "op", "reference", "output"), CASES, ids=IDS)
def test_get_config_json_round_trips(key, op, reference, output):
    cfg = op.get_config()
    assert json.loads(json.dumps(cfg)) == cfg
    rebuilt = type(op)(**cfg)
    assert rebuilt.get_config() == cfg
    assert type(op).__name__ in repr(op)
