"""Tests for the operator combinators ``Augment``, ``Tap``, ``ApplyToEach``."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xr_toolz import ApplyToEach, Augment, Operator, Sequential, Tap


# ---------- helper ops -----------------------------------------------------


class _AddDoubled(Operator):
    """Toy op: returns a Dataset with a single ``doubled`` variable."""

    def __init__(self, variable: str = "x") -> None:
        self.variable = variable

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return xr.Dataset({f"{self.variable}_doubled": ds[self.variable] * 2})

    def get_config(self) -> dict[str, str]:
        return {"variable": self.variable}


class _ReturnsDataArray(Operator):
    """Toy op that returns a DataArray, not a Dataset (for type-error tests)."""

    def _apply(self, ds: xr.Dataset) -> xr.DataArray:
        return ds["x"] * 2


@pytest.fixture
def ds() -> xr.Dataset:
    return xr.Dataset(
        {
            "x": (("t",), np.arange(4, dtype=float)),
            "y": (("t",), np.arange(4, dtype=float) + 10.0),
        },
        coords={"t": np.arange(4)},
    )


# ---------- Augment --------------------------------------------------------


def test_augment_merges_inner_output(ds: xr.Dataset) -> None:
    out = Augment(_AddDoubled("x"))(ds)
    assert set(out.data_vars) == {"x", "y", "x_doubled"}
    np.testing.assert_array_equal(out["x_doubled"].values, ds["x"].values * 2)


def test_augment_preserves_input_variables(ds: xr.Dataset) -> None:
    out = Augment(_AddDoubled("y"))(ds)
    np.testing.assert_array_equal(out["x"].values, ds["x"].values)
    np.testing.assert_array_equal(out["y"].values, ds["y"].values)


def test_augment_inside_sequential_threads_outputs(ds: xr.Dataset) -> None:
    pipe = Sequential([Augment(_AddDoubled("x")), Augment(_AddDoubled("x_doubled"))])
    out = pipe(ds)
    assert "x_doubled_doubled" in out.data_vars
    np.testing.assert_array_equal(out["x_doubled_doubled"].values, ds["x"].values * 4)


def test_augment_rejects_non_operator() -> None:
    with pytest.raises(TypeError, match="Operator"):
        Augment(lambda ds: ds)  # type: ignore[arg-type]


def test_augment_rejects_non_dataset_inner_output(ds: xr.Dataset) -> None:
    with pytest.raises(TypeError, match="Dataset"):
        Augment(_ReturnsDataArray())(ds)


def test_augment_get_config_is_json_safe_introspection() -> None:
    """get_config() emits a JSON-safe introspection record.

    Augment's config is *not* constructor-replayable — the record
    holds a serialized ``{"class", "config"}`` dict where the
    constructor expects a live Operator instance. We assert only the
    shape needed for printing/logging/diffing pipeline structure.
    """
    op = Augment(_AddDoubled("x"))
    cfg = op.get_config()
    payload = json.loads(json.dumps(cfg))
    assert payload == {"inner": {"class": "_AddDoubled", "config": {"variable": "x"}}}


def test_augment_config_is_not_constructor_replayable() -> None:
    """Document the known limitation: Augment(**get_config()) fails."""
    op = Augment(_AddDoubled("x"))
    cfg = op.get_config()
    with pytest.raises(TypeError):
        Augment(**cfg)  # type: ignore[arg-type]


# ---------- Tap ------------------------------------------------------------


def test_tap_passes_input_through(ds: xr.Dataset) -> None:
    out = Tap(lambda _ds: None)(ds)
    xr.testing.assert_identical(out, ds)


def test_tap_invokes_side_effect(ds: xr.Dataset) -> None:
    seen: list[xr.Dataset] = []
    Tap(seen.append)(ds)
    assert len(seen) == 1
    xr.testing.assert_identical(seen[0], ds)


def test_tap_uses_callable_name_when_no_name_given(ds: xr.Dataset) -> None:
    def my_logger(_ds: xr.Dataset) -> None:
        pass

    op = Tap(my_logger)
    assert op.name == "my_logger"
    cfg = op.get_config()
    assert cfg == {"name": "my_logger", "side_effect": "<callable>"}


def test_tap_explicit_name_overrides_default(ds: xr.Dataset) -> None:
    op = Tap(lambda d: None, name="dump_qc")
    assert op.name == "dump_qc"


def test_tap_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        Tap(42)  # type: ignore[arg-type]


def test_tap_inside_sequential_does_not_alter_data(ds: xr.Dataset) -> None:
    seen: list[int] = []
    pipe = Sequential(
        [
            Tap(lambda d: seen.append(len(d.data_vars))),
            Augment(_AddDoubled("x")),
        ]
    )
    out = pipe(ds)
    assert seen == [2]
    assert "x_doubled" in out.data_vars


# ---------- ApplyToEach ----------------------------------------------------


def test_apply_to_each_runs_one_inner_per_value(ds: xr.Dataset) -> None:
    op = ApplyToEach(_AddDoubled("x"), kwarg="variable", values=["x", "y"])
    out = op(ds)
    assert "x_doubled" in out.data_vars
    assert "y_doubled" in out.data_vars
    np.testing.assert_array_equal(out["x_doubled"].values, ds["x"].values * 2)
    np.testing.assert_array_equal(out["y_doubled"].values, ds["y"].values * 2)


def test_apply_to_each_rejects_non_operator_prototype() -> None:
    with pytest.raises(TypeError, match="prototype"):
        ApplyToEach("not an op", kwarg="variable", values=["x"])  # type: ignore[arg-type]


def test_apply_to_each_rejects_unknown_kwarg() -> None:
    with pytest.raises(ValueError, match="kwarg 'nonsense'"):
        ApplyToEach(_AddDoubled("x"), kwarg="nonsense", values=["x"])


def test_apply_to_each_rejects_non_dataset_inner_output(ds: xr.Dataset) -> None:
    op = ApplyToEach(
        _AddDoubled("x"),
        kwarg="variable",
        values=["x"],
    )
    # Swap the prototype to one that returns a DataArray so the type
    # check inside ``_apply`` fires — _ReturnsDataArray has no config
    # keys, so a fresh ApplyToEach build would also reject the kwarg.
    op.prototype = _ReturnsDataArray()  # type: ignore[assignment]
    with pytest.raises((TypeError, ValueError)):
        op(ds)


def test_apply_to_each_rejects_composite_prototype() -> None:
    """Composite prototypes (Sequential, etc.) fail the leaf-config contract."""
    inner = Sequential([_AddDoubled("x")])
    with pytest.raises(TypeError, match="cannot be reconstructed"):
        ApplyToEach(inner, kwarg="operators", values=[[_AddDoubled("y")]])


def test_apply_to_each_rejects_augment_prototype() -> None:
    """Augment is also a combinator with a serialized inner record."""
    inner = Augment(_AddDoubled("x"))
    with pytest.raises(TypeError, match="cannot be reconstructed"):
        ApplyToEach(inner, kwarg="inner", values=[_AddDoubled("y")])


def test_apply_to_each_get_config_is_json_safe_introspection() -> None:
    """get_config() emits a JSON-safe introspection record.

    Like Augment, the prototype's nested config is a serialized
    ``{"class", "config"}`` dict, not a live Operator. The config is
    suitable for printing/logging but not for ``ApplyToEach(**cfg)``
    reconstruction.
    """
    op = ApplyToEach(_AddDoubled("x"), kwarg="variable", values=["x", "y"])
    cfg = op.get_config()
    payload = json.loads(json.dumps(cfg))
    assert payload["prototype"]["class"] == "_AddDoubled"
    assert payload["kwarg"] == "variable"
    assert payload["values"] == ["x", "y"]


def test_apply_to_each_config_is_not_constructor_replayable() -> None:
    """Document the known limitation: ApplyToEach(**get_config()) fails."""
    op = ApplyToEach(_AddDoubled("x"), kwarg="variable", values=["x", "y"])
    cfg = op.get_config()
    with pytest.raises(TypeError):
        ApplyToEach(**cfg)  # type: ignore[arg-type]


def test_apply_to_each_inside_sequential(ds: xr.Dataset) -> None:
    pipe = Sequential(
        [
            ApplyToEach(_AddDoubled("x"), kwarg="variable", values=["x", "y"]),
        ]
    )
    out = pipe(ds)
    assert {"x_doubled", "y_doubled"} <= set(out.data_vars)


def test_apply_to_each_threads_results_through_sweep(ds: xr.Dataset) -> None:
    """Regression: each rebuilt op must run against the threaded result.

    A later ``values`` entry can consume a variable produced by an
    earlier one — this is what makes ``ApplyToEach`` equivalent to
    ``Sequential([Augment(prototype.__class__(...)) for v in values])``.
    Sweeping ``values=["x", "x_doubled"]`` over ``_AddDoubled`` only
    works if the second pass sees the first pass's ``x_doubled``
    output; otherwise it raises ``KeyError`` because the original
    input has no ``x_doubled``.
    """
    op = ApplyToEach(_AddDoubled("x"), kwarg="variable", values=["x", "x_doubled"])
    out = op(ds)
    assert "x_doubled" in out.data_vars
    assert "x_doubled_doubled" in out.data_vars
    np.testing.assert_array_equal(out["x_doubled_doubled"].values, ds["x"].values * 4)

    # Equivalence check: building the same fan-out manually with
    # ``Sequential([Augment(...)])`` produces the same dataset.
    manual = Sequential(
        [
            Augment(_AddDoubled("x")),
            Augment(_AddDoubled("x_doubled")),
        ]
    )(ds)
    xr.testing.assert_identical(out, manual)
