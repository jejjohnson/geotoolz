"""Tests for Branch, Switch."""

from __future__ import annotations

import pytest

from geotoolz.core import Branch, Const, Identity, Operator, Switch


class _Tag(Operator):
    """Returns the constructor's label regardless of input."""

    def __init__(self, label: str) -> None:
        self.label = label

    def _apply(self, _x) -> str:
        return self.label

    def get_config(self) -> dict:
        return {"label": self.label}


class TestBranch:
    def test_predicate_true_runs_if_true(self) -> None:
        op = Branch(
            predicate=lambda x: x > 0, if_true=_Tag("pos"), if_false=_Tag("neg")
        )
        assert op(1) == "pos"
        assert op(-1) == "neg"

    def test_default_if_false_is_identity(self) -> None:
        op = Branch(predicate=lambda x: False, if_true=_Tag("never"))
        assert op(42) == 42  # Identity passthrough

    def test_rejects_non_operator_arms(self) -> None:
        with pytest.raises(TypeError, match="if_true"):
            Branch(predicate=lambda x: True, if_true="not an op")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="if_false"):
            Branch(
                predicate=lambda x: True,
                if_true=Identity(),
                if_false="not an op",  # type: ignore[arg-type]
            )

    def test_forbid_in_yaml(self) -> None:
        assert Branch.forbid_in_yaml is True

    def test_config_includes_arm_classes(self) -> None:
        op = Branch(
            predicate=lambda x: x,
            if_true=_Tag("a"),
            if_false=_Tag("b"),
        )
        cfg = op.get_config()
        assert cfg["if_true"]["class"] == "_Tag"
        assert cfg["if_false"]["class"] == "_Tag"


class TestSwitch:
    def test_dispatches_on_key(self) -> None:
        op = Switch(
            key=lambda gt: gt["sensor"],
            cases={"S2": _Tag("s2"), "Landsat": _Tag("landsat")},
        )
        assert op({"sensor": "S2"}) == "s2"
        assert op({"sensor": "Landsat"}) == "landsat"

    def test_default_identity_when_no_match(self) -> None:
        op = Switch(key=lambda x: x, cases={"a": _Tag("hit")})
        # Unmatched key falls through to Identity by default.
        assert op("missing") == "missing"

    def test_custom_default(self) -> None:
        op = Switch(
            key=lambda x: x,
            cases={"a": _Tag("hit")},
            default=Const("fallback"),
        )
        assert op("missing") == "fallback"

    def test_rejects_non_operator_case(self) -> None:
        with pytest.raises(TypeError, match="case 'a'"):
            Switch(key=lambda x: x, cases={"a": "bad"})  # type: ignore[dict-item]

    def test_rejects_non_operator_default(self) -> None:
        with pytest.raises(TypeError, match="default"):
            Switch(
                key=lambda x: x,
                cases={"a": _Tag("ok")},
                default="bad",  # type: ignore[arg-type]
            )

    def test_forbid_in_yaml(self) -> None:
        assert Switch.forbid_in_yaml is True
