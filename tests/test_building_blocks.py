"""Tests for Identity, Const, Lambda, Sink."""

from __future__ import annotations

from geotoolz.core import Const, Identity, Lambda, Operator, Sink


class TestIdentity:
    def test_returns_input_unchanged(self) -> None:
        op = Identity()
        for val in (0, "x", [1, 2], None):
            assert op(val) is val

    def test_repr_and_config(self) -> None:
        op = Identity()
        assert repr(op) == "Identity()"
        assert op.get_config() == {}

    def test_not_forbid_in_yaml(self) -> None:
        assert Identity.forbid_in_yaml is False


class TestConst:
    def test_returns_fixed_value(self) -> None:
        op = Const(42)
        assert op(0) == 42
        assert op("anything") == 42
        # Const should accept being called with no args too — useful in
        # graph-eval contexts where an input has no upstream.
        assert op() == 42

    def test_config_includes_type(self) -> None:
        cfg = Const("hello").get_config()
        assert cfg["value_type"] == "str"


class TestLambda:
    def test_applies_callable(self) -> None:
        op = Lambda(lambda x: x * 3, name="triple")
        assert op(4) == 12

    def test_repr_uses_name(self) -> None:
        op = Lambda(lambda x: x, name="my_lambda")
        assert "my_lambda" in repr(op)

    def test_forbid_in_yaml(self) -> None:
        assert Lambda.forbid_in_yaml is True

    def test_is_operator(self) -> None:
        assert isinstance(Lambda(lambda x: x), Operator)


class TestSink:
    def test_side_effect_runs_and_input_passes_through(self) -> None:
        seen: list[int] = []
        op = Sink(seen.append, name="collector")
        out = op(5)
        assert out == 5
        assert seen == [5]

    def test_forbid_in_yaml(self) -> None:
        assert Sink.forbid_in_yaml is True

    def test_repr_uses_name(self) -> None:
        op = Sink(lambda _gt: None, name="checkpoint")
        assert "checkpoint" in repr(op)
