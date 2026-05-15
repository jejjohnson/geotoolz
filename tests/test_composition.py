"""Tests for Fanout."""

from __future__ import annotations

import pytest

from geotoolz.core import Fanout, Operator


class _Add(Operator):
    def __init__(self, n: int) -> None:
        self.n = n

    def _apply(self, x: int) -> int:
        return x + self.n

    def get_config(self) -> dict:
        return {"n": self.n}


class TestFanout:
    def test_one_input_multiple_outputs(self) -> None:
        op = Fanout({"a": _Add(1), "b": _Add(10), "c": _Add(100)})
        result = op(0)
        assert result == {"a": 1, "b": 10, "c": 100}

    def test_empty_branches_rejected(self) -> None:
        with pytest.raises(TypeError, match="at least one"):
            Fanout({})

    def test_non_operator_branch_rejected(self) -> None:
        with pytest.raises(TypeError, match="'bad'"):
            Fanout({"bad": "not an operator"})  # type: ignore[dict-item]

    def test_config_records_branch_classes(self) -> None:
        op = Fanout({"a": _Add(1), "b": _Add(2)})
        cfg = op.get_config()
        assert set(cfg["branches"]) == {"a", "b"}
        assert cfg["branches"]["a"]["class"] == "_Add"
        assert cfg["branches"]["a"]["config"] == {"n": 1}

    def test_not_forbid_in_yaml(self) -> None:
        # Fanout itself holds no closures — its branches may, but Fanout
        # is YAML-safe.
        assert Fanout.forbid_in_yaml is False
