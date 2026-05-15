"""Pickleability tests — every YAML-safe operator must round-trip.

The §11.2 sharp edge says "operator graph as audit artifact" depends on
pickle working. Lambdas, closures, and unbound methods break silently —
catch the regression here, early, rather than at deploy time.

Operators flagged ``forbid_in_yaml = True`` are expected to hold
closures and are not tested for pickleability.
"""

from __future__ import annotations

import pickle

import pytest

from geotoolz.core import (
    Const,
    Fanout,
    Graph,
    Identity,
    Input,
    Operator,
    Sequential,
    ShapeTrace,
    Snapshot,
)


class _Inc(Operator):
    def _apply(self, x: int) -> int:
        return x + 1

    def get_config(self) -> dict:
        return {}


YAML_SAFE_INSTANCES: list[Operator] = [
    Identity(),
    Const(42),
    ShapeTrace(),
    Sequential([Identity(), Identity()]),
    _Inc(),
]


@pytest.mark.parametrize("op", YAML_SAFE_INSTANCES)
def test_yaml_safe_operators_roundtrip(op: Operator) -> None:
    restored = pickle.loads(pickle.dumps(op))
    assert isinstance(restored, type(op))
    assert restored.get_config() == op.get_config()


def test_sequential_chain_roundtrips() -> None:
    pipe = Sequential([_Inc(), _Inc(), _Inc()])
    restored = pickle.loads(pickle.dumps(pipe))
    assert isinstance(restored, Sequential)
    assert restored(0) == 3


def test_graph_with_diamond_roundtrips() -> None:
    x = Input("x")
    a = _Inc()(x)
    b = _Inc()(a)
    g = Graph(inputs={"x": x}, outputs={"a": a, "b": b})
    restored = pickle.loads(pickle.dumps(g))
    result = restored(x=0)
    assert result == {"a": 1, "b": 2}


def test_fanout_roundtrips() -> None:
    op = Fanout({"a": _Inc(), "b": _Inc()})
    restored = pickle.loads(pickle.dumps(op))
    assert restored(10) == {"a": 11, "b": 11}


def test_snapshot_controller_roundtrips() -> None:
    """Snapshot itself (the controller, not the _SnapshotTap) round-trips."""
    snap = Snapshot()
    pipe = Sequential([snap.at("k")])
    pipe(99)
    assert snap["k"] == 99

    restored_snap = pickle.loads(pickle.dumps(snap))
    assert restored_snap["k"] == 99
