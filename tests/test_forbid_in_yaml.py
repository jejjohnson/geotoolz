"""Verify the ``forbid_in_yaml`` flag is set on operators that hold
closures (and unset on the others).

The flag is documented for future YAML loader enforcement (per
``tips_n_tricks.md`` §"Round-trip discipline"). No runtime enforcement
in v0.1 — the test just locks the contract for each shipped operator.
"""

from __future__ import annotations

import pytest

from geotoolz.core import (
    Branch,
    Const,
    Fanout,
    Graph,
    Identity,
    Lambda,
    ModelOp,
    Operator,
    Sequential,
    ShapeTrace,
    Sink,
    Snapshot,
    Switch,
    Tap,
)


YAML_FORBIDDEN: list[type[Operator]] = [
    Tap,
    Lambda,
    Sink,
    Branch,
    Switch,
    ModelOp,
]

YAML_SAFE: list[type[Operator]] = [
    Identity,
    Const,
    Sequential,
    ShapeTrace,
    Fanout,
    Graph,
]


@pytest.mark.parametrize("cls", YAML_FORBIDDEN)
def test_forbidden_in_yaml(cls: type[Operator]) -> None:
    assert cls.forbid_in_yaml is True, (
        f"{cls.__name__} holds runtime closures; must set forbid_in_yaml = True"
    )


@pytest.mark.parametrize("cls", YAML_SAFE)
def test_safe_in_yaml(cls: type[Operator]) -> None:
    assert cls.forbid_in_yaml is False, (
        f"{cls.__name__} does not hold closures; forbid_in_yaml must be False"
    )


def test_snapshot_controller_not_an_operator() -> None:
    """`Snapshot` is a controller, not an Operator; the flag concept
    doesn't apply to it."""
    assert not isinstance(Snapshot(), Operator)
