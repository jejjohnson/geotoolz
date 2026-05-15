"""Smoke test for hydra-zen ``builds()`` round-trip.

Per `geotoolz.md` §6.6, every YAML-safe Operator's ``get_config()``
should round-trip through hydra-zen. We test a handful of representative
ops — full coverage lives in a separate integration suite once domain
operators land.

The whole module is gated by ``importorskip`` so environments without
the optional ``[hydra]`` extra skip cleanly.
"""

from __future__ import annotations

import pytest


hydra_zen = pytest.importorskip("hydra_zen")

from geotoolz.core import Identity, Operator, Sequential, ShapeTrace


class _Scale(Operator):
    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def _apply(self, x: float) -> float:
        return x * self.factor

    def get_config(self) -> dict:
        return {"factor": self.factor}


def test_builds_roundtrips_simple_operator() -> None:
    op = _Scale(factor=2.5)
    cfg = hydra_zen.builds(_Scale, **op.get_config())
    restored = hydra_zen.instantiate(cfg)
    assert restored.factor == 2.5
    assert restored(4.0) == 10.0


def test_builds_roundtrips_identity() -> None:
    cfg = hydra_zen.builds(Identity, **Identity().get_config())
    restored = hydra_zen.instantiate(cfg)
    assert isinstance(restored, Identity)
    assert restored(123) == 123


def test_builds_roundtrips_shape_trace() -> None:
    op = ShapeTrace(mode="diff_only")
    cfg = hydra_zen.builds(ShapeTrace, **op.get_config())
    restored = hydra_zen.instantiate(cfg)
    assert restored.mode == "diff_only"


def test_sequential_get_config_lists_inner_class_names() -> None:
    """`Sequential.get_config()` returns a description suitable for YAML
    *authoring*, even if Sequential itself isn't built via ``builds()``
    (that's a higher-level wiring concern we defer)."""
    pipe = Sequential([_Scale(2.0), _Scale(3.0)])
    cfg = pipe.get_config()
    assert [op["class"] for op in cfg["operators"]] == ["_Scale", "_Scale"]
    assert [op["config"] for op in cfg["operators"]] == [
        {"factor": 2.0},
        {"factor": 3.0},
    ]
