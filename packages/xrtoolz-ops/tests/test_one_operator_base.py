"""Gate C1: every public xrops operator subclasses the one base, xrops.Operator."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pipekit

import xrops


def _public_operators() -> dict[str, type]:
    found: dict[str, type] = {}
    for info in pkgutil.walk_packages(xrops.__path__, prefix="xrops."):
        if "._src" in info.name or info.name.startswith("xrops._"):
            continue
        module = importlib.import_module(info.name)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if inspect.isclass(obj) and issubclass(obj, pipekit.Operator):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return found


def test_every_public_operator_is_an_xrops_operator() -> None:
    operators = _public_operators()
    assert len(operators) > 100  # the walk found the families
    strays = sorted(
        k for k, cls in operators.items() if not issubclass(cls, xrops.Operator)
    )
    assert not strays, f"not on xrops.Operator: {strays}"
