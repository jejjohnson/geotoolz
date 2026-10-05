"""Enforce the scoped-package docstring bar on xrsklearn's public API.

ruff's pydocstyle rules treat everything under ``_src/`` as private, so
they never flag a missing or thin docstring on the public classes that
live there. This test walks the public surface instead: every public
class and method needs a docstring with a runnable ``Example:`` (``>>>``,
executed by ``make test-doctest``), ``Args:`` when it takes parameters,
and ``Returns:`` when it returns something.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import xrsklearn
from xrsklearn._src.accessor import (
    SklearnDataArrayAccessor,
    SklearnDatasetAccessor,
    SklearnDataTreeAccessor,
)


CLASSES: list[type] = [
    xrsklearn.XarrayEstimator,
    xrsklearn.SklearnOp,
    SklearnDataArrayAccessor,
    SklearnDatasetAccessor,
    SklearnDataTreeAccessor,
]


def _public_methods(cls: type) -> Iterator[tuple[str, Callable[..., Any]]]:
    """Public methods defined in xrsklearn (not inherited from sklearn / pipekit)."""
    seen: set[str] = set()
    for klass in cls.__mro__:
        if not klass.__module__.startswith("xrsklearn"):
            continue
        for name, member in vars(klass).items():
            if name.startswith("_") or name in seen:
                continue
            fn = getattr(member, "fn", member)  # unwrap sklearn's available_if
            if inspect.isfunction(fn):
                seen.add(name)
                yield name, fn


METHODS = [
    pytest.param(fn, id=f"{cls.__name__}.{name}")
    for cls in CLASSES
    for name, fn in _public_methods(cls)
]


def _params(fn: Callable[..., Any]) -> list[str]:
    return [p for p in inspect.signature(fn).parameters if p != "self"]


def _has_runnable_example(doc: str) -> bool:
    return "Example:" in doc and ">>>" in doc


def test_every_public_symbol_is_covered() -> None:
    exported = {
        name for name in xrsklearn.__all__ if inspect.isclass(getattr(xrsklearn, name))
    }
    assert exported <= {cls.__name__ for cls in CLASSES}


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_class_docstring(cls: type) -> None:
    doc = inspect.getdoc(cls) or ""
    assert _has_runnable_example(doc), f"{cls.__name__} needs a runnable Example"
    if cls.__init__ is not object.__init__ and _params(cls.__init__):
        if cls.__module__.endswith("accessor"):
            return  # accessor __init__ takes the xarray object, documented by xarray
        assert "Args:" in doc, f"{cls.__name__} needs an Args section"


@pytest.mark.parametrize("fn", METHODS)
def test_method_docstring(fn: Callable[..., Any]) -> None:
    doc = inspect.getdoc(fn) or ""
    name = fn.__qualname__
    assert doc, f"{name} has no docstring"
    assert _has_runnable_example(doc), f"{name} needs a runnable Example"
    if _params(fn):
        assert "Args:" in doc, f"{name} needs an Args section"
    returns = inspect.signature(fn).return_annotation
    if returns not in (None, "None", inspect.Signature.empty):
        assert "Returns:" in doc, f"{name} needs a Returns section"
