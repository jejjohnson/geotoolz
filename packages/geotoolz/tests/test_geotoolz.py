"""Top-level package smoke tests."""

from __future__ import annotations

from typing import Any

import geotoolz


def test_import() -> None:
    assert geotoolz is not None


def test_version_attribute() -> None:
    assert isinstance(geotoolz.__version__, str)
    assert geotoolz.__version__.count(".") >= 2


def test_core_re_exports_at_top_level() -> None:
    """Public symbols should be reachable as ``gz.X`` (re-exported from pipekit)."""
    import pipekit

    pipekit_names = (
        "Operator",
        "Sequential",
        "Graph",
        "Input",
        "Node",
        "Tap",
        "Snapshot",
        "ShapeTrace",
        "Branch",
        "Switch",
        "Fanout",
        "Identity",
        "Const",
        "Lambda",
        "Sink",
    )
    for name in pipekit_names:
        assert getattr(geotoolz, name) is getattr(pipekit, name), name
    # ModelOp is geotoolz-specific (built on top of pipekit.Operator).
    from geotoolz.learn import ModelOp

    assert geotoolz.ModelOp is ModelOp


def test_from_state_refuses_forbid_in_yaml() -> None:
    """pipekit >= 0.0.2 refuses to rebuild a ``forbid_in_yaml`` operator.

    Guards the pipekit pin: with 0.0.1 this "succeeded" and returned an
    operator whose ``geometries == '<list n=0>'``.
    """
    import pytest
    from pipekit import Operator

    from geotoolz.geom import Rasterize

    op = Rasterize(geometries=[])
    assert type(op).forbid_in_yaml
    with pytest.raises(RuntimeError, match="forbid_in_yaml"):
        Operator.from_state(op.state)


def test_no_operator_overrides_call() -> None:
    """Operators implement ``_apply``; ``Operator.__call__`` owns dispatch.

    Overriding ``__call__`` bypasses graph construction (``Node``) and the
    post-apply hook, so the operator cannot be used with ``Input``.
    """
    from _helpers import all_operator_classes

    offenders = [
        c.__qualname__ for c in all_operator_classes() if "__call__" in vars(c)
    ]
    assert offenders == []


def _multi_input_operators() -> list[Any]:
    from geotoolz.compositing import BlendMatched, StackMatched
    from geotoolz.geom import coregister as co

    return [
        StackMatched(),
        BlendMatched(),
        co.RasterToRasterLike(),
        co.RasterToPoints(),
        co.PointsToRaster(),
        co.RasterToPointCloud(),
        co.PointCloudToRaster(),
        co.VectorToRasterAgg(agg="count"),
    ]


def test_formerly_call_overriding_operators_support_graph_mode() -> None:
    """Regression for #135: these used to override ``__call__``."""
    import inspect

    from pipekit import Input, Node

    for op in _multi_input_operators():
        params = inspect.signature(op._apply).parameters.values()
        n_inputs = sum(1 for p in params if p.default is inspect.Parameter.empty)
        node = op(*(Input(f"x{i}") for i in range(n_inputs)))
        assert isinstance(node, Node), type(op).__name__
        assert node.operator is op


def _families() -> list[Any]:
    from pathlib import Path

    root = Path(geotoolz.__file__).parent
    return sorted(
        p for p in root.iterdir() if (p / "__init__.py").is_file() and p.name != "_src"
    )


def test_family_layout() -> None:
    """Every family follows the canonical two-tier layout (#162).

    ``family/__init__.py`` re-exports only (no ``def`` / ``class``);
    ``family/_src/array.py`` holds the Tier-A numpy primitives and
    ``family/_src/operators.py`` the Tier-B Operators. Optional
    ``family/_src/<topic>.py`` modules hold constants / tables / helpers.
    """
    import ast

    families = _families()
    assert len(families) >= 19
    problems: list[str] = []
    for family in families:
        name = family.name
        if not (family / "_src" / "__init__.py").is_file():
            problems.append(f"{name}: missing _src/")
        for module in ("array.py", "operators.py"):
            if not (family / "_src" / module).is_file():
                problems.append(f"{name}: missing _src/{module}")
        tree = ast.parse((family / "__init__.py").read_text(encoding="utf-8"))
        problems.extend(
            f"{name}/__init__.py defines {node.name!r}"
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        )
    assert problems == []


#: Public operators deliberately *not* re-exported at the top level; the
#: justification for each is in the ``geotoolz/__init__.py`` docstring.
_TOP_LEVEL_EXCEPTIONS = frozenset(
    {
        "geotoolz.augment.Compose",  # reads like pipekit's generic ``compose``
        "geotoolz.io.SinkOperator",  # abstract base; next to pipekit's ``Sink``
        "geotoolz.io.SourceOperator",  # abstract base for custom readers
    }
)
#: Optional-extra modules: ``import geotoolz`` must not require their deps.
_OPTIONAL_MODULES = frozenset({"geotoolz.patch_ops"})


def _public_modules() -> list[Any]:
    """Every public ``geotoolz.*`` module: families and their subnamespaces."""
    import importlib
    import pkgutil

    return [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(geotoolz.__path__, "geotoolz.")
        if not any(part.startswith("_") for part in info.name.split("."))
    ]


def test_public_operators_exported() -> None:
    """Every public Operator (and the SCL / QA constants) is top-level (#164)."""
    from pipekit import Operator

    assert len(geotoolz.__all__) == len(set(geotoolz.__all__))
    modules = _public_modules()
    assert {"geotoolz.geom.coregister", "geotoolz.qa"} <= {m.__name__ for m in modules}
    problems: list[str] = []
    for module in modules:
        names = getattr(module, "__all__", None)
        if names is None:
            problems.append(f"{module.__name__}: no __all__")
            continue
        if len(names) != len(set(names)):
            problems.append(f"{module.__name__}: duplicate names in __all__")
        if module.__name__ in _OPTIONAL_MODULES:
            continue
        for name in names:
            obj = getattr(module, name)
            qualified = f"{module.__name__}.{name}"
            if not (isinstance(obj, type) and issubclass(obj, Operator)):
                continue
            if qualified in _TOP_LEVEL_EXCEPTIONS:
                assert name not in geotoolz.__all__, f"stale exception {qualified}"
            elif getattr(geotoolz, name, None) is not obj:
                problems.append(f"{qualified} is not geotoolz.{name}")
    assert problems == []
    qa_constants = [n for n in geotoolz.qa.__all__ if n.isupper() and "_" in n]
    assert {"SCL_CLOUDS_AND_INVALID", "SENSOR_QA_REGISTRY"} <= set(qa_constants)
    for name in [*qa_constants, "SCL"]:
        assert getattr(geotoolz, name) is getattr(geotoolz.qa, name), name
        assert name in geotoolz.__all__


def test_one_home_per_public_name() -> None:
    """No object is exported from two geotoolz families (#164).

    A package re-exporting its own submodules' names is one home;
    ``plume`` re-exporting ``segment.otsu_threshold`` is two.
    """
    import types

    homes: dict[int, dict[str, str]] = {}
    for module in _public_modules():
        family = module.__name__.split(".")[1]
        for name in module.__all__:
            obj = getattr(module, name)
            if isinstance(obj, types.ModuleType | int | float | str | bool):
                continue
            homes.setdefault(id(obj), {}).setdefault(
                family, f"{module.__name__}.{name}"
            )
    shared = sorted(sorted(w.values()) for w in homes.values() if len(w) > 1)
    assert shared == []


def test_removed_modules_are_gone() -> None:
    """Removed aliases stay removed — no deprecation shims (#164)."""
    import importlib

    import pytest

    for module in ("geotoolz.cloud", "geotoolz.model", "geotoolz.readers"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
    # Readers moved to geotoolz-products (`geoproducts`).
    for name in ("cloud", "model", "readers", "SensorReader"):
        assert not hasattr(geotoolz, name)
        assert name not in geotoolz.__all__
    assert not hasattr(geotoolz.viz, "ToDisplayRange")
    assert geotoolz.ModelOp is geotoolz.learn.ModelOp
