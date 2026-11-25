"""Lazy import of optional-extra dependencies with an actionable error.

Families that need a third-party package outside the base install (e.g.
``viz`` → matplotlib, ``learn`` → scikit-learn / joblib, ``io.WriteZarr``
→ zarr) import it through :func:`optional_module` at call time, so
``import geoops`` works on a base install and only *using* the
operator fails — with an ``ImportError`` that names the extra to add.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from pipekit_ops import import_optional


def optional_module(module: str, extra: str, *, feature: str) -> ModuleType:
    """Import ``module``, or raise an ``ImportError`` naming the extra.

    It is `pipekit_ops.import_optional` with geotoolz-ops's install command. A module
    that is installed and fails while it is imported raises its own error.

    Args:
        module: Dotted module name to import (e.g. ``"sklearn.impute"``).
        extra: The geotoolz extra that provides it (e.g. ``"learn"``).
        feature: What needs it, for the message (e.g. ``"ModelOp.save"``).

    Returns:
        The imported module.

    Raises:
        ImportError: ``module`` is not installed; the message names the
            ``geotoolz-ops[extra]`` install command.
    """
    return import_optional(module, install=f"geotoolz-ops[{extra}]", needed_by=feature)


def optional_root(module: str, extra: str, *, feature: str) -> ModuleType:
    """``import a.b`` lazily: import ``module`` and return its top package ``a``.

    Example:
        >>> optional_root("json.decoder", "base", feature="doc").__name__
        'json'
    """
    optional_module(module, extra, feature=feature)
    return sys.modules[module.split(".")[0]]


def optional_name(module: str, name: str, extra: str, *, feature: str) -> Any:
    """``from module import name`` lazily, with the extra's install hint.

    ``name`` may be an attribute of ``module`` or one of its submodules
    (``from georeader import read``).

    Example:
        >>> optional_name("json", "decoder", "base", feature="doc").__name__
        'json.decoder'
    """
    parent = optional_module(module, extra, feature=feature)
    try:
        return getattr(parent, name)
    except AttributeError:
        return optional_module(f"{module}.{name}", extra, feature=feature)
