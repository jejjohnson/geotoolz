"""Lazy import of optional-extra dependencies with an actionable error.

Families that need a third-party package outside the base install (e.g.
``viz`` → matplotlib, ``learn`` → scikit-learn / joblib, ``io.WriteZarr``
→ zarr) import it through :func:`import_optional` at call time, so
``import geoops`` works on a base install and only *using* the
operator fails — with an ``ImportError`` that names the extra to add.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any


def import_optional(module: str, extra: str, *, feature: str) -> ModuleType:
    """Import ``module``, or raise an ``ImportError`` naming the extra.

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
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires the optional dependency {module.split('.')[0]!r}, "
            f"which is not installed. Install it with "
            f"`pip install 'geotoolz-ops[{extra}]'`."
        ) from exc


def optional_root(module: str, extra: str, *, feature: str) -> ModuleType:
    """``import a.b`` lazily: import ``module`` and return its top package ``a``.

    Example:
        >>> optional_root("json.decoder", "base", feature="doc").__name__
        'json'
    """
    import_optional(module, extra, feature=feature)
    return sys.modules[module.split(".")[0]]


def optional_name(module: str, name: str, extra: str, *, feature: str) -> Any:
    """``from module import name`` lazily, with the extra's install hint.

    ``name`` may be an attribute of ``module`` or one of its submodules
    (``from georeader import read``).

    Example:
        >>> optional_name("json", "decoder", "base", feature="doc").__name__
        'json.decoder'
    """
    parent = import_optional(module, extra, feature=feature)
    try:
        return getattr(parent, name)
    except AttributeError:
        return import_optional(f"{module}.{name}", extra, feature=feature)
