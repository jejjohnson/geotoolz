"""Lazy import of optional-extra dependencies with an actionable error.

Families that need a third-party package outside the base install (e.g.
``viz`` → matplotlib, ``learn`` → scikit-learn / joblib, ``io.WriteZarr``
→ zarr) import it through :func:`import_optional` at call time, so
``import geotoolz`` works on a base install and only *using* the
operator fails — with an ``ImportError`` that names the extra to add.
"""

from __future__ import annotations

import importlib
from types import ModuleType


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
            ``geotoolz[extra]`` install command.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires the optional dependency {module.split('.')[0]!r}, "
            f"which is not installed. Install it with "
            f"`pip install 'geotoolz[{extra}]'`."
        ) from exc
