"""Install hints for optional extras."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def install_hint(extra: str) -> str:
    """The ``pip install`` command for a ``geotoolz-products`` extra.

    Examples:
        >>> install_hint("operators")
        "pip install 'geotoolz-products[operators]'"
    """
    return f"pip install 'geotoolz-products[{extra}]'"


def missing_extra(feature: str, extra: str) -> ImportError:
    """An ``ImportError`` naming the extra that ``feature`` needs."""
    return ImportError(f"{feature} needs the [{extra}] extra: {install_hint(extra)}")


def require(module: str, feature: str, extra: str) -> ModuleType:
    """Import ``module``, or raise an ``ImportError`` naming the extra.

    The one way optional dependencies are imported: at use time, so the
    package installs without them, and with an error that tells the user
    which extra to install.

    Args:
        module: Module to import (``"h5py"``, ``"geotoolz.viz"``).
        feature: What needs it, for the message (``"goes.TrueColor"``).
        extra: The ``geotoolz-products`` extra that provides it.

    Raises:
        ImportError: ``module`` is not installed; names ``extra``.

    Examples:
        >>> require("json", "a feature", "none").__name__
        'json'
    """
    try:
        return import_module(module)
    except ImportError as exc:
        raise missing_extra(feature, extra) from exc
