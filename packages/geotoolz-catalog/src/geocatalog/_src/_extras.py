"""Optional extras: one import helper, one install-hint format."""

from __future__ import annotations

import importlib
from types import ModuleType


def install_hint(extra: str) -> str:
    """The pip command that installs ``extra`` (``geotoolz-catalog[extra]``)."""
    return f"pip install 'geotoolz-catalog[{extra}]'"


def missing_extra(
    feature: str, extra: str, *, packages: str | None = None
) -> ImportError:
    """An `ImportError` saying which extra ``feature`` needs and how to get it.

    Args:
        feature: What the user tried to use, e.g. ``"`STACSource`"``.
        extra: The ``geotoolz-catalog`` extra that provides it.
        packages: The underlying distributions, offered as an alternative
            install (``pip install <packages>``).
    """
    message = (
        f"{feature} requires the [{extra}] extra; install with `{install_hint(extra)}`"
    )
    if packages:
        message += f" (or `pip install {packages}`)"
    return ModuleNotFoundError(message + ".")


def require_extra(module: str, extra: str, *, feature: str | None = None) -> ModuleType:
    """Import ``module`` or raise `missing_extra` naming ``extra``."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise missing_extra(feature or f"`{module}`", extra) from exc


__all__ = ["install_hint", "missing_extra", "require_extra"]
