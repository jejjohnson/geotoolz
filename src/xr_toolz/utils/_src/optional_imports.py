"""Optional dependency import helpers."""

from __future__ import annotations

import importlib
from typing import Any


def _require_optional(
    module: str,
    *,
    extra: str,
    feature: str | None = None,
    package: str | None = None,
) -> Any:
    """Import an optional dependency module or raise an install hint."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on optional installs
        package_name = package or module
        name = feature or module
        raise ImportError(
            f"{name} requires {package_name}. "
            f"Install with: pip install 'xr_toolz[{extra}]'"
        ) from exc


__all__ = ["_require_optional"]
