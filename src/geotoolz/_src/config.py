"""Shared ``get_config()`` coercion helpers.

Every Operator family emits its constructor parameters through
``get_config()`` so pipelines round-trip through YAML / hydra-zen.
Those serialisers only accept JSON builtins, so config values need
coercing: numpy scalars and arrays, datetimes, paths, and nested
containers thereof. :func:`jsonable` is the one recursive coercion,
replacing the per-module ``_jsonable`` / ``_to_jsonable`` /
``_stat_as_jsonable`` variants that had drifted apart.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import PurePath
from typing import Any

import numpy as np


__all__ = ["jsonable"]


def jsonable(value: Any) -> Any:
    """Recursively coerce a config value into JSON/YAML-safe builtins.

    Args:
        value: Any ``get_config()`` leaf or container. Dicts and
            lists/tuples are converted recursively (tuples become lists,
            per strict-JSON convention); numpy arrays become (nested)
            lists of Python scalars; numpy scalars become their Python
            equivalents; ``datetime`` / ``date`` become ISO-8601 strings;
            paths become strings. Everything else passes through.

    Returns:
        A structure of dicts, lists, and JSON-safe scalars.
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    return value
