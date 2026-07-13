"""Shared serialization helpers for axis ``get_config`` methods.

Every patcher axis (geometry / sampler / window / aggregation) exposes a
``get_config() -> dict`` for YAML round-trip. Most of those configs are a
plain dump of the dataclass's init fields with a small scalar coercion —
this module centralises that boilerplate:

- `jsonable_scalar` — numpy generic / ``datetime64`` / ``timedelta64`` →
  plain Python value.
- `config_from_fields` — dataclass instance → ``{field: coerced value}``.
- `axis_envelope` — the ``{"class": ..., "config": ...}`` wrapper the
  Patcher-level ``get_config`` methods build around each axis.

Axes whose config is *not* a field dump (e.g. summarising counts like
``{"n_anchors": len(...)}``) keep their hand-written ``get_config``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


def jsonable_scalar(value: Any) -> Any:
    """Coerce a numpy scalar to a YAML-friendly Python value.

    ``datetime64`` / ``timedelta64`` become ``str(value)`` so YAML stays
    portable; other numpy generics unwrap via ``.item()``; everything else
    passes through unchanged.

    Args:
        value: Any scalar-like value.

    Returns:
        The coerced Python scalar, or ``value`` unchanged.
    """
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable(value: Any) -> Any:
    """`jsonable_scalar` extended over sequences — tuples/lists become lists."""
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return jsonable_scalar(value)


def config_from_fields(obj: Any, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Serialise a dataclass axis into its ``get_config`` dict.

    Walks the dataclass's init fields in declaration order (private
    ``init=False`` accumulator state is skipped automatically) and coerces
    each value with `jsonable_scalar`; tuples and lists are rebuilt as
    lists of coerced elements, matching the hand-written convention of
    ``list(self.size)``-style dumps.

    Args:
        obj: A dataclass instance.
        exclude: Field names to omit (e.g. fields needing custom handling).

    Returns:
        ``{field_name: coerced_value}`` in field declaration order.
    """
    return {
        f.name: _jsonable(getattr(obj, f.name))
        for f in dataclasses.fields(obj)
        if f.init and f.name not in exclude
    }


def axis_envelope(obj: Any) -> dict[str, Any]:
    """Wrap one axis as ``{"class": <type name>, "config": <get_config()>}``.

    Args:
        obj: An axis instance exposing ``get_config()``.

    Returns:
        The class-name + config envelope used by Patcher-level configs.
    """
    return {"class": type(obj).__name__, "config": obj.get_config()}
