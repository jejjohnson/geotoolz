"""Shared ``get_config()`` coercion helpers.

Every Operator family emits its constructor parameters through
``get_config()`` so pipelines round-trip through YAML / hydra-zen.
Those serialisers only accept JSON builtins, so config values need
coercing: numpy scalars and arrays, bytes, datetimes, paths, and nested
containers thereof. :func:`jsonable` is the one recursive coercion
(``strict=True`` for debug payloads that must always serialise), so
operator modules carry no per-family ``*_as_jsonable`` variants.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import PurePath
from typing import Any

import numpy as np
from pipekit._base.operator import callable_name, nested_config


# ``nested_config`` (pipekit's canonical ``{"class", "config"}`` payload for an
# operator nested in another's config) and ``callable_name`` (display name for
# a callable in a debug config) are re-exported here because pipekit does not
# expose them at the top level.
__all__ = [
    "as_tuple",
    "callable_name",
    "jsonable",
    "mapping_from_pairs",
    "mapping_to_pairs",
    "nested_config",
    "reject_config_summary",
]


def jsonable(value: Any, *, strict: bool = False) -> Any:
    """Recursively coerce a config value into JSON/YAML-safe builtins.

    Args:
        value: Any ``get_config()`` leaf or container. Dicts (and other
            mappings) and lists/tuples are converted recursively (tuples
            become lists, per strict-JSON convention; keys become
            strings); numpy arrays become (nested) lists of Python
            scalars; numpy scalars become their Python equivalents;
            ``bytes`` are decoded as UTF-8 (undecodable bytes replaced);
            ``datetime`` / ``date`` become ISO-8601 strings; paths become
            strings.
        strict: What to do with anything left over. ``False`` (default)
            passes it through unchanged -- the caller knows the value
            is serialisable, or wants the runtime object back. ``True``
            guarantees the result survives ``json.dumps(...,
            allow_nan=False)``: any value that is not ``None`` / ``bool``
            / ``int`` / ``str`` / finite ``float`` becomes its ``repr``
            string (so non-finite floats become ``'nan'`` / ``'inf'``,
            and runtime objects such as nested estimators, callables or
            ``RandomState`` instances become e.g. ``'StandardScaler()'``)
            instead of being dropped.

    Returns:
        A structure of dicts, lists, and JSON-safe scalars.
    """
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, strict=strict) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, strict=strict) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist(), strict=strict)
    if isinstance(value, np.generic):
        # ``.item()`` can itself yield bytes / datetime (``np.bytes_``,
        # ``np.datetime64``), so coerce the Python scalar in turn.
        return jsonable(value.item(), strict=strict)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if strict and not _is_strict_json_scalar(value):
        return repr(value)
    return value


def _is_strict_json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def as_tuple(value: Any) -> Any:
    """Normalise a list or tuple constructor argument to a tuple.

    JSON and YAML have no tuple type, so a config reloaded through
    ``Operator.from_state`` hands the constructor a list where a tuple
    was emitted. Constructors call this on tuple-typed parameters so the
    reloaded operator is identical; scalars and ``None`` pass through.

    Args:
        value: The constructor argument.

    Returns:
        ``tuple(value)`` for a non-string sequence (list, tuple, OmegaConf
        ``ListConfig``), otherwise ``value``.
    """
    if _is_sequence(value):
        return tuple(value)
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def mapping_to_pairs(mapping: Mapping[Any, Any] | None) -> list[list[Any]] | None:
    """Emit a mapping config value as ``[[key, value], ...]``.

    ``Operator.from_state`` rejects dict config values (they are reserved
    for nested-operator payloads) and JSON would stringify non-string
    keys, so mapping-valued parameters are emitted as a list of pairs.
    Constructors accept that form back through :func:`mapping_from_pairs`.

    Args:
        mapping: The mapping to emit, or ``None``.

    Returns:
        A JSON-safe list of ``[key, value]`` pairs, or ``None``.
    """
    if mapping is None:
        return None
    return [[jsonable(k), jsonable(v)] for k, v in mapping.items()]


def mapping_from_pairs(
    value: Mapping[Any, Any] | Iterable[Any] | None,
) -> dict[Any, Any] | None:
    """Accept a mapping or its ``[[key, value], ...]`` config form.

    Args:
        value: A mapping, an iterable of ``(key, value)`` pairs as emitted
            by :func:`mapping_to_pairs`, or ``None``.

    Returns:
        A ``dict`` (list-valued keys become tuples), or ``None``.

    Raises:
        ValueError: if an item of the iterable form is not a pair.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    out: dict[Any, Any] = {}
    for item in value:
        if not _is_sequence(item) or len(item) != 2:
            raise ValueError(f"expected a mapping or [key, value] pairs; got {item!r}")
        key, val = item
        out[as_tuple(key)] = val
    return out


def reject_config_summary(value: Any, name: str) -> Any:
    """Refuse a debug summary passed back as a runtime constructor argument.

    Operators with an *optional* runtime argument (an array, a callable)
    stay reloadable when it is unset, and summarise it as a dict in
    ``get_config`` when it is set. ``Operator.from_state`` refuses dict
    values, but a YAML / Hydra loader would hand the summary back to the
    constructor; this turns that into a clear error.

    Args:
        value: The constructor argument.
        name: Parameter name for the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: if ``value`` is a mapping (a config summary).
    """
    if isinstance(value, Mapping):
        raise TypeError(
            f"{name} got a config summary; runtime values cannot be rebuilt "
            "from a config — construct the operator in code instead."
        )
    return value
