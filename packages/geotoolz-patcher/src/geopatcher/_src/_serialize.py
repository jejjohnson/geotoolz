"""Shared serialization helpers for axis and patcher ``get_config`` methods.

Every patcher axis (geometry / sampler / window / aggregation) exposes a
``get_config() -> dict`` for YAML round-trip. Most of those configs are a
plain dump of the dataclass's init fields with a small scalar coercion —
this module centralises that boilerplate:

- `jsonable_scalar` — numpy generic / ``datetime64`` / ``timedelta64`` →
  plain Python value.
- `config_from_fields` — dataclass instance → ``{field: coerced value}``.
- `axis_envelope` — the ``{"class": ..., "config": ...}`` wrapper every
  nested component (an axis inside a patcher, a stencil inside a stencil
  axis, a patcher inside a matched patcher) is serialised as.
- `patcher_config` — the one ``get_config`` every patcher family uses: its
  init fields, with each component nested via `axis_envelope`.
- `from_config` — the inverse: rebuild an object from its envelope,
  resolving nested envelopes recursively.

The contract (enforced by ``tests/test_get_config.py``): for every object
whose ``forbid_in_yaml`` is ``False``, ``from_config(axis_envelope(obj))``
rebuilds an equivalent object — same config, same behaviour — and a leaf
axis also rebuilds as ``type(obj)(**obj.get_config())``. Objects whose
state cannot be expressed as JSON (closures, polygons, backend-native
anchors) set ``forbid_in_yaml = True``; their ``get_config`` is a debug
summary and `from_config` refuses them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
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
    """`jsonable_scalar` extended over containers.

    Tuples, lists and arrays become lists and mappings become dicts, each
    with their elements coerced recursively.
    """
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value]
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    return jsonable_scalar(value)


def config_from_fields(obj: Any, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Serialise a dataclass axis into its ``get_config`` dict.

    Walks the dataclass's init fields in declaration order (private
    ``init=False`` accumulator state is skipped automatically) and coerces
    each value with `jsonable_scalar`; tuples, lists and arrays are
    rebuilt as lists and mappings as dicts, with every element coerced
    recursively.

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
    """Wrap one component as ``{"class": <type name>, "config": <get_config()>}``.

    Args:
        obj: An axis, stencil or patcher instance exposing ``get_config()``.

    Returns:
        The class-name + config envelope used by every nested config.
    """
    return {"class": type(obj).__name__, "config": obj.get_config()}


def qualified_name(cls: type) -> str:
    """``module.qualname`` of ``cls``; the bare name for a builtin.

    Args:
        cls: Any class (in practice an exception class from ``retry_on``).

    Returns:
        ``"OSError"`` for a builtin, else e.g.
        ``"rasterio.errors.RasterioIOError"``.
    """
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def _nested(value: Any) -> Any:
    """Config value of one patcher field — components become envelopes."""
    if isinstance(value, type):
        return qualified_name(value)
    if hasattr(value, "get_config"):
        return axis_envelope(value)
    if isinstance(value, Mapping):
        return {k: _nested(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_nested(v) for v in value]
    return _jsonable(value)


def patcher_config(obj: Any) -> dict[str, Any]:
    """The ``get_config`` shared by every patcher family.

    Dumps the dataclass's init fields in declaration order. Each component
    exposing ``get_config`` (an axis, an inner patcher, a per-source
    aggregator inside a mapping) nests as an `axis_envelope`; classes
    (``retry_on`` exception types) serialise as their `qualified_name`;
    everything else is coerced like `config_from_fields`.

    Args:
        obj: A patcher dataclass instance.

    Returns:
        A JSON-safe dict that `from_config` turns back into the patcher.
    """
    return {
        f.name: _nested(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init
    }


def _is_envelope(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"class", "config"}
        and isinstance(value["class"], str)
        and isinstance(value["config"], Mapping)
    )


def _subclasses(cls: type) -> list[type]:
    out = [cls]
    for sub in cls.__subclasses__():
        out.extend(_subclasses(sub))
    return out


def _registry() -> dict[str, list[type]]:
    """Every loaded geopatcher component class, keyed by ``__name__``.

    Built from the axis / stencil / patcher roots plus all their loaded
    subclasses, so a user-defined axis subclass resolves once imported.
    """
    from geopatcher._src.matched.patcher import (
        MatchedSpatialPatcher,
        MatchedSpatioTemporalPatcher,
        MatchedTemporalPatcher,
    )
    from geopatcher._src.spatial.aggregation import SpatialAggregation
    from geopatcher._src.spatial.geometry import SpatialGeometry
    from geopatcher._src.spatial.patcher import AsyncSpatialPatcher, SpatialPatcher
    from geopatcher._src.spatial.sampler import SpatialSampler
    from geopatcher._src.spatial.window import SpatialWindow
    from geopatcher._src.spatial_time import SpatioTemporalPatcher
    from geopatcher._src.time.aggregation import TemporalAggregation
    from geopatcher._src.time.geometry import TemporalGeometry
    from geopatcher._src.time.patcher import TemporalPatcher
    from geopatcher._src.time.sampler import TemporalSampler
    from geopatcher._src.time.stencils import Stencil
    from geopatcher._src.time.window import TemporalWindow

    roots: tuple[type, ...] = (
        SpatialGeometry,
        SpatialSampler,
        SpatialWindow,
        SpatialAggregation,
        TemporalGeometry,
        TemporalSampler,
        TemporalWindow,
        TemporalAggregation,
        Stencil,
        SpatialPatcher,
        AsyncSpatialPatcher,
        TemporalPatcher,
        SpatioTemporalPatcher,
        MatchedSpatialPatcher,
        MatchedTemporalPatcher,
        MatchedSpatioTemporalPatcher,
    )
    registry: dict[str, list[type]] = {}
    for root in roots:
        for cls in _subclasses(root):
            entries = registry.setdefault(cls.__name__, [])
            if cls not in entries:
                entries.append(cls)
    return registry


def _resolve(value: Any, registry: dict[str, list[type]]) -> Any:
    if _is_envelope(value):
        return _build(value, registry)
    if isinstance(value, Mapping):
        return {k: _resolve(v, registry) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, registry) for v in value]
    return value


def _build(envelope: Mapping[str, Any], registry: dict[str, list[type]]) -> Any:
    name = envelope["class"]
    candidates = registry.get(name, [])
    if not candidates:
        raise TypeError(
            f"from_config: {name!r} is not a loaded geopatcher axis, stencil or "
            "patcher class (import the module defining it first)."
        )
    if len(candidates) > 1:
        raise TypeError(
            f"from_config: class name {name!r} is ambiguous — defined in "
            f"{sorted(c.__module__ for c in candidates)}."
        )
    (cls,) = candidates
    if getattr(cls, "forbid_in_yaml", False):
        raise RuntimeError(
            f"from_config cannot rebuild {name}: it is marked forbid_in_yaml "
            "(its config is a debug summary, not a faithful round-trip). "
            "Rebuild it in code instead."
        )
    config = {k: _resolve(v, registry) for k, v in envelope["config"].items()}
    return cls(**config)


def from_config(envelope: Mapping[str, Any]) -> Any:
    """Rebuild an axis, stencil or patcher from its `axis_envelope`.

    Nested envelopes (the axes inside a patcher, the stencil inside a
    stencil axis, the per-source aggregators of a matched patcher) are
    rebuilt recursively. The class is looked up by name among the loaded
    geopatcher component classes and their subclasses.

    Args:
        envelope: ``{"class": <name>, "config": <get_config()>}`` — e.g.
            ``axis_envelope(obj)`` after a JSON / YAML round-trip.

    Returns:
        A new instance equivalent to the one the envelope was taken from.

    Raises:
        TypeError: if the envelope is malformed or names an unknown or
            ambiguous class.
        RuntimeError: if the class is marked ``forbid_in_yaml``.

    Examples:
        >>> geom = SpatialRectangular(size=(64, 64), boundary="pad")
        >>> from_config(axis_envelope(geom)).get_config() == geom.get_config()
        True
        >>> stencil = TimeStencil("-9h", "3h", "1h", closed="both")
        >>> from_config(json.loads(json.dumps(axis_envelope(stencil)))) == stencil
        True
        >>> from_config({"class": "SpatialCustom", "config": {}})
        Traceback (most recent call last):
        RuntimeError: from_config cannot rebuild SpatialCustom: ...
    """
    if not _is_envelope(envelope):
        raise TypeError(
            "from_config expects a {'class': <name>, 'config': {...}} envelope, "
            f"got {envelope!r}."
        )
    return _build(envelope, _registry())
