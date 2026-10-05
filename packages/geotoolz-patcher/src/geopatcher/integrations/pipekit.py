"""Operator wrappers around `geopatcher` — `GridSampler`, `ApplyToChips`, `Stitch`.

Thin glue between the four-axis Patcher framework and `pipekit.Operator`,
so a sliding-window inference pipeline composes inside a `Sequential` or
`Graph`::

    from pipekit import Sequential
    from geopatcher.integrations.pipekit import (
        GridSampler, ApplyToChips, Stitch,
    )

    pipe = Sequential([
        GridSampler(patcher=patcher),
        ApplyToChips(operator=model_op),
        Stitch(aggregation=SpatialOverlapAdd(), domain=field.domain),
    ])

Optional extra: install the ``[pipekit]`` extra to pull in pipekit.
While `pipekit` is pre-PyPI, use ``uv sync --extra pipekit`` (or
``uv pip install "git+https://github.com/jejjohnson/geopatcher@main#egg=geopatcher[pipekit]"``)
so uv can resolve the git source declared in this repo's
``pyproject.toml``. Once `pipekit` ships to PyPI, plain
``pip install 'geopatcher[pipekit]'`` will work too.

Importing this module without pipekit installed raises a friendly
``ImportError`` pointing at the right extra.
"""

from __future__ import annotations

from typing import Any, ClassVar


try:
    from pipekit import Operator
    from pipekit._base.operator import nested_config
except ImportError as _e:  # pragma: no cover - exercised when [pipekit] is missing
    raise ImportError(
        "geopatcher.integrations.pipekit requires the `pipekit` package. "
        "Install the [pipekit] extra with `uv sync --extra pipekit` "
        "(or `uv pip install 'geopatcher[pipekit]'`); plain "
        "`pip install` will work once pipekit reaches PyPI."
    ) from _e

from geopatcher import Patch, SpatialAggregation, SpatialPatcher, axis_envelope


class GridSampler(Operator):
    """Operator: ``Field → list[Patch]`` — yields the Patcher's patches.

    Materialises the iterator into a list so downstream operators don't
    need to know about lazy iteration; users who want streaming should
    consume ``patcher.split`` directly.

    Args:
        patcher: The `SpatialPatcher` to drive.

    Note:
        ``forbid_in_yaml = True`` — `patcher` is a runtime `SpatialPatcher`
        (not a `pipekit.Operator`), so pipekit's `from_state` cannot rebuild
        it and `pipekit.check_pickleable` cannot look inside it; the flag
        makes the lint report this operator, so a closure-bearing axis
        (e.g. `SpatialCustom`) is not hidden. `get_config()` nests the
        patcher as a ``{"class", "config"}`` envelope, which
        `geopatcher.from_config` rebuilds when the patcher is serialisable.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, patcher: SpatialPatcher) -> None:
        self.patcher = patcher

    def _apply(self, field: Any) -> list[Patch]:
        return list(self.patcher.split(field))

    def get_config(self) -> dict[str, Any]:
        return {"patcher": axis_envelope(self.patcher)}


class ApplyToChips(Operator):
    """Operator: ``list[Patch] → list[Patch]`` — map ``operator`` over each patch.

    The inner operator runs against each ``patch.data`` and the result
    replaces ``patch.data``; ``anchor`` / ``indices`` / ``weights`` are
    preserved so downstream `Stitch` can reconstruct the field.

    Args:
        operator: The per-chip operator (any `pipekit.Operator`).
    """

    forbid_in_yaml: ClassVar[bool] = False

    def __init__(self, *, operator: Operator) -> None:
        self.operator = operator

    def _apply(self, patches: list[Patch]) -> list[Patch]:
        return [p.with_data(self.operator(p.data)) for p in patches]

    def get_config(self) -> dict[str, Any]:
        return {"operator": nested_config(self.operator)}


class Stitch(Operator):
    """Operator: ``list[Patch] → field`` — wraps a `SpatialAggregation`.

    Pairs with `GridSampler` + `ApplyToChips` to express ``split →
    operator → merge`` as a three-step `Sequential`. The ``domain``
    argument is supplied at construction (commonly ``field.domain``) so
    the resulting `Operator` has a single positional input (the list of
    patches) and slots into the linear pipeline.

    Args:
        aggregation: The `SpatialAggregation` to apply.
        domain: The `Domain` the patches were drawn from. Required
            because the aggregation's output shape is fixed by the
            domain.

    Note:
        ``forbid_in_yaml = True`` — `domain` is a runtime `Domain`
        protocol object (CRS, affine transform, shape) and isn't in
        general JSON-serialisable, so the constructor cannot be
        round-tripped from `get_config()` alone. `get_config()` emits a
        debug record (class names + nested configs) for introspection /
        logging, not for replay.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, aggregation: SpatialAggregation, domain: Any) -> None:
        self.aggregation = aggregation
        self.domain = domain

    def _apply(self, patches: list[Patch]) -> Any:
        return self.aggregation.merge(patches, self.domain)

    def get_config(self) -> dict[str, Any]:
        return {
            "aggregation": axis_envelope(self.aggregation),
            "domain": {"class": type(self.domain).__name__},
        }


__all__ = ["ApplyToChips", "GridSampler", "Stitch"]
