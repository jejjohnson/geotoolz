"""Small composable building blocks — Identity, Const, Lambda, Sink.

Tiny operators that on their own look trivial but in combination unlock
common patterns:

- `Identity` — explicit no-op, the right thing in a `Branch.if_false`,
  a `Switch.default`, or any "structurally need an Operator" slot.
- `Const` — return a fixed `GeoTensor` regardless of input. Test
  fixtures, `Switch` defaults, golden values.
- `Lambda` — inline-function escape hatch when writing a full subclass
  is overkill. Flagged ``forbid_in_yaml = True`` because closures don't
  round-trip.
- `Sink` — terminal write that *returns the input*. Composes (unlike
  a write op that returns `None`); useful for checkpointing and "save
  intermediates" patterns.

See `tips_n_tricks.md` §"Small but load-bearing building blocks".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

from geotoolz.core._src.operator import Carrier, Operator


# Identity-preserving carrier type. ``Identity`` and ``Sink`` return
# whatever was passed in; this lets static type checkers narrow.
_C = TypeVar("_C")


class Identity(Operator):
    """Explicit no-op. Returns its input unchanged.

    Use as a default arm in ``Branch`` / ``Switch`` rather than passing
    ``None`` — it serialises, composes, shows up in ``repr()``, and makes
    pipeline structure self-documenting.

    Examples:
        ``Branch(predicate=is_clean, if_true=Identity(), if_false=cleanup)``
    """

    def _apply(self, gt: _C) -> _C:
        return gt


class Const(Operator):
    """Return a fixed value regardless of input.

    Useful for golden test fixtures, as a synthetic source in a
    ``Switch`` default, or anywhere the pipeline needs a stand-in.

    Args:
        value: The value to return on every call. Typically a `GeoTensor`
            for in-pipeline use; can be any object for test scaffolding.

    Examples:
        Build a deterministic test pipeline::

            test_pipeline = Sequential([
                Const(synthetic_gt),    # ignores input
                real_pipeline,
            ])
    """

    def __init__(self, value: Any) -> None:
        self.value = value

    def _apply(self, _: Carrier = None) -> Any:
        return self.value

    def get_config(self) -> dict[str, Any]:
        # Best-effort debug repr — the actual value may not be JSON-safe.
        return {
            "value_type": type(self.value).__name__,
            "value_shape": getattr(self.value, "shape", None),
        }


class Lambda(Operator):
    """Inline-function escape hatch.

    Holds a user callable and applies it. The callable's signature is
    `fn(gt) -> result` — same contract as `Operator._apply`. Use when
    writing a full `Operator` subclass would be overkill for a one-off
    transform.

    ``forbid_in_yaml = True`` — the closure cannot round-trip to YAML
    faithfully. ``get_config()`` returns a debug repr only. The first
    time a `Lambda` recurs in your code, promote it to a real `Operator`
    subclass.

    Args:
        fn: A callable `(input) -> output`.
        name: Display name for ``repr()`` and provenance. Defaults to
            ``"lambda"``.

    Examples:
        ``Lambda(lambda gt: gt * 0.0001, name="scale_to_reflectance")``
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, fn: Callable[[Any], Any], *, name: str = "lambda") -> None:
        self.fn = fn
        self.name = name

    def _apply(self, gt: Any) -> Any:
        return self.fn(gt)

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}


class Sink(Operator):
    """Composable terminal write — performs a side effect and returns
    the input unchanged.

    Unlike a write op that returns `None` and breaks the pipe,
    ``Sink(write_fn)`` keeps the GeoTensor flowing. Useful for
    checkpointing long pipelines, debugging ("what did step 3 produce?"),
    and branching analysis (write an intermediate, continue with the
    final product).

    ``forbid_in_yaml = True`` — the write callable is a closure.

    Args:
        write_fn: A callable `(gt) -> Any` whose return value is ignored.
            Typical use: ``lambda gt: georeader.save_cog(gt, "/path.tif")``.
        name: Display name for ``repr()`` / provenance. Default
            ``"sink"``.

    Examples:
        ``Sink(lambda gt: georeader.save_cog(gt, "/intermediate.tif"))``
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        write_fn: Callable[[Any], Any],
        *,
        name: str = "sink",
    ) -> None:
        self.write_fn = write_fn
        self.name = name

    def _apply(self, gt: _C) -> _C:
        self.write_fn(gt)
        return gt

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}
