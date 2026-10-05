"""Callback hook protocol and dispatch helpers for patcher observability."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable


UNKNOWN_TOTAL = -1


@runtime_checkable
class PatcherHook(Protocol):
    """Optional callbacks emitted by patcher split and merge operations.

    Hook objects may implement any subset of these methods. The patchers
    dispatch callbacks dynamically, in order, and convert hook exceptions into
    warnings so observability code cannot interrupt patch generation.

    `on_patch_start` and `on_patch_done` accept an optional trailing
    ``coord_value`` (or ``None`` for integer geometries). `_dispatch` trims
    to the callback's actual arity, so single-arg hooks written for the
    pre-coord protocol still receive only ``anchor`` without warnings.
    See ADR-004 in ``docs/decisions.md`` ("Hook payload extension").

    `on_patch_skipped` fires (instead of ``on_patch_start`` /
    ``on_patch_done``) for each patch a ``journal`` already holds, so
    ``on_split_start``'s total is accounted for on a resumed run.
    `on_error` receives the original exception — whether the patcher
    re-raises it or its ``on_error`` policy swallows it.
    """

    def on_split_start(self, n_anchors: int) -> None: ...

    def on_patch_start(self, anchor: Any, coord_value: Any = None) -> None: ...

    def on_patch_done(
        self,
        anchor: Any,
        runtime_s: float,
        bytes_: int,
        coord_value: Any = None,
    ) -> None: ...

    def on_patch_skipped(self, anchor: Any, coord_value: Any = None) -> None: ...

    def on_split_end(self) -> None: ...

    def on_merge_start(self, n_patches: int) -> None: ...

    def on_merge_end(self, output_bytes: int) -> None: ...

    def on_error(self, anchor: Any, exc: Exception) -> None: ...


def _as_hooks(hooks: Iterable[PatcherHook] | None) -> tuple[PatcherHook, ...]:
    """Convert optional hooks to a materialized tuple.

    This ensures one-shot iterables, such as generators, can be reused across
    multiple callback dispatch calls throughout a split or merge operation.
    """
    return () if hooks is None else tuple(hooks)


_UNBOUNDED_ARITY = 1 << 30


def _positional_arity(callback: Callable[..., Any]) -> int:
    """Return the number of positional args the callback can accept.

    Used by `_dispatch` to trim trailing arguments so a hook written for the
    pre-coord protocol (``on_patch_start(self, anchor)``) still works after
    the patcher started passing the optional ``coord_value`` slot. Returns a
    very large value if introspection fails (e.g. builtins) so the dispatcher
    falls back to passing every arg, matching the prior behaviour.

    Bound methods are cached by their underlying function (``__func__``),
    never by the bound method itself: that would keep ``__self__`` — the
    hook object (a span, a progress bar, …) — alive in the cache.
    """
    func = getattr(callback, "__func__", None)
    if inspect.ismethod(callback) and inspect.isfunction(func):
        # ``self`` is the first positional parameter of the function.
        return max(_function_arity(func) - 1, 0)
    if inspect.isfunction(callback):
        return _function_arity(callback)
    # Callable instances, builtins, partials: uncached so we never hold
    # a strong reference to user objects.
    return _signature_arity(callback)


@lru_cache(maxsize=256)
def _function_arity(func: Callable[..., Any]) -> int:
    """`_signature_arity` cached on a plain function (holds no hook object)."""
    return _signature_arity(func)


def _signature_arity(callback: Callable[..., Any]) -> int:
    """Positional arity of ``callback`` (`_UNBOUNDED_ARITY` for ``*args``)."""
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):
        return _UNBOUNDED_ARITY
    n = 0
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return _UNBOUNDED_ARITY
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            n += 1
    return n


def _dispatch(hooks: Iterable[PatcherHook], method: str, *args: Any) -> None:
    """Call ``method`` on each hook that implements it.

    Patcher methods call this after materialising user-provided iterables with
    `_as_hooks`, so generator-backed hook lists are safe to reuse across all
    callbacks in a split or merge lifecycle. The warning `stacklevel` points to
    the direct `_dispatch` caller, which is usually the patcher method or
    private helper that emitted the callback.

    Trailing args beyond the callback's positional arity are dropped so the
    coord-value extension stays backwards compatible with single-arg hooks.

    Hook failures are intentionally downgraded to warnings: callbacks are
    observability side effects and must not change patcher correctness.
    """
    for hook in hooks:
        callback = getattr(hook, method, None)
        if callback is None:
            continue
        try:
            n = _positional_arity(callback)
            callback(*args[:n])
        except Exception as exc:
            warnings.warn(
                f"PatcherHook.{method} failed on {type(hook).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _len_or_unknown(values: Iterable[Any]) -> int:
    """Return ``len(values)`` or `UNKNOWN_TOTAL` for unsized iterables."""
    try:
        return len(values)  # type: ignore[arg-type]
    except TypeError:
        return UNKNOWN_TOTAL


def _nbytes(value: Any) -> int:
    """Best-effort byte count for patch data and aggregation outputs.

    Pure attribute probing — never materialises a lazy carrier: direct
    ``.nbytes`` first (NumPy, `GeoTensor`, dask arrays and xarray objects,
    all sized from dtype x shape without computing anything), then
    ``.data.nbytes`` for wrappers that expose their backing array under
    ``data``, and only then ``.values.nbytes`` (e.g. a pandas frame). The
    ``.values`` probe comes last because on a lazy xarray object it would
    compute the chunk graph — those always answer ``.nbytes`` first.

    A mapping (a `MatchedField` read's ``{source: data}``, a per-source
    merge result) or a list / tuple counts the sum of its values, so the
    matched patchers report every member's bytes, not ``0``.
    """
    if isinstance(value, Mapping):
        return sum(_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(v) for v in value)
    for attr in (None, "data", "values"):
        candidate = value if attr is None else getattr(value, attr, None)
        if candidate is None:
            continue
        nbytes = getattr(candidate, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
    return 0
