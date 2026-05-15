"""Observer operators — identity Operators with side effects.

`Tap`, `Snapshot`, `ShapeTrace` let you observe a pipeline mid-flight
without breaking the chain. They follow the same `GeoTensor → GeoTensor`
contract as transforms; the GeoTensor flows through unchanged while
something useful happens on the side.

See `tips_n_tricks.md` §"Inspection / introspection (Tap family)".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

from geotoolz.core._src.operator import Carrier, Operator


# All three observers are identity-preserving — they return whatever was
# passed in, so the carrier type survives through them. The ``Carrier``
# alias is also imported for use in the ``Tap`` / ``_SnapshotTap`` /
# ``ShapeTrace`` callback signatures (the ``fn`` / ``printer`` callables
# *consume* a carrier but don't transform it).
_C = TypeVar("_C")


class Tap(Operator):
    """Identity operator with a side effect.

    Calls ``fn(gt)`` and returns ``gt`` unchanged. The return value of
    ``fn`` is ignored — `Tap` is for side effects, not transforms. If
    you want to transform, use `Lambda` or write a real `Operator`.

    ``forbid_in_yaml = True`` because the callback closure can't
    round-trip.

    Args:
        fn: A callable ``(gt) -> Any`` invoked for its side effect.
        name: Display name for ``repr()`` / provenance. Default ``"tap"``.

    Examples:
        Log NaN fraction between steps::

            Sequential([
                MaskClouds(...),
                Tap(lambda gt: print(f"NaN: {np.isnan(gt).mean():.1%}")),
                NDVI(...),
            ])
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, fn: Callable[[Carrier], Any], *, name: str = "tap") -> None:
        self.fn = fn
        self.name = name

    def _apply(self, gt: _C) -> _C:
        self.fn(gt)
        return gt

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}


class _SnapshotTap(Operator):
    """Private operator returned by `Snapshot.at(...)`.

    Stores the passing GeoTensor in the controller's dict under the given
    key. Closures over the controller's dict — does not own state itself,
    so multiple ``snap.at("k1")``, ``snap.at("k2")`` share one snapshot
    store.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, store: dict[str, Any], key: str) -> None:
        self._store = store
        self._key = key

    def _apply(self, gt: _C) -> _C:
        self._store[self._key] = gt
        return gt

    def get_config(self) -> dict[str, Any]:
        return {"key": self._key}


class Snapshot:
    """Controller that produces snapshot-taking operators.

    Not an `Operator` itself — `Snapshot.at(key)` returns the operator
    you drop into a `Sequential`. After the pipeline runs, every named
    intermediate is available via ``snap[key]``.

    Stores *references*, not copies — if a downstream op mutates the
    array in place, your snapshot sees the mutation too. Add explicit
    copies in exploratory work if needed.

    Examples:
        ::

            snap = Snapshot()
            pipe = Sequential([
                op1, snap.at("after_op1"),
                op2, snap.at("after_op2"),
            ])
            pipe(gt)
            print(snap.keys())             # dict_keys(["after_op1", "after_op2"])
            inspect = snap["after_op1"]
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def at(self, key: str) -> _SnapshotTap:
        """Return an operator that captures the GeoTensor under ``key``."""
        return _SnapshotTap(self._store, key)

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def keys(self):
        return self._store.keys()

    def items(self):
        return self._store.items()

    def clear(self) -> None:
        self._store.clear()


class ShapeTrace(Operator):
    """Log shape, dtype, CRS, transform at every step.

    Drop one between steps of a `Sequential` to see what's happening to
    the carrier. Optional ``mode="diff_only"`` skips lines that don't
    change anything from the previous trace.

    Args:
        printer: Callable used to print each line. Default
            ``builtins.print``. Override with ``log.info`` etc.
        mode: ``"every"`` (default) logs every call; ``"diff_only"``
            suppresses output when nothing changed since the last call.

    Examples:
        ::

            Sequential([
                ShapeTrace(),
                op1,
                ShapeTrace(),
                op2,
                ShapeTrace(),
            ])(gt)
    """

    _MODES: ClassVar[tuple[str, ...]] = ("every", "diff_only")

    def __init__(
        self,
        *,
        printer: Callable[[str], Any] = print,
        mode: str = "every",
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {mode!r}")
        self._printer = printer
        self.mode = mode
        self._last_line: str | None = None

    def _describe(self, gt: Carrier) -> str:
        shape = getattr(gt, "shape", None)
        dtype = getattr(gt, "dtype", None)
        crs = getattr(gt, "crs", None)
        return f"shape={shape} dtype={dtype} crs={crs}"

    def _apply(self, gt: _C) -> _C:
        line = self._describe(gt)
        if self.mode == "diff_only" and line == self._last_line:
            return gt
        self._printer(line)
        self._last_line = line
        return gt

    def get_config(self) -> dict[str, Any]:
        return {"mode": self.mode}
