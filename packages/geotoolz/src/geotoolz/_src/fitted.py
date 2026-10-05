"""Shared plumbing for fitted operators (the ``fit`` / ``transform`` seam).

Fitted operators (the ``normalize`` scalers, ``matched_filter.MatchedFilter``,
``restore.MNF``, ``learn.SklearnOp``) expose sklearn-style ``fit(x) -> self``
and a pure ``transform(x)`` so they satisfy
:class:`pipekit.protocols.FittableTransformer`. Fitted state lives in
trailing-underscore attributes (``mean_``, ``cov_op_``, ...) that are never
constructor parameters, so ``get_config()`` -- which mirrors the
constructor -- never serialises it.

When an operator learns its state on the first call, ``_apply`` routes
through :func:`fit_once`: the check-fit-publish sequence runs under a
per-instance lock, so concurrent first calls under ``pipekit.ThreadMap`` fit
exactly once and every call transforms with the same, fully published
state. *Which* item the state is learned from is still whichever call
takes the lock first; fit explicitly (``op.fit(x)``) before parallel use for
deterministic results.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class PicklableLock:
    """A re-entrant lock that pickles / deep-copies to a fresh, unlocked lock.

    Operators are pickled for ``ProcessMap`` and deep-copied by some
    callers; a raw ``threading.RLock`` supports neither.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __enter__(self) -> bool:
        return self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()

    def __reduce__(self) -> tuple[type[PicklableLock], tuple[()]]:
        return (PicklableLock, ())


def fit_lock(op: Any) -> PicklableLock:
    """The per-instance fit lock of ``op``, created on first use.

    ``dict.setdefault`` is atomic, so two threads racing on the first
    call still share a single lock.
    """
    return op.__dict__.setdefault("_fit_lock", PicklableLock())


def fit_once(op: Any, x: Any, is_fitted: Callable[[], bool]) -> None:
    """Call ``op.fit(x)`` unless ``is_fitted()``; at most once across threads.

    Double-checked: the cheap unlocked check keeps already-fitted calls
    lock-free; the locked re-check makes concurrent first calls fit once.
    ``op.fit`` must publish its state only after computing all of it.
    """
    if is_fitted():
        return
    with fit_lock(op):
        if not is_fitted():
            op.fit(x)
