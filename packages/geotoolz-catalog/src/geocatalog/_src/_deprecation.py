"""Deprecation shims for renamed keywords and attributes (one minor release)."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


def _warn(old: str, new: str, where: str, stacklevel: int) -> None:
    warnings.warn(
        f"{where}: `{old}` is deprecated, use `{new}`; the old name will be "
        "removed in the next minor release.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def renamed_kwargs(**renames: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Accept old keyword names, mapping them to the new ones with a warning.

    ``@renamed_kwargs(backend="kind")`` lets ``f(backend=x)`` keep working
    as ``f(kind=x)`` (with a `DeprecationWarning`); passing both raises
    `TypeError`.
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for old, new in renames.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(
                            f"{fn.__qualname__}() got both `{old}` (deprecated) "
                            f"and `{new}`"
                        )
                    _warn(old, new, f"{fn.__qualname__}()", stacklevel=3)
                    kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        if inspect.iscoroutinefunction(fn):  # keep `async def` introspectable
            inspect.markcoroutinefunction(wrapper)
        return wrapper

    return decorate


def deprecated_alias(new: str) -> Any:
    """A property forwarding to attribute ``new``, with a warning."""

    def getter(self: Any) -> Any:
        _warn(getter._old_name, new, type(self).__name__, stacklevel=3)  # type: ignore[attr-defined]
        return getattr(self, new)

    def setter(self: Any, value: Any) -> None:
        _warn(getter._old_name, new, type(self).__name__, stacklevel=3)  # type: ignore[attr-defined]
        setattr(self, new, value)

    class _Alias(property):
        def __set_name__(self, owner: type, name: str) -> None:
            getter._old_name = name  # type: ignore[attr-defined]

    return _Alias(getter, setter)


__all__ = ["deprecated_alias", "renamed_kwargs"]
