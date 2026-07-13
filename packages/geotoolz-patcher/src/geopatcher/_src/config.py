"""Process-wide configuration toggles for `geopatcher`.

Currently houses one flag: `strict`. When enabled, `streaming_safe`
violations escalate from a `RuntimeWarning` to a `RuntimeError`. The
default is permissive — interactive / notebook callers see a warning
and decide what to do; batch / CI callers can lock down by calling
`set_strict(True)` (or exporting `GEOPATCHER_STRICT=1`).

See `docs/decisions.md` (ADR-006) for rationale.
"""

from __future__ import annotations

import os


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


_strict: bool = _truthy(os.environ.get("GEOPATCHER_STRICT"))


def get_strict() -> bool:
    """Return the current value of the process-wide ``strict`` flag."""
    return _strict


def set_strict(value: bool) -> None:
    """Set the process-wide ``strict`` flag.

    When ``True``, `streaming_safe = False` aggregations raise instead
    of emitting a warning. The flag is read by
    `_warn_if_unsafe_streaming` and any future check that wants the same
    fail-fast contract.
    """
    global _strict
    _strict = bool(value)


__all__ = ["get_strict", "set_strict"]
