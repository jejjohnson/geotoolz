"""Deterministic colour palette helper for multi-method panels.

Used to keep colours consistent across iso PSD, space-time PSD, and
time-series figures so a reader can track ``"DUACS = green"`` through
every panel of a benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterable


# Default palette cycle — the matplotlib ``tab:`` family. Picked for
# print legibility and to match the style guide most ocean-paper
# notebooks use already.
_DEFAULT_CYCLE: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)


def method_palette(
    names: Iterable[str],
    cycle: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return a deterministic ``{name: colour}`` mapping.

    The order is determined by sorting ``names`` alphabetically (so the
    same set of methods always yields the same mapping regardless of
    input order), then cycling through ``cycle``.

    Args:
        names: Iterable of method names. Duplicates are de-duplicated.
        cycle: Iterable of matplotlib colour specifiers. Defaults to the
            ``tab:*`` family.

    Returns:
        ``dict`` mapping each unique name to a colour.
    """
    palette = tuple(cycle) if cycle is not None else _DEFAULT_CYCLE
    if not palette:
        raise ValueError("`cycle` must contain at least one colour.")
    unique = sorted(set(names))
    return {name: palette[i % len(palette)] for i, name in enumerate(unique)}


__all__ = ["method_palette"]
