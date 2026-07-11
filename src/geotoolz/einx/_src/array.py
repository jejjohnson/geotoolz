"""Tier-A helpers — einx pattern analysis, pure string processing.

No ``einx`` import happens here: the spatial-survival check is static
analysis of the pattern text, so it is importable (and testable) even
without the ``[einx]`` extra installed. The carrier-aware dispatch that
actually calls einx lives in ``operators.py``.

The survival rule (design decision for geotoolz issue #69, Q3): a
pattern *preserves spatial structure* iff

1. the trailing two top-level axes of its output expression are exactly
   the bare spatial axis names (default ``("y", "x")``), and
2. neither spatial axis appears inside a composed / bracketed group
   anywhere in the pattern (composition means the axis is being split
   or merged, so its size — and therefore the geotransform — changes).

Transposes, renames, spatial reductions, channels-last outputs, pooled
axes, bracketed vmap axes, and patterns with no explicit ``->`` all
count as *not survived* — the result then carries no geotransform and
is returned as a plain array. This is deliberately conservative:
`georeader.GeoTensor.array_as_geotensor` requires the trailing two dims
to keep their sizes, so only size-and-position-preserving patterns can
rewrap safely.
"""

from __future__ import annotations

import re


__all__ = ["output_axes", "spatial_survives"]

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


def _tokenize(expr: str, *, pattern: str) -> list[str]:
    """Split one einx expression into top-level tokens.

    A bare axis name is one token; a whole parenthesized / bracketed
    group (``(y py)``, ``[c]``) is one composite token including its
    delimiters. Whitespace and commas separate tokens at depth zero.

    Args:
        expr: One side (or comma-separated part) of an einx pattern.
        pattern: The full pattern, used only for error messages.

    Returns:
        The top-level tokens of ``expr``.

    Raises:
        ValueError: On unbalanced brackets.
    """
    tokens: list[str] = []
    current: list[str] = []
    stack: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for ch in expr:
        if ch in _OPEN:
            if not stack:
                flush()
            stack.append(ch)
            current.append(ch)
        elif ch in _CLOSE:
            if not stack or _CLOSE[ch] != stack[-1]:
                raise ValueError(f"unbalanced {ch!r} in einx pattern: {pattern!r}")
            stack.pop()
            current.append(ch)
            if not stack:
                flush()
        elif (ch.isspace() or ch == ",") and not stack:
            flush()
        else:
            current.append(ch)
    if stack:
        raise ValueError(f"unbalanced {stack[-1]!r} in einx pattern: {pattern!r}")
    flush()
    return tokens


def output_axes(pattern: str) -> list[str] | None:
    """Return the top-level axis tokens of a pattern's output expression.

    The output expression is everything after the *last* ``->``.

    Args:
        pattern: An einx pattern string, e.g. ``"c y x -> y x"``.

    Returns:
        The list of top-level output tokens, or ``None`` when the
        pattern has no explicit ``->`` (einx's implicit-output forms
        cannot be analyzed statically here).

    Raises:
        ValueError: On unbalanced brackets in the pattern.
    """
    if "->" not in pattern:
        return None
    return _tokenize(pattern.rsplit("->", 1)[1], pattern=pattern)


def spatial_survives(
    pattern: str,
    spatial_axes: tuple[str, str] = ("y", "x"),
) -> bool:
    """Return whether an einx pattern preserves the carrier's spatial grid.

    True iff the output expression's trailing two top-level tokens are
    exactly the bare ``spatial_axes`` names in order, AND neither
    spatial axis appears inside a composed / bracketed group anywhere
    in the pattern (composition splits or merges the axis, changing its
    size and invalidating the geotransform).

    Args:
        pattern: An einx pattern string.
        spatial_axes: The (row, column) axis names, default ``("y", "x")``.

    Returns:
        Whether a ``GeoTensor`` carrier's transform / CRS remain valid
        for the output.

    Examples:
        >>> spatial_survives("c y x -> y x")
        True
        >>> spatial_survives("c y x -> x y")               # transposed
        False
        >>> spatial_survives("c y x -> y x c")             # channels-last
        False
        >>> spatial_survives("c (y py) (x px) -> c y x")   # pooled: resized
        False
        >>> spatial_survives("band y x, sig band -> sig y x")
        True
    """
    tokens = output_axes(pattern)
    if tokens is None or len(tokens) < 2:
        return False
    if tokens[-2] != spatial_axes[0] or tokens[-1] != spatial_axes[1]:
        return False
    # Reject patterns that compose/split a spatial axis anywhere: the
    # axis size changes, so the trailing dims can't keep their extent.
    sides = pattern.split("->")
    for side in sides:
        for token in _tokenize(side, pattern=pattern):
            if token[0] in _OPEN and any(
                re.search(rf"(?<![\w.]){re.escape(ax)}(?![\w.])", token)
                for ax in spatial_axes
            ):
                return False
    return True
