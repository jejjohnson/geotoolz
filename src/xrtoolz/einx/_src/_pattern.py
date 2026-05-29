"""Light parser for einx patterns — dim-name extraction only.

The bridge delegates the *semantics* of a pattern to einx itself
(einx parses the same string at apply time). This module only needs to
know which dim names appear on which side so the Layer-0 functions can:

- check inputs carry the right dims,
- determine output dim names + order,
- pull sizes from input signatures / kwargs for ``compute_output_signature``.

An ``Element`` is either a bare axis name (``str``) or a parenthesised
group of names (``tuple[str, ...]``) produced by einx's merge/split
syntax. One level of nesting is supported; deeper nesting and the
bracket ``[...]`` reduction marker are rejected with
:class:`PatternError` (use the explicit ``in -> out`` arrow form).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xrtoolz.einx._src.errors import PatternError
from xrtoolz.signature import Signature


Element = str | tuple[str, ...]

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class EinxPattern:
    """Parsed view of an einx pattern.

    Attributes:
        raw: The original pattern string (passed verbatim to einx).
        inputs: One element list per input slot (left of ``->``).
        output: Element list for the output slot (right of ``->``).
    """

    raw: str
    inputs: tuple[tuple[Element, ...], ...]
    output: tuple[Element, ...]

    @property
    def output_dims(self) -> tuple[str, ...]:
        """Output dim names in result order (groups underscore-joined)."""
        return tuple(_element_name(el) for el in self.output)

    def flat_input_names(self, index: int) -> tuple[str, ...]:
        """Flat axis names of input slot ``index`` (groups expanded)."""
        return _flatten(self.inputs[index])

    def output_axis_names(self) -> tuple[str, ...]:
        """All atomic axis names appearing in the output (groups expanded)."""
        return _flatten(self.output)


def _element_name(element: Element) -> str:
    return element if isinstance(element, str) else "_".join(element)


def _flatten(elements: Sequence[Element]) -> tuple[str, ...]:
    names: list[str] = []
    for el in elements:
        if isinstance(el, str):
            names.append(el)
        else:
            names.extend(el)
    return tuple(names)


def _parse_slot(slot: str) -> tuple[Element, ...]:
    """Parse one ``->``-delimited slot into ordered elements."""
    if "[" in slot or "]" in slot:
        raise PatternError(
            "Bracket reduction syntax '[...]' is not supported by the "
            "xrtoolz.einx bridge; use the explicit 'in -> out' arrow form."
        )
    if "..." in slot:
        raise PatternError(
            "Ellipsis '...' is not supported by the xrtoolz.einx bridge yet; "
            "name every axis explicitly."
        )
    elements: list[Element] = []
    i = 0
    n = len(slot)
    while i < n:
        ch = slot[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            close = slot.find(")", i)
            if close == -1:
                raise PatternError(f"Unbalanced '(' in pattern slot: {slot!r}")
            inner = slot[i + 1 : close]
            if "(" in inner:
                raise PatternError(f"Nested parentheses are not supported: {slot!r}")
            members = _NAME.findall(inner)
            if not members:
                raise PatternError(f"Empty group in pattern slot: {slot!r}")
            elements.append(tuple(members))
            i = close + 1
            continue
        if ch == ")":
            raise PatternError(f"Unbalanced ')' in pattern slot: {slot!r}")
        match = _NAME.match(slot, i)
        if not match:
            raise PatternError(f"Unexpected character {ch!r} in pattern: {slot!r}")
        elements.append(match.group())
        i = match.end()
    return tuple(elements)


def parse_pattern(pattern: str) -> EinxPattern:
    """Parse an einx pattern string into an :class:`EinxPattern`.

    Args:
        pattern: einx pattern of the form ``"in0, in1, ... -> out"``.

    Raises:
        PatternError: if the pattern lacks exactly one ``->``, contains
            unsupported syntax (brackets / ellipsis / nested groups),
            or is otherwise malformed.
    """
    if pattern.count("->") != 1:
        raise PatternError(f"Pattern must contain exactly one '->'; got {pattern!r}.")
    lhs, rhs = pattern.split("->")
    input_slots = tuple(_parse_slot(slot) for slot in lhs.split(","))
    output = _parse_slot(rhs)
    return EinxPattern(raw=pattern, inputs=input_slots, output=output)


def infer_output_signature(
    pattern: str,
    input_sigs: Sequence[Signature],
    kwargs: Mapping[str, Any],
) -> Signature:
    """Infer the output :class:`Signature` of a labeled einx call.

    Sizes are taken from ``kwargs`` first, then from any input signature
    carrying the axis. A merged-group output dim is the product of its
    members' sizes when all are known, else ``None`` (unknown). The
    output dtype is the numpy promotion of the input dtypes.

    Args:
        pattern: einx pattern string.
        input_sigs: input signatures in pattern order.
        kwargs: ``shape_kwargs`` supplying sizes for new axes.
    """
    parsed = parse_pattern(pattern)
    known: dict[str, int | None] = {}
    for sig in input_sigs:
        for name, size in sig.dims.items():
            known.setdefault(name, size)
    for name, size in kwargs.items():
        if isinstance(size, int):
            known[name] = size

    dims: dict[str, int | None] = {}
    for el in parsed.output:
        if isinstance(el, str):
            dims[el] = known.get(el)
        else:
            sizes = [known.get(member) for member in el]
            if any(s is None for s in sizes):
                dims[_element_name(el)] = None
            else:
                product = 1
                for s in sizes:
                    if s is not None:
                        product *= s
                dims[_element_name(el)] = product

    dtypes = [sig.dtype for sig in input_sigs if sig.dtype is not None]
    dtype = _promote(dtypes) if dtypes else None
    return Signature(dims, dtype=dtype)


def _promote(dtypes: Sequence[Any]) -> Any:
    import numpy as np

    result = dtypes[0]
    for dt in dtypes[1:]:
        result = np.promote_types(result, dt)
    return result


__all__ = [
    "EinxPattern",
    "Element",
    "infer_output_signature",
    "parse_pattern",
]
