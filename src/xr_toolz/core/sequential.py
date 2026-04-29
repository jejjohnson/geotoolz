"""Linear chain of single-input operators."""

from __future__ import annotations

from typing import Any

from xr_toolz.core.operator import Operator


class Sequential(Operator):
    """A pipeline of single-input operators, applied left to right.

    ``Sequential`` is itself an :class:`Operator`, so pipelines nest::

        preprocess = Sequential([ValidateCoords(), Regrid(grid)])
        full = Sequential([preprocess, RemoveClimatology(clim)])
    """

    def __init__(self, operators: list[Operator]):
        self.operators = list(operators)

    def _apply(self, ds: Any) -> Any:
        for op in self.operators:
            ds = op(ds)
        return ds

    def get_config(self) -> dict[str, Any]:
        return {
            "operators": [
                {"class": op.__class__.__name__, "config": op.get_config()}
                for op in self.operators
            ]
        }

    def describe(self, *, max_width: int = 88) -> str:
        """Pretty-print the pipeline as an indented tree.

        Nested ``Sequential`` instances are expanded inline so the full
        composition is visible. Each operator is rendered as
        ``ClassName(k=v, ...)`` from its :meth:`get_config`; long config
        strings are wrapped at ``max_width`` with continuation lines
        aligned under the operator name.
        """
        return "\n".join(self._describe_lines(max_width=max_width))

    def _describe_lines(self, *, max_width: int = 88) -> list[str]:
        header = f"Sequential ({len(self.operators)} ops)"
        lines = [header]
        n = len(self.operators)
        for i, op in enumerate(self.operators):
            is_last = i == n - 1
            branch = "└── " if is_last else "├── "
            cont = "    " if is_last else "│   "
            inner_width = max_width - len(branch)
            if isinstance(op, Sequential):
                child = op._describe_lines(max_width=inner_width)
            else:
                child = _format_op(op, max_width=inner_width)
            lines.append(branch + child[0])
            lines.extend(cont + ln for ln in child[1:])
        return lines

    def __repr__(self) -> str:
        return f"Sequential({self.operators!r})"

    def __or__(self, other: Operator) -> Sequential:
        """Flatten chains: ``seq | op`` and ``seq | other_seq`` stay flat."""
        rhs = other.operators if isinstance(other, Sequential) else [other]
        return Sequential([*self.operators, *rhs])


def _format_op(op: Operator, *, max_width: int) -> list[str]:
    """Render a single operator as one or more lines.

    Single line when ``ClassName(k=v, ...)`` fits within ``max_width``;
    otherwise the kwargs are split onto continuation lines indented
    under the open paren.
    """
    name = op.__class__.__name__
    config = op.get_config()
    if not config:
        return [f"{name}()"]
    parts = [f"{k}={v!r}" for k, v in config.items()]
    one_line = f"{name}({', '.join(parts)})"
    if len(one_line) <= max_width:
        return [one_line]
    indent = " " * (len(name) + 1)
    head = f"{name}("
    last = len(parts) - 1
    body = []
    for j, part in enumerate(parts):
        prefix = head if j == 0 else indent
        suffix = ")" if j == last else ","
        body.append(prefix + part + suffix)
    return body
