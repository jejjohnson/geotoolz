"""Graph composition — symbolic operator graphs.

`Sequential` covers the linear case; `Graph` covers the rest: branching
outputs, multi-input fusion, diamond dependencies. Construction is
symbolic — calling an `Operator` on `Input` / `Node` instances builds up
a graph; running it is `_apply(**inputs_by_name)`.

Pattern:

    img = Input("image")
    ref = Input("reference")
    ndvi    = NDVI(...)(img)                # Node(operator=NDVI, parents=(img,))
    rmse    = RMSE(...)(ndvi, ref)          # multi-input Node

    g = Graph(inputs={"image": img, "reference": ref},
              outputs={"ndvi": ndvi, "rmse": rmse})

    result = g(image=img_gt, reference=ref_gt)  # {"ndvi": GeoTensor, "rmse": float}

See `geotoolz.md` §4.2 (dual-mode `__call__`) and §6.3 (Graph spec).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from geotoolz.core._src.operator import Carrier, Operator


@dataclass(eq=False)
class Input:
    """A named entry point into a `Graph`.

    `Input` instances are placeholders during graph construction.
    `Operator.__call__` recognises them as graph mode (same as `Node`).
    `Graph._apply` consumes the keyword `**inputs` mapping by name.

    The dataclass disables ``__eq__`` so `Input` instances compare by
    identity — necessary for the ``id(...)``-keyed evaluation cache.
    """

    name: str
    parents: tuple[Any, ...] = field(default_factory=tuple)
    operator: Operator | None = None


@dataclass(eq=False)
class Node:
    """A non-input vertex in a `Graph`.

    Created automatically by `Operator.__call__` when any argument is an
    `Input` or another `Node`. Carries the operator and its parents
    (other `Input` / `Node` instances).

    Like `Input`, equality is by identity for the evaluation cache.
    """

    operator: Operator
    parents: tuple[Any, ...]


class Graph(Operator):
    """A symbolic operator graph with multiple inputs and outputs.

    Construction is implicit — calling operators on `Input` / `Node`
    instances builds the graph; ``Graph(inputs=..., outputs=...)`` wraps
    the result. ``_apply(**inputs)`` evaluates the graph in topological
    order.

    Inherits from `Operator` so a `Graph` satisfies the same interface
    as any other operator. ``Operator.__call__`` dispatches keyword args
    straight through to ``Graph._apply``; positional args are unused.

    Args:
        inputs: Map of ``input-name → Input`` placeholders. The keys are
            the keyword names accepted by ``__call__``.
        outputs: Map of ``output-name → Node`` (or ``Input``, if the
            output is a direct passthrough). The keys are the keys of the
            returned dict.

    Examples:
        Two-input, two-output graph::

            img = Input("image")
            ref = Input("reference")
            ndvi = NDVI(red_idx=2, nir_idx=3)(img)
            rmse = RMSE(axis=(-2, -1))(ndvi, ref)

            g = Graph(
                inputs={"image": img, "reference": ref},
                outputs={"ndvi": ndvi, "rmse": rmse},
            )
            result = g(image=img_gt, reference=ref_gt)
            # {"ndvi": GeoTensor, "rmse": scalar}

    Raises:
        ValueError: if the graph contains a cycle, if an output node
            isn't reachable from any input, or if an `Input` referenced
            by a node isn't declared in ``inputs``.
    """

    def __init__(
        self,
        inputs: dict[str, Input],
        outputs: dict[str, Node | Input],
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self._order = self._topological_sort()

    def _topological_sort(self) -> list[Node]:
        """Return a topological ordering of internal `Node`s.

        Inputs are not included (they are supplied by the caller, not
        computed). Output `Input`s aren't either — they pass straight
        through.
        """
        declared_inputs = set(map(id, self.inputs.values()))
        order: list[Node] = []
        visited: set[int] = set()
        on_stack: set[int] = set()

        def visit(node: Input | Node) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            if node_id in on_stack:
                raise ValueError(
                    "Cycle detected in graph — operator graphs must be DAGs."
                )
            on_stack.add(node_id)
            for parent in node.parents:
                visit(parent)
            on_stack.discard(node_id)
            visited.add(node_id)
            # Inputs are sources, not work to do during _apply.
            if isinstance(node, Input):
                if node_id not in declared_inputs:
                    raise ValueError(
                        f"Input {node.name!r} is referenced by an output but "
                        f"not declared in `inputs=`."
                    )
                return
            order.append(node)

        for output in self.outputs.values():
            visit(output)
        return order

    def _apply(self, *args: Carrier, **inputs: Carrier) -> dict[str, Any]:
        """Evaluate the graph with the supplied inputs.

        Accepts inputs either positionally (bound to declared `Input`s in
        construction order) or by keyword. The positional form makes
        single-input graphs compose with `Sequential` and lets `Graph`s
        nest inside other `Graph`s — both shapes route values through
        `Operator.__call__`, which only knows how to splat positionally.

        Args:
            *args: One value per declared `Input`, in declaration order.
                Mutually exclusive with ``**inputs``.
            **inputs: One value per declared `Input`, keyed by name.

        Returns:
            ``{output-name: result}`` for each declared output.
        """
        if args and inputs:
            raise TypeError(
                "Graph._apply accepts either positional args (bound to inputs "
                "in declaration order) or keyword inputs, not both."
            )
        if args:
            if len(args) != len(self.inputs):
                raise TypeError(
                    f"Graph expected {len(self.inputs)} positional argument(s) "
                    f"to bind to inputs {list(self.inputs)}, got {len(args)}."
                )
            inputs = dict(zip(self.inputs, args, strict=True))

        missing = set(self.inputs) - set(inputs)
        if missing:
            raise ValueError(f"Graph missing required input(s): {sorted(missing)}")

        cache: dict[int, Any] = {
            id(self.inputs[name]): inputs[name] for name in self.inputs
        }
        for node in self._order:
            node_args = tuple(cache[id(p)] for p in node.parents)
            # Route through __call__ so nested operators (Graph, Sequential)
            # get their own dispatch, not just bare _apply.
            cache[id(node)] = node.operator(*node_args)

        return {name: cache[id(node)] for name, node in self.outputs.items()}

    def get_config(self) -> dict[str, Any]:
        """Best-effort config — node operators' configs, by output name.

        Graphs are inherently runtime-defined (the topology comes from
        Python object identity), so the config is a debug repr rather than
        a faithful YAML round-trip. Future YAML support would store the
        topology as a list of (op, parent-keys) records.
        """
        return {
            "inputs": list(self.inputs),
            "outputs": {
                name: {
                    "class": type(node.operator).__name__
                    if isinstance(node, Node)
                    else "Input",
                    "config": node.operator.get_config()
                    if isinstance(node, Node)
                    else {},
                }
                for name, node in self.outputs.items()
            },
        }

    def __repr__(self) -> str:
        ins = ", ".join(self.inputs)
        outs = ", ".join(self.outputs)
        return f"Graph(inputs=[{ins}], outputs=[{outs}])"
