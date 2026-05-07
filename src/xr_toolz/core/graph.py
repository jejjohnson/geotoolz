"""Functional Graph API — Input, Node, and Graph.

Mirrors the Keras functional API: build a computation DAG by calling
operators on symbolic :class:`Input` nodes, then compile the graph into
a :class:`Graph` operator. The DAG is executed eagerly in topological
order when the graph is called with concrete data.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from xr_toolz.core.operator import Operator
from xr_toolz.core.signature import Signature


class Node:
    """A symbolic placeholder for an intermediate result in a Graph.

    Nodes are created automatically when operators are called on
    :class:`Input` instances or other ``Node`` instances. Users do not
    instantiate these directly.
    """

    def __init__(
        self,
        operator: Operator | None,
        parents: tuple[Node, ...],
        name: str | None = None,
    ):
        self.operator = operator
        self.parents = tuple(parents)
        self.name = name

    def __repr__(self) -> str:
        if self.operator is None:
            return f"Input(name={self.name!r})"
        return f"Node({self.operator!r}, parents={len(self.parents)})"


class Input(Node):
    """A named entry point into a computation graph.

    Inputs have no parents and no operator. They are pure placeholders
    bound to real datasets at execution time.
    """

    def __init__(self, name: str):
        super().__init__(operator=None, parents=(), name=name)


class Graph(Operator):
    """A computation DAG compiled from symbolic :class:`Node` connections.

    ``Graph`` is itself an :class:`~xr_toolz.core.operator.Operator`, so
    graphs nest: a single-input / single-output graph can be a step
    inside a :class:`~xr_toolz.core.sequential.Sequential`, and any
    graph can be a node inside a larger graph.

    Calling conventions:

    - ``graph(**kwargs)``: bind each :class:`Input` by its declared
      name and return a dict keyed by output name.
    - ``graph(ds)``: positional shortcut for single-input /
      single-output graphs; returns the raw output value instead of a
      dict.
    - ``graph(some_node)``: symbolic mode — returns a new ``Node`` so
      the graph can be composed into another graph being built.
    """

    def __init__(
        self,
        inputs: dict[str, Input],
        outputs: dict[str, Node],
    ):
        self.inputs = dict(inputs)
        self.outputs = dict(outputs)
        self._execution_order = _topological_sort(self.inputs, self.outputs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Symbolic mode: if any positional argument is a Node, build a
        # composite node that defers execution to this graph.
        if any(isinstance(a, Node) for a in args):
            return Node(operator=self, parents=args)
        return self._apply(*args, **kwargs)

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        bindings = self._bind(args, kwargs)
        results = self._execute(bindings)
        if args and not kwargs and len(self.outputs) == 1:
            (only_output,) = self.outputs.values()
            return results[id(only_output)]
        return {name: results[id(node)] for name, node in self.outputs.items()}

    def get_config(self) -> dict[str, Any]:
        """Serialize the graph topology.

        Output nodes are flattened into a list of ``{class, config,
        parents}`` records using each node's index in execution order
        for cross-references. Raw data state (anything the operators
        close over) is *not* captured here.
        """
        idx = {id(node): i for i, node in enumerate(self._execution_order)}
        steps = []
        for i, node in enumerate(self._execution_order):
            if node.operator is None:
                steps.append({"index": i, "input": node.name})
                continue
            steps.append(
                {
                    "index": i,
                    "class": node.operator.__class__.__name__,
                    "config": node.operator.get_config(),
                    "parents": [idx[id(p)] for p in node.parents],
                }
            )
        return {
            "inputs": {name: idx[id(node)] for name, node in self.inputs.items()},
            "outputs": {name: idx[id(node)] for name, node in self.outputs.items()},
            "steps": steps,
        }

    def compute_output_signature(
        self,
        input_signature: Signature | dict[str, Signature],
    ) -> Signature | dict[str, Signature]:
        """Propagate signatures through the graph without executing data.

        Mirrors :meth:`__call__`'s positional-shortcut behaviour: a bare
        :class:`Signature` is allowed when the graph has exactly one
        :class:`Input`, and the return type matches the output count
        (single ``Signature`` for a 1-output graph, ``dict`` for a
        multi-output graph). This keeps nested-graph composition working
        — an outer operator wrapping an inner multi-output graph passes
        a single parent signature here and gets back a dict.
        """
        if isinstance(input_signature, Signature):
            if len(self.inputs) != 1:
                raise ValueError(
                    "Single Signature input requires exactly one graph input; "
                    f"got {len(self.inputs)} inputs. Pass a "
                    "{name: Signature} dict for multi-input graphs."
                )
            input_signatures = {next(iter(self.inputs)): input_signature}
            outputs = self._compute_signatures(input_signatures)
            if len(self.outputs) == 1:
                return outputs[next(iter(self.outputs))]
            return outputs
        return self._compute_signatures(input_signature)

    def summary(self, input_signatures: Signature | dict[str, Signature]) -> str:
        """Render input/output signatures for each graph node."""
        from xr_toolz.core.sequential import _format_summary_table

        if isinstance(input_signatures, Signature):
            if len(self.inputs) != 1:
                raise ValueError(
                    "Single Signature summary requires exactly one graph input; "
                    f"got {len(self.inputs)} inputs."
                )
            input_signatures = {next(iter(self.inputs)): input_signatures}
        missing = set(self.inputs) - set(input_signatures)
        extra = set(input_signatures) - set(self.inputs)
        if missing or extra:
            raise ValueError(
                f"Graph input signatures mismatch. missing={sorted(missing)}, "
                f"unexpected={sorted(extra)}"
            )
        cache: dict[int, Signature] = {
            id(node): input_signatures[name] for name, node in self.inputs.items()
        }
        rows: list[tuple[str, str, str, str]] = []
        for i, node in enumerate(self._execution_order):
            if node.operator is None:
                continue
            parent_signatures = tuple(cache[id(parent)] for parent in node.parents)
            input_signature = (
                parent_signatures[0]
                if len(parent_signatures) == 1
                else parent_signatures
            )
            output_signature = node.operator.compute_output_signature(input_signature)
            cache[id(node)] = output_signature
            rows.append(
                (
                    str(i),
                    repr(node.operator),
                    _format_graph_signature(input_signature),
                    output_signature.format(),
                )
            )
        return _format_summary_table(
            f"Graph ({_format_count(len(self.inputs), 'input')}, "
            f"{_format_count(len(self.outputs), 'output')})",
            rows,
        )

    def _bind(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[int, Any]:
        if args and not kwargs:
            if len(args) != 1 or len(self.inputs) != 1:
                raise ValueError(
                    "Positional call requires exactly one input and one "
                    f"argument; got {len(args)} args and "
                    f"{len(self.inputs)} inputs."
                )
            (only_input,) = self.inputs.values()
            return {id(only_input): args[0]}

        missing = set(self.inputs) - set(kwargs)
        extra = set(kwargs) - set(self.inputs)
        if missing or extra:
            raise ValueError(
                f"Graph inputs mismatch. missing={sorted(missing)}, "
                f"unexpected={sorted(extra)}"
            )
        return {id(node): kwargs[name] for name, node in self.inputs.items()}

    def _execute(self, bindings: dict[int, Any]) -> dict[int, Any]:
        cache: dict[int, Any] = dict(bindings)
        for node in self._execution_order:
            if id(node) in cache:
                continue
            if node.operator is None:
                raise ValueError(f"Input node {node.name!r} was not bound to data.")
            parent_values = tuple(cache[id(p)] for p in node.parents)
            cache[id(node)] = node.operator._apply(*parent_values)
        return cache

    def _compute_signatures(
        self,
        input_signatures: dict[str, Signature],
    ) -> dict[str, Signature]:
        missing = set(self.inputs) - set(input_signatures)
        extra = set(input_signatures) - set(self.inputs)
        if missing or extra:
            raise ValueError(
                f"Graph input signatures mismatch. missing={sorted(missing)}, "
                f"unexpected={sorted(extra)}"
            )
        cache: dict[int, Signature] = {
            id(node): input_signatures[name] for name, node in self.inputs.items()
        }
        for node in self._execution_order:
            if id(node) in cache:
                continue
            if node.operator is None:
                # We're propagating signatures here, not executing data —
                # pin the wording so failures during compute_output_signature
                # / summary point at the right thing.
                raise ValueError(
                    f"Graph input {node.name!r} has no signature; pass it "
                    "via the input_signatures mapping."
                )
            parent_signatures = tuple(cache[id(parent)] for parent in node.parents)
            input_signature = (
                parent_signatures[0]
                if len(parent_signatures) == 1
                else parent_signatures
            )
            cache[id(node)] = node.operator.compute_output_signature(input_signature)
        return {name: cache[id(node)] for name, node in self.outputs.items()}

    def __repr__(self) -> str:
        # Operator base __repr__ would dump the full topology via
        # get_config(), which produces a multi-hundred-char string and
        # blows up Sequential.summary tables when a Graph is nested
        # inside another Graph or a Sequential. Keep it terse — describe()
        # / get_config() expose the full topology when needed.
        return f"Graph(inputs={list(self.inputs)!r}, outputs={list(self.outputs)!r})"

    def describe(self) -> str:
        """Pretty-print the graph structure."""
        lines = [
            f"Graph ({len(self.inputs)} inputs, {len(self.outputs)} outputs):",
            f"  Inputs: {list(self.inputs.keys())}",
        ]
        for i, node in enumerate(self._execution_order):
            if node.operator is None:
                continue
            parent_labels = [p.name or f"<node {id(p)}>" for p in node.parents]
            lines.append(f"  [{i}] {node.operator!r} ← {parent_labels}")
        lines.append(f"  Outputs: {list(self.outputs.keys())}")
        return "\n".join(lines)


def _topological_sort(
    inputs: dict[str, Input],
    outputs: dict[str, Node],
) -> list[Node]:
    """Kahn's algorithm over the reachable subgraph rooted at ``outputs``."""

    reachable: dict[int, Node] = {}
    stack: list[Node] = list(outputs.values())
    while stack:
        node = stack.pop()
        if id(node) in reachable:
            continue
        reachable[id(node)] = node
        stack.extend(node.parents)

    input_ids = {id(node) for node in inputs.values()}
    for node_id in input_ids:
        if node_id not in reachable:
            raise ValueError("At least one declared Input is not used by any output.")

    in_degree: dict[int, int] = {
        nid: sum(1 for p in node.parents if id(p) in reachable)
        for nid, node in reachable.items()
    }
    children: dict[int, list[Node]] = {nid: [] for nid in reachable}
    for node in reachable.values():
        for parent in node.parents:
            if id(parent) in reachable:
                children[id(parent)].append(node)

    queue: deque[Node] = deque(
        reachable[nid] for nid, deg in in_degree.items() if deg == 0
    )
    ordered: list[Node] = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for child in children[id(node)]:
            in_degree[id(child)] -= 1
            if in_degree[id(child)] == 0:
                queue.append(child)

    if len(ordered) != len(reachable):
        raise ValueError("Graph contains a cycle.")
    return ordered


def _format_graph_signature(signature: Signature | tuple[Signature, ...]) -> str:
    if isinstance(signature, tuple):
        return ", ".join(sig.format() for sig in signature)
    return signature.format()


def _format_count(count: int, noun: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix}"
