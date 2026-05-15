"""End-to-end composition tests exercising multiple primitives together."""

from __future__ import annotations

from geotoolz.core import (
    Branch,
    Fanout,
    Graph,
    Identity,
    Input,
    Operator,
    Sequential,
    Snapshot,
    Switch,
    Tap,
)


class _Add(Operator):
    def __init__(self, n: int) -> None:
        self.n = n

    def _apply(self, x: int) -> int:
        return x + self.n

    def get_config(self) -> dict:
        return {"n": self.n}


def test_sequential_with_tap_branch_and_snapshot() -> None:
    """Sequential containing observers, control flow, and identity."""
    snap = Snapshot()
    seen: list[int] = []

    pipe = Sequential(
        [
            _Add(1),
            Tap(seen.append),  # observer
            snap.at("after_tap"),  # observer (private subclass)
            Branch(
                predicate=lambda x: x > 0,
                if_true=_Add(10),
                if_false=Identity(),
            ),  # control flow
            snap.at("final"),
        ]
    )

    out = pipe(0)
    assert out == 11  # 0 + 1 + 10
    assert seen == [1]
    assert snap["after_tap"] == 1
    assert snap["final"] == 11


def test_fanout_after_sequential() -> None:
    """Pipe a Sequential's output through a Fanout."""
    preprocess = Sequential([_Add(1), _Add(2)])
    products = Fanout(
        {
            "plus10": _Add(10),
            "plus100": _Add(100),
        }
    )

    result = products(preprocess(0))
    assert result == {"plus10": 13, "plus100": 103}


def test_graph_with_internal_sequential_and_fanout() -> None:
    """A Graph whose nodes contain composite operators."""
    x = Input("x")
    pre = Sequential([_Add(1), _Add(2)])(x)  # x → x + 3
    g = Graph(inputs={"x": x}, outputs={"y": pre})
    assert g(x=0) == {"y": 3}


def test_switch_dispatches_into_sequential() -> None:
    """Switch routes the *same* input through different Sequential pipelines."""
    # The key callable and the case operators all receive the same value.
    # Here we route on parity of the input integer: even → +1, odd → +100,+100.
    pipe = Switch(
        key=lambda x: "even" if x % 2 == 0 else "odd",
        cases={
            "even": Sequential([_Add(1)]),
            "odd": Sequential([_Add(100), _Add(100)]),
        },
    )
    assert pipe(4) == 5
    assert pipe(3) == 203
