"""DimArray: data, dims and labels, nothing else (PK-P7, pipekit#101).

What a geo carrier becomes when a kernel consumes one of its spatial dims
(per-column statistics consume ``y`` and keep ``x``). A pytree whose only
leaf is the data.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, eq=False)
class DimArray:
    """A raw array (NumPy, JAX or dask) with named dims and static labels."""

    data: Any
    dims: tuple[str, ...]
    labels: tuple[tuple[str, tuple[Any, ...]], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(self.dims))
        object.__setattr__(self, "labels", tuple(self.labels))
        if len(self.dims) != len(self.data.shape):
            raise ValueError(f"dims {self.dims} for data of shape {self.data.shape}")

    @property
    def shape(self) -> tuple[int, ...]:
        """The data's shape."""
        return tuple(self.data.shape)

    def labels_of(self, dim: str) -> tuple[Any, ...] | None:
        """Static labels of one dim, or None."""
        return dict(self.labels).get(dim)

    def __repr__(self) -> str:
        return f"DimArray[{' '.join(self.dims)}]{self.shape}"


def _build(data: Any, dims: Any, labels: Any) -> DimArray:
    d = object.__new__(DimArray)
    object.__setattr__(d, "data", data)
    object.__setattr__(d, "dims", tuple(dims))
    object.__setattr__(d, "labels", tuple(labels))
    return d


def _register() -> None:
    try:
        import jax
    except ImportError:  # pragma: no cover
        return
    jax.tree_util.register_pytree_node(
        DimArray,
        lambda d: ((d.data,), (d.dims, d.labels)),
        lambda aux, leaves: _build(leaves[0], aux[0], aux[1]),
    )


_register()
