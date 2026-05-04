"""Tier B/C — learned resolution-change operators (D12, F3.4).

Per D12, ``Downscale`` is *learned* refinement (super-resolution) and
``Upscale`` is *learned* aggregation (subgrid-scale surrogate); the
deterministic counterparts ``Refine`` / ``Coarsen`` live in
:mod:`._src.grid_to_grid`.

Per the F3.5 resolution of D12 Q1, both operators are pure callable
wrappers — they do not carry ``patch_size`` / ``overlap`` constructor
args. Patch tiling is delegated to ``xrpatcher`` upstream of the
operator (compose ``patcher | Downscale(model) | unpatcher``).

The wrapped ``model`` is duck-typed: it can be a
:class:`~xr_toolz.inference.modelop.ModelOp`, any other
:class:`~xr_toolz.core.Operator`, or a plain callable that maps an
input container (Dataset / DataArray / array) to an output container.
No framework imports here (D4).
"""

from __future__ import annotations

from typing import Any

from xr_toolz.core import Operator


class _ResolutionOp(Operator):
    """Common base for learned resolution-change operators.

    Stores a callable ``model`` and an optional ``target_grid`` (a
    Dataset, DataArray, or any object the user wants to attach as a
    reference output specification — not used inside ``_apply``; the
    model is responsible for emitting the right resolution).
    """

    def __init__(self, model: Any, *, target_grid: Any = None) -> None:
        if not callable(model):
            raise TypeError(
                f"model must be callable (Operator / ModelOp / function), "
                f"got {type(model).__name__}"
            )
        self.model = model
        self.target_grid = target_grid

    def _apply(self, data: Any) -> Any:
        return self.model(data)

    def get_config(self) -> dict[str, Any]:
        model_repr = (
            type(self.model).__name__
            if not isinstance(self.model, type)
            else self.model.__name__
        )
        return {
            "model": f"<{model_repr}>",
            "target_grid": "<grid>" if self.target_grid is not None else None,
        }


class Downscale(_ResolutionOp):
    """Learned refinement (super-resolution).

    Wraps a model that maps a coarse-resolution input to a
    fine-resolution output. Per D12, ``Refine`` is the deterministic
    counterpart in :mod:`xr_toolz.interpolate._src.grid_to_grid`.
    """


class Upscale(_ResolutionOp):
    """Learned aggregation (subgrid-scale surrogate).

    Wraps a model that maps a fine-resolution input to a
    coarse-resolution output. Per D12, ``Coarsen`` is the deterministic
    counterpart in :mod:`xr_toolz.interpolate._src.grid_to_grid`.
    """


__all__ = ["Downscale", "Upscale"]
