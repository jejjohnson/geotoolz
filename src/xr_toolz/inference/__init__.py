"""Inference operators — wrap trained models as Layer-1 Operators.

Per design decision D4, this module is **framework-agnostic**: it never
imports ``sklearn``, ``jax``, ``torch``, or ``equinox`` at module load
time. Backend-specific subclasses (``SklearnModelOp``, ``JaxModelOp``)
defer their imports until first use.

The classes are not re-exported from :mod:`xr_toolz` itself; users
opt-in with ``from xr_toolz.inference import ModelOp`` so that simply
``import xr_toolz`` never pulls a heavy ML stack into ``sys.modules``.
"""

from __future__ import annotations

from xr_toolz.inference.modelop import JaxModelOp, ModelOp, SklearnModelOp


__all__ = [
    "JaxModelOp",
    "ModelOp",
    "SklearnModelOp",
]
