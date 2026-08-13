"""Inference operators — wrap trained models as Layer-1 Operators.

Per design decision D4, this module is **framework-agnostic**: it never
imports ``sklearn``, ``jax``, ``torch``, or ``equinox`` at module load
time. Backend-specific subclasses (``SklearnModelOp``, ``JaxModelOp``)
defer their imports until first use.

The classes are not re-exported from :mod:`xrtoolz` itself; users
opt-in with ``from xrtoolz.inference import ModelOp`` so that simply
``import xrtoolz`` never pulls a heavy ML stack into ``sys.modules``.
"""

from __future__ import annotations

from xrtoolz.inference.modelop import JaxModelOp, ModelOp, SklearnModelOp


__all__ = [
    "JaxModelOp",
    "ModelOp",
    "SklearnModelOp",
]
