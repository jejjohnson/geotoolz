# Inference

Wrap a trained model as a Layer-1 `Operator` so it drops into a
`Sequential` or `Graph` like any other step. `ModelOp` is framework-
agnostic (duck-typed `predict` / `__call__`); `SklearnModelOp` and
`JaxModelOp` add framework-specific adapters. The ML backends are imported
lazily, so installing JAX or scikit-learn is only required for the wrapper
you actually use.

::: xrtoolz.inference.ModelOp

::: xrtoolz.inference.SklearnModelOp

::: xrtoolz.inference.JaxModelOp
