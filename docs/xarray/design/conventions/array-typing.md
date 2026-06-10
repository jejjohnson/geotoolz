# Array typing & docstring conventions

This note is the standard the codebase is being swept toward. It makes the
numpy ↔ xarray boundary explicit and self-documenting.

## Two layers

`xrtoolz` keeps a strict two-layer split inside every domain submodule:

| Layer | Speaks | Signature style | Lives in |
|-------|--------|-----------------|----------|
| **Operators / primitives** | `xr.DataArray` / `xr.Dataset` | `dim: str` (named dims), coords & attrs preserved | `operators.py`, `_src/<topic>.py` |
| **numpy kernels** | `np.ndarray` | `axis: int` (positional axes), pure array math | `_src/_<topic>_kernels.py` |

Everything users touch — operators and the pure-function primitives — takes
and returns xarray objects. Raw numpy only appears in the private kernels,
which the xarray layer wraps (adding coord/attr handling and dim→axis
translation).

## numpy kernels MUST be jaxtyped

Every array parameter and return of a numpy kernel is annotated with
[`jaxtyping`](https://docs.kidger.site/jaxtyping/) so the **dtype and shape**
are part of the signature:

```python
from __future__ import annotations

import numpy as np
from jaxtyping import Float


def partial(
    x: Float[np.ndarray, "*shape"],
    *,
    axis: int,
    spacing: float = 1.0,
) -> Float[np.ndarray, "*shape"]:
    """Partial derivative ∂x/∂<axis> via 2nd-order central differences.

    Args:
        x: Real-valued field of arbitrary shape ``(*shape)``.
        axis: Axis to differentiate along.
        spacing: Uniform sample spacing along ``axis``.

    Returns:
        The derivative, same shape ``(*shape)`` as ``x``.
    """
    ...
```

Rules:

- **Annotations only.** We do **not** wrap kernels in `@jaxtyped`/runtime
  checkers — the annotations are documentation and static types, so there is
  zero runtime cost. `from __future__ import annotations` keeps them as
  strings.
- **Dtype** picks the narrowest correct jaxtyping class: `Float`, `Int`,
  `Bool`, `Complex`, or `Num` (any numeric) — e.g. an FFT kernel returns
  `Complex[np.ndarray, ...]`.
- **Shape axes** are named to carry meaning *and* express relationships:
  - semantic names when the kernel is dimension-aware: `Float[np.ndarray, "lat lon"]`;
  - a shared name across params/returns expresses "same shape": reuse
    `"*shape"` (or `"n"`, `"lat lon"`) on both sides;
  - `*name` for a variadic run of axes (leading batch dims, arbitrary rank):
    `Float[np.ndarray, "*batch n"]`, `Float[np.ndarray, "*shape"]`.
- **Document the shape in the docstring** too (Args/Returns), so it reads
  without knowing jaxtyping.

### Tooling

`jaxtyping` is a core dependency (it only pulls `wadler-lindig`; no JAX). ruff
ignores `F722`, `F821`, and `UP037` repo-wide because pyflakes / pyupgrade
read the shape strings (`"lat lon"`, `"*shape"`) as forward-ref annotations
and would otherwise flag the named axes as undefined or strip their required
quotes — this is jaxtyping's recommended config. `ty` understands the
annotations with no extra configuration and remains the undefined-name safety
net.

## Docstrings

Google-style, on every public operator, primitive, and kernel. Operators
follow this template (see `geo.Reduce` / `interpolate.Coarsen` as exemplars):

```python
class Thing(Operator):
    """One-sentence purpose.

    Optional 1–3 sentence extended description.

    Args:
        param: Type + meaning; say if required/optional and the default.

    Returns:
        The xarray result — name the produced variables / dims when useful.
    """
```

Kernels additionally state array **shapes** in Args/Returns (mirroring the
jaxtyping axes).

## Reference implementation

`xrtoolz.calc` is the worked example of this convention end-to-end:
`_src/_calc_kernels.py` holds the jaxtyped numpy kernels; `_src/operators.py`
and the geometry modules are the xarray layer with full Google-style
docstrings. New/updated modules should match it.
