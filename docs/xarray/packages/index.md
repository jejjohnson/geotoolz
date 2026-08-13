# The Workspace Packages

`xrtoolz` is a uv workspace of five packages. The main `xrtoolz`
distribution carries the domain operator families and depends on the
other four, so `pip install xrtoolz` gives you the whole stack — but
each foundation layer also installs and works on its own.

| Distribution | Import | Scope | Standalone deps |
|---|---|---|---|
| [`xrtoolz-core`](core.md) | `xrcore` | xarray-aware `Operator` (DataTree dispatch) + `Signature` | pipekit, xarray |
| [`xrtoolz-grad`](grad.md) | `xrgrad` | Finite-difference calculus on labeled grids | xarray, finitediffx |
| [`xrtoolz-einx`](einx.md) | `xreinx` | xarray ↔ einx named-tensor bridge | xrcore, einx |
| [`xrtoolz-sklearn`](sklearn.md) | `xrsklearn` | scikit-learn ↔ xarray bridge | xrcore, scikit-learn |
| `xrtoolz` | `xrtoolz` | geo, ocn, budgets, interpolate, transforms, metrics, viz, … | all of the above |

## Which import do I use?

New code imports the scoped packages directly — `import xrgrad`,
`import xreinx`, `from xrsklearn import XarrayEstimator`, `from xrcore
import Operator, Signature`. The historical paths (`xrtoolz.calc`,
`xrtoolz.einx`, `xrtoolz.utils.XarrayEstimator`,
`xrtoolz.transforms.SklearnOp`) keep working through the 0.x series via
re-export shims; the renamed modules emit one `DeprecationWarning`
naming the replacement.

## Dependency shape

`xrcore` sits at the bottom (it is how every operator package speaks
pipekit's `Operator` protocol with xarray carriers). `xrgrad` is fully
standalone — it is also the only member that pulls the finitediffx/JAX
stack, so `xrtoolz-core`, `xrtoolz-einx`, and `xrtoolz-sklearn` install
without JAX. The main `xrtoolz` package consumes all four.
