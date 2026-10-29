# The Workspace Packages

`xrtoolz` is a uv workspace of seven packages. `xrtoolz-ops` (import
`xrops`) carries the domain operator families, and the `xrtoolz`
umbrella holds no code and installs the other six, so `pip install
xrtoolz` gives you the whole stack — but each package also installs and
works on its own.

| Distribution | Import | Scope | Standalone deps |
|---|---|---|---|
| [`xrtoolz-core`](core.md) | `xrcore` | xarray-aware `Operator` (DataTree dispatch) + `Signature` | pipekit, xarray |
| [`xrtoolz-grad`](grad.md) | `xrgrad` | Finite-difference calculus on labeled grids | xarray, finitediffx |
| [`xrtoolz-einx`](einx.md) | `xreinx` | xarray ↔ einx named-tensor bridge | xrcore, einx |
| [`xrtoolz-sklearn`](sklearn.md) | `xrsklearn` | scikit-learn ↔ xarray bridge | xrcore, scikit-learn |
| [`xrtoolz-products`](reader.md) | `xrproducts` | Authenticated archive readers (CMEMS, CDS, AEMET) | numpy, pandas, xarray |
| `xrtoolz-ops` | `xrops` | geo, ocn, budgets, interpolate, transforms, metrics, viz, … | all of the above |
| `xrtoolz` | `xrtoolz` | the umbrella: no code, installs the family | all of the above |

## Which import do I use?

Import each package under its own name — `import xrops`, `import xrgrad`,
`import xreinx`, `from xrsklearn import XarrayEstimator`, `from xrcore
import Operator, Signature`. There are no compatibility paths: the old
`xrtoolz.*` modules (and the `xrtoolz.calc`, `xrtoolz.einx`,
`xrtoolz.utils.XarrayEstimator` and `xrtoolz.transforms.SklearnOp`
re-exports) were removed when the operators moved to `xrops`.

## Dependency shape

`xrcore` sits at the bottom (it is how every operator package speaks
pipekit's `Operator` protocol with xarray carriers). `xrgrad` is fully
standalone — it is also the only member that pulls the finitediffx/JAX
stack, so `xrtoolz-core`, `xrtoolz-einx`, and `xrtoolz-sklearn` install
without JAX. The main `xrtoolz` package consumes all five — it depends on
`xrtoolz-products` for the shared typed request vocabulary and CF
`Variable` registry, whose service clients stay behind optional extras.
