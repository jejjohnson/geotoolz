# The Workspace Packages

`xrtoolz` is a uv workspace of three packages. `xrtoolz-ops` (import
`xrops`) carries the domain operator families, and the `xrtoolz`
umbrella holds no code and installs the other two, so `pip install
xrtoolz` gives you the whole stack — but each package also installs and
works on its own.

| Distribution | Import | Scope | Standalone deps |
|---|---|---|---|
| [`xrtoolz-products`](reader.md) | `xrproducts` | Authenticated archive readers (CMEMS, CDS, AEMET) | numpy, pandas, xarray |
| `xrtoolz-ops` | `xrops` | geo, ocn, budgets, interpolate, transforms, metrics, viz, calculus ([`xrops.grad`](grad.md)), … | xrtoolz-products, pipekit-array, geotoolz-ops |
| `xrtoolz` | `xrtoolz` | the umbrella: no code, installs the family | all of the above |

## Which import do I use?

Import each package under its own name — `import xrops`, `from xrops import grad`,
`from xrops import Operator`, `from xrops.learn import XarrayEstimator`, `from pipekit import Signature`. There are no compatibility paths: the old
`xrtoolz.*` modules (and the `xrtoolz.calc`, `xrtoolz.einx`,
`xrtoolz.utils.XarrayEstimator` and `xrtoolz.transforms.SklearnOp`
re-exports) were removed when the operators moved to `xrops`.

## Dependency shape

`xrops.Operator` (in `xrops._src`) is how every xrops operator speaks pipekit's `Operator` protocol with xarray carriers. The calculus in `xrops.grad` runs on geoops's difference kernels, so no member pulls JAX. The main `xrtoolz` package consumes both — it depends on
`xrtoolz-products` for the shared typed request vocabulary and CF
`Variable` registry, whose service clients stay behind optional extras.
