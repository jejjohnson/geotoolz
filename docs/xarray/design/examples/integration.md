---
status: draft
version: 0.1.0
---

!!! note "Imports in this page are from the original `geo_toolz` layout"
    These design docs were adapted from the `geo_toolz` design study.
    Code snippets use the original feature-based import paths
    (`geo_toolz.<module>`). In `xr_toolz`, the domain-agnostic
    operations live under `xr_toolz.geo.<module>`; physics-specific
    operations live under `xr_toolz.ocn` / `xr_toolz.atm` /
    `xr_toolz.rs`. See `xr_toolz/__init__.py` and
    `xr_toolz/geo/__init__.py` for the current export surface.

# Layer 3 — Integration Examples

Cross-library patterns: sklearn, xrpatcher, xarray_sklearn, and ecosystem composition.

---

## With sklearn (SklearnOp)

### PCA in a geo_toolz pipeline

```python
from geo_toolz.core import Sequential
from geo_toolz.validation import ValidateCoords
from geo_toolz.regrid import Regrid
from geo_toolz.detrend import RemoveClimatology
from geo_toolz.sklearn import SklearnOp
from sklearn.decomposition import PCA

clim = CalculateClimatology(freq="day", smoothing=60)(ds_train)

pipeline = Sequential([
    ValidateCoords(),
    Regrid(target_lon=target_lon, target_lat=target_lat, method="linear"),
    RemoveClimatology(clim),
    SklearnOp(PCA(n_components=10), sample_dim="time", new_feature_dim="mode"),
])

# Raw satellite data → regridded → detrended → first 10 EOFs
pcs = pipeline(ds_raw)
```

### NaN-safe scaling on ocean data

```python
from geo_toolz.sklearn import SklearnOp
from sklearn.preprocessing import StandardScaler

# nan_policy="mask" drops NaN ocean/land cells, scales, then re-inserts NaN
scale_op = SklearnOp(StandardScaler(), sample_dim="time", nan_policy="mask")
ds_scaled = scale_op(ds_ocean)
```

---

## With xrpatcher (Patch-wise Processing)

### Apply a pipeline per spatial patch — bounded memory

```python
from xrpatcher import XRDAPatcher
from geo_toolz.core import Sequential
from geo_toolz.detrend import RemoveClimatology
from geo_toolz.inference import JaxModelOp

patcher = XRDAPatcher(
    da=ssh_global,
    patches={"lat": 64, "lon": 64},
    strides={"lat": 64, "lon": 64},
)

infer_pipeline = Sequential([
    RemoveClimatology(clim),
    JaxModelOp(trained_model, sample_dim="time", jit=True, batch_size=1024),
])

# Each 64x64 patch: preprocess → predict, bounded memory
predictions = [infer_pipeline(patcher[i]) for i in range(len(patcher))]
full_prediction = patcher.reconstruct(predictions)
```

### Overlapping patches for smooth spatial metrics

```python
from xrpatcher import XRDAPatcher
from geo_toolz.metrics import RMSE

# 50% overlap to reduce boundary artifacts
patcher_pred = XRDAPatcher(da=ds_pred, patches={"lat": 64, "lon": 64}, strides={"lat": 32, "lon": 32})
patcher_ref  = XRDAPatcher(da=ds_ref,  patches={"lat": 64, "lon": 64}, strides={"lat": 32, "lon": 32})

rmse_op = RMSE(variable="ssh", dims=["time"])
regional_rmse = [rmse_op(patcher_pred[i], patcher_ref[i]) for i in range(len(patcher_pred))]
```

---

## With xarray_sklearn + xrpatcher (Patch-wise ML)

### Per-region EOF analysis on large grids

```python
from xrpatcher import XRDAPatcher
from geo_toolz.core import Sequential
from geo_toolz.detrend import RemoveClimatology
from geo_toolz.sklearn import SklearnOp
from sklearn.decomposition import PCA

pipeline = Sequential([
    RemoveClimatology(clim),
    SklearnOp(PCA(n_components=5), sample_dim="time", new_feature_dim="mode"),
])

patcher = XRDAPatcher(da=ssh_global, patches={"lat": 64, "lon": 64}, strides={"lat": 64, "lon": 64})

# Each 64x64 region gets its own detrending + local EOF decomposition
regional_eofs = [pipeline(patcher[i]) for i in range(len(patcher))]
```

### Spatiotemporal patches for batch normalisation

```python
from xrpatcher import XRDAPatcher
from geo_toolz.core import Sequential
from geo_toolz.sklearn import SklearnOp
from sklearn.preprocessing import MinMaxScaler

patcher = XRDAPatcher(
    da=ssh_global,
    patches={"time": 90, "lat": 32, "lon": 32},
    strides={"time": 90, "lat": 32, "lon": 32},
)

normalise = Sequential([
    ValidateCoords(),
    SklearnOp(MinMaxScaler(), sample_dim="time"),
])

# Each chunk independently normalised — ready for ML temporal windows
normalised = [normalise(patcher[i]) for i in range(len(patcher))]
```

---

## With ekalmX (DA pre/post-processing)

### Preprocess observations, evaluate analysis output

```python
import geo_toolz as gt
import ekalmx

# Preprocessing
raw_obs = xr.open_dataset("satellite_ssh.nc")
pipeline = gt.Sequential([
    gt.validation.ValidateCoords(),
    gt.subset.SubsetBBox(lon=(-80, 0), lat=(20, 60)),
    gt.regrid.Regrid(target_grid=model_grid),
    gt.detrend.RemoveClimatology(clim),
])
obs_clean = pipeline(raw_obs)
obs_values = jnp.array(obs_clean["ssh_anomaly"].values)

# ekalmX assimilation
result = letkf.assimilate(ensemble, obs_values, ...)

# Evaluation
analysis_ds = xr.Dataset({"ssh": (["ens", "y", "x"], result.particles)})
eval_pipeline = gt.Sequential([
    gt.metrics.RMSE(variable="ssh", reference=truth_dataset),
    gt.spectral.PSD(variable="ssh", dim="x"),
])
evaluation = eval_pipeline(analysis_ds)
```

---

## With gaussx (Structured Covariance for Spatial Modeling)

### Spatiotemporal GP on preprocessed satellite data

geo_toolz preprocesses the xarray data; gaussx provides the structured covariance operators for a Kronecker GP on the resulting regular grid.

```python
import jax.numpy as jnp
import lineax as lx
from gaussx.operators import KroneckerOperator
from gaussx.ops import solve, logdet
from gaussx.distributions import MultivariateNormal
from gaussx.solvers import AutoSolver
import geo_toolz as gt

# --- geo_toolz: preprocess satellite SSH to a regular grid ---
preprocess = gt.Sequential([
    gt.validation.ValidateCoords(),
    gt.subset.SubsetBBox(lon_bnds=(-30, 45), lat_bnds=(25, 65)),
    gt.regrid.Regrid(target_lon=lon_grid, target_lat=lat_grid, method="linear"),
    gt.detrend.RemoveClimatology(clim),
])

ds_clean = preprocess(xr.open_dataset("satellite_ssh.nc"))

# Flatten the regular grid to a vector for GP modeling
y = jnp.array(ds_clean["ssh_anomaly"].values.reshape(-1))

# --- gaussx: Kronecker GP on the regular grid ---
# Because geo_toolz regridded to a regular grid, the covariance factorizes
K_space = lx.MatrixLinearOperator(rbf_kernel(lon_grid, lat_grid))  # (n_space, n_space)
K_time = lx.MatrixLinearOperator(matern_kernel(time_grid))          # (n_time, n_time)

# Kronecker: (n_space * n_time) × (n_space * n_time) — never materialized
K = KroneckerOperator(K_space, K_time)
K_noisy = K + lx.DiagonalLinearOperator(jnp.full(len(y), noise_var))

# Log marginal likelihood — O(n_space^3 + n_time^3) via Kronecker decomposition
alpha = solve(K_noisy, y)
lml = -0.5 * (y @ alpha + logdet(K_noisy) + len(y) * jnp.log(2 * jnp.pi))

# Posterior predictive distribution — structured MVN
posterior = MultivariateNormal(
    loc=K @ alpha,
    cov_operator=K_noisy,
    solver=AutoSolver(),
)
```

### Ensemble covariance from DA output — structured diagnostics

geo_toolz evaluates ekalmX analysis fields; gaussx provides the low-rank ensemble covariance for structured diagnostics.

```python
import jax.numpy as jnp
from gaussx.recipes import ensemble_covariance
from gaussx.ops import diag, trace, eigvals
import geo_toolz as gt

# --- geo_toolz: evaluate DA analysis ensemble ---
analysis_ds = xr.Dataset({"ssh": (["ens", "lat", "lon"], analysis_particles)})

eval_pipeline = gt.Sequential([
    gt.metrics.RMSE(variable="ssh", dims=["ens"], reference=truth_ds),
])
rmse_map = eval_pipeline(analysis_ds)

# --- gaussx: structured covariance diagnostics ---
# Flatten ensemble to (N_e, N_x) for gaussx
particles = jnp.array(analysis_ds["ssh"].values.reshape(N_e, -1))

# Ensemble covariance as low-rank operator (rank N_e - 1, never forms N_x × N_x)
P = ensemble_covariance(particles)

# Structured diagnostics — all exploit the low-rank structure
spread = jnp.sqrt(diag(P))                    # per-gridpoint spread, O(N_e * N_x)
total_variance = trace(P)                      # scalar, O(N_e * N_x)
evals = eigvals(P)                             # spectrum of ensemble covariance
effective_rank = jnp.sum(evals > 1e-8)         # how many modes are active

# Reshape spread back to spatial grid for xarray
spread_ds = xr.Dataset({
    "ensemble_spread": (["lat", "lon"], spread.reshape(n_lat, n_lon)),
}, coords=analysis_ds.coords)
```

---

## With somax (Model Output Processing)

### Evaluate ocean model output against observations

```python
import geo_toolz as gt
import somax

# Run somax SWM simulation
solution = swm.integrate(state0, t0=0, t1=86400*365, dt=300.0)

# Convert to xarray for geo_toolz
model_ds = xr.Dataset({"ssh": (["time", "lat", "lon"], solution.ys[:, 0, :, :])})

# geo_toolz: preprocess + evaluate
eval_pipeline = gt.Sequential([
    gt.regrid.Regrid(target_lon=obs_lon, target_lat=obs_lat),
    gt.metrics.RMSE(variable="ssh", dims=["time"], reference=obs_clean),
])
scores = eval_pipeline(model_ds)
```

---

## With vardax (Pre/Post-processing for VarDA)

```python
# geo_toolz preprocesses observations for vardax
preprocess = gt.Sequential([gt.validation.ValidateCoords(), gt.regrid.Regrid(grid)])
obs_clean = preprocess(raw_satellite)

# vardax does the reconstruction
x_recon = vardanet(batch)

# geo_toolz evaluates the reconstruction
recon_ds = xr.Dataset({"ssh": (["time", "lat", "lon"], x_recon)})
scores = gt.metrics.RMSE(variable="ssh", reference=truth_ds)(recon_ds, truth_ds)
```

---

## With xtremax (Extreme Event Analysis)

```python
from xtremax.xarray import block_maxima
from xtremax.models import nonstationary_gev

# geo_toolz preprocesses, then xtremax analyzes extremes
preprocess = gt.Sequential([gt.validation.ValidateCoords(), gt.detrend.RemoveClimatology(clim)])
ds_clean = preprocess(xr.open_dataset("daily_temperature.nc"))

# xtremax: extract annual maxima and fit GEV
annual_max = block_maxima.temporal_block_maxima(ds_clean["tmax"], block="year")
# ... fit nonstationary GEV with GMST covariate
```

---

## With pyrox-gp (GP Spatial Interpolation)

```python
# geo_toolz regrids to a regular grid → enables Kronecker GP
preprocess = gt.Sequential([gt.validation.ValidateCoords(), gt.regrid.Regrid(regular_grid)])
ds_regular = preprocess(sparse_station_data)

# pyrox-gp: Kronecker GP on the regular grid
from pyrox_gp.models import GPPrior, gp_factor
from pyrox_gp.kernels import RBF

# Spatial interpolation via GP
gp_prior = GPPrior(kernel=RBF(variance=1.0, lengthscale=100e3), solver=CholeskySolver(), X=coords)
```

---

## With methanex (Methane Retrieval Pipeline)

```python
# geo_toolz preprocesses satellite radiance before methane retrieval
preprocess = gt.Sequential([
    gt.validation.ValidateCoords(),
    gt.subset.SubsetBBox(lon_bnds=(-100, -95), lat_bnds=(30, 35)),
])
radiance_clean = preprocess(xr.open_dataset("emit_l1b.nc"))

# methanex takes over for the retrieval
from methanex.xarray import retrieve_methane_xr
ch4 = retrieve_methane_xr(radiance_clean, config=retrieval_config)

# geo_toolz evaluates the retrieval
eval_pipeline = gt.metrics.RMSE(variable="ch4_enhancement", reference=truth_ds)
```

---

## With fairkl (Fair ML on Geospatial Data)

```python
from fairkl.sklearn import FairKernelRidgeRegressor

# geo_toolz preprocesses, then fairkl provides fair prediction
preprocess = gt.Sequential([gt.validation.ValidateCoords(), gt.regrid.Regrid(grid)])
ds_clean = preprocess(xr.open_dataset("air_quality.nc"))

# Fair kernel ridge: predict health outcome decorrelated from income
from geo_toolz.sklearn import SklearnOp
fair_op = SklearnOp(FairKernelRidgeRegressor(hsic_weight=0.2), sample_dim="time")
pred = fair_op(ds_clean)
```

---

## With xr_assimilate (Full DA Orchestration)

geo_toolz is the data interface layer for xr_assimilate:

```python
from xr_assimilate import Assimilator, DataModule

# geo_toolz: preprocess raw data
preprocess = gt.Sequential([gt.validation.ValidateCoords(), gt.regrid.Regrid(grid)])
obs_clean = preprocess(raw_satellite)

# xr_assimilate: full DA pipeline
data = DataModule(observations=obs_clean, context=background_ds, ...)
assimilator.fit(data, ...)
analysis = assimilator.cycle(data)

# geo_toolz: evaluate analysis
eval_pipeline = gt.Sequential([gt.metrics.RMSE(...), gt.spectral.PSD(...)])
scores = eval_pipeline(analysis, reference)
```

---

## Composition Patterns

| Pattern | Libraries | Use Case |
|---|---|---|
| Linear preprocessing | `Sequential([Validate, Regrid, Detrend])` | Standard data cleaning |
| Preprocess → infer | `Sequential + ModelOp(model)` | ML prediction pipeline |
| Multi-model comparison | `Graph` with branching inference + metrics | Model benchmarking |
| Patch-wise inference | `xrpatcher` + `Sequential` per patch | Large-grid bounded-memory |
| Per-region EOF | `xrpatcher` + `SklearnOp(PCA)` | Spatially-varying decomposition |
| Ensemble DA | geo_toolz → ekalmX → geo_toolz | State estimation workflows |
| Learned DA | geo_toolz → vardax → geo_toolz | ML-augmented reconstruction |
| Full DA orchestration | geo_toolz → xr_assimilate → geo_toolz | End-to-end DA pipeline |
| Ocean model evaluation | somax output → geo_toolz metrics/spectral | Compare model vs observations |
| Extreme analysis | geo_toolz → xtremax | Temperature/precipitation extremes |
| Kronecker GP | geo_toolz.Regrid → gaussx.KroneckerOp | Spatiotemporal GP on regular grid |
| Ensemble diagnostics | geo_toolz.metrics + gaussx.ensemble_cov | Spread/spectrum from DA output |
| Methane retrieval | geo_toolz → methanex → geo_toolz | Satellite methane pipeline |
| Fair spatial ML | geo_toolz → fairkl.FairKernelRidge | Fairness-constrained geospatial |
| GP interpolation | geo_toolz.Regrid → pyrox-gp GP | Spatial field interpolation |
| Hydra config | `builds(Sequential, [...])` | Reproducible experiment configs |
