# ODC-3.1 — Spectral KE flux + dual-cascade companions (enstrophy flux, integral / Taylor scales, slope fitter, compensated spectra)

**Source survey item:** [ocean-data-challenges-survey.md §3.1](ocean-data-challenges-survey.md)
**Status:** proposed
**Maps to upstream:** `src/mod_powerspec.py` from `2024c_DC_4DMedSea-ESA`.

---

## 1. Motivation

The 2024c MedSea repo's `mod_powerspec.py` (635 LOC) is mostly redundant
with `xrft` + our existing `power_spectrum` / `cross_spectrum` /
`coherence`. The genuinely new piece is **`spectra_flux(u, v, …)`** —
the **kinetic-energy spectral flux Π(k)** that diagnoses energy
exchange between scales:

> *"Energy injected at scale k_inj cascades upscale (large eddies) or
> downscale (dissipation) — the sign and shape of Π(k) tells you
> which."*

Spectral KE flux is the standard mesoscale-turbulence diagnostic and
absent from xr_toolz. While we're in the neighborhood, four
companion primitives that are universally used alongside KE flux —
**all sharing the same Fourier machinery** — fold in for ~30 LOC each:

- **Enstrophy spectral flux Π<sub>Z</sub>(k)** — the dual-cascade
  partner; KE and enstrophy both cascade in 2-D / QG turbulence but in
  opposite directions. Every paper that shows Π(k) shows Π<sub>Z</sub>(k).
- **Integral scale ℓ + Taylor microscale λ** — energy-weighted scalar
  length scales; sibling reductions of any PSD (`∫ψ/∫k^n·ψ` for
  `n=1, 2`).
- **Spectral slope fitter** — fits `E(k) ∝ k^{−n}` over a chosen
  inertial range. Used to discriminate dynamical regimes (`-5/3` KE
  inverse cascade, `-3` enstrophy cascade, `-2` SQG, `-11/3`
  sub-mesoscale frontogenesis).
- **Compensated spectrum** — `k^n · E(k)` viz primitive that makes
  inertial-range slopes legible (true `k^{-n}` regime appears flat).

This issue ships all five in one PR, all in
[`transforms/_src/fourier.py`](../../src/xr_toolz/transforms/_src/fourier.py)
alongside the existing `power_spectrum` / `cross_spectrum`.

## 2. User stories

### 2.1 KE flux + enstrophy flux on a 2-D velocity field (primary)

> *I have a gridded `(u, v)` snapshot and want both KE and enstrophy
> spectral fluxes to check the dual cascade.*

```python
import xarray as xr
from xr_toolz.transforms import ke_spectral_flux, enstrophy_spectral_flux

ds_uv = xr.open_dataset("velocity_snapshot.nc")  # has u, v on (x, y)

ds_ke = ke_spectral_flux(ds_uv["u"], ds_uv["v"], dim=("x", "y"))
ds_z  = enstrophy_spectral_flux(ds_uv["u"], ds_uv["v"], dim=("x", "y"))

# Each Dataset has data_vars: transfer (T(k)), flux (Π(k))
# coord: freq_r (isotropic wavenumber)
```

### 2.2 Time-mean flux

```python
ds_ke = ke_spectral_flux(
    ds["u"], ds["v"],
    dim=("x", "y"),
    avg_dims=("time",),
)
```

### 2.3 Diagnose inertial-range slope

> *I want the slope of `E(k)` between 50 km and 200 km wavelengths.*

```python
from xr_toolz.transforms import power_spectrum, fit_spectral_slope

psd = power_spectrum(ds["ssh"], dim=("x", "y"), isotropic=True)

slope, intercept = fit_spectral_slope(
    psd,
    wavenumber_dim="freq_r",
    k_min=1 / 200.0,    # 1/km
    k_max=1 / 50.0,
)
print(f"slope = {slope:.2f}")   # e.g. -2.0 → SQG
```

### 2.4 Compensated spectrum for a clean visual

```python
from xr_toolz.transforms import compensated_spectrum

psd_compensated = compensated_spectrum(psd, wavenumber_dim="freq_r", exponent=2.0)
psd_compensated.plot()    # flat plateau wherever true k^{-2} regime holds
```

### 2.5 Single-number length scales

```python
from xr_toolz.transforms import integral_scale

ell    = integral_scale(psd, wavenumber_dim="freq_r", moment=1)   # integral scale
lambda_T = integral_scale(psd, wavenumber_dim="freq_r", moment=2) # Taylor microscale
```

### 2.6 Operators inside a Sequential

```python
from xr_toolz.transforms import KESpectralFlux
from xr_toolz.core import Sequential

pipeline = Sequential([
    KESpectralFlux(u_var="u", v_var="v", dim=("x", "y"), avg_dims=("time",)),
    # downstream: plot or compute integral diagnostics
])
```

## 3. What we already have / what's missing

| Capability | Current | This proposal |
|---|---|---|
| `power_spectrum` (1-D and 2-D isotropic) | [`fourier.py:43`](../../src/xr_toolz/transforms/_src/fourier.py) | unchanged |
| `cross_spectrum`, `coherence`, `stft` | [`fourier.py`](../../src/xr_toolz/transforms/_src/fourier.py) | unchanged |
| Window / detrend kwargs | via `xrft` | reuse |
| Radial binning (2-D → 1-D iso) | via `xrft.isotropic_power_spectrum` | reuse |
| KE spectral flux | — | **add** `ke_spectral_flux` |
| Enstrophy spectral flux | — | **add** `enstrophy_spectral_flux` |
| Integral scale / Taylor microscale | — | **add** `integral_scale(..., moment=)` |
| Spectral slope fitter | — | **add** `fit_spectral_slope` |
| Compensated spectrum | — | **add** `compensated_spectrum` |
| Operator wrappers | — | **add** `KESpectralFlux`, `EnstrophySpectralFlux` |

## 4. Design

### 4.1 Algorithm — KE spectral flux

Following Frisch (1995) / Aluie (2018) / standard mesoscale practice:

```text
1. Window + detrend u, v.
2. Fourier-space gradients via 1j·2π·k:
     ∂u/∂x = ifft(2π·1j·k_x · fft(u)); etc. for ∂u/∂y, ∂v/∂x, ∂v/∂y.
3. Advection terms in physical space:
     φ₁ = u·∂u/∂x + v·∂u/∂y
     φ₂ = u·∂v/∂x + v·∂v/∂y
4. 2-D KE transfer:
     T(kx, ky) = -Re(û*·F[φ₁] + v̂*·F[φ₂]) / (Ni·Nj)²
5. Radially integrate to T(k).
6. Cumulative sum from high-k:
     Π(k) = Σ_{k' ≥ k} T(k')   ≡  np.cumsum(T[::-1])[::-1]
```

`Π(k)` is the rate at which KE crosses scale `k` from larger scales
toward smaller scales. Sign convention is the standard upstream
convention.

### 4.2 Algorithm — enstrophy spectral flux

Same template with vorticity `ζ = ∂v/∂x − ∂u/∂y` in place of `(u, v)`:

```text
T_Z = -Re(ẑ* · F[u·∂ζ/∂x + v·∂ζ/∂y]) / (Ni·Nj)²
Π_Z(k) = Σ_{k' ≥ k} T_Z(k')
```

Computes ζ from the same Fourier-space gradients of u, v — no
additional FFT cost beyond what KE flux already pays.

### 4.3 Tier B — primitives

```python
# src/xr_toolz/transforms/_src/fourier.py

def ke_spectral_flux(
    u: xr.DataArray, v: xr.DataArray, *,
    dim: Sequence[str],                         # 2 spatial dims, e.g. ("x", "y")
    window: str | None = "tukey",
    detrend: str | None = "linear",
    avg_dims: Sequence[str] | None = None,
    return_2d: bool = False,
) -> xr.Dataset:
    """Kinetic-energy spectral flux Π(k) from 2-D velocity (u, v).

    Returns Dataset with:
      - transfer: T(k), the per-scale KE transfer rate
      - flux:     Π(k), cumulative integral from k_max downward
      - (transfer_2d if ``return_2d=True``)
    """

def enstrophy_spectral_flux(
    u: xr.DataArray, v: xr.DataArray, *,
    dim: Sequence[str],
    window: str | None = "tukey",
    detrend: str | None = "linear",
    avg_dims: Sequence[str] | None = None,
    return_2d: bool = False,
) -> xr.Dataset:
    """Enstrophy spectral flux Π_Z(k). Same structure as ke_spectral_flux."""

def integral_scale(
    psd: xr.DataArray, *,
    wavenumber_dim: str = "freq_r",
    moment: int = 1,
) -> xr.DataArray:
    """Energy-weighted length scale.

    moment=1 → integral scale ℓ = ∫ψ dk / ∫k·ψ dk
    moment=2 → Taylor microscale λ where λ² = ∫ψ dk / ∫k²·ψ dk
    moment=p → generalized: ℓ_p = (∫ψ dk / ∫k^p·ψ dk)^(1/p) for p > 1
    """

def fit_spectral_slope(
    psd: xr.DataArray, *,
    wavenumber_dim: str = "freq_r",
    k_min: float, k_max: float,
) -> tuple[float, float]:
    """Linear fit on log-log over (k_min, k_max). Returns (slope, intercept).

    `numpy.polyfit(log(k), log(psd), 1)` over the inertial range.
    """

def compensated_spectrum(
    psd: xr.DataArray, *,
    wavenumber_dim: str = "freq_r",
    exponent: float,
) -> xr.DataArray:
    """Return psd * k^exponent. Flat where psd ∝ k^{-exponent}."""
```

`fit_spectral_slope` and `compensated_spectrum` are dim-agnostic — work
on isotropic 2-D PSDs, 1-D along-track PSDs, frequency PSDs, anything
1-D in the chosen dim.

### 4.4 Implementation re-use

Both flux functions share an internal helper for the 4 Fourier-space
gradients:

```python
def _fourier_uv_gradients(u, v, *, dim, window, detrend, avg_dims):
    """Return (du_dx, du_dy, dv_dx, dv_dy, k_radial, freq_r_coord, fft_norm)."""
```

Used by `ke_spectral_flux` (forms `φ₁`, `φ₂`) and
`enstrophy_spectral_flux` (forms ζ then `u·∇ζ`). Saves ~40 LOC of
duplication.

### 4.5 Layer-1 Operators

```python
# src/xr_toolz/transforms/operators.py
class KESpectralFlux(Operator):
    """Single-Dataset KE spectral flux operator.

    Reads u and v from the Dataset by name; returns a Dataset with
    transfer + flux variables.
    """

    def __init__(self, *,
                 u_var: str, v_var: str,
                 dim: Sequence[str],
                 window: str | None = "tukey",
                 detrend: str | None = "linear",
                 avg_dims: Sequence[str] | None = None,
                 return_2d: bool = False): ...

class EnstrophySpectralFlux(Operator):     # mirrors KESpectralFlux exactly
```

`integral_scale`, `fit_spectral_slope`, and `compensated_spectrum` are
small enough to skip Operator promotion in v1 — primitive functions
suffice.

## 5. Library leverage

| Need | Library |
|---|---|
| FFT / IFFT / window / detrend | `xrft` (already a dep, used by `power_spectrum`) |
| `1j·k` derivative | `xrft.fft(...) * (1j * 2π * k)` (manual) |
| Radial 2-D → 1-D binning | `xrft.isotropic_power_spectrum`'s internals; alternative: `numpy.histogram(weights=...)` |
| Cumulative integral | `numpy.cumsum` |
| Log-log slope fit | `numpy.polyfit` |
| Power-law multiplication | xarray broadcasting |

No new dependencies. Pure xrft + numpy + xarray.

## 6. Public API surface

```python
# Tier B primitives
xr_toolz.transforms.ke_spectral_flux(u, v, *, dim, window, detrend,
                                     avg_dims, return_2d)
xr_toolz.transforms.enstrophy_spectral_flux(u, v, *, dim, window, detrend,
                                            avg_dims, return_2d)
xr_toolz.transforms.integral_scale(psd, *, wavenumber_dim, moment)
xr_toolz.transforms.fit_spectral_slope(psd, *, wavenumber_dim, k_min, k_max)
xr_toolz.transforms.compensated_spectrum(psd, *, wavenumber_dim, exponent)

# Operators
xr_toolz.transforms.KESpectralFlux(...)
xr_toolz.transforms.EnstrophySpectralFlux(...)
```

All re-exported from `xr_toolz.transforms.__init__`.

## 7. Tests

| Test | Asserts |
|---|---|
| `ke_spectral_flux` Taylor–Green vortex | T(k) peaks at the imposed wavenumber; off-peak ≈ 0 |
| `ke_spectral_flux` KE conservation | `T(k)` integrated over all k ≈ 0 to fp tol |
| `ke_spectral_flux` flux endpoints | `Π(0) ≈ -Π(k_max) ≈ 0` (pure transfer, no source/sink) |
| `ke_spectral_flux` `avg_dims=("time",)` | output shape correct; mean of per-step fluxes matches |
| `ke_spectral_flux` `return_2d=True` | exposes `transfer_2d` with correct shape |
| `enstrophy_spectral_flux` on Taylor–Green | Π_Z opposite sign to Π_KE in the inertial range (dual cascade) |
| `enstrophy_spectral_flux` conservation | enstrophy budget closes within fp tol |
| `integral_scale` moment=1 on Gaussian PSD | matches analytic mean wavenumber |
| `integral_scale` moment=2 on Gaussian PSD | matches analytic Taylor microscale |
| `integral_scale` moment=1 on monochromatic spike | returns spike's wavenumber |
| `fit_spectral_slope` on synthetic `k^{-5/3}` | slope ≈ -5/3 within 1% |
| `fit_spectral_slope` on synthetic `k^{-3}` | slope ≈ -3 within 1% |
| `fit_spectral_slope` `k_min`/`k_max` window honoured | excluded points don't affect fit |
| `compensated_spectrum` flatness | `k^n · k^{-n}` ≈ const (fp noise) |
| `KESpectralFlux` Operator round-trip via `get_config` | reconstructed produces identical Dataset |
| `EnstrophySpectralFlux` Operator round-trip | identical Dataset |

Target: ~16 cases.

## 8. Out of scope

- **`wavenumber_spectra`, `cross_spectra`, `_tukey`, `_hanning`** —
  duplicates of existing `power_spectrum` / `cross_spectrum` + `xrft`
  window/detrend.
- **`fill_nan`** — covered by ODC-2.2 `fillnan_laplacian` and existing
  `fillnan_*` family.
- **APE / buoyancy spectral flux Π_b(k)** — full energy-budget
  closure (KE ↔ APE exchange). Separate proposal if needed.
- **Helmholtz spectral decomposition** (`E_rot(k)` + `E_div(k)`) —
  important for sub-mesoscale; bigger lift; separate proposal.
- **Structure functions** `S_p(r)` — physical-space dual; different
  machinery; separate concern.
- **Coarse-graining flux** Π(ℓ) (Aluie / Eyink) — alternative to
  spectral flux that doesn't assume periodic BCs; separate paradigm.
- **Tier A array-kernel split** — current Fourier module is xarray-only
  via `xrft`. Match the convention.

## 9. Effort

≈190 LOC implementation + ≈140 LOC tests. Single PR.

| Slice | LOC |
|---|---|
| `_fourier_uv_gradients` shared helper | 30 |
| `ke_spectral_flux` | 40 |
| `enstrophy_spectral_flux` | 30 |
| `integral_scale` (with `moment=` kwarg) | 15 |
| `fit_spectral_slope` | 15 |
| `compensated_spectrum` | 5 |
| `KESpectralFlux`, `EnstrophySpectralFlux` operators | 40 |
| Tests | ~140 |
| Docs / re-exports | 15 |

## 10. Risks / open questions

1. **Where it lives.** Two options: (a)
   [`transforms/_src/fourier.py`](../../src/xr_toolz/transforms/_src/fourier.py)
   alongside `power_spectrum` (recommended — math is generic Fourier
   transfer), (b) new `ocn/_src/spectral_flux.py` (since the formula
   assumes incompressible momentum advection). **Recommend (a)**.
2. **Sign convention.** `Π(k)` = downscale flux (energy from large
   to small) per upstream convention. Document explicitly; sign matters
   when comparing across papers.
3. **Default window/detrend.** Upstream defaults to `Tukey` + `Both`
   (mean + linear). Closest xrft equivalent: `window='tukey'`,
   `detrend='linear'`. Document the small mismatch with upstream
   "Both" (which detrends along both axes separately).
4. **Radial binning.** Reuse `xrft.isotropic_power_spectrum`'s binning
   strategy when possible. If not directly callable on a transfer
   field, write a minimal `_radial_integrate(field_2d, k_2d, k_1d)`
   helper.
5. **Periodic BC assumption.** All FFT-based fluxes assume periodic
   boundary conditions. Document; users with non-periodic domains
   should window aggressively (Tukey with small α, or Hanning).
6. **`integral_scale` moment > 2.** Generalized `(∫ψ / ∫k^p·ψ)^(1/p)`
   handles arbitrary `moment`. Document common cases (1, 2); allow
   anything via the kwarg.
7. **Operator promotion for slope/integral/compensated.** Skip in v1
   — primitive functions are ergonomic enough. Promote later if a
   Sequential calls for them.
