# xrgrad — NaN and land-mask-aware differentiation

**Status:** accepted (2026-08-18) · tracks [#269](https://github.com/jejjohnson/xrtoolz/issues/269)

## Problem

Finite-difference stencils propagate NaN. Every NaN cell contaminates a
halo of roughly `accuracy` cells in each differentiated direction, so a
vorticity or divergence map of a masked ocean field loses a strip of
valid water all the way around every coastline — silently, since nothing
warns and the result is simply NaN there.

This matters more for xrgrad than for a generic differentiation library,
because its primary consumers are ocean fields with land masks. Before
this note, xrgrad did not mention NaN anywhere: no mitigation, and no
documentation of the behaviour.

## Options

### A — Documented recipe, no API change

Fill the mask (`xrtoolz.interpolate.fillnan_*`), differentiate, re-apply
the mask. Zero implementation risk; the coastal bias persists but is at
least written down.

### B — Per-point stencil selection

Near a mask boundary, fall back to one-sided stencils that touch only
valid cells. Most accurate, and the option the issue expected to be
hardest: `finitediffx` applies one stencil per axis, so this looked like
it needed per-cell stencil bookkeeping and a mask-dilation pass keyed to
the stencil width.

It turns out not to. **NaN is itself the support detector.** Any stencil
that reads a NaN produces NaN, including through a zero centre
coefficient, because `0 * nan` is `nan`. So differentiating the masked
field three times — central, forward, backward — and taking per point
the first variant that came back finite is exactly "use the widest
stencil that fits on valid data", with no mask arithmetic at all and no
dependency on the library's internal stencil-sizing rules.

### C — Built-in fill-extrapolate

`nan_policy="fill-extrapolate"`: mirror values across the mask edge
internally, differentiate, re-mask. A curated version of A, first-order
at the boundary.

## Measurement

`docs/design/examples/xrgrad-nan-mask-comparison.py` scores each option
on `sin(2λ)cos(φ)` over a 121×91 grid with a synthetic coastline (a
straight western shelf plus a curved bay), reporting RMS error relative
to the mean field gradient, binned on distance from land in cells.
Option C is represented by its best case, the harmonic fill in A/C.

RMS relative error (`lost` = fraction of cells returned as NaN):

| distance | propagate | fill-then-mask | adaptive |
|---|---|---|---|
| **accuracy=1** | | | |
| 1 (coastal) | 5.5e-05, **lost 62%** | 6.5e-01, lost 0% | **7.2e-03**, lost 0% |
| 2 | 7.8e-05 | 7.8e-05 | 7.8e-05 |
| ≥3 | identical | identical | identical |
| **accuracy=2** | | | |
| 1 (coastal) | 5.5e-05, **lost 62%** | 6.5e-01, lost 0% | **1.4e-04**, lost 0% |
| 2 | 7.8e-05 | 7.8e-05 | 7.8e-05 |
| ≥3 | identical | identical | identical |

Three things decide it:

1. **Fill-then-mask is far worse than losing the cells.** At 6.5e-01 it
   is roughly four orders of magnitude above the interior error and ~90×
   worse than adaptive. Harmonic relaxation produces a smooth field that
   satisfies `∇²u = 0` — precisely the wrong thing to differentiate,
   since the fill's gradient is an artefact of the fill, not the data.
   It converts a visible gap into an invisible wrong answer.
2. **Adaptive costs one order of accuracy at the coast, nothing
   elsewhere.** Coastal cells fall from the central stencil's order to
   the one-sided stencil's, hence 7.2e-03 at `accuracy=1` improving to
   1.4e-04 at `accuracy=2`. Raising `accuracy` therefore buys back
   coastal quality specifically — and it never costs cells, because the
   selection descends through lower accuracies for any region too narrow
   for the requested stencil. Without that descent a two-cell channel
   would vanish exactly when the caller asked for more accuracy, which
   would make the option anti-monotone in its own quality knob.
3. **Adaptive is bit-identical in the interior** — the comparison
   asserts `max |adaptive - propagate| == 0` for cells more than four
   from land — so it cannot regress unmasked work.

## Decision

**Option B, as `nan_policy="adaptive"`, opt-in, in the xrgrad kernel.**

```python
xrgrad.partial(da, "lon", geometry="spherical", nan_policy="adaptive")
```

`nan_policy="propagate"` stays the default: the change is opt-in because
one-sided coastal values are a different numerical object from centred
interior ones, and a caller should say they want them.

The mask is re-imposed on the output. Land is NaN because there is no
data there, which is a property of the result rather than of the
arithmetic — it does not rest on the stencil happening to read the
centre point.

Placement is in xrgrad rather than as an `xrtoolz` wrapper, because
stencil selection *is* differentiation. A wrapper cannot choose a
stencil; it can only pre-process the field, which is option A.

Option A remains available and is documented: the `fillnan_*` family in
`xrtoolz.interpolate` is the right tool when a caller wants a gap-free
field, and the measurement above is the reason not to reach for it as a
derivative-accuracy mitigation.

## Consequences

- Cost is bounded by `3 * accuracy` stencil passes per requested axis,
  against one for `propagate`, and applies only to the axes where the
  policy is asked for. The sweep returns as soon as no finite input cell
  is left unresolved, so in practice:

  | field | passes |
  |---|---|
  | no NaN at all | 1 — the centred stencil covers everything |
  | ordinary mask, `accuracy=1` | up to 3 |
  | ordinary mask, `accuracy=n` | 3, since the coast resolves at the requested accuracy |
  | mask with strips narrower than the requested stencil | up to `3n`, descending only while such cells remain |

  Each `(shape, accuracy, method)` combination is a separate
  `finitediffx` JIT compilation, so the descent costs compilations as
  well as evaluations the first time a shape is seen. That upper bound
  is reached exactly in the case where the guide suggests raising
  `accuracy` on a heavily fragmented mask; it is the price of recovery
  being monotone in `accuracy` rather than losing narrow channels.
- `method=` is not meaningful with `nan_policy="adaptive"`, since the
  fallback chain *is* method selection; passing a non-central method
  raises.
- Coastal cells carry one order lower truncation error than the
  interior. Documented in the operator docstrings and the package guide.
- Composes with `periodic=`: the wrap is applied before stencil
  selection, so a periodic axis with a mask gets both.
- Threading `nan_policy` through the `xrtoolz.ocn` kinematics call sites
  is a mechanical follow-up, deliberately not in the landing PR.
