"""OP-O8 measurements for geostack#79: an L3 forward model inside a differentiated cost.

Two ways to write the 4D-Var cost J(x₀) over one window:

(A) L3: the forward model returns a GeoMapSeries (a carrier rebuilt per
    evaluation) and H(x) is ``sample`` on the carrier, from its transform.
(B) L1: the forward model returns raw frames and H(x) uses bilinear weights
    precomputed once outside the cost.

Problem (from #79): a 256 × 256 state, T = 8 output frames, N = 20,000
observations padded to a fixed N. The model is linear diffusion with
5 steps per output frame. Reported per option: traces of
``jit(value_and_grad(cost))`` over three windows with different real
observation counts, compile time, and the median per-call time.

Run from packages/geotoolz-array: ``uv run python spike/bench_op_o8.py``.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).parent))
from proto.carrier import GeoArray
from proto.sample import sample


H = W = 256
T_OUT = 8
STEPS = 5
N = 20_000
KAPPA_DT = 0.2  # κΔt / Δx², stable below 0.25
SIGMA_B, SIGMA_O = 1.0, 0.1
T0 = jnp.asarray([20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0])
CRS = "EPSG:32613"
REPEATS = 200


def step(x: jax.Array, mask: jax.Array) -> jax.Array:
    """xₖ₊₁ = xₖ + κΔt ∇²xₖ on ocean cells (5-point Laplacian)."""
    # ∇²x ≈ x[i±1, j] + x[i, j±1] − 4x[i, j]                      (H, W)
    lap = (
        jnp.roll(x, 1, 0)
        + jnp.roll(x, -1, 0)
        + jnp.roll(x, 1, 1)
        + jnp.roll(x, -1, 1)
        - 4 * x
    )
    return jnp.where(mask, x + KAPPA_DT * lap, 0.0)


def l1_frames(x0: jax.Array, mask: jax.Array) -> jax.Array:
    """L1 loop: T_OUT frames, STEPS model steps apart.            (H, W) → (T, H, W)"""

    def one_frame(x: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        x = jax.lax.fori_loop(0, STEPS, lambda _, s: step(s, mask), x)
        return x, x

    _, frames = jax.lax.scan(one_frame, x0, None, length=T_OUT)
    return frames


def forward_l3(x0_map: GeoArray, mask: jax.Array) -> GeoArray:
    """L3: the same loop, returned as a GeoMapSeries GeoArray['time y x']."""
    frames = l1_frames(x0_map.data, mask)
    return x0_map.with_data(frames, ("time", "y", "x"))


def weights(xy: np.ndarray, valid: np.ndarray) -> tuple[jax.Array, ...]:
    """Bilinear corner indices and weights from the transform, once, on the host."""
    t = np.asarray(T0)
    col = (xy[:, 0] - t[2]) / t[0] - 0.5
    row = (xy[:, 1] - t[5]) / t[4] - 0.5
    col = np.where(valid, col, 0.0)
    row = np.where(valid, row, 0.0)
    i0 = np.clip(np.floor(row), 0, H - 2).astype(int)
    j0 = np.clip(np.floor(col), 0, W - 2).astype(int)
    return (
        jnp.asarray(i0),
        jnp.asarray(j0),
        jnp.asarray(row - i0),
        jnp.asarray(col - j0),
    )


def sample_l1(
    frames: jax.Array, tidx: jax.Array, w: tuple[jax.Array, ...], valid: jax.Array
) -> jax.Array:
    """H(x) on raw frames with precomputed weights.                 (T, H, W) → (N,)"""
    i0, j0, wr, wc = w
    # gather the four corners per point, never frames[tidx] (N × H × W)
    f00 = frames[tidx, i0, j0]
    f01 = frames[tidx, i0, j0 + 1]
    f10 = frames[tidx, i0 + 1, j0]
    f11 = frames[tidx, i0 + 1, j0 + 1]
    val = (1 - wr) * ((1 - wc) * f00 + wc * f01) + wr * ((1 - wc) * f10 + wc * f11)
    return jnp.where(valid, val, 0.0)


def make_window(rng: np.random.Generator, n_real: int) -> dict[str, jax.Array]:
    """n_real observations at random times and places, padded to N with NaN."""
    t = np.asarray(T0)
    cols = rng.uniform(1, W - 2, n_real)
    rows = rng.uniform(1, H - 2, n_real)
    xy = np.stack([t[2] + t[0] * cols, t[5] + t[4] * rows], axis=1)
    pad = N - n_real
    xy = np.concatenate([xy, np.full((pad, 2), np.nan)])
    valid = np.arange(N) < n_real
    tidx = np.concatenate([rng.integers(0, T_OUT, n_real), np.zeros(pad, int)])
    y = np.concatenate([rng.normal(size=n_real), np.full(pad, np.nan)])
    return {
        "xy": jnp.asarray(xy),
        "valid": jnp.asarray(valid),
        "tidx": jnp.asarray(tidx),
        "y": jnp.asarray(y),
        "w": weights(xy, valid),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    mask = jnp.asarray(rng.random((H, W)) > 0.05)  # 95 % ocean
    xb = jnp.zeros((H, W))
    background = GeoArray(xb, T0, ("y", "x"), CRS)
    windows = [make_window(rng, n) for n in (20_000, 14_000, 9_500)]
    x0 = jnp.asarray(rng.normal(size=(H, W)))
    traces = {"A": 0, "B": 0}

    def cost_a(x: jax.Array, obs: dict[str, jax.Array]) -> jax.Array:
        traces["A"] += 1
        # J_b = ½ ‖x − x_b‖² / σ_b²
        jb = 0.5 * jnp.sum(jnp.where(mask, x - xb, 0.0) ** 2) / SIGMA_B**2
        traj = forward_l3(background.with_data(x), mask)  # GeoArray['time y x']
        # H(x) from the carrier's transform, each point on its own frame       (N,)
        hx = sample(traj, obs["xy"], obs["valid"], obs["tidx"])
        r = jnp.where(obs["valid"], obs["y"] - hx, 0.0)
        return jb + 0.5 * jnp.sum(r**2) / SIGMA_O**2

    def cost_b(x: jax.Array, obs: dict[str, jax.Array]) -> jax.Array:
        traces["B"] += 1
        jb = 0.5 * jnp.sum(jnp.where(mask, x - xb, 0.0) ** 2) / SIGMA_B**2
        frames = l1_frames(x, mask)  # (T, H, W)
        hx = sample_l1(frames, obs["tidx"], obs["w"], obs["valid"])
        r = jnp.where(obs["valid"], obs["y"] - hx, 0.0)
        return jb + 0.5 * jnp.sum(r**2) / SIGMA_O**2

    # compile both, then time them interleaved (alternating order each round),
    # so drift on a shared 2-core machine hits A and B alike
    fns, first, grads = {}, {}, {}
    for name, cost in (("A", cost_a), ("B", cost_b)):
        vg = jax.jit(jax.value_and_grad(cost))
        t0 = time.perf_counter()
        grads[name] = jax.block_until_ready(vg(x0, windows[0]))
        first[name] = time.perf_counter() - t0
        for w in windows[1:]:
            jax.block_until_ready(vg(x0, w))
        fns[name] = vg
    times: dict[str, list[float]] = {"A": [], "B": []}
    for k in range(REPEATS):
        for name in ("A", "B") if k % 2 == 0 else ("B", "A"):
            w = windows[k % len(windows)]
            t0 = time.perf_counter()
            jax.block_until_ready(fns[name](x0, w))
            times[name].append(time.perf_counter() - t0)
    # paired ratio per round: drift between rounds cancels
    paired = sorted(a / b for a, b in zip(times["A"], times["B"], strict=True))
    q = len(paired) // 4
    results: dict[str, dict[str, float]] = {}
    for name in ("A", "B"):
        per_call = statistics.median(times[name])
        results[name] = {
            "traces_over_3_windows": traces[name],
            "compile_s": first[name] - per_call,
            "per_call_ms": per_call * 1e3,
            "per_call_ms_min": min(times[name]) * 1e3,
            # machine-independent: XLA's own estimate for the compiled program
            **{
                k: float(v)
                for k, v in (
                    fns[name].lower(x0, windows[0]).compile().cost_analysis() or {}
                ).items()
                if k in ("flops", "bytes accessed")
            },
        }
    ja, ga = grads["A"]
    jb_, gb = grads["B"]
    ratio = statistics.median(paired)
    rule_a = ratio <= 1.10 and results["A"]["traces_over_3_windows"] == 1
    out = {
        "problem": {"H": H, "W": W, "T": T_OUT, "steps_per_frame": STEPS, "N": N},
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "device": str(jax.devices()[0]),
        },
        "A_l3_carrier": results["A"],
        "B_l1_loop": results["B"],
        "A_over_B_per_call": ratio,
        "A_over_B_interquartile": [paired[q], paired[-q - 1]],
        "J_rel_diff": float(abs(ja - jb_) / abs(jb_)),
        "grad_max_rel_diff": float(jnp.max(jnp.abs(ga - gb)) / jnp.max(jnp.abs(gb))),
        "rule": "A if its per-call time is within 10 % of B's and it traces once",
        "decision": "A" if rule_a else "B",
    }
    dest = Path(__file__).parent / "results" / "op_o8.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
