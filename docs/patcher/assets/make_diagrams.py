"""Generate the conceptual PNGs used by the geopatcher docs.

Run from the repo root::

    uv run python docs/assets/make_diagrams.py

The script is deterministic (fixed seed, synthetic data only — no network)
and overwrites the committed PNGs in this directory:

- ``four-axes.png``        — Geometry × Sampler × Window × Aggregation
- ``patch-lifecycle.png``  — Field → split → patches → operator → merge
- ``boundary-modes.png``   — clip / pad / drop boundary behaviors
- ``overlap-add.png``      — schematic of overlap-add with feather weights

Diagrams use only matplotlib + numpy so they reproduce identically on any
machine with the project's ``[docs]`` group installed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(0)


def _save(fig: plt.Figure, name: str) -> None:
    out = HERE / name
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE.parent.parent)}")


# ---------------------------------------------------------------------------
# 1. four-axes.png — the orthogonal axes of the patcher framework
# ---------------------------------------------------------------------------
def four_axes() -> None:
    """Visualise Geometry / Sampler / Window / Aggregation as a 2×2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    fig.suptitle(
        "geopatcher — four orthogonal axes",
        fontsize=15,
        weight="bold",
    )

    # --- Geometry: a 24x24 field with two patch shapes -------------------
    ax = axes[0, 0]
    ax.set_title("Geometry — shape of the neighborhood", fontsize=11)
    ax.set_xlim(0, 24)
    ax.set_ylim(24, 0)
    ax.set_aspect("equal")
    ax.add_patch(
        mpatches.Rectangle((2, 2), 8, 8, edgecolor="#2962ff", facecolor="#bbdefb", lw=2)
    )
    ax.text(6, 6, "rect\n8×8", ha="center", va="center", fontsize=9)
    ax.add_patch(
        mpatches.Circle((17, 14), 5, edgecolor="#c62828", facecolor="#ffcdd2", lw=2)
    )
    ax.text(17, 14, "disk\nr=5", ha="center", va="center", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Sampler: anchor placement ----------------------------------------
    ax = axes[0, 1]
    ax.set_title("Sampler — where anchors go", fontsize=11)
    ax.set_xlim(0, 24)
    ax.set_ylim(24, 0)
    ax.set_aspect("equal")
    grid = np.array([(x, y) for x in range(3, 24, 6) for y in range(3, 24, 6)])
    ax.scatter(grid[:, 0], grid[:, 1], c="#1565c0", s=50, label="RegularStride")
    jitter = grid + RNG.uniform(-1.5, 1.5, size=grid.shape)
    ax.scatter(
        jitter[:, 0], jitter[:, 1], c="#ef6c00", s=50, marker="x", label="Jittered"
    )
    random = RNG.uniform(2, 22, size=(8, 2))
    ax.scatter(
        random[:, 0], random[:, 1], c="#2e7d32", s=50, marker="^", label="Random"
    )
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Window: per-cell weights --------------------------------------
    ax = axes[1, 0]
    ax.set_title("Window — boundary treatment / per-cell weights", fontsize=11)
    n = 32
    x = np.linspace(-1, 1, n)
    boxcar = np.ones(n)
    hann = 0.5 * (1 + np.cos(np.pi * x))
    tukey = np.minimum(boxcar, 0.5 * (1 + np.cos(np.pi * (np.abs(x) - 0.5) / 0.5)))
    tukey = np.where(np.abs(x) <= 0.5, 1.0, tukey)
    gauss = np.exp(-(x**2) / (2 * 0.4**2))
    ax.plot(boxcar, label="Boxcar", lw=2)
    ax.plot(hann, label="Hann", lw=2)
    ax.plot(tukey, label="Tukey(α=0.5)", lw=2)
    ax.plot(gauss, label="Gaussian(σ=0.4)", lw=2)
    ax.set_xlabel("patch index")
    ax.set_ylabel("weight")
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Aggregation: local → global merge -------------------------------
    ax = axes[1, 1]
    ax.set_title("Aggregation — local outputs → global field", fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    boxes = [
        ("OverlapAdd", 1.0, 7.0, "#bbdefb"),
        ("Mean", 5.5, 7.0, "#c8e6c9"),
        ("WeightedSum", 1.0, 4.0, "#ffe0b2"),
        ("InvVarWMean", 5.5, 4.0, "#f8bbd0"),
        ("HardVote", 1.0, 1.0, "#d1c4e9"),
        ("ApproxQuantile", 5.5, 1.0, "#cfd8dc"),
    ]
    for name, x0, y0, color in boxes:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y0),
                3.5,
                1.6,
                boxstyle="round,pad=0.1",
                edgecolor="#333",
                facecolor=color,
                lw=1.2,
            )
        )
        ax.text(x0 + 1.75, y0 + 0.8, name, ha="center", va="center", fontsize=9)

    _save(fig, "four-axes.png")


# ---------------------------------------------------------------------------
# 2. patch-lifecycle.png — Field → split → patches → operator → merge
# ---------------------------------------------------------------------------
def patch_lifecycle() -> None:
    """Visualise the end-to-end lifecycle as a pipeline of arrays."""
    H = W = 32
    field = np.outer(
        np.linspace(0, 1, H), np.linspace(0, 1, W)
    ) + 0.2 * RNG.standard_normal((H, W))

    fig = plt.figure(figsize=(14, 4.8))
    fig.suptitle(
        "Patch lifecycle:  Field → split → patches → operator → merge → output",
        fontsize=13,
        weight="bold",
    )

    gs = fig.add_gridspec(2, 5, height_ratios=[3, 0.5], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(field, cmap="viridis")
    ax0.set_title("1. Field\n(32×32 raster)", fontsize=10)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(field, cmap="viridis")
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            ax1.add_patch(
                mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 8, 8, fill=False, edgecolor="white", lw=1
                )
            )
    ax1.set_title("2. split\n(8×8 anchors)", fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 2])
    sample = field[:8, :8]
    ax2.imshow(sample, cmap="viridis")
    ax2.set_title("3. Patch\n(one 8×8 chip)", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(1.0 - sample, cmap="viridis")
    ax3.set_title("4. operator\n(per-patch op)", fontsize=10)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs[0, 4])
    ax4.imshow(1.0 - field, cmap="viridis")
    ax4.set_title("5. merge\n(global output)", fontsize=10)
    ax4.set_xticks([])
    ax4.set_yticks([])

    # arrows under each transition
    arrow_ax = fig.add_subplot(gs[1, :])
    arrow_ax.axis("off")
    for i, label in enumerate(
        ["patcher.split", "iterate", "operator", "patcher.merge"]
    ):
        arrow_ax.annotate(
            "",
            xy=(0.21 + 0.21 * i, 0.5),
            xytext=(0.09 + 0.21 * i, 0.5),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#444"),
        )
        arrow_ax.text(
            0.15 + 0.21 * i,
            0.0,
            label,
            ha="center",
            va="center",
            fontsize=9,
            transform=arrow_ax.transAxes,
        )

    _save(fig, "patch-lifecycle.png")


# ---------------------------------------------------------------------------
# 3. boundary-modes.png — clip / pad / drop on a small grid
# ---------------------------------------------------------------------------
def boundary_modes() -> None:
    """Show how the three boundary policies treat an edge-overflowing patch."""
    grid = np.outer(np.linspace(0, 1, 12), np.linspace(0, 1, 12))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    fig.suptitle(
        "Boundary policy — what happens at the edge of the field?",
        fontsize=13,
        weight="bold",
    )

    # Base — show the 12x12 field with a 6x6 patch anchored at (8, 8)
    ax = axes[0]
    ax.imshow(grid, cmap="viridis", extent=(0, 12, 12, 0))
    ax.add_patch(
        mpatches.Rectangle(
            (8, 8), 6, 6, edgecolor="red", facecolor="none", lw=2, ls="--"
        )
    )
    ax.set_xlim(-1, 15)
    ax.set_ylim(15, -1)
    ax.set_title("Anchor at (8, 8)\n6×6 patch overflows", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # drop — anchor is never emitted
    ax = axes[1]
    ax.imshow(grid, cmap="viridis", extent=(0, 12, 12, 0))
    ax.text(11, 11, "✗", color="red", ha="center", va="center", fontsize=28)
    ax.set_xlim(-1, 15)
    ax.set_ylim(15, -1)
    ax.set_title('"drop" (default)\nanchor never emitted', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # pad — patch is full size, padded with nodata
    ax = axes[2]
    padded = np.full((6, 6), np.nan)
    padded[:4, :4] = grid[8:12, 8:12]
    ax.imshow(padded, cmap="viridis", extent=(8, 14, 14, 8), vmin=0, vmax=1)
    ax.add_patch(
        mpatches.Rectangle(
            (12, 8), 2, 6, facecolor="lightgray", edgecolor="gray", hatch="//", lw=1
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (8, 12), 6, 2, facecolor="lightgray", edgecolor="gray", hatch="//", lw=1
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (8, 8), 6, 6, edgecolor="red", facecolor="none", lw=2, ls="--"
        )
    )
    ax.set_xlim(-1, 15)
    ax.set_ylim(15, -1)
    ax.set_title('"pad" — full 6×6\nnodata in overflow', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # shrink — patch is smaller at the edge
    ax = axes[3]
    ax.imshow(grid[8:12, 8:12], cmap="viridis", extent=(8, 12, 12, 8), vmin=0, vmax=1)
    ax.add_patch(
        mpatches.Rectangle((8, 8), 4, 4, edgecolor="red", facecolor="none", lw=2)
    )
    ax.set_xlim(-1, 15)
    ax.set_ylim(15, -1)
    ax.set_title('"shrink" — smaller patch\nweights crop to match', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    _save(fig, "boundary-modes.png")


# ---------------------------------------------------------------------------
# 4. overlap-add.png — schematic of feathered overlap-add
# ---------------------------------------------------------------------------
def overlap_add() -> None:
    """Show three overlapping Hann patches summing to a flat reconstruction."""
    n = 96
    patch_size = 48
    stride = 24
    x = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        "Overlap-add reconstruction with feathered (Hann) windows",
        fontsize=13,
        weight="bold",
    )

    # 1. three overlapping patches with Hann tapers
    ax = axes[0]
    colors = ["#1565c0", "#ef6c00", "#2e7d32"]
    weights_sum = np.zeros(n)
    for i, anchor in enumerate([0, stride, 2 * stride]):
        local = np.arange(patch_size)
        hann = 0.5 * (1 - np.cos(2 * np.pi * local / (patch_size - 1)))
        full = np.zeros(n)
        end = anchor + patch_size
        if end > n:
            full[anchor:] = hann[: n - anchor]
        else:
            full[anchor:end] = hann
        ax.plot(x, full, color=colors[i], lw=2, label=f"patch @ {anchor}")
        ax.fill_between(x, 0, full, color=colors[i], alpha=0.2)
        weights_sum += full
    ax.set_ylabel("weight w(x)")
    ax.set_title(
        "1. Three overlapping patches with Hann (feather) windows", fontsize=10
    )
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # 2. sum of weights
    ax = axes[1]
    ax.plot(x, weights_sum, color="#6a1b9a", lw=2)
    ax.fill_between(x, 0, weights_sum, color="#6a1b9a", alpha=0.2)
    ax.set_ylabel("Σ w(x)")
    ax.set_title("2. Accumulated weights (denominator)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. normalised reconstruction (constant 1 on the interior)
    ax = axes[2]
    signal_sum = weights_sum.copy()
    normalised = np.divide(
        signal_sum, weights_sum, out=np.zeros_like(signal_sum), where=weights_sum > 0
    )
    ax.plot(x, normalised, color="#c62828", lw=2)
    ax.fill_between(x, 0, normalised, color="#c62828", alpha=0.15)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Σ(w·y) / Σw")
    ax.set_xlabel("position")
    ax.set_title("3. Normalised reconstruction (flat on the interior)", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "overlap-add.png")


def main() -> None:
    four_axes()
    patch_lifecycle()
    boundary_modes()
    overlap_add()


if __name__ == "__main__":
    main()
