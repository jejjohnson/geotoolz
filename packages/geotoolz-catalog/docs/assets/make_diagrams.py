"""Generate concept diagrams for geocatalog docs.

Reproducible matplotlib script that emits the PNGs referenced by
``docs/concepts.md``. Uses only synthetic data — no network, no
external services.

Usage:

    cd docs/assets && python make_diagrams.py

Writes three PNGs into the current directory:

- ``catalog-architecture.png`` — Source/Bundle/Catalog/Slice/Loader flow.
- ``backend-comparison.png`` — InMemory vs DuckDB query latency curves.
- ``set-algebra.png`` — query / intersect / union as overlapping AOIs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent


# A muted, print-friendly palette. Same five colours used across all
# three figures so they read as a set.
PALETTE = {
    "source": "#4C72B0",
    "bundle": "#55A868",
    "catalog": "#C44E52",
    "slice": "#8172B2",
    "loader": "#CCB974",
    "muted": "#999999",
}


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: str,
    subtitle: str | None = None,
) -> tuple[float, float]:
    """Draw a rounded-rectangle node, return its (right, mid_y) anchor."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color + "22",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2 + (0.06 if subtitle else 0.0),
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#222",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.12,
            subtitle,
            ha="center",
            va="center",
            fontsize=8,
            color="#555",
            style="italic",
        )
    return x + w, y + h / 2


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>,head_width=4,head_length=6",
        linewidth=1.2,
        color="#444",
        mutation_scale=12,
        shrinkA=4,
        shrinkB=6,
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + 0.08
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8, color="#444")


def architecture_diagram(out: Path) -> None:
    """Source -> Bundle -> Catalog -> Slice -> Loader -> GeoTensor."""
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.set_axis_off()

    # Row of boxes.
    y = 1.5
    h = 1.0
    w = 1.7
    nodes = [
        ("Source", "STAC / CMR /\nEarthAccess", PALETTE["source"], 0.2),
        ("Bundle", "queries +\nmatchups", PALETTE["bundle"], 2.2),
        ("Catalog", "InMemory or\nDuckDB", PALETTE["catalog"], 4.2),
        ("GeoSlice", "bbox + interval\n+ CRS + res", PALETTE["slice"], 6.2),
        ("Loader", "load_raster()\nload_vector()", PALETTE["loader"], 8.2),
    ]

    anchors: list[tuple[float, float]] = []
    for label, sub, color, x in nodes:
        right, mid = _box(ax, x, y, w, h, label, color, subtitle=sub)
        anchors.append((right, mid))

    # Arrows between successive boxes.
    for i in range(len(nodes) - 1):
        start = anchors[i]
        left_next = (nodes[i + 1][3], start[1])
        _arrow(ax, start, left_next)

    # Trailing GeoTensor sink.
    _box(ax, 10.1, y + 0.2, 0.85, 0.6, "GeoTensor", PALETTE["muted"])
    _arrow(ax, anchors[-1], (10.1, anchors[-1][1]))

    # Title + caption.
    ax.text(
        5.5,
        3.5,
        "geocatalog data flow",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        5.5,
        0.55,
        "Catalogs are the index. GeoSlice is the unit of work. "
        "Loaders are the materialisation step.",
        ha="center",
        va="center",
        fontsize=9,
        color="#555",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def backend_comparison(out: Path) -> None:
    """InMemory vs DuckDB latency vs row count (synthetic)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=140)
    n = np.logspace(2, 7, 60)

    # Synthetic curves chosen to match the design's published thresholds:
    # - InMemory: ~constant low latency up to ~1e5 rows, then climbs as
    #   the GeoDataFrame stops fitting in L3 cache.
    # - DuckDB: roughly flat with a slope ~ log(n) because bbox-column
    #   pushdown skips row-groups proportional to AOI selectivity.
    in_memory_ms = 0.4 + (n / 1e5) ** 1.8 * 0.6
    duckdb_ms = 6.0 + np.log10(n) * 1.4

    ax.plot(
        n,
        in_memory_ms,
        color=PALETTE["catalog"],
        linewidth=2.2,
        label="InMemoryGeoCatalog (R-tree + IntervalIndex)",
    )
    ax.plot(
        n,
        duckdb_ms,
        color=PALETTE["source"],
        linewidth=2.2,
        label="DuckDBGeoCatalog (GeoParquet 1.1 bbox pushdown)",
    )

    ax.axvspan(1e5, 1e7, color=PALETTE["source"], alpha=0.05)
    ax.text(
        3e5,
        0.18,
        "DuckDB favoured: 10^5+ rows, remote URIs",
        fontsize=9,
        color="#333",
    )

    ax.axvspan(1e2, 1e5, color=PALETTE["catalog"], alpha=0.05)
    ax.text(
        3e2,
        0.18,
        "InMemory favoured: 10^5 or less rows, local",
        fontsize=9,
        color="#333",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("catalog rows (log)")
    ax.set_ylabel("query latency, ms (log)")
    ax.set_title(
        "Backend comparison: InMemory vs DuckDB\n"
        "(illustrative — synthetic curves, not measured)",
        fontsize=12,
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def set_algebra(out: Path) -> None:
    """query / intersect / union visualised as overlapping AOIs."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), dpi=140)

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bbb")

    # --- query ---
    axes[0].set_title("query(catalog, slice)", fontsize=11, fontweight="bold")
    # All catalog footprints.
    rng = np.random.default_rng(42)
    foots = rng.uniform(0.3, 8.7, size=(8, 2))
    for cx, cy in foots:
        rect = mpatches.Rectangle(
            (cx, cy), 1.2, 1.0, edgecolor=PALETTE["catalog"], facecolor="none", lw=1.0
        )
        axes[0].add_patch(rect)
    # AOI slice.
    aoi = mpatches.Rectangle(
        (3.5, 2.0),
        3.5,
        2.3,
        edgecolor=PALETTE["slice"],
        facecolor=PALETTE["slice"] + "30",
        lw=2.0,
        label="query AOI",
    )
    axes[0].add_patch(aoi)
    axes[0].text(
        5.25,
        4.5,
        "returns rows that overlap",
        ha="center",
        fontsize=8.5,
        color="#444",
    )

    # --- intersect ---
    axes[1].set_title("intersect(left, right)", fontsize=11, fontweight="bold")
    left = mpatches.Rectangle(
        (1.0, 1.5),
        4.5,
        3.0,
        edgecolor=PALETTE["source"],
        facecolor=PALETTE["source"] + "30",
        lw=2.0,
    )
    right = mpatches.Rectangle(
        (3.5, 2.5),
        5.0,
        2.5,
        edgecolor=PALETTE["bundle"],
        facecolor=PALETTE["bundle"] + "30",
        lw=2.0,
    )
    axes[1].add_patch(left)
    axes[1].add_patch(right)
    overlap = mpatches.Rectangle(
        (3.5, 2.5),
        2.0,
        2.0,
        edgecolor=PALETTE["catalog"],
        facecolor=PALETTE["catalog"] + "60",
        lw=2.5,
        hatch="///",
    )
    axes[1].add_patch(overlap)
    axes[1].text(2.0, 1.0, "left", fontsize=10, color=PALETTE["source"])
    axes[1].text(7.0, 5.2, "right", fontsize=10, color=PALETTE["bundle"])
    axes[1].text(
        4.5,
        0.5,
        "rows in both (space AND time)",
        ha="center",
        fontsize=8.5,
        color="#444",
    )

    # --- union ---
    axes[2].set_title("union(a, b)", fontsize=11, fontweight="bold")
    a = mpatches.Rectangle(
        (1.0, 1.5),
        4.0,
        3.0,
        edgecolor=PALETTE["source"],
        facecolor=PALETTE["source"] + "40",
        lw=2.0,
    )
    b = mpatches.Rectangle(
        (4.0, 2.5),
        5.0,
        2.5,
        edgecolor=PALETTE["bundle"],
        facecolor=PALETTE["bundle"] + "40",
        lw=2.0,
    )
    axes[2].add_patch(a)
    axes[2].add_patch(b)
    axes[2].text(2.0, 1.0, "a", fontsize=10, color=PALETTE["source"])
    axes[2].text(7.5, 5.2, "b", fontsize=10, color=PALETTE["bundle"])
    axes[2].text(
        5.0,
        0.5,
        "rows from either (UNION ALL — no dedup)",
        ha="center",
        fontsize=8.5,
        color="#444",
    )

    fig.suptitle("Catalog set algebra", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    architecture_diagram(HERE / "catalog-architecture.png")
    backend_comparison(HERE / "backend-comparison.png")
    set_algebra(HERE / "set-algebra.png")
    print(f"wrote 3 PNGs to {HERE}")


if __name__ == "__main__":
    main()
