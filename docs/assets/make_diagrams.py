"""Generate the concept PNGs referenced from the docs.

Run from the repo root::

    uv run --group docs python docs/assets/make_diagrams.py

Produces (under ``docs/assets/``):

* ``composition-shapes.png`` — Sequential / Graph / Branch / Switch as
  boxes-and-arrows.
* ``operator-lifecycle.png`` — input → forward → output with type
  annotations.
* ``pipeline-ecosystem.png`` — how ``geocatalog``, ``geotoolz``, and
  ``geopatcher`` fit together.

The script uses only ``matplotlib`` + synthetic data so it runs in any
sandbox; no network, no real GeoTensors required.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent

NODE_FACE = "#EEF2FF"
NODE_EDGE = "#4F46E5"
ACCENT = "#4F46E5"
SUBTLE = "#94A3B8"
TEXT = "#0F172A"
GROUP_FACE = "#F8FAFC"
GROUP_EDGE = "#CBD5E1"


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    face: str = NODE_FACE,
    edge: str = NODE_EDGE,
) -> tuple[float, float, float, float]:
    """Draw a rounded box centred on (x, y); return its (x, y, w, h)."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=TEXT)
    return x, y, w, h


def _arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = SUBTLE,
    style: str = "-",
) -> None:
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color=color,
        linestyle=style,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def _group(ax, x: float, y: float, w: float, h: float, title: str) -> None:
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0,
        facecolor=GROUP_FACE,
        edgecolor=GROUP_EDGE,
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(
        x - w / 2 + 0.1,
        y + h / 2 - 0.18,
        title,
        ha="left",
        va="center",
        fontsize=9,
        color=SUBTLE,
        fontstyle="italic",
    )


def make_composition_shapes() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    fig.suptitle("Composition shapes", fontsize=13, color=TEXT, y=0.98)

    # --- Sequential ---
    ax = axes[0, 0]
    ax.set_title("Sequential — linear chain", fontsize=10.5, color=TEXT)
    for i, label in enumerate(["op_a", "op_b", "op_c"]):
        _box(ax, 1.0 + i * 1.5, 1.0, 1.1, 0.55, label)
    _arrow(ax, 1.55, 1.0, 1.95, 1.0)
    _arrow(ax, 3.05, 1.0, 3.45, 1.0)
    _arrow(ax, 0.2, 1.0, 0.45, 1.0)
    _arrow(ax, 4.55, 1.0, 4.85, 1.0)
    ax.text(0.2, 1.25, "x", fontsize=9, color=TEXT)
    ax.text(4.85, 1.25, "y", fontsize=9, color=TEXT)
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 2)
    ax.axis("off")

    # --- Graph ---
    ax = axes[0, 1]
    ax.set_title("Graph — branching / fan-in", fontsize=10.5, color=TEXT)
    _box(ax, 0.6, 1.5, 0.9, 0.5, "Input")
    _box(ax, 2.0, 2.1, 1.0, 0.5, "scale")
    _box(ax, 2.0, 0.9, 1.0, 0.5, "mask")
    _box(ax, 3.5, 1.5, 1.0, 0.5, "apply")
    _box(ax, 4.9, 1.5, 1.0, 0.5, "ndvi")
    _arrow(ax, 1.05, 1.55, 1.5, 2.05)
    _arrow(ax, 1.05, 1.45, 1.5, 0.95)
    _arrow(ax, 2.5, 2.05, 3.0, 1.6)
    _arrow(ax, 2.5, 0.95, 3.0, 1.4)
    _arrow(ax, 4.0, 1.5, 4.4, 1.5)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 2.8)
    ax.axis("off")

    # --- Branch ---
    ax = axes[1, 0]
    ax.set_title("Branch — runtime 2-way fork", fontsize=10.5, color=TEXT)
    _box(ax, 0.7, 1.0, 0.7, 0.5, "x")
    _box(ax, 2.2, 1.0, 1.1, 0.5, "predicate?", face="#FEF3C7", edge="#D97706")
    _box(ax, 3.9, 1.7, 1.1, 0.5, "if_true")
    _box(ax, 3.9, 0.3, 1.1, 0.5, "if_false")
    _box(ax, 5.4, 1.0, 0.7, 0.5, "y")
    _arrow(ax, 1.05, 1.0, 1.65, 1.0)
    _arrow(ax, 2.75, 1.15, 3.35, 1.65)
    _arrow(ax, 2.75, 0.85, 3.35, 0.35)
    _arrow(ax, 4.45, 1.65, 5.05, 1.15)
    _arrow(ax, 4.45, 0.35, 5.05, 0.85)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2.2)
    ax.axis("off")

    # --- Switch ---
    ax = axes[1, 1]
    ax.set_title("Switch — N-way dispatch by key", fontsize=10.5, color=TEXT)
    _box(ax, 0.7, 1.0, 0.7, 0.5, "x")
    _box(ax, 2.2, 1.0, 0.9, 0.5, "key(x)", face="#FEF3C7", edge="#D97706")
    _box(ax, 4.0, 1.8, 1.1, 0.5, "case 'S2'")
    _box(ax, 4.0, 1.0, 1.1, 0.5, "case 'L8'")
    _box(ax, 4.0, 0.2, 1.1, 0.5, "default")
    _box(ax, 5.7, 1.0, 0.7, 0.5, "y")
    _arrow(ax, 1.05, 1.0, 1.75, 1.0)
    _arrow(ax, 2.65, 1.15, 3.45, 1.75)
    _arrow(ax, 2.65, 1.0, 3.45, 1.0)
    _arrow(ax, 2.65, 0.85, 3.45, 0.25)
    _arrow(ax, 4.55, 1.75, 5.35, 1.2)
    _arrow(ax, 4.55, 1.0, 5.35, 1.0)
    _arrow(ax, 4.55, 0.25, 5.35, 0.8)
    ax.set_xlim(0, 6.3)
    ax.set_ylim(0, 2.4)
    ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "composition-shapes.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_operator_lifecycle() -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_title("Operator lifecycle", fontsize=12, color=TEXT, pad=10)

    _box(ax, 1.0, 2.0, 1.7, 0.7, "GeoTensor\n(in)", face="#ECFEFF", edge="#0891B2")
    _box(ax, 3.4, 2.0, 1.7, 0.7, "op.__call__(x)")
    _box(ax, 5.8, 2.65, 1.9, 0.7, "op._apply(x)\n(eager)")
    _box(
        ax,
        5.8,
        1.35,
        1.9,
        0.7,
        "new Node\n(graph mode)",
        face="#FEF3C7",
        edge="#D97706",
    )
    _box(ax, 8.2, 2.0, 1.7, 0.7, "GeoTensor\n(out)", face="#ECFEFF", edge="#0891B2")

    _box(ax, 3.4, 0.5, 1.7, 0.55, "op.get_config()", face="#F5F3FF", edge="#7C3AED")
    _box(ax, 6.0, 0.5, 1.9, 0.55, "YAML / Hydra-zen", face="#F5F3FF", edge="#7C3AED")

    _arrow(ax, 1.85, 2.0, 2.55, 2.0)
    _arrow(ax, 4.25, 2.1, 4.85, 2.55)
    _arrow(ax, 4.25, 1.9, 4.85, 1.45)
    _arrow(ax, 6.75, 2.5, 7.35, 2.1)

    _arrow(ax, 4.25, 0.5, 5.05, 0.5, color=ACCENT, style="--")
    ax.text(2.6, 2.25, "value", fontsize=8.5, color=SUBTLE)
    ax.text(4.4, 2.6, "value path", fontsize=8.5, color=SUBTLE)
    ax.text(4.4, 1.05, "Input/Node path", fontsize=8.5, color=SUBTLE)
    ax.text(4.4, 0.65, "round-trips", fontsize=8.5, color=ACCENT, fontstyle="italic")

    ax.set_xlim(0, 9.4)
    ax.set_ylim(0, 3.3)
    ax.axis("off")

    fig.tight_layout()
    out = HERE / "operator-lifecycle.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_pipeline_ecosystem() -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_title(
        "The pipeline ecosystem — geocatalog → geotoolz → geopatcher",
        fontsize=12,
        color=TEXT,
        pad=10,
    )

    # group rects — centred at y=2.0, each 2.8 wide x 3.2 tall
    _group(ax, 1.9, 2.0, 2.8, 3.2, "geocatalog — discover & load")
    _group(ax, 5.7, 2.0, 2.8, 3.2, "geotoolz — operate")
    _group(ax, 9.5, 2.0, 2.8, 3.2, "geopatcher (via geotoolz.patch_ops)")

    # geocatalog: stack STAC -> LoadScene vertically
    _box(ax, 1.9, 3.0, 1.8, 0.55, "STAC catalogue", face="#FEF9C3", edge="#CA8A04")
    _box(ax, 1.9, 2.0, 1.8, 0.55, "LoadScene")
    _arrow(ax, 1.9, 2.7, 1.9, 2.3)

    # geotoolz: three boxes vertical
    _box(ax, 5.7, 3.0, 1.8, 0.55, "Scale")
    _box(ax, 5.7, 2.0, 1.8, 0.55, "CloudMask")
    _box(ax, 5.7, 1.0, 1.8, 0.55, "NDVI")
    _arrow(ax, 5.7, 2.7, 5.7, 2.3)
    _arrow(ax, 5.7, 1.7, 5.7, 1.3)

    # geopatcher: three boxes vertical
    _box(ax, 9.5, 3.0, 1.9, 0.55, "GridSampler")
    _box(ax, 9.5, 2.0, 1.9, 0.55, "ApplyToChips")
    _box(ax, 9.5, 1.0, 1.9, 0.55, "Stitch")
    _arrow(ax, 9.5, 2.7, 9.5, 2.3)
    _arrow(ax, 9.5, 1.7, 9.5, 1.3)

    # cross-package arrows (GeoTensor flows between groups)
    _arrow(ax, 2.85, 2.0, 4.75, 3.0, color=ACCENT)
    _arrow(ax, 6.65, 1.0, 8.55, 3.0, color=ACCENT)

    ax.text(3.35, 2.7, "GeoTensor", fontsize=9, color=ACCENT, fontstyle="italic")
    ax.text(7.15, 2.05, "GeoTensor", fontsize=9, color=ACCENT, fontstyle="italic")

    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.2, 4.0)
    ax.axis("off")

    legend = [
        mpatches.Patch(facecolor=NODE_FACE, edgecolor=NODE_EDGE, label="Operator"),
        mpatches.Patch(
            facecolor="#FEF9C3", edgecolor="#CA8A04", label="External source"
        ),
    ]
    ax.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.tight_layout()
    out = HERE / "pipeline-ecosystem.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    for path in (
        make_composition_shapes(),
        make_operator_lifecycle(),
        make_pipeline_ecosystem(),
    ):
        print(f"wrote {path.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
