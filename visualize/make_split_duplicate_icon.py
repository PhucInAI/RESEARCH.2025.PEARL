# make_split_duplicate_icon.py
# Generates a transparent PNG icon illustrating:
# (A) Split: one Gaussian -> k smaller Gaussians along principal axis
# (B) Duplicate: one Gaussian -> two overlapping copies (slight offset)
#
# Output: assets/figures/split_duplicate_icon.png

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

FIGURES_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_split_duplicate_icon(
    out_path=FIGURES_DIR / "split_duplicate_icon.png",
    k_children=3,
    seed=0,
):
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(8.0, 3.2), dpi=200)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    # Canvas coordinates
    # Left panel: split
    # Right panel: duplicate
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    # ----------------------------
    # (A) SPLIT
    # ----------------------------
    ax.text(1.9, 3.6, "Split", fontsize=12, fontweight="bold", ha="center")

    # Parent Gaussian (ellipse)
    parent_center = (1.9, 2.0)
    parent_w, parent_h, parent_ang = 1.8, 1.1, 25
    parent = Ellipse(parent_center, parent_w, parent_h, angle=parent_ang,
                     edgecolor="black", facecolor="none", linewidth=2)
    ax.add_patch(parent)
    ax.text(parent_center[0], 0.45, "large $s_i$", fontsize=10, ha="center")

    # Arrow to children
    ax.add_patch(FancyArrowPatch((2.95, 2.0), (4.1, 2.0),
                                arrowstyle="-|>", mutation_scale=16,
                                linewidth=2, color="black"))

    # Children Gaussians along principal axis
    # Principal axis direction (unit vector) from angle
    theta = np.deg2rad(parent_ang)
    dir_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    child_base = np.array([4.8, 2.0], dtype=float)
    offsets = np.linspace(-0.7, 0.7, k_children)

    for j, off in enumerate(offsets):
        c = child_base + off * dir_vec
        child = Ellipse((c[0], c[1]),
                        parent_w / np.sqrt(2.2),
                        parent_h / np.sqrt(2.2),
                        angle=parent_ang,
                        edgecolor="black",
                        facecolor="none",
                        linewidth=2)
        ax.add_patch(child)

    ax.text(4.8, 0.45, "children: $\\mu_i+\\delta_j$, $s_i/\\sqrt{\\gamma}$", fontsize=10, ha="center")

    # ----------------------------
    # (B) DUPLICATE
    # ----------------------------
    ax.text(8.1, 3.6, "Duplicate", fontsize=12, fontweight="bold", ha="center")

    # Original Gaussian
    orig_center = (7.0, 2.0)
    orig_w, orig_h, orig_ang = 1.3, 0.9, -15
    orig = Ellipse(orig_center, orig_w, orig_h, angle=orig_ang,
                   edgecolor="black", facecolor="none", linewidth=2)
    ax.add_patch(orig)
    ax.text(orig_center[0], 0.45, "small $s_i$", fontsize=10, ha="center")

    # Arrow to duplicated copies
    ax.add_patch(FancyArrowPatch((7.75, 2.0), (8.6, 2.0),
                                arrowstyle="-|>", mutation_scale=16,
                                linewidth=2, color="black"))

    # Duplicate pair (slight offset)
    dup_center = np.array([9.2, 2.0], dtype=float)
    off = np.array([0.18, 0.10], dtype=float)

    dup1 = Ellipse((dup_center[0] - off[0], dup_center[1] - off[1]),
                   orig_w, orig_h, angle=orig_ang,
                   edgecolor="black", facecolor="none", linewidth=2)
    dup2 = Ellipse((dup_center[0] + off[0], dup_center[1] + off[1]),
                   orig_w, orig_h, angle=orig_ang,
                   edgecolor="black", facecolor="none", linewidth=2)

    ax.add_patch(dup1)
    ax.add_patch(dup2)
    ax.text(9.2, 0.45, "copy params\n(+ tiny offset)", fontsize=10, ha="center")

    # Optional small note
    ax.text(5.0, 0.05, r"Budgeted selection under $N_{\max}$ chooses split vs duplicate", fontsize=10, ha="center")

    plt.tight_layout(pad=0)
    plt.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_split_duplicate_icon(out_path=FIGURES_DIR / "split_duplicate_icon.png", k_children=3)
