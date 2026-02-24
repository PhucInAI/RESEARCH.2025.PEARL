# make_budget_control_icon.py
# Creates a paper-ready PNG (transparent) illustrating budget control N_max:
# - baseline curve that would exceed the budget
# - PEARL curve capped below N_max
# - dashed horizontal line at N_max
#
# Output: budget_control_Nmax.png

import numpy as np
import matplotlib.pyplot as plt


def make_budget_control_icon(
    out_path="budget_control_Nmax.png",
    Nmax=50000,
    T=15000,
    steps=300,
    show_baseline=True,
    show_densify_marks=False,
):
    # Synthetic example curves (edit if you have real logs)
    t = np.linspace(0, T, steps)

    # Baseline: grows and can exceed Nmax
    baseline = 12000 + 42000 * (1 - np.exp(-t / (0.35 * T)))  # saturates ~54000
    # PEARL: grows but respects budget (soft cap)
    pearl_raw = 12000 + 52000 * (1 - np.exp(-t / (0.25 * T)))
    pearl = np.minimum(pearl_raw, Nmax * 0.995)

    # Densification event markers (optional)
    densify_steps = np.linspace(0.15 * T, 0.85 * T, 6)

    # Draw
    plt.figure(figsize=(6.2, 3.6), dpi=200)
    ax = plt.gca()

    if show_baseline:
        ax.plot(t, baseline, label="Baseline N(t)")

    ax.plot(t, pearl, label="PEARL N(t)")

    # Nmax cap line
    ax.axhline(Nmax, linestyle="--", linewidth=1.5)
    ax.text(t[int(0.02 * steps)], Nmax * 1.01, r"$N_{\max}$", va="bottom")

    # Shade infeasible region above Nmax (optional but clear)
    ax.fill_between(t, Nmax, max(baseline.max() if show_baseline else pearl.max(), Nmax) * 1.02, alpha=0.15)

    # Mark densification steps
    if show_densify_marks:
        for ds in densify_steps:
            ax.axvline(ds, linestyle=":", linewidth=1.0)
        ax.text(densify_steps[0], ax.get_ylim()[0], "densify", rotation=90, va="bottom", ha="left")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of Gaussians")
    ax.set_title("Budget control under hard cap $N_{\\max}$")
    ax.legend(loc="lower right", frameon=True)

    ax.set_xlim(0, T)
    # tighten y-limits so the cap is visible
    ymin = min(pearl.min(), baseline.min() if show_baseline else pearl.min()) * 0.95
    ymax = max(baseline.max() if show_baseline else pearl.max(), Nmax) * 1.05
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    make_budget_control_icon(
        out_path="budget_control_Nmax.png",
        Nmax=50000,
        T=15000,
        show_baseline=True,
        show_densify_marks=False,
    )
