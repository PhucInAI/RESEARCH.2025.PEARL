from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "assets" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- data (Curacao, SeaThru-NeRF) ---
budgets = np.array([10_000, 20_000, 50_000, 100_000, 200_000, 1_000_000, 10_000_000], dtype=float)
psnr  = np.array([26.913822174072266, 27.479276657104492, 28.441143035888672, 29.126787185668945, 29.2712, 29.872440338134766, 29.54])
ssim  = np.array([0.7817281484603882, 0.8228389620780945, 0.8642091751098633, 0.9008803367614746, 0.912, 0.9202315807342529, 0.918])
lpips = np.array([0.5087003707885742, 0.4067353904247284, 0.3088158071041107, 0.22225245833396912, 0.191, 0.16798260807991028, 0.155])

# --- CVPR-ish matplotlib settings (no explicit colors) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.2,
})

def _tick(v: float) -> str:
    v = int(v)
    if v >= 1_000_000: return f"{v//1_000_000}M"
    if v >= 1_000:     return f"{v//1_000}K"
    return str(v)

tick_labels = [_tick(v) for v in budgets]

def plot_curve(y, ylabel, out_pdf, lower_is_better=False):
    fig = plt.figure(figsize=(3.3, 2.4), dpi=300)  # ~single-column width
    ax = fig.add_subplot(111)
    ax.plot(budgets, y, marker="o", linestyle="-")
    ax.set_xscale("log")
    ax.set_xlabel(r"Gaussian budget $N_{\max}$ (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(budgets)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, linestyle=":", linewidth=0.6, which="both")
    if lower_is_better:
        ax.set_title("lower is better", pad=2.0)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

plot_curve(psnr,  "PSNR (dB)", RESULTS_DIR / "ablation_budget_psnr.pdf")
plot_curve(ssim,  "SSIM",      RESULTS_DIR / "ablation_budget_ssim.pdf")
plot_curve(lpips, "LPIPS",     RESULTS_DIR / "ablation_budget_lpips.pdf", lower_is_better=True)