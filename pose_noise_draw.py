import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---- load ----
data = json.loads(Path("pose_noise_report.json").read_text())
df = pd.DataFrame(data["results"]).sort_values(["rot_sigma_deg", "trans_sigma"])

# ---- CVPR-ish matplotlib settings (no explicit colors) ----
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

rot_levels = sorted(df["rot_sigma_deg"].unique().tolist())
trans_levels = sorted(df["trans_sigma"].unique().tolist())

linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
markers = ["o", "s", "D", "^", "v"]

def plot_metric(metric_mean, metric_std, ylabel, outbase):
    fig = plt.figure(figsize=(3.3, 2.4), dpi=300)  # ~single-column
    ax = fig.add_subplot(111)

    for idx, r in enumerate(rot_levels):
        sub = df[df["rot_sigma_deg"] == r].sort_values("trans_sigma")
        x = sub["trans_sigma"].to_numpy()
        y = sub[metric_mean].to_numpy()
        e = sub[metric_std].to_numpy()

        ax.errorbar(
            x, y, yerr=e,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            capsize=2.0,
            label=fr"$\sigma_R={r:g}^\circ$",
        )

    ax.set_xlabel(r"Translation noise $\sigma_t$ (scene units)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(trans_levels)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, handlelength=1.6, columnspacing=0.8)

    fig.tight_layout(pad=0.2)
    fig.savefig(f"{outbase}.pdf", bbox_inches="tight")
    fig.savefig(f"{outbase}.png", bbox_inches="tight")
    plt.close(fig)

plot_metric("psnr_mean",  "psnr_std",  "PSNR (dB)", "pose_noise_psnr")
plot_metric("ssim_mean",  "ssim_std",  "SSIM",      "pose_noise_ssim")
plot_metric("lpips_mean", "lpips_std", "LPIPS",     "pose_noise_lpips")