import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- load ---
data = json.loads(Path("medium_mismatch_report.json").read_text())
df = pd.DataFrame(data["results"]).sort_values("attn_scale")

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

x = df["attn_scale"].to_numpy()

def save_line_with_err(y_mean, y_std, ylabel, outbase, lower_is_better=False):
    fig = plt.figure(figsize=(3.3, 2.4), dpi=300)  # ~single-column
    ax = fig.add_subplot(111)
    ax.errorbar(x, y_mean, yerr=y_std, marker="o", linestyle="-", capsize=2.0)
    ax.set_xlabel(r"Attenuation scale $\alpha_\beta$ (applied to $\beta(\mathbf{v})$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(x.tolist()))
    ax.grid(True, linestyle=":", linewidth=0.6)
    if lower_is_better:
        ax.set_title("lower is better", pad=2.0)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{outbase}.pdf", bbox_inches="tight")
    fig.savefig(f"{outbase}.png", bbox_inches="tight")
    plt.close(fig)

# PSNR / SSIM / LPIPS
save_line_with_err(df["psnr_mean"],  df["psnr_std"],  "PSNR (dB)", "medium_mismatch_psnr")
save_line_with_err(df["ssim_mean"],  df["ssim_std"],  "SSIM",      "medium_mismatch_ssim")
save_line_with_err(df["lpips_mean"], df["lpips_std"], "LPIPS",     "medium_mismatch_lpips", lower_is_better=True)

# Corr(depth, abs error)
fig = plt.figure(figsize=(3.3, 2.4), dpi=300)
ax = fig.add_subplot(111)
ax.errorbar(x, df["depth_abs_err_corr_mean"], yerr=df["depth_abs_err_corr_std"],
            marker="o", linestyle="-", capsize=2.0)
ax.axhline(0.0, linewidth=0.8, linestyle="--")
ax.set_xlabel(r"Attenuation scale $\alpha_\beta$")
ax.set_ylabel(r"Corr(depth, abs error)")
ax.set_xticks(sorted(x.tolist()))
ax.grid(True, linestyle=":", linewidth=0.6)
fig.tight_layout(pad=0.2)
fig.savefig("medium_mismatch_depthcorr.pdf", bbox_inches="tight")
fig.savefig("medium_mismatch_depthcorr.png", bbox_inches="tight")
plt.close(fig)

# Depth-binned PSNR proxy (5 depth quantile bins)
bins = np.stack(df["depth_bin_psnr_q20_40_60_80"].to_numpy())  # (N,5)

linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
markers = ["o", "s", "D", "^", "v"]
labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

fig = plt.figure(figsize=(3.3, 2.4), dpi=300)
ax = fig.add_subplot(111)
for b in range(5):
    ax.plot(x, bins[:, b], marker=markers[b], linestyle=linestyles[b], label=labels[b])

ax.set_xlabel(r"Attenuation scale $\alpha_\beta$")
ax.set_ylabel("Depth-binned PSNR proxy (dB)")
ax.set_xticks(sorted(x.tolist()))
ax.grid(True, linestyle=":", linewidth=0.6)
ax.legend(frameon=False, ncol=2, handlelength=1.6, columnspacing=0.8)
fig.tight_layout(pad=0.2)
fig.savefig("medium_mismatch_depthbins.pdf", bbox_inches="tight")
fig.savefig("medium_mismatch_depthbins.png", bbox_inches="tight")
plt.close(fig)