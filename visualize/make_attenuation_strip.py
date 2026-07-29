from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pearl.ai_logger import aiLogger

FIGURES_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def make_attenuation_strip(
    out_path=FIGURES_DIR / "attenuation_strip_rgb.png",
    W=1200, H=360,
    beta=(2.2, 1.2, 0.6),   # relative attenuation strengths for (R,G,B)
    dmax=1.0,              # normalized distance range
):
    """
    Creates a 3-row RGB attenuation strip:
      row 0: Red channel fades fastest
      row 1: Green fades medium
      row 2: Blue fades slowest
    The strip intensity is multiplied by T(d)=exp(-beta*d).
    """
    # Coordinate along the strip (distance)
    x = np.linspace(0.0, dmax, W, dtype=np.float32)

    # Compute transmittance per channel
    T = np.stack([np.exp(-b * x) for b in beta], axis=0)  # (3, W)

    # Base colors for each row: pure R, pure G, pure B
    base = np.array([
        [1.0, 0.0, 0.0],  # red row
        [0.0, 1.0, 0.0],  # green row
        [0.0, 0.0, 1.0],  # blue row
    ], dtype=np.float32)  # (3,3)

    # Build image: 3 bands, each band color multiplied by its T
    img = np.zeros((H, W, 3), dtype=np.float32)
    band_h = H // 3
    for i in range(3):
        y0 = i * band_h
        y1 = (i + 1) * band_h if i < 2 else H
        img[y0:y1, :, :] = base[i][None, None, :] * T[i][None, :, None]

    # Plot with curves overlay
    plt.figure(figsize=(W / 150, H / 150), dpi=150)
    plt.imshow(img, origin="upper")
    plt.axis("off")

    # Overlay decay curves per band (white curve)
    for i, b in enumerate(beta):
        y_center = (i + 0.5) * band_h
        curve = np.exp(-b * x)  # (W,)
        # map curve value (0..1) to y position within the band (top->bottom)
        y = (i * band_h) + (1.0 - curve) * (band_h - 1)
        plt.plot(np.arange(W), y, linewidth=2)

    # Optional labels
    # plt.text(10, 20, r"$T(d)=\exp(-\beta(v) d)$", fontsize=14)

    plt.tight_layout(pad=0)
    plt.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()
    aiLogger.info(f"Saved: {out_path}")

if __name__ == "__main__":
    make_attenuation_strip()
