# export_pearl_components.py
# Generates: (1) component box PNGs, (2) input/pred/residual/edge PNGs,
# (3) connections.csv describing all arrows you should draw in draw.io.

import argparse
import os
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from pearl.ai_logger import aiLogger

FIGURES_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"


# -----------------------------
# Utilities
# -----------------------------
def mkdirp(p): os.makedirs(p, exist_ok=True)

def load_rgb(path, max_side=1200):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

def resize_to(arr, H, W):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im = im.resize((W, H), Image.BICUBIC)
    return np.asarray(im).astype(np.float32) / 255.0

def ensure_same_size(a, b):
    Ha, Wa = a.shape[:2]
    Hb, Wb = b.shape[:2]
    if (Ha, Wa) == (Hb, Wb):
        return a, b
    # resize b to a
    b2 = resize_to(b, Ha, Wa)
    return a, b2

def save_rgb(arr, path):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im.save(path)

def save_heatmap(gray01, path, cmap="magma", vmin=0.0, vmax=1.0, title=None):
    plt.figure(figsize=(6, 4))
    plt.imshow(gray01, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=10)
    plt.tight_layout(pad=0.05)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def robust_normalize(x, lo=1.0, hi=99.0):
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    y = (x - a) / (b - a + 1e-8)
    return np.clip(y, 0.0, 1.0)

def compute_residual(input_img, pred_img, mode="l2"):
    # input_img treated as reference (GT), pred_img as prediction
    diff = pred_img - input_img
    if mode == "l1":
        e = np.mean(np.abs(diff), axis=2)
    else:  # l2
        e = np.sqrt(np.sum(diff * diff, axis=2) + 1e-8)
    e_vis = robust_normalize(e, 1, 99)
    return e, e_vis

def sobel_edge(gray01):
    # simple Sobel without scipy
    # gray01: HxW float
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    Ky = np.array([[1,  2,  1],
                   [0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)

    H, W = gray01.shape
    pad = 1
    g = np.pad(gray01, ((pad, pad), (pad, pad)), mode="edge")
    gx = np.zeros((H, W), dtype=np.float32)
    gy = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            patch = g[y:y+3, x:x+3]
            gx[y, x] = np.sum(patch * Kx)
            gy[y, x] = np.sum(patch * Ky)
    mag = np.sqrt(gx*gx + gy*gy)
    mag_vis = robust_normalize(mag, 5, 99)
    return mag, mag_vis


# -----------------------------
# Box rendering (PNG assets for draw.io)
# -----------------------------
def load_font(size):
    # Try common fonts; fall back to default
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def wrap_text(draw, text, font, max_w):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)

def render_box_png(out_path, title, body, size=(900, 260), bg=(245,245,245), border=(40,40,40)):
    W, H = size
    im = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(im)

    # rounded rect
    r = 28
    x0, y0 = 8, 8
    x1, y1 = W-8, H-8
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=bg+(255,), outline=border+(255,), width=4)

    font_t = load_font(32)
    font_b = load_font(24)

    # Title
    title_bbox = draw.textbbox((0,0), title, font=font_t)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((W-title_w)//2, 16), title, fill=(15,15,15,255), font=font_t)

    # Body (wrapped)
    max_w = W - 60
    body_wrapped = wrap_text(draw, body, font_b, max_w)
    draw.multiline_text((30, 70), body_wrapped, fill=(25,25,25,255), font=font_b, spacing=6)

    im.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_image", required=True, help="Reference/GT image (same view as prediction)")
    ap.add_argument("--pred_image", required=True, help="Rendered prediction image")
    ap.add_argument("--out_dir", default=str(FIGURES_DIR / "pearl_assets"), help="Output directory for draw.io assets")
    ap.add_argument("--residual_mode", default="l2", choices=["l2", "l1"], help="Residual type")
    args = ap.parse_args()

    out_dir = args.out_dir
    mkdirp(out_dir)
    mkdirp(os.path.join(out_dir, "images"))
    mkdirp(os.path.join(out_dir, "boxes"))

    # -----------------------------
    # 1) Export your images (input, pred, residual, edge)
    # -----------------------------
    inp = load_rgb(args.input_image)
    pred = load_rgb(args.pred_image)
    inp, pred = ensure_same_size(inp, pred)

    save_rgb(inp,  os.path.join(out_dir, "images", "C01_input.png"))
    save_rgb(pred, os.path.join(out_dir, "images", "C02_pred.png"))

    e, e_vis = compute_residual(inp, pred, mode=args.residual_mode)
    save_heatmap(e_vis, os.path.join(out_dir, "images", "C03_residual.png"),
                 cmap="magma", title=f"Residual ({args.residual_mode}, robust norm)")

    gray = (0.299*inp[:,:,0] + 0.587*inp[:,:,1] + 0.114*inp[:,:,2]).astype(np.float32)
    _, edge_vis = sobel_edge(gray)
    save_heatmap(edge_vis, os.path.join(out_dir, "images", "C04_edge.png"),
                 cmap="gray", title="Edge strength (Sobel, robust norm)")

    # -----------------------------
    # 2) Export boxes (components) as PNG
    # -----------------------------
    BOX = os.path.join(out_dir, "boxes")

    components = [
        ("B01_inputs", "Inputs",
         "Posed underwater images (I_m, K_m, T_m). Optional depth priors D_m.",
         (242,246,255)),
        ("B02_init", "Initialization (M3)",
         "SfM/COLMAP points → initial Gaussians. Add complexity-based seeds in high-frequency / high-depth-variance regions.",
         (240,255,246)),
        ("B03_gaussians", "Gaussian Scene",
         "Primitives: mean μ_i ∈ R^3, covariance Σ_i, opacity α_i, SH color coefficients a_{i,lm}.",
         (247,247,247)),
        ("B04_project", "Project → 2D Gaussians",
         "Project 3D Gaussians to image: (u_i, Σ_i^{2D}) using camera (K,T).",
         (255,247,242)),
        ("B05_render", "Alpha Compositing",
         "Front-to-back compositing: w_{p,k}=T_{p,k-1} α_{i_k}(p), T_{p,k} = Π_{j≤k}(1-α_{i_j}(p)).",
         (255,247,242)),
        ("B06_mednet", "Medium MLP (M5)",
         "Direction-conditioned network: E(v) → (c_med(v), β(v), optional σ_n(v)).",
         (239,255,253)),
        ("B07_attn", "Depth-linked Attenuation (M5)",
         "Beer–Lambert: T_med(d,v)=exp(-β(v)·d). Modulate Gaussian color: c_i(v) ⊙ T_med(d_i,v).",
         (239,255,253)),
        ("B08_bg", "Background / Veiling (M5)",
         "Background term: c_bg(v,D)=c_med(v) ⊙ (1 - T_med(D,v)).",
         (239,255,253)),
        ("B09_loss", "Loss (M4)",
         "Photometric loss + SSIM, plus edge-weighted term using Sobel edge map g_p.",
         (255,242,246)),
        ("B10_opt", "Optimizer",
         "Adam updates Gaussian parameters Θ and medium parameters φ by backprop through renderer.",
         (242,242,255)),
        ("B11_errattrib", "Error Attribution (M1)",
         "Pixel residual e_p attributed to Gaussians: E_i = Σ_p (w̃_{p,i} e_p)/Σ_p w̃_{p,i}. EMA: Ē_i.",
         (255,253,235)),
        ("B12_score", "Densify Score (M1)",
         "Combine gradient activity g_i with attributed error: h_i=g_i(1+λ_E Ē_i/(Ē_mean+ε)).",
         (255,253,235)),
        ("B13_budget", "Budget Control (M2)",
         "Select split/duplicate operations under hard budget N_max (rank by h_i).",
         (255,253,235)),
        ("B14_splitdup", "Split / Duplicate (M2)",
         "Split large Gaussians; duplicate small ones. Create children means/scale adjustments; add to set.",
         (255,253,235)),
        ("B15_output", "Outputs",
         "Optimized Gaussians + medium model. Real-time novel-view synthesis rendering.",
         (247,247,247)),
    ]

    for cid, title, body, bg in components:
        render_box_png(
            os.path.join(BOX, f"{cid}.png"),
            title=title,
            body=body,
            bg=bg
        )

    # -----------------------------
    # 3) Export connections (arrows) as CSV for draw.io
    # -----------------------------
    # Node IDs are the PNG filenames (without extension). Use these to connect in draw.io.
    connections = [
        # Image insets
        ("C01_input", "B01_inputs", "I_m"),
        ("B01_inputs", "B02_init", "poses / intrinsics"),
        ("B02_init", "B03_gaussians", "initialize primitives"),

        # Rendering path
        ("B03_gaussians", "B04_project", "project"),
        ("B04_project", "B05_render", "splat + blend"),
        ("B06_mednet", "B07_attn", "β(v), c_med(v)"),
        ("B07_attn", "B05_render", "modulate colors"),
        ("B08_bg", "B05_render", "background/veiling"),
        ("B05_render", "C02_pred", "render prediction"),

        # Residual + edge (computed from input & pred)
        ("C01_input", "C03_residual", "reference"),
        ("C02_pred", "C03_residual", "compare"),
        ("C01_input", "C04_edge", "Sobel"),

        # Loss + optimizer
        ("C02_pred", "B09_loss", "prediction"),
        ("C01_input", "B09_loss", "reference"),
        ("C04_edge", "B09_loss", "weights"),
        ("B09_loss", "B10_opt", "backprop gradients"),
        ("B10_opt", "B03_gaussians", "update Θ"),
        ("B10_opt", "B06_mednet", "update φ"),

        # Densification loop
        ("C03_residual", "B11_errattrib", "e_p"),
        ("B11_errattrib", "B12_score", "Ē_i"),
        ("B03_gaussians", "B12_score", "g_i / size"),
        ("B12_score", "B13_budget", "rank by h_i"),
        ("B13_budget", "B14_splitdup", "select ops under N_max"),
        ("B14_splitdup", "B03_gaussians", "add/refine primitives"),

        # Final outputs
        ("B03_gaussians", "B15_output", "final Θ"),
        ("B06_mednet", "B15_output", "final φ"),
        ("B15_output", "C02_pred", "novel-view render (inference)"),
    ]

    with open(os.path.join(out_dir, "connections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_id", "to_id", "label"])
        for a, b, lab in connections:
            w.writerow([a, b, lab])

    aiLogger.info(f"Wrote assets to: {out_dir}")
    aiLogger.info(" - images/: input/pred/residual/edge PNGs")
    aiLogger.info(" - boxes/: component PNGs")
    aiLogger.info(" - connections.csv: all arrows to draw in draw.io")


if __name__ == "__main__":
    main()
