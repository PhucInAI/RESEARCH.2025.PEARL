import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras

from pearl.ai_logger import aiLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "assets" / "results"


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    # inverse of softplus for y>0, stable piecewise
    # softplus(x)=log(1+exp(x))
    # inv(y)=log(exp(y)-1)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(y).clamp_min(1e-12)))


def logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


def make_medium_hook(model, rgb_scale=1.0, bs_scale=1.0, attn_scale=1.0):
    """
    Returns a forward hook for model.medium_mlp that rescales the *activated* medium terms exactly.
    base[:3]  -> sigmoid -> medium_rgb
    base[3:6] -> softplus(+bias) -> medium_bs
    base[6:9] -> softplus(+bias) -> medium_attn
    """
    bias = float(getattr(model, "medium_density_bias", 0.0))

    def hook(module, inputs, output):
        base = output
        # base shape (..., 9)
        if not torch.is_tensor(base) or base.shape[-1] != 9:
            return output

        base2 = base.clone()

        # rgb: sigmoid -> scale -> back to pre-sigmoid via logit
        if rgb_scale != 1.0:
            rgb = torch.sigmoid(base2[..., :3])
            rgb2 = (rgb * float(rgb_scale)).clamp(0.0, 1.0)
            base2[..., :3] = logit(rgb2)

        # bs: softplus(x+bias) -> scale -> back via inv_softplus - bias
        if bs_scale != 1.0:
            x = base2[..., 3:6]
            y = F.softplus(x + bias)
            y2 = (y * float(bs_scale)).clamp_min(1e-12)
            base2[..., 3:6] = inv_softplus(y2) - bias

        # attn: softplus(x+bias) -> scale -> back
        if attn_scale != 1.0:
            x = base2[..., 6:9]
            y = F.softplus(x + bias)
            y2 = (y * float(attn_scale)).clamp_min(1e-12)
            base2[..., 6:9] = inv_softplus(y2) - bias

        return base2

    return hook


def get_eval_dataloader(pipeline):
    dm = pipeline.datamanager
    if hasattr(dm, "fixed_indices_eval_dataloader"):
        return dm.fixed_indices_eval_dataloader
    if hasattr(dm, "eval_dataloader"):
        return dm.eval_dataloader
    raise RuntimeError("Could not find an eval dataloader on pipeline.datamanager.")


@torch.no_grad()
def eval_with_hook(pipeline, rgb_scale, bs_scale, attn_scale) -> Dict[str, float]:
    model = pipeline.model
    model.eval()

    # attach hook
    h = model.medium_mlp.register_forward_hook(
        make_medium_hook(model, rgb_scale=rgb_scale, bs_scale=bs_scale, attn_scale=attn_scale)
    )

    dl = get_eval_dataloader(pipeline)

    psnr_list, ssim_list, lpips_list = [], [], []
    corr_list = []  # corr(depth, abs_err) per image
    # depth-binned MSE accumulators (5 bins by depth quantiles)
    bin_mse_sum = np.zeros(5, dtype=np.float64)
    bin_mse_cnt = np.zeros(5, dtype=np.float64)

    try:
        for item in dl:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                camera, batch = item
            elif isinstance(item, dict) and "camera" in item:
                camera, batch = item["camera"], item
            else:
                raise RuntimeError(f"Unexpected dataloader item type: {type(item)}")

            assert isinstance(camera, Cameras)
            if camera.shape[0] != 1:
                camera = camera[0:1]

            outputs = model.get_outputs_for_camera(camera.to(model.device))
            metrics, _ = model.get_image_metrics_and_images(outputs, batch)

            psnr_list.append(metrics["psnr"])
            ssim_list.append(metrics["ssim"])
            lpips_list.append(metrics["lpips"])

            # --- detailed reporting: depth/error coupling + depth-binned PSNR proxy ---
            # Use model's internal GT handling for consistent resizing/background rules.
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])  # (H,W,3)
            pred = torch.clamp(outputs["pred_image"], 0.0, 1.0)  # (H,W,3)
            depth = outputs["depth"][..., 0]  # (H,W)
            alpha = outputs["accumulation"][..., 0]  # (H,W)

            valid = (alpha > 0.01) & torch.isfinite(depth)
            if valid.any():
                err = (pred - gt).abs().mean(dim=-1)  # (H,W)

                d = depth[valid].flatten().float()
                e = err[valid].flatten().float()
                # Pearson corr (depth vs abs_err)
                d0 = d - d.mean()
                e0 = e - e.mean()
                denom = (d0.std() * e0.std()).clamp_min(1e-12)
                corr = float((d0 * e0).mean() / denom)
                corr_list.append(corr)

                # depth quantile bins
                qs = torch.quantile(d, torch.tensor([0.2, 0.4, 0.6, 0.8], device=d.device))
                bins = torch.bucketize(depth, qs)  # 0..4
                mse_map = ((pred - gt) ** 2).mean(dim=-1)  # (H,W)

                for b in range(5):
                    m = valid & (bins == b)
                    if m.any():
                        bin_mse_sum[b] += float(mse_map[m].sum().item())
                        bin_mse_cnt[b] += float(m.sum().item())
    finally:
        h.remove()

    def _mean(x): return float(np.mean(x)) if len(x) else float("nan")
    def _std(x): return float(np.std(x)) if len(x) else float("nan")

    # depth-binned PSNR proxy from accumulated MSE
    bin_psnr = []
    for b in range(5):
        if bin_mse_cnt[b] > 0:
            mse = bin_mse_sum[b] / bin_mse_cnt[b]
            bin_psnr.append(float(-10.0 * np.log10(max(mse, 1e-12))))
        else:
            bin_psnr.append(float("nan"))

    return {
        "rgb_scale": float(rgb_scale),
        "bs_scale": float(bs_scale),
        "attn_scale": float(attn_scale),
        "psnr_mean": _mean(psnr_list),
        "psnr_std": _std(psnr_list),
        "ssim_mean": _mean(ssim_list),
        "ssim_std": _std(ssim_list),
        "lpips_mean": _mean(lpips_list),
        "lpips_std": _std(lpips_list),
        "depth_abs_err_corr_mean": _mean(corr_list),
        "depth_abs_err_corr_std": _std(corr_list),
        "depth_bin_psnr_q20_40_60_80": bin_psnr,  # 5 bins
        "n_images": int(len(psnr_list)),
    }


def parse_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--attn-scales", type=str, default="0.5,0.75,1,1.25,1.5,2.0")
    ap.add_argument("--bs-scales", type=str, default="1.0")   # keep one-by-one unless you want a grid
    ap.add_argument("--rgb-scales", type=str, default="1.0")  # keep one-by-one unless you want a grid
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "medium_mismatch_report.json")
    args = ap.parse_args()

    _, pipeline, _, _ = eval_setup(args.load_config, eval_num_rays_per_chunk=None, test_mode="test")

    attn_list = parse_list(args.attn_scales)
    bs_list = parse_list(args.bs_scales)
    rgb_list = parse_list(args.rgb_scales)

    results = []
    for a in attn_list:
        for b in bs_list:
            for r in rgb_list:
                res = eval_with_hook(pipeline, rgb_scale=r, bs_scale=b, attn_scale=a)
                results.append(res)
                aiLogger.info(res)

    payload = {
        "load_config": str(args.load_config),
        "attn_scales": attn_list,
        "bs_scales": bs_list,
        "rgb_scales": rgb_list,
        "results": results,
        "reporting": {
            "depth_abs_err_corr": "Pearson corr between rendered depth and per-pixel absolute RGB error (valid pixels only).",
            "depth_bin_psnr": "PSNR proxy computed from MSE accumulated in 5 depth quantile bins (0-20,20-40,...,80-100%).",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    aiLogger.info(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


# python medium_mismatch_stress.py \
#   --load-config /path/to/outputs/.../config.yml \
#   --attn-scales 0.5,0.75,1,1.25,1.5,2.0 \
#   --out medium_mismatch_report.json