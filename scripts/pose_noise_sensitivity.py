import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "assets" / "results"


def _skew(w: torch.Tensor) -> torch.Tensor:
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    return torch.stack(
        [
            torch.stack([O, -wz, wy], dim=-1),
            torch.stack([wz, O, -wx], dim=-1),
            torch.stack([-wy, wx, O], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Exponential map from axis-angle vector w (...,3) to rotation matrix (...,3,3)."""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / theta
    K = _skew(k)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(K.shape[:-2] + (3, 3))
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


def perturb_c2w_camera_frame(
    c2w: torch.Tensor,
    rot_sigma_deg: float,
    trans_sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Apply SE(3) noise in the camera frame:
      R' = R @ Exp(w),  t' = t + R @ delta
    c2w: (3,4)
    """
    dtype, device = c2w.dtype, c2w.device
    R = c2w[:, :3]
    t = c2w[:, 3]

    rot_sigma = float(rot_sigma_deg) * np.pi / 180.0
    w = torch.randn(3, generator=generator, device=device, dtype=dtype) * rot_sigma
    d = torch.randn(3, generator=generator, device=device, dtype=dtype) * float(trans_sigma)

    dR = so3_exp(w)
    Rn = R @ dR
    tn = t + R @ d

    out = c2w.clone()
    out[:, :3] = Rn
    out[:, 3] = tn
    return out


def get_eval_dataloader(pipeline):
    dm = pipeline.datamanager
    if hasattr(dm, "fixed_indices_eval_dataloader"):
        return dm.fixed_indices_eval_dataloader
    if hasattr(dm, "eval_dataloader"):
        return dm.eval_dataloader
    raise RuntimeError("Could not find an eval dataloader on pipeline.datamanager.")


@torch.no_grad()
def eval_once(pipeline, rot_sigma_deg: float, trans_sigma: float, seed: int) -> Dict[str, float]:
    model = pipeline.model
    model.eval()

    dl = get_eval_dataloader(pipeline)

    g = torch.Generator(device=model.device)
    g.manual_seed(seed)

    psnr_list, ssim_list, lpips_list = [], [], []

    for item in dl:
        # FullImageDataManager typically yields (camera, batch); be robust.
        if isinstance(item, (tuple, list)) and len(item) == 2:
            camera, batch = item
        elif isinstance(item, dict) and "camera" in item:
            camera, batch = item["camera"], item
        else:
            raise RuntimeError(f"Unexpected dataloader item type: {type(item)}")

        assert isinstance(camera, Cameras), "Expected Cameras from eval dataloader."
        if camera.shape[0] != 1:
            camera = camera[0:1]

        cam_noisy = camera.to(model.device)
        c2w = cam_noisy.camera_to_worlds[0].clone()  # (3,4)
        c2w_noisy = perturb_c2w_camera_frame(c2w, rot_sigma_deg, trans_sigma, g)
        cam_noisy.camera_to_worlds[0] = c2w_noisy

        outputs = model.get_outputs_for_camera(cam_noisy)
        metrics, _ = model.get_image_metrics_and_images(outputs, batch)

        psnr_list.append(metrics["psnr"])
        ssim_list.append(metrics["ssim"])
        lpips_list.append(metrics["lpips"])

    def _mean(x): return float(np.mean(x)) if len(x) else float("nan")
    def _std(x): return float(np.std(x)) if len(x) else float("nan")

    return {
        "rot_sigma_deg": float(rot_sigma_deg),
        "trans_sigma": float(trans_sigma),
        "psnr_mean": _mean(psnr_list),
        "psnr_std": _std(psnr_list),
        "ssim_mean": _mean(ssim_list),
        "ssim_std": _std(ssim_list),
        "lpips_mean": _mean(lpips_list),
        "lpips_std": _std(lpips_list),
        "n_images": int(len(psnr_list)),
    }


def parse_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-config", type=Path, required=True, help="Path to Nerfstudio config.yml")
    ap.add_argument("--rot-sigmas-deg", type=str, default="0,0.25,0.5,1,2,5")
    ap.add_argument("--trans-sigmas", type=str, default="0,0.0025,0.005,0.01,0.02")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "pose_noise_report.json")
    args = ap.parse_args()

    _, pipeline, _, _ = eval_setup(args.load_config, eval_num_rays_per_chunk=None, test_mode="test")

    rot_list = parse_list(args.rot_sigmas_deg)
    trans_list = parse_list(args.trans_sigmas)

    results = []
    for r in rot_list:
        for t in trans_list:
            # deterministic but different per gridpoint
            grid_seed = args.seed + int(round(r * 1000)) * 100000 + int(round(t * 1e6))
            res = eval_once(pipeline, r, t, grid_seed)
            results.append(res)
            print(res)

    payload = {
        "load_config": str(args.load_config),
        "rot_sigmas_deg": rot_list,
        "trans_sigmas": trans_list,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

# python pose_noise_sensitivity.py \
#   --load-config /path/to/outputs/.../config.yml \
#   --rot-sigmas-deg 0,0.5,1,2,5 \
#   --trans-sigmas 0,0.005,0.01,0.02 \
#   --out pose_noise_report.json