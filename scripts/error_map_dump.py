import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pearl.ai_logger import aiLogger
from pearl.ns_cli import pearl_eval_setup
from pearl.run_paths import run_artifact_dir

DIR_NAME = "error_maps"


def get_eval_dataloader(pipeline):
    dm = pipeline.datamanager
    if hasattr(dm, "fixed_indices_eval_dataloader"):
        return dm.fixed_indices_eval_dataloader
    if hasattr(dm, "eval_dataloader"):
        return dm.eval_dataloader
    raise RuntimeError("Could not find an eval dataloader on pipeline.datamanager.")


# ========================================================================
# Entry point
# ========================================================================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--vmax", type=float, default=0.25,
                    help="Fixed error-colormap ceiling so maps compare across models.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to the run's analysis dir under assets/results/.")
    args = ap.parse_args()

    out_dir = args.out or run_artifact_dir(args.load_config, "analysis") / DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    _, pipeline, _, _ = pearl_eval_setup(
        args.load_config, eval_num_rays_per_chunk=None, test_mode="test"
    )
    model = pipeline.model
    model.eval()

    index = []
    for i, (camera, batch) in enumerate(get_eval_dataloader(pipeline)):
        if camera.shape[0] != 1:
            camera = camera[0:1]
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        pred = torch.clamp(outputs["pred_image"], 0.0, 1.0)
        err = (pred - gt).abs().mean(dim=-1)

        stem = f"{i:03d}"
        plt.imsave(out_dir / f"{stem}_gt.png", gt.cpu().numpy())
        plt.imsave(out_dir / f"{stem}_pred.png", pred.cpu().numpy())
        plt.imsave(out_dir / f"{stem}_err.png", err.cpu().numpy(),
                   cmap="magma", vmin=0.0, vmax=args.vmax)
        index.append({"image": i, "mean_abs_err": float(err.mean().item())})
        aiLogger.info(f"{stem}: mean |err| {index[-1]['mean_abs_err']:.4f}")

    (out_dir / "index.json").write_text(json.dumps(
        {"load_config": str(args.load_config), "vmax": args.vmax, "images": index}, indent=2
    ))
    aiLogger.info(f"Saved {len(index)} views to {out_dir}")


if __name__ == "__main__":
    main()


# python scripts/error_map_dump.py \
#   --load-config assets/results/<scene>/<timestamp>/best/config.yml
# -> assets/results/<scene>/<timestamp>/analysis/error_maps/NNN_{gt,pred,err}.png
# Use the same --vmax for every model when building comparison figures.
