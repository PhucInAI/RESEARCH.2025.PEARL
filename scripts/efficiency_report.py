import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from pearl.ai_logger import aiLogger
from pearl.ns_cli import pearl_eval_setup
from pearl.run_paths import run_artifact_dir

REPORT_NAME = "efficiency.json"


# ========================================================================
# Measurements
# ========================================================================
def gpu_info() -> dict:
    """Read GPU name and current power limit from nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    name, power = [s.strip() for s in out.split(",")]
    return {"gpu": name, "power_limit": power}


def get_eval_dataloader(pipeline):
    dm = pipeline.datamanager
    if hasattr(dm, "fixed_indices_eval_dataloader"):
        return dm.fixed_indices_eval_dataloader
    if hasattr(dm, "eval_dataloader"):
        return dm.eval_dataloader
    raise RuntimeError("Could not find an eval dataloader on pipeline.datamanager.")


@torch.no_grad()
def measure_fps(pipeline, n_warmup: int) -> dict:
    """Time full-frame renders over all eval views at native resolution."""
    model = pipeline.model
    model.eval()
    dl = get_eval_dataloader(pipeline)

    cameras = []
    for camera, _ in dl:
        if camera.shape[0] != 1:
            camera = camera[0:1]
        cameras.append(camera.to(model.device))

    for camera in cameras[:n_warmup]:
        model.get_outputs_for_camera(camera)

    # Few eval views per scene (interval-8 split): repeat passes so the timed
    # sample holds at least 20 renders.
    n_passes = max(1, -(-20 // len(cameras)))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_passes):
        for camera in cameras:
            model.get_outputs_for_camera(camera)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_renders = n_passes * len(cameras)
    peak_bytes = torch.cuda.max_memory_allocated()
    return {
        "n_images": len(cameras),
        "n_timed_renders": n_renders,
        "resolution": [int(cameras[0].width.item()), int(cameras[0].height.item())],
        "render_seconds_total": elapsed,
        "fps_measured": n_renders / elapsed,
        "peak_inference_mem_gb": peak_bytes / 1024**3,
    }


def train_time_hours(ckpt_path: Path) -> float | None:
    """Estimate training wall-time from step checkpoint mtimes, if available."""
    steps = sorted(ckpt_path.parent.glob("step-*.ckpt"))
    if len(steps) < 2:
        return None
    return (steps[-1].stat().st_mtime - steps[0].stat().st_mtime) / 3600.0


# ========================================================================
# Entry point
# ========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to the run's analysis dir under assets/results/.")
    args = ap.parse_args()

    out_path = args.out or run_artifact_dir(args.load_config, "analysis") / REPORT_NAME

    config, pipeline, ckpt_path, step = pearl_eval_setup(
        args.load_config, eval_num_rays_per_chunk=None, test_mode="test"
    )

    payload = {
        "load_config": str(args.load_config),
        "checkpoint": str(ckpt_path),
        "step": int(step),
        "gaussian_count": int(pipeline.model.num_points),
        "checkpoint_size_mb": ckpt_path.stat().st_size / 1024**2,
        "train_time_hours_est": train_time_hours(Path(ckpt_path)),
        **gpu_info(),
        **measure_fps(pipeline, args.n_warmup),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    aiLogger.info(payload)
    aiLogger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


# python scripts/efficiency_report.py \
#   --load-config assets/results/<scene>/<timestamp>/best/config.yml
# -> assets/results/<scene>/<timestamp>/analysis/efficiency.json
# NOTE: fps/memory are measured on the current GPU at its current power limit
# (recorded in the report); training time is estimated from ckpt mtimes and is
# only available when pointing at a run config with step-*.ckpt files.
