"""Efficiency + metrics report for a WaterSplatting baseline run.

Standalone mirror of scripts/efficiency_report.py for the baseline `ws`
conda env, where the pearl package is not installed (ADR-013): the KI-007
allowlist (pearl/ns_cli.py), the measurement helpers
(scripts/efficiency_report.py) and the metric walker
(scripts/eval_all_ckpts.py) are duplicated here, and stdlib logging
replaces aiLogger — the sanctioned deviations from the house rules.
"""


import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

# KI-007: allowlist the numpy globals torch 2.6 refuses (pearl/ns_cli.py)
torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])

from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ws_report")


# ========================================================================
# Metric extraction (copied from scripts/eval_all_ckpts.py)
# ========================================================================
def _numeric_list(x):
    return isinstance(x, list) and x and all(isinstance(v, (int, float)) for v in x)


def _collect_numbers(obj):
    """Collect all numeric values inside obj (dict/list), ignoring booleans."""
    nums = []
    if isinstance(obj, dict):
        for v in obj.values():
            nums.extend(_collect_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            nums.extend(_collect_numbers(v))
    else:
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            nums.append(float(obj))
    return nums


def extract_metric(data, name: str):
    """Robustly find a scalar metric from arbitrary nerfstudio metrics.json layouts."""
    target = name.lower()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if target == kl or target in kl:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return float(v)
                    if _numeric_list(v):
                        return fmean(v)
                    if isinstance(v, (dict, list)):
                        nums = _collect_numbers(v)
                        if nums:
                            return fmean(nums)
                got = walk(v)
                if got is not None:
                    return got
        elif isinstance(obj, list):
            for v in obj:
                got = walk(v)
                if got is not None:
                    return got
        return None

    return walk(data)


# ========================================================================
# Measurements (copied from scripts/efficiency_report.py)
# ========================================================================
def gpu_info() -> dict:
    """Read GPU name and current power limit from nvidia-smi."""
    # Must never kill the report: a broken nvidia-smi (Exec format error on the
    # 2026-08-14 wave) is cosmetic next to the measurements already taken.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        name, power = [s.strip() for s in out.split(",")]
        return {"gpu": name, "power_limit": power}
    except Exception as exc:  # noqa: BLE001 - any nvidia-smi failure is non-fatal
        return {"gpu": None, "power_limit": None, "gpu_info_error": repr(exc)}


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


def memlog_stats(csv_path: Path | None) -> dict:
    """Peak training memory + wall-time estimate from the nvidia-smi 10 s poll."""
    # WaterSplatting keeps only the latest checkpoint, so the ckpt-mtime
    # estimate efficiency_report.py uses is unavailable — the poll stands in.
    if csv_path is None or not Path(csv_path).exists():
        return {}
    rows = [int(line) for line in Path(csv_path).read_text().split() if line.strip().isdigit()]
    if not rows:
        return {}
    return {
        "train_peak_mem_gb": max(rows) / 1024,
        "train_time_hours_est": len(rows) * 10 / 3600.0,
    }


# ========================================================================
# Entry point
# ========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, default=None,
                    help="ns-eval metrics.json to fold into the report.")
    ap.add_argument("--memlog", type=Path, default=None,
                    help="nvidia-smi memory.used poll captured during training.")
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-warmup", type=int, default=3)
    args = ap.parse_args()

    config, pipeline, ckpt_path, step = eval_setup(
        args.load_config, eval_num_rays_per_chunk=None, test_mode="test"
    )

    metrics_data = {}
    if args.metrics is not None and args.metrics.exists():
        metrics_data = json.loads(args.metrics.read_text())

    payload = {
        "method": "water-splatting",
        "scene": args.scene,
        "budget": "uncapped",
        "iterations": int(config.max_num_iterations),
        "split": "colmap interval-8 (nerfstudio 1.1.4 default, same as PEARL)",
        "load_config": str(args.load_config),
        "checkpoint": str(ckpt_path),
        "step": int(step),
        "gaussian_count": int(pipeline.model.num_points),
        "checkpoint_size_mb": Path(ckpt_path).stat().st_size / 1024**2,
        "psnr": extract_metric(metrics_data, "psnr"),
        "ssim": extract_metric(metrics_data, "ssim"),
        "lpips": extract_metric(metrics_data, "lpips"),
        "fps_ns_eval": extract_metric(metrics_data, "fps"),
        **memlog_stats(args.memlog),
        **gpu_info(),
        **measure_fps(pipeline, args.n_warmup),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    log.info(payload)
    log.info(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


# python scripts/ws_report.py \
#   --load-config assets/outputs/base_ws_<scene>/water-splatting/<ts>/config.yml \
#   --metrics assets/results/baselines/water-splatting/<scene>/metrics.json \
#   --memlog assets/results/baselines/memlogs/base_ws_<scene>.mem.csv \
#   --scene <scene> --out assets/results/baselines/water-splatting/<scene>/efficiency.json
# NOTE: fps/memory are measured on the current GPU at its current power limit
# (recorded in the report); tables keep the published 275 W FPS policy and use
# N_final / model size / peak memory from here (CURRENT_TASK 2026-08-10).
