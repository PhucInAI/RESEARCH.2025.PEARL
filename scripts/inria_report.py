"""Report N_final / model size / metrics for an INRIA-family baseline run.

Serves SeaSplat (--method seasplat) and 3D-UIR (--method 3duir), running
inside each repo's own conda env where the pearl package is not installed
(ADR-013): stdlib logging plus a local copy of eval_all_ckpts.py's metric
walker are the sanctioned deviations from the house rules.

Metrics source, in order of preference:
  3duir     the repo's own metrics.py output   (<model-dir>/results.json)
  seasplat  the repo's in-train eval           (<model-dir>/eval_metrics.json),
            else computed here from test/with_water renders vs the every-8th
            holdout of the staged view's images (the split train.py --eval
            used), with the repo's psnr/ssim and LPIPS(alex) when importable.
"""


import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import fmean

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("inria_report")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".JPG", ".PNG"}


# ========================================================================
# PLY header stats (stdlib only — never reads the body)
# ========================================================================
def ply_stats(ply: Path) -> dict:
    """Read gaussian count from the ply header and model size from st_size."""
    count = None
    with ply.open("rb") as f:
        while True:
            line = f.readline()
            if not line or line.strip() == b"end_header":
                break
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
    if count is None:
        raise SystemExit(f"No 'element vertex' line in {ply}")
    return {"gaussian_count": count, "ply_size_mb": ply.stat().st_size / 1024**2}


def aux_model_files(method: str, model_dir: Path, iteration: int) -> dict:
    """Size the non-ply files an inference load needs (water/appearance nets)."""
    # chkpnt<iter>.pth is training state (optimizer included) — not counted.
    if method == "seasplat":
        candidates = [
            model_dir / f"attenuate_{iteration}.pth",
            model_dir / f"backscatter_{iteration}.pth",
            model_dir / f"bg_{iteration}.pth",
        ]
    else:
        candidates = [model_dir / "point_cloud" / f"iteration_{iteration}" / "model_weights.pth"]
    files = [p for p in candidates if p.exists()]
    return {
        "aux_files": [p.name for p in files],
        "aux_size_mb": sum(p.stat().st_size for p in files) / 1024**2,
    }


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
    """Robustly find a scalar metric in an arbitrary metrics-JSON layout."""
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


def test_section(data):
    """Narrow a metrics JSON to its test/holdout subtree when one exists."""
    # SeaSplat's eval_metrics.json holds train AND test sections; the greedy
    # walker used to return whichever came first (train — hence the inflated
    # 2026-08-14 numbers, e.g. Curasao 35.25 vs the true test 29.59).
    def find(obj, needle):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if needle in str(k).lower() and isinstance(v, (dict, list)):
                    return v
            for v in obj.values():
                got = find(v, needle)
                if got is not None:
                    return got
        elif isinstance(obj, list):
            for v in obj:
                got = find(v, needle)
                if got is not None:
                    return got
        return None

    for needle in ("test", "holdout"):
        sub = find(data, needle)
        if sub is not None:
            return sub, needle
    return data, "whole-file"


def fold_metrics_json(path: Path, source: str) -> dict:
    """Pull psnr/ssim/lpips out of a repo-written metrics JSON (test split only)."""
    data, section = test_section(json.loads(path.read_text()))
    return {
        "psnr": extract_metric(data, "psnr"),
        "ssim": extract_metric(data, "ssim"),
        # SeaSplat's metrics code spells the key "LIPPS" — tolerate both.
        "lpips": extract_metric(data, "lpips") or extract_metric(data, "lipps"),
        "metrics_source": f"{source} [{section} section]",
    }


# ========================================================================
# Memlog reduction
# ========================================================================
def memlog_stats(csv_path: Path | None, prefix: str, with_hours: bool = False) -> dict:
    """Peak memory (and optional wall-time) from an nvidia-smi 10 s poll."""
    if csv_path is None or not Path(csv_path).exists():
        return {}
    rows = [int(line) for line in Path(csv_path).read_text().split() if line.strip().isdigit()]
    if not rows:
        return {}
    stats = {f"{prefix}_peak_mem_gb": max(rows) / 1024}
    if with_hours:
        stats[f"{prefix}_time_hours_est"] = len(rows) * 10 / 3600.0
    return stats


# ========================================================================
# SeaSplat fallback: compute metrics from rendered files
# ========================================================================
def _load_lpips(device):
    """Best-effort LPIPS(alex): vendored lpipsPyTorch, then the pip package."""
    try:
        from lpipsPyTorch import lpips as vendored
        return "lpipsPyTorch(alex)", lambda a, b: vendored(a, b, net_type="alex").item()
    except Exception:
        pass
    try:
        import lpips as lpips_pkg
        net = lpips_pkg.LPIPS(net="alex").to(device)
        return "lpips-pip(alex)", lambda a, b: net(a, b, normalize=True).item()
    except Exception:
        return None, None


def compute_rendered_metrics(model_dir: Path, view: Path, repo: Path) -> dict:
    """Compare test/with_water renders to the every-8th image holdout."""
    try:
        import torch
        from PIL import Image
        from torchvision.transforms import functional as tf
        sys.path.insert(0, str(repo))
        from utils.image_utils import psnr as repo_psnr
        from utils.loss_utils import ssim as repo_ssim
    except Exception as exc:
        log.warning(f"metric imports failed ({exc}) — psnr/ssim/lpips left null")
        return {}

    renders_dir = model_dir / "test" / "with_water"
    renders = sorted(p for p in renders_dir.iterdir() if p.suffix in IMAGE_SUFFIXES)
    all_images = sorted(p for p in (view / "images").iterdir() if p.suffix in IMAGE_SUFFIXES)
    gts = all_images[::8]  # llffhold-8 test set: idx % 8 == 0 on name-sorted images
    if not renders or len(renders) != len(gts):
        log.warning(f"render/gt count mismatch ({len(renders)} vs {len(gts)}) — metrics left null")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_impl, lpips_fn = _load_lpips(device)

    psnrs, ssims, lpipss = [], [], []
    with torch.no_grad():
        for render_path, gt_path in zip(renders, gts):
            render = tf.to_tensor(Image.open(render_path))[:3].unsqueeze(0).to(device)
            gt = tf.to_tensor(Image.open(gt_path))[:3].unsqueeze(0).to(device)
            if render.shape != gt.shape:
                log.warning(f"shape mismatch {render_path.name} vs {gt_path.name} — metrics left null")
                return {}
            psnrs.append(repo_psnr(render, gt).mean().item())
            ssims.append(repo_ssim(render, gt).mean().item())
            if lpips_fn is not None:
                lpipss.append(lpips_fn(render, gt))

    return {
        "psnr": fmean(psnrs),
        "ssim": fmean(ssims),
        "lpips": fmean(lpipss) if lpipss else None,
        "lpips_impl": lpips_impl,
        "metrics_source": "computed (test/with_water vs every-8th images)",
    }


# ========================================================================
# Entry point
# ========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["seasplat", "3duir"], required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--view", type=Path, required=True,
                    help="Staged INRIA view the run trained on (images + sparse/0).")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--train-memlog", type=Path, default=None)
    ap.add_argument("--render-memlog", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ply = args.model_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not ply.exists():
        raise SystemExit(f"Missing {ply} — training incomplete?")

    stats = ply_stats(ply)
    aux = aux_model_files(args.method, args.model_dir, args.iteration)

    if args.method == "3duir":
        results = args.model_dir / "results.json"
        metrics = (fold_metrics_json(results, "3duir metrics.py (results.json)")
                   if results.exists() else {})
    else:
        results = args.model_dir / "eval_metrics.json"
        metrics = (fold_metrics_json(results, "seasplat in-train eval (eval_metrics.json)")
                   if results.exists() else {})
        if metrics.get("psnr") is None:
            metrics = compute_rendered_metrics(args.model_dir, args.view, args.repo) or metrics

    payload = {
        "method": args.method,
        "scene": args.scene,
        "budget": "uncapped (INRIA ADC densification, no primitive cap)",
        "iterations": args.iteration,
        "split": "llffhold-8 (--eval; test = idx % 8 == 0 on name-sorted images)",
        "model_dir": str(args.model_dir),
        "model_file": str(ply),
        **stats,
        **aux,
        "model_size_mb": stats["ply_size_mb"] + aux["aux_size_mb"],
        **metrics,
        **memlog_stats(args.train_memlog, "train", with_hours=True),
        **memlog_stats(args.render_memlog, "render"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    log.info(payload)
    log.info(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


# python scripts/inria_report.py --method seasplat \
#   --model-dir assets/outputs/baselines/seasplat/<scene> \
#   --view assets/data/inria_views/<scene> \
#   --repo ../seasplat --scene <scene> \
#   --train-memlog assets/results/baselines/memlogs/base_seasplat_<scene>.mem.csv \
#   --render-memlog assets/results/baselines/memlogs/base_seasplat_<scene>.render.mem.csv \
#   --out assets/results/baselines/seasplat/<scene>/efficiency.json
# NOTE: peak render memory comes from the nvidia-smi poll around the render
# step (process-level, labeled as such); FPS is deliberately not measured —
# tables keep the published 275 W FPS policy (CURRENT_TASK 2026-08-10).
