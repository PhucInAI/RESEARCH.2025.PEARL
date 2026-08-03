import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from statistics import fmean
from math import inf

from pearl.ai_logger import aiLogger
from pearl.run_paths import absolutize_config_text, run_artifact_dir

STEP_RE = re.compile(r"step-(\d+)\.ckpt$")
CKPT_STEP_RE = re.compile(r"step-\d+\.ckpt")

def replace_scalar_line(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"(?m)^(?P<indent>\s*){re.escape(key)}:\s*.*$")
    m = pat.search(text)
    if m:
        indent = m.group("indent")
        return pat.sub(f"{indent}{key}: {value}", text, count=1)
    lines = text.splitlines(True)
    if not lines:
        return f"{key}: {value}\n"
    lines.insert(1, f"{key}: {value}\n")
    return "".join(lines)

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
    """
    Robustly find a scalar metric from arbitrary nerfstudio metrics.json layouts.
    Looks for keys containing 'psnr' / 'ssim' / 'lpips' (case-insensitive).
    If value is list/dict, averages numeric leaves.
    """
    target = name.lower()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                # match exact or substring keys like "test/psnr", "eval_psnr", etc.
                if target == kl or target in kl:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return float(v)
                    if _numeric_list(v):
                        return fmean(v)
                    if isinstance(v, (dict, list)):
                        nums = _collect_numbers(v)
                        if nums:
                            return fmean(nums)
                # recurse
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

def scrub_step_text(text: str) -> str:
    """Null the checkpoint-resume keys so a copied config names no step."""
    # Unlike replace_scalar_line, never insert a key that the config lacks.
    for key in ("load_step", "load_checkpoint"):
        pat = re.compile(rf"(?m)^(?P<indent>\s*){re.escape(key)}:\s*.*$")
        text = pat.sub(lambda m: f"{m.group('indent')}{key}: null", text)
    return text

def scrub_step_json(obj):
    """Strip every step reference from a metrics report, keeping all values."""
    if isinstance(obj, dict):
        return {k: scrub_step_json(v) for k, v in obj.items() if str(k).lower() != "step"}
    if isinstance(obj, list):
        return [scrub_step_json(v) for v in obj]
    if isinstance(obj, str) and CKPT_STEP_RE.search(obj):
        return "best.ckpt"
    return obj

def export_best(run_dir: Path, cfg_path: Path, eval_root: Path, best: dict) -> Path:
    """Copy the winning checkpoint, its config and its metrics into best/, step-free."""
    best_dir = run_artifact_dir(cfg_path, "best")

    # Byte copy by choice: the .ckpt still records its step internally (top-level
    # "step", Adam step counters, scheduler last_epoch). Scrubbing those would mean
    # re-saving, so best.ckpt stays identical to the source and only the surrounding
    # filenames and text artifacts are step-free.
    src_ckpt = run_dir / "nerfstudio_models" / best["ckpt"]
    assert src_ckpt.exists(), f"Missing {src_ckpt}"
    shutil.copy2(src_ckpt, best_dir / "best.ckpt")

    # Absolutised so best/config.yml loads from any cwd; pearl-viewer / pearl-render /
    # pearl-eval then find best.ckpt as its sibling (ADR-009).
    config_text = absolutize_config_text(scrub_step_text(cfg_path.read_text()))
    (best_dir / "config.yml").write_text(config_text)

    src_metrics = eval_root / f"step_{best['step']:09d}" / "metrics.json"
    assert src_metrics.exists(), f"Missing {src_metrics}"
    metrics = scrub_step_json(json.loads(src_metrics.read_text()))
    (best_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    aiLogger.info(f"best checkpoint exported to {best_dir}")
    return best_dir

def run_sweep(args, cfg_path: Path, models_dir: Path, eval_root: Path) -> list:
    """Run ns-eval over every step-*.ckpt of the run and collect its metrics."""
    ckpts = sorted(
        [p for p in models_dir.glob("step-*.ckpt") if STEP_RE.search(p.name)],
        key=lambda p: int(STEP_RE.search(p.name).group(1)),
    )
    assert ckpts, f"No step-*.ckpt found in {models_dir}"

    # ns-eval rebuilds load_dir from output_dir on every load, so absolutising the
    # config is what points the sweep at models_dir — writing load_dir never did.
    base_text = absolutize_config_text(cfg_path.read_text())

    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    results = []
    for idx, ckpt in enumerate(ckpts):
        step = int(STEP_RE.search(ckpt.name).group(1))
        step_dir = eval_root / f"step_{step:09d}"
        step_dir.mkdir(exist_ok=True)

        text = replace_scalar_line(base_text, "load_checkpoint", "null")
        text = replace_scalar_line(text, "load_step", str(step))

        tmp_cfg = step_dir / "config_eval.yml"
        tmp_cfg.write_text(text)

        out_json = step_dir / "metrics.json"
        cmd = ["ns-eval", "--load-config", str(tmp_cfg), "--output-path", str(out_json)]
        aiLogger.info(" ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

        data = json.loads(out_json.read_text())
        if args.debug_first and idx == 0:
            aiLogger.debug(f"top-level keys: {list(data.keys())[:50]}")

        psnr = extract_metric(data, "psnr")
        ssim = extract_metric(data, "ssim")
        lpips = extract_metric(data, "lpips")

        results.append({"step": step, "ckpt": ckpt.name, "psnr": psnr, "ssim": ssim, "lpips": lpips})

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--metric", choices=["psnr", "ssim", "psnr+ssim"], default="psnr")
    ap.add_argument("--device", default=None)
    ap.add_argument("--debug-first", action="store_true",
                    help="Print top-level keys of the first metrics.json for inspection.")
    ap.add_argument("--copy-best-only", action="store_true",
                    help="Skip ns-eval; rebuild best/ from an existing ckpt_sweep/all_results.json.")
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg_path = run_dir / "config.yml"
    models_dir = run_dir / "nerfstudio_models"
    assert cfg_path.exists(), f"Missing {cfg_path}"
    assert models_dir.exists(), f"Missing {models_dir}"

    eval_root = run_artifact_dir(cfg_path, "ckpt_sweep")

    def score(r):
        # Missing metrics get -inf so they never win; avoids NoneType crash.
        p = r["psnr"] if isinstance(r["psnr"], (int, float)) else -inf
        s = r["ssim"] if isinstance(r["ssim"], (int, float)) else -inf
        if args.metric == "psnr":
            return p
        if args.metric == "ssim":
            return s
        return p + 100.0 * s

    if args.metric != "psnr":
        aiLogger.warning(
            f"--metric {args.metric}: best/ will not follow the published protocol, "
            "which reports the PSNR-maximising checkpoint (paper Table 3)."
        )

    if args.copy_best_only:
        all_results = eval_root / "all_results.json"
        assert all_results.exists(), f"Missing {all_results}; run the sweep first."
        results = json.loads(all_results.read_text())
    else:
        results = run_sweep(args, cfg_path, models_dir, eval_root)
        (eval_root / "all_results.json").write_text(json.dumps(results, indent=2))

    # If *all* scores are -inf, metrics were not found at all.
    best = max(results, key=score)
    if score(best) == -inf:
        aiLogger.error("Could not find metrics in any metrics.json. Use --debug-first to inspect keys.")
        return

    if not args.copy_best_only:
        (eval_root / f"BEST_{args.metric}.json").write_text(json.dumps(best, indent=2))
    aiLogger.info(f"BEST: {best}")
    export_best(run_dir, cfg_path, eval_root, best)

if __name__ == "__main__":
    main()