import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from statistics import fmean
from math import inf

from pearl.ai_logger import aiLogger
from pearl.run_paths import run_artifact_dir

STEP_RE = re.compile(r"step-(\d+)\.ckpt$")

def posixpath_block(p: Path, indent: str = "") -> str:
    parts = p.resolve().parts
    lines = [f"{indent}load_dir: !!python/object/apply:pathlib.PosixPath"]
    for part in parts:
        lines.append(f"{indent}- {part}")
    return "\n".join(lines) + "\n"

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

def replace_block_load_dir(text: str, models_dir: Path) -> str:
    lines = text.splitlines(True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)load_dir:\s*(.*)\s*$", line)
        if not m:
            out.append(line)
            i += 1
            continue
        indent = m.group(1)
        i += 1
        while i < len(lines) and re.match(rf"^{re.escape(indent)}-\s+", lines[i]):
            i += 1
        out.append(posixpath_block(models_dir, indent=indent))
        return "".join(out) + "".join(lines[i:])
    head = lines[0] if lines else ""
    rest = "".join(lines[1:]) if len(lines) > 1 else ""
    return head + posixpath_block(models_dir, indent="") + rest

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--metric", choices=["psnr", "ssim", "psnr+ssim"], default="psnr")
    ap.add_argument("--device", default=None)
    ap.add_argument("--debug-first", action="store_true",
                    help="Print top-level keys of the first metrics.json for inspection.")
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg_path = run_dir / "config.yml"
    models_dir = run_dir / "nerfstudio_models"
    assert cfg_path.exists(), f"Missing {cfg_path}"
    assert models_dir.exists(), f"Missing {models_dir}"

    ckpts = sorted(
        [p for p in models_dir.glob("step-*.ckpt") if STEP_RE.search(p.name)],
        key=lambda p: int(STEP_RE.search(p.name).group(1)),
    )
    assert ckpts, f"No step-*.ckpt found in {models_dir}"

    eval_root = run_artifact_dir(cfg_path, "ckpt_sweep")

    base_text = cfg_path.read_text()

    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    results = []
    for idx, ckpt in enumerate(ckpts):
        step = int(STEP_RE.search(ckpt.name).group(1))
        step_dir = eval_root / f"step_{step:09d}"
        step_dir.mkdir(exist_ok=True)

        text = base_text
        text = replace_scalar_line(text, "load_checkpoint", "null")
        text = replace_scalar_line(text, "load_step", str(step))
        text = replace_block_load_dir(text, models_dir)

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

    def score(r):
        # Missing metrics get -inf so they never win; avoids NoneType crash.
        p = r["psnr"] if isinstance(r["psnr"], (int, float)) else -inf
        s = r["ssim"] if isinstance(r["ssim"], (int, float)) else -inf
        if args.metric == "psnr":
            return p
        if args.metric == "ssim":
            return s
        return p + 100.0 * s

    # If *all* scores are -inf, metrics were not found at all.
    best = max(results, key=score)
    if score(best) == -inf:
        aiLogger.error("Could not find metrics in any metrics.json. Use --debug-first to inspect keys.")
        (eval_root / "all_results.json").write_text(json.dumps(results, indent=2))
        return

    (eval_root / "all_results.json").write_text(json.dumps(results, indent=2))
    (eval_root / f"BEST_{args.metric}.json").write_text(json.dumps(best, indent=2))
    aiLogger.info(f"BEST: {best}")

if __name__ == "__main__":
    main()