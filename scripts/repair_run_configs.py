"""One-shot KI-001 repair: absolutise output_dir and data in stored run configs.

Runs saved before 2026-07 wrote `output_dir: outputs` and `data: data/<scene>`,
which resolved only through the root symlinks removed in 2026-07. nerfstudio
rebuilds the checkpoint dir from output_dir on every load, so those runs are
loadable from the repo root only after this rewrite. Idempotent.
"""

import argparse
from pathlib import Path

from pearl.ai_logger import aiLogger
from pearl.run_paths import OUTPUTS_ROOT, absolutize_config_text

ap = argparse.ArgumentParser()
ap.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
ap.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing.")
args = ap.parse_args()

configs = sorted(args.outputs_root.glob("*/*/*/config.yml"))
assert configs, f"No run configs found under {args.outputs_root}"

changed = 0
for config_path in configs:
    text = config_path.read_text()
    repaired = absolutize_config_text(text)
    if repaired == text:
        continue
    changed += 1
    if args.dry_run:
        aiLogger.info(f"would repair {config_path}")
    else:
        config_path.write_text(repaired)
        aiLogger.info(f"repaired {config_path}")

aiLogger.info(f"{changed}/{len(configs)} run configs {'need repair' if args.dry_run else 'repaired'}")
