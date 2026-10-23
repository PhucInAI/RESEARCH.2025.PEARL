#!/usr/bin/env bash
#SBATCH --job-name=pearl_rev_job2
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                 # EDIT: pin a GPU type if supported, e.g. gpu:a100:1
#SBATCH --time=60:00:00              # 10 runs x (~1.5 h train + ~2.5 h sweep) + margin
#SBATCH --output=assets/outputs/slurm/%x-%j.out
#SBATCH --error=assets/outputs/slurm/%x-%j.err
#
# PEARL revision batch 2/2 (revision_manifest.md): 10 of the Phase-3 component
# ablations under the published protocol (R2-#5/#3/#1) — the three 3-scene
# triads (no-error-densification, no-edge-loss, no-attn-regularization) plus
# the depth-attenuation render ablation. The 2 rev_nobudget_* runs live in
# scripts/revision_job1.sbatch.
#
#     sbatch scripts/revision_job2.sbatch          # from the repo root
#
# May run concurrently with job 1. Idempotent: completed trainings
# (step-000019999.ckpt) are reused, sweeps skip evaluated checkpoints, so a
# requeue/relaunch never redoes or overwrites finished work. Disk: ~64G per
# run transient (checkpoint sweep); budget ~640G free before launching.
set -uo pipefail

# --- cluster modules + conda env -----------------------------------
# module load cuda/11.8 gcc/11       # EDIT to your cluster's module names
ROOT_DIR="${PEARL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

CONDA_PROFILE="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
CONDA_ENV="${PEARL_CONDA_ENV:-pearl}"
if [[ ! -f "$CONDA_PROFILE" ]]; then
    echo "Conda profile not found: $CONDA_PROFILE — set CONDA_SH." >&2
    exit 1
fi
# nerfstudio's conda hooks read $ZSH_VERSION unguarded — suspend nounset
set +u
source "$CONDA_PROFILE"
conda activate "$CONDA_ENV"
set -u

mkdir -p assets/outputs/slurm

# --- batch bookkeeping (same layout as the local runners) ----------
PAPER=assets/results/paper
LOGS=$PAPER/logs
MEMLOGS=$PAPER/memlogs
mkdir -p "$LOGS" "$MEMLOGS"
BATCHLOG=$PAPER/revision_runs.log
FINAL_CKPT=step-000019999.ckpt
JOB="job2[${SLURM_JOB_ID:-local}]"

gpu_ok() { nvidia-smi -L >/dev/null 2>&1; }

echo "Start at:  $(date)"
echo "Job:       ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Env:       $CONDA_ENV"
nvidia-smi -L || true

# --- the ten runs: RUNID|scene|variant flags -----------------------
RUNS=(
  "rev_noerr_Curasao|Curasao|--pipeline.model.enable-error-densification False"
  "rev_noerr_IUI3-RedSea|IUI3-RedSea|--pipeline.model.enable-error-densification False"
  "rev_noerr_Panama|Panama|--pipeline.model.enable-error-densification False"
  "rev_noedge_Curasao|Curasao|--pipeline.model.enable-edge-loss False"
  "rev_noedge_IUI3-RedSea|IUI3-RedSea|--pipeline.model.enable-edge-loss False"
  "rev_noedge_Panama|Panama|--pipeline.model.enable-edge-loss False"
  "rev_noattnreg_Curasao|Curasao|--pipeline.model.enable-attn-regularization False"
  "rev_noattnreg_IUI3-RedSea|IUI3-RedSea|--pipeline.model.enable-attn-regularization False"
  "rev_noattnreg_Panama|Panama|--pipeline.model.enable-attn-regularization False"
  "rev_nodepthattn_Curasao|Curasao|--pipeline.model.enable-depth-attenuation False"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r RUNID SCENE FLAGS <<<"$spec"

  if ! gpu_ok; then
    echo "$(date -Is) $JOB GPU-DEAD — aborting batch before $RUNID" | tee -a "$BATCHLOG"
    break
  fi

  TS=$(ls "assets/outputs/$RUNID/pearl" 2>/dev/null | sort | tail -1)
  if [ -n "$TS" ] && [ ! -f "assets/outputs/$RUNID/pearl/$TS/nerfstudio_models/$FINAL_CKPT" ]; then
    echo "$(date -Is) $JOB SKIP $RUNID (incomplete training at $TS — resolve manually)" | tee -a "$BATCHLOG"
    continue
  fi

  if [ -z "$TS" ]; then
    echo "$(date -Is) $JOB TRAIN $RUNID" | tee -a "$BATCHLOG"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=10 \
      >"$MEMLOGS/$RUNID.mem.csv" &
    MEMPID=$!
    if ns-train pearl --vis viewer+tensorboard --viewer.quit-on-train-completion True \
      --steps-per-save 200 --output-dir assets/outputs --experiment-name "$RUNID" \
      $FLAGS \
      --data "assets/data/undistorted_seathrunerf_dataset/$SCENE" \
      colmap --downscale-factor 1 --colmap-path sparse --images-path images \
      >"$LOGS/$RUNID.train.log" 2>&1; then
      kill $MEMPID 2>/dev/null
      TS=$(ls "assets/outputs/$RUNID/pearl" | sort | tail -1)
    else
      kill $MEMPID 2>/dev/null
      echo "$(date -Is) $JOB TRAIN-FAIL $RUNID" | tee -a "$BATCHLOG"
      continue
    fi
  else
    echo "$(date -Is) $JOB REUSE $RUNID training ($TS)" | tee -a "$BATCHLOG"
  fi

  if [ -f "assets/results/$SCENE/$TS/ckpt_sweep/all_results.json" ]; then
    echo "$(date -Is) $JOB DONE $RUNID (sweep already complete)" | tee -a "$BATCHLOG"
    continue
  fi
  if ! gpu_ok; then
    echo "$(date -Is) $JOB GPU-DEAD — aborting batch before sweep of $RUNID" | tee -a "$BATCHLOG"
    break
  fi
  echo "$(date -Is) $JOB SWEEP $RUNID ($TS)" | tee -a "$BATCHLOG"
  if python scripts/eval_all_ckpts.py --run-dir "assets/outputs/$RUNID/pearl/$TS" \
    >"$LOGS/$RUNID.sweep.log" 2>&1; then
    echo "$(date -Is) $JOB DONE $RUNID" | tee -a "$BATCHLOG"
  else
    echo "$(date -Is) $JOB SWEEP-FAIL $RUNID" | tee -a "$BATCHLOG"
  fi
done
echo "$(date -Is) $JOB BATCH EXITED" | tee -a "$BATCHLOG"
echo "Finished at: $(date)"
