#!/usr/bin/env bash
#SBATCH --job-name=pearl_base_ws
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                 # EDIT: pin a GPU type if supported, e.g. gpu:a100:1
#SBATCH --time=24:00:00              # 4 x (uncapped 15k-iter train + eval + report) + margin
#SBATCH --output=assets/outputs/slurm/%x-%j.out
#SBATCH --error=assets/outputs/slurm/%x-%j.err
#
# WaterSplatting baseline batch (ADR-013): the 4 SeaThru-NeRF scenes under
# the method's own published defaults — 15k iterations, UNCAPPED
# densification — on the same undistorted data and the same nerfstudio
# colmap interval-8 split as PEARL. Per scene: training run, ns-eval
# metrics.json (+ renders), and efficiency.json with N_final / model size /
# fps / peak inference memory (R2-#6). An OOM on an uncapped scene is
# acceptable, reportable evidence — the batch continues past it.
#
#     sbatch scripts/baseline_ws_job.sbatch        # from the repo root
#
# Prereq: bash scripts/setup_baseline_envs.sh ws   (env + upstream clone)
set -uo pipefail

# --- cluster modules + conda env -----------------------------------
# The cluster has no cuda modulefiles and ships CUDA 12.2 at
# /usr/local/cuda; nerfstudio's gsplat compiles kernels against $CUDA_HOME
# on first use, so point it at a toolkit (ADR-014).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"   # EDIT for another cluster
export PATH="$CUDA_HOME/bin:$PATH"
ROOT_DIR="${PEARL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

CONDA_PROFILE="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
# `wsplat` is a clone of the production `pearl` env with the pearl plugin
# removed and water-splatting installed — same nerfstudio/torch/gsplat as
# PEARL, which is both the fair basis for the efficiency table and the only
# stack that imports at all on python 3.12 (ADR-015, KI-010).
CONDA_ENV="${WS_CONDA_ENV:-wsplat}"
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

# --- batch bookkeeping (baseline layout, ADR-013) ------------------
BASE=assets/results/baselines
RESULTS=$BASE/water-splatting
LOGS=$BASE/logs
MEMLOGS=$BASE/memlogs
mkdir -p "$RESULTS" "$LOGS" "$MEMLOGS"
BATCHLOG=$BASE/baseline_runs.log
FINAL_CKPT=step-000014999.ckpt   # 15k iters; save_only_latest keeps exactly one
JOB="ws[${SLURM_JOB_ID:-local}]"

gpu_ok() { nvidia-smi -L >/dev/null 2>&1; }

fail() {
    # fail <marker> <runid> <log> — a failure line is useless without its
    # cause: the 2026-08-14 wave read as four OOMs and was an import error.
    echo "$(date -Is) $JOB $1 $2" | tee -a "$BATCHLOG"
    [ -f "$3" ] && tail -3 "$3" | sed 's/^/    /' | tee -a "$BATCHLOG"
}

echo "Start at:  $(date)"
echo "Job:       ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Env:       $CONDA_ENV"
nvidia-smi -L || true

for SCENE in Curasao IUI3-RedSea JapaneseGradens-RedSea Panama; do
  RUNID=base_ws_$SCENE

  if ! gpu_ok; then
    echo "$(date -Is) $JOB GPU-DEAD — aborting batch before $RUNID" | tee -a "$BATCHLOG"
    break
  fi

  TS=$(ls "assets/outputs/$RUNID/water-splatting" 2>/dev/null | sort | tail -1)
  if [ -n "$TS" ] && [ ! -f "assets/outputs/$RUNID/water-splatting/$TS/nerfstudio_models/$FINAL_CKPT" ]; then
    echo "$(date -Is) $JOB SKIP $RUNID (incomplete training at $TS — resolve manually)" | tee -a "$BATCHLOG"
    continue
  fi

  if [ -z "$TS" ]; then
    echo "$(date -Is) $JOB TRAIN $RUNID" | tee -a "$BATCHLOG"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=10 \
      >"$MEMLOGS/$RUNID.mem.csv" &
    MEMPID=$!
    if ns-train water-splatting --vis viewer+tensorboard --viewer.quit-on-train-completion True \
      --output-dir assets/outputs --experiment-name "$RUNID" \
      --data "assets/data/undistorted_seathrunerf_dataset/$SCENE" \
      colmap --downscale-factor 1 --colmap-path sparse --images-path images \
      >"$LOGS/$RUNID.train.log" 2>&1; then
      kill $MEMPID 2>/dev/null
      TS=$(ls "assets/outputs/$RUNID/water-splatting" | sort | tail -1)
    else
      kill $MEMPID 2>/dev/null
      fail TRAIN-FAIL "$RUNID" "$LOGS/$RUNID.train.log"
      continue
    fi
  else
    echo "$(date -Is) $JOB REUSE $RUNID training ($TS)" | tee -a "$BATCHLOG"
  fi

  CFG="assets/outputs/$RUNID/water-splatting/$TS/config.yml"
  mkdir -p "$RESULTS/$SCENE"

  if [ -f "$RESULTS/$SCENE/metrics.json" ]; then
    echo "$(date -Is) $JOB SKIP-EVAL $RUNID (metrics.json exists)" | tee -a "$BATCHLOG"
  else
    if ! gpu_ok; then
      echo "$(date -Is) $JOB GPU-DEAD — aborting batch before eval of $RUNID" | tee -a "$BATCHLOG"
      break
    fi
    echo "$(date -Is) $JOB EVAL $RUNID ($TS)" | tee -a "$BATCHLOG"
    if ! python scripts/ns_eval_safe.py --load-config "$CFG" \
      --output-path "$RESULTS/$SCENE/metrics.json" \
      --render-output-path "$RESULTS/$SCENE/renders" \
      >"$LOGS/$RUNID.eval.log" 2>&1; then
      fail EVAL-FAIL "$RUNID" "$LOGS/$RUNID.eval.log"
      continue
    fi
  fi

  if [ -f "$RESULTS/$SCENE/efficiency.json" ]; then
    echo "$(date -Is) $JOB DONE $RUNID (report exists)" | tee -a "$BATCHLOG"
    continue
  fi
  echo "$(date -Is) $JOB REPORT $RUNID" | tee -a "$BATCHLOG"
  if python scripts/ws_report.py --load-config "$CFG" \
    --metrics "$RESULTS/$SCENE/metrics.json" \
    --memlog "$MEMLOGS/$RUNID.mem.csv" \
    --scene "$SCENE" \
    --out "$RESULTS/$SCENE/efficiency.json" \
    >"$LOGS/$RUNID.report.log" 2>&1; then
    echo "$(date -Is) $JOB DONE $RUNID" | tee -a "$BATCHLOG"
  else
    fail REPORT-FAIL "$RUNID" "$LOGS/$RUNID.report.log"
  fi
done
echo "$(date -Is) $JOB BATCH EXITED" | tee -a "$BATCHLOG"
echo "Finished at: $(date)"
