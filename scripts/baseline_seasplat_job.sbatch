#!/usr/bin/env bash
#SBATCH --job-name=pearl_base_seasplat
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                 # EDIT: pin a GPU type if supported, e.g. gpu:a100:1
#SBATCH --time=24:00:00              # 4 x (uncapped 30k-iter train + render + report) + margin
#SBATCH --output=assets/outputs/slurm/%x-%j.out
#SBATCH --error=assets/outputs/slurm/%x-%j.err
#
# SeaSplat baseline batch (ADR-013): the 4 SeaThru-NeRF scenes under the
# method's own published defaults — 30k iterations, UNCAPPED densification
# (densify_until_iter 15000), --do_seathru --seathru_from_iter 10000 — on
# staged views of the same undistorted data with the every-8th --eval
# holdout. Deliberate deltas from bare upstream defaults, both
# protocol-neutral (ADR-013):
#   -r 1                      forced native resolution — Curasao/Panama are
#                             1794 px wide and INRIA silently downscales
#                             anything >1600 px; PEARL trains at downscale 1
#   save/test/checkpoint      trimmed to iteration 30000 only (default writes
#                             multi-GB chkpnt/ply snapshots at 1k/7k/15k);
#                             checkpoint 30000 also writes the attenuate/
#                             backscatter .pth that render_uw.py --seathru needs
# train.py ignores -m and writes to <source>/experiments/<MMDDYYYY>/<exp>
# (train.py:1035), so every finished run is adopted into
# assets/outputs/baselines/seasplat/<scene> before anything downstream runs
# (KI-012) — that adoption also salvages runs stranded by earlier jobs.
# An OOM on an uncapped scene is acceptable, reportable evidence.
#
#     sbatch scripts/baseline_seasplat_job.sbatch  # from the repo root
#
# Prereq: bash scripts/setup_baseline_envs.sh seasplat
set -uo pipefail

# --- cluster modules + conda env -----------------------------------
# No toolkit needed at runtime: d-g-r and simple-knn are compiled once by
# setup_baseline_envs.sh (system CUDA 12.2 against the cu121 wheels, ADR-014).
ROOT_DIR="${PEARL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

CONDA_PROFILE="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
CONDA_ENV="${SEASPLAT_CONDA_ENV:-seasplat}"
REPO="${SEASPLAT_REPO:-$(dirname "$ROOT_DIR")/seasplat}"
if [[ ! -f "$CONDA_PROFILE" ]]; then
    echo "Conda profile not found: $CONDA_PROFILE — set CONDA_SH." >&2
    exit 1
fi
set +u
source "$CONDA_PROFILE"
conda activate "$CONDA_ENV"
set -u

mkdir -p assets/outputs/slurm

# --- batch bookkeeping (baseline layout, ADR-013) ------------------
BASE=assets/results/baselines
RESULTS=$BASE/seasplat
LOGS=$BASE/logs
MEMLOGS=$BASE/memlogs
mkdir -p "$RESULTS" "$LOGS" "$MEMLOGS"
BATCHLOG=$BASE/baseline_runs.log
ITER=30000
EXP=test          # train.py --exp default; also the leaf of its forced path
JOB="seasplat[${SLURM_JOB_ID:-local}]"

gpu_ok() { nvidia-smi -L >/dev/null 2>&1; }

fail() {
    # fail <marker> <runid> <log> — a failure line is useless without its
    # cause; the 2026-08-14 wave logged bare RENDER-FAIL/REPORT-FAIL for
    # what was a path bug (KI-012).
    echo "$(date -Is) $JOB $1 $2" | tee -a "$BATCHLOG"
    [ -f "$3" ] && tail -3 "$3" | sed 's/^/    /' | tee -a "$BATCHLOG"
}

adopt_seasplat_run() {
    # adopt_seasplat_run <views/scene> <out> — train.py overwrites -m with
    # <source_path>/experiments/<MMDDYYYY>/<exp> (train.py:1035), so the run
    # lands inside the staged view. Move it to the canonical baseline layout:
    # render, report and reuse detection all key off <out>, and derived
    # artifacts must never live under assets/data (ADR-007, ADR-013, KI-012).
    local real
    [ -d "$2" ] && return 1          # never nest a run inside an existing one
    real=$(ls -dt "$1"/experiments/*/"$EXP" 2>/dev/null | head -1)
    [ -n "$real" ] && [ -f "$real/point_cloud/iteration_$ITER/point_cloud.ply" ] || return 1
    mkdir -p "$(dirname "$2")" && mv "$real" "$2"
}

echo "Start at:  $(date)"
echo "Job:       ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Env:       $CONDA_ENV ($REPO)"
nvidia-smi -L || true

# --- stage INRIA-layout views (idempotent symlinks) ----------------
# The INRIA loader wants <scene>/{images, sparse/0}; our undistorted
# dataset holds the COLMAP .bin files directly under sparse/.
VIEWS=assets/data/inria_views
for SCENE in Curasao IUI3-RedSea JapaneseGradens-RedSea Panama; do
  mkdir -p "$VIEWS/$SCENE/sparse"
  ln -sfn "$ROOT_DIR/assets/data/undistorted_seathrunerf_dataset/$SCENE/images" "$VIEWS/$SCENE/images"
  ln -sfn "$ROOT_DIR/assets/data/undistorted_seathrunerf_dataset/$SCENE/sparse" "$VIEWS/$SCENE/sparse/0"
done

for SCENE in Curasao IUI3-RedSea JapaneseGradens-RedSea Panama; do
  RUNID=base_seasplat_$SCENE
  OUT="$ROOT_DIR/assets/outputs/baselines/seasplat/$SCENE"

  if ! gpu_ok; then
    echo "$(date -Is) $JOB GPU-DEAD — aborting batch before $RUNID" | tee -a "$BATCHLOG"
    break
  fi

  # A run left in the source tree by an earlier job is adopted, not retrained;
  # its render-time memory poll never ran, so drop the stub memlog too.
  if adopt_seasplat_run "$VIEWS/$SCENE" "$OUT"; then
    echo "$(date -Is) $JOB ADOPT $RUNID (moved out of $VIEWS/$SCENE)" | tee -a "$BATCHLOG"
    rm -f "$MEMLOGS/$RUNID.render.mem.csv"
  fi

  if [ -f "$OUT/point_cloud/iteration_$ITER/point_cloud.ply" ]; then
    echo "$(date -Is) $JOB REUSE $RUNID training" | tee -a "$BATCHLOG"
  elif [ -f "$OUT/cfg_args" ]; then
    echo "$(date -Is) $JOB SKIP $RUNID (incomplete training at $OUT — resolve manually)" | tee -a "$BATCHLOG"
    continue
  else
    echo "$(date -Is) $JOB TRAIN $RUNID" | tee -a "$BATCHLOG"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=10 \
      >"$MEMLOGS/$RUNID.mem.csv" &
    MEMPID=$!
    # -m is passed for the record only: train.py forces its own model_path.
    if (cd "$REPO" && python train.py \
        -s "$ROOT_DIR/$VIEWS/$SCENE" -m "$OUT" -r 1 --eval --exp "$EXP" \
        --do_seathru --seathru_from_iter 10000 \
        --test_iterations "$ITER" --save_iterations "$ITER" --checkpoint_iterations "$ITER") \
      >"$LOGS/$RUNID.train.log" 2>&1; then
      kill $MEMPID 2>/dev/null
      adopt_seasplat_run "$VIEWS/$SCENE" "$OUT" \
        || fail ADOPT-FAIL "$RUNID" "$LOGS/$RUNID.train.log"
    else
      kill $MEMPID 2>/dev/null
      fail TRAIN-FAIL "$RUNID" "$LOGS/$RUNID.train.log"
      continue
    fi
  fi

  # render_uw regenerates test/with_water if the in-train eval did not.
  # An adopted run already carries those renders but no memory poll, so the
  # memlog is part of the condition — that poll is the R2-#6 column.
  if ls "$OUT/test/with_water"/* >/dev/null 2>&1 && [ -s "$MEMLOGS/$RUNID.render.mem.csv" ]; then
    echo "$(date -Is) $JOB SKIP-RENDER $RUNID (test/with_water + memlog exist)" | tee -a "$BATCHLOG"
  else
    if ! gpu_ok; then
      echo "$(date -Is) $JOB GPU-DEAD — aborting batch before render of $RUNID" | tee -a "$BATCHLOG"
      break
    fi
    echo "$(date -Is) $JOB RENDER $RUNID" | tee -a "$BATCHLOG"
    # 1 s poll: a handful of test views renders in seconds, and a 10 s poll
    # can miss the peak entirely — this feeds the R2-#6 memory column.
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=1 \
      >"$MEMLOGS/$RUNID.render.mem.csv" &
    MEMPID=$!
    # -s/-r/--eval are repeated here on purpose: render_uw.py builds its
    # parser WITHOUT INRIA's `sentinel=True` (render_uw.py:247), so every
    # unset flag keeps its default and `get_combined_args` overwrites the
    # stored cfg_args with it — an empty source_path ("Could not recognize
    # scene type!"), resolution -1 and eval=False (an empty test set).
    (cd "$REPO" && python render_uw.py -m "$OUT" \
        -s "$ROOT_DIR/$VIEWS/$SCENE" -r 1 --eval \
        --iteration "$ITER" --skip_train --seathru) \
      >"$LOGS/$RUNID.render.log" 2>&1 \
      || fail RENDER-FAIL "$RUNID" "$LOGS/$RUNID.render.log"
    kill $MEMPID 2>/dev/null
  fi

  if [ -f "$RESULTS/$SCENE/efficiency.json" ]; then
    echo "$(date -Is) $JOB DONE $RUNID (report exists)" | tee -a "$BATCHLOG"
    continue
  fi
  mkdir -p "$RESULTS/$SCENE"
  echo "$(date -Is) $JOB REPORT $RUNID" | tee -a "$BATCHLOG"
  if python scripts/inria_report.py --method seasplat \
    --model-dir "$OUT" --view "$ROOT_DIR/$VIEWS/$SCENE" --repo "$REPO" \
    --scene "$SCENE" --iteration "$ITER" \
    --train-memlog "$MEMLOGS/$RUNID.mem.csv" \
    --render-memlog "$MEMLOGS/$RUNID.render.mem.csv" \
    --out "$RESULTS/$SCENE/efficiency.json" \
    >"$LOGS/$RUNID.report.log" 2>&1; then
    echo "$(date -Is) $JOB DONE $RUNID" | tee -a "$BATCHLOG"
  else
    fail REPORT-FAIL "$RUNID" "$LOGS/$RUNID.report.log"
  fi
done
echo "$(date -Is) $JOB BATCH EXITED" | tee -a "$BATCHLOG"
echo "Finished at: $(date)"
