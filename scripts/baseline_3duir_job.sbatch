#!/usr/bin/env bash
#SBATCH --job-name=pearl_base_3duir
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                 # EDIT: pin a GPU type if supported, e.g. gpu:a100:1
#SBATCH --time=24:00:00              # 4 x (uncapped 30k-iter train + render + metrics) + margin
#SBATCH --output=assets/outputs/slurm/%x-%j.out
#SBATCH --error=assets/outputs/slurm/%x-%j.err
#
# 3D-UIR baseline batch (ADR-013): the 4 SeaThru-NeRF scenes under the
# method's own published protocol — 30k iterations, UNCAPPED densification,
# every-8th --eval holdout, WITH its Depth-Anything-V2 pseudo-depth
# pipeline (README steps 1-5): DAv2 ViT-L depths → make_depth_scale.py →
# train -d. Without -d the DRO depth loss silently deactivates
# (train.py gates on depth_reliable), so depths are required for a
# faithful baseline. Deliberate explicit flags, protocol-neutral:
#   -r 1                 forced native resolution (INRIA silently downscales
#                        the 1794 px Curasao/Panama images to 1600 px)
#   --iterations 30000   upstream's published schedule, pinned explicitly
#                        (+ save/test trimmed to 30000)
# Own staged views (uir_views) with a REAL sparse/0 dir of .bin symlinks:
# make_depth_scale.py writes depth_params.json into sparse/0/, which must
# not land inside the shared dataset. Metrics come from the repo's own
# render.py + metrics.py (results.json). An OOM on an uncapped scene is
# acceptable, reportable evidence.
#
#     sbatch scripts/baseline_3duir_job.sbatch     # from the repo root
#
# Prereq: bash scripts/setup_baseline_envs.sh 3duir
set -uo pipefail

# --- cluster modules + conda env -----------------------------------
# No toolkit needed at runtime: d-g-r, simple-knn, fused-ssim and tcnn are
# compiled once by setup_baseline_envs.sh against the env's own CUDA 11.8
# (the cluster's system toolkit is 12.2, a major mismatch for cu118, ADR-014).
ROOT_DIR="${PEARL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

CONDA_PROFILE="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
CONDA_ENV="${UIR_CONDA_ENV:-uir3d}"
REPO="${UIR_REPO:-$(dirname "$ROOT_DIR")/3D-UIR}"
DAV2="${DAV2:-$(dirname "$ROOT_DIR")/Depth-Anything-V2}"
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
RESULTS=$BASE/3d-uir
LOGS=$BASE/logs
MEMLOGS=$BASE/memlogs
mkdir -p "$RESULTS" "$LOGS" "$MEMLOGS"
BATCHLOG=$BASE/baseline_runs.log
ITER=30000
JOB="3duir[${SLURM_JOB_ID:-local}]"

gpu_ok() { nvidia-smi -L >/dev/null 2>&1; }

fail() {
    # fail <marker> <runid> <log> — a failure line is useless without its
    # cause: the 2026-08-14 wave read as four DEPTH-FAILs and was a numpy
    # ABI mismatch inside DAv2 (KI-011).
    echo "$(date -Is) $JOB $1 $2" | tee -a "$BATCHLOG"
    [ -f "$3" ] && tail -3 "$3" | sed 's/^/    /' | tee -a "$BATCHLOG"
}

echo "Start at:  $(date)"
echo "Job:       ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Env:       $CONDA_ENV ($REPO)"
nvidia-smi -L || true

# --- stage 3D-UIR views (idempotent) -------------------------------
# Unlike the seasplat views, sparse/0 is a REAL directory of per-file
# symlinks: make_depth_scale.py writes depth_params.json next to the
# .bin files, and that must not land inside the shared dataset.
VIEWS=assets/data/uir_views
for SCENE in Curasao IUI3-RedSea JapaneseGradens-RedSea Panama; do
  mkdir -p "$VIEWS/$SCENE/sparse/0"
  ln -sfn "$ROOT_DIR/assets/data/undistorted_seathrunerf_dataset/$SCENE/images" "$VIEWS/$SCENE/images"
  for BIN in cameras.bin images.bin points3D.bin; do
    ln -sfn "$ROOT_DIR/assets/data/undistorted_seathrunerf_dataset/$SCENE/sparse/$BIN" "$VIEWS/$SCENE/sparse/0/$BIN"
  done
done

for SCENE in Curasao IUI3-RedSea JapaneseGradens-RedSea Panama; do
  RUNID=base_3duir_$SCENE
  OUT="$ROOT_DIR/assets/outputs/baselines/3d-uir/$SCENE"

  if ! gpu_ok; then
    echo "$(date -Is) $JOB GPU-DEAD — aborting batch before $RUNID" | tee -a "$BATCHLOG"
    break
  fi

  if [ -f "$OUT/point_cloud/iteration_$ITER/point_cloud.ply" ]; then
    echo "$(date -Is) $JOB REUSE $RUNID training" | tee -a "$BATCHLOG"
  elif [ -f "$OUT/cfg_args" ]; then
    echo "$(date -Is) $JOB SKIP $RUNID (incomplete training at $OUT — resolve manually)" | tee -a "$BATCHLOG"
    continue
  else
    # --- DAv2 pseudo-depths + depth params (3D-UIR README steps 3-4) ---
    if [ "$(ls "$VIEWS/$SCENE/depths" 2>/dev/null | wc -l)" -lt "$(ls "$VIEWS/$SCENE/images/" | wc -l)" ]; then
      echo "$(date -Is) $JOB DEPTHS $RUNID (DAv2 vitl)" | tee -a "$BATCHLOG"
      if ! (cd "$DAV2" && python run.py --encoder vitl --pred-only --grayscale \
          --img-path "$ROOT_DIR/$VIEWS/$SCENE/images" \
          --outdir "$ROOT_DIR/$VIEWS/$SCENE/depths") \
        >"$LOGS/$RUNID.depth.log" 2>&1; then
        fail DEPTH-FAIL "$RUNID" "$LOGS/$RUNID.depth.log"
        continue
      fi
    fi
    if [ ! -f "$VIEWS/$SCENE/sparse/0/depth_params.json" ]; then
      if ! (cd "$REPO" && python utils/make_depth_scale.py \
          --base_dir "$ROOT_DIR/$VIEWS/$SCENE" \
          --depths_dir "$ROOT_DIR/$VIEWS/$SCENE/depths") \
        >>"$LOGS/$RUNID.depth.log" 2>&1; then
        fail DEPTHSCALE-FAIL "$RUNID" "$LOGS/$RUNID.depth.log"
        continue
      fi
    fi

    echo "$(date -Is) $JOB TRAIN $RUNID" | tee -a "$BATCHLOG"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=10 \
      >"$MEMLOGS/$RUNID.mem.csv" &
    MEMPID=$!
    # --disable_viewer: train.py binds TCP 6009 for the SIBR viewer, and a
    # SeaSplat run on the same node binds it unconditionally — a headless
    # batch has no use for either, and the collision kills training.
    if (cd "$REPO" && python train.py \
        -s "$ROOT_DIR/$VIEWS/$SCENE" -m "$OUT" -r 1 --eval -d depths \
        --iterations "$ITER" --save_iterations "$ITER" --test_iterations "$ITER" \
        --disable_viewer) \
      >"$LOGS/$RUNID.train.log" 2>&1; then
      kill $MEMPID 2>/dev/null
    else
      kill $MEMPID 2>/dev/null
      fail TRAIN-FAIL "$RUNID" "$LOGS/$RUNID.train.log"
      continue
    fi
  fi

  if [ -d "$OUT/test/ours_$ITER/renders" ] && ls "$OUT/test/ours_$ITER/renders"/* >/dev/null 2>&1; then
    echo "$(date -Is) $JOB SKIP-RENDER $RUNID (test/ours_$ITER exists)" | tee -a "$BATCHLOG"
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
    (cd "$REPO" && python render.py -m "$OUT" --iteration "$ITER" --skip_train) \
      >"$LOGS/$RUNID.render.log" 2>&1 \
      || fail RENDER-FAIL "$RUNID" "$LOGS/$RUNID.render.log"
    kill $MEMPID 2>/dev/null
  fi

  if [ -f "$OUT/results.json" ]; then
    echo "$(date -Is) $JOB SKIP-METRICS $RUNID (results.json exists)" | tee -a "$BATCHLOG"
  else
    echo "$(date -Is) $JOB METRICS $RUNID" | tee -a "$BATCHLOG"
    (cd "$REPO" && python metrics.py -m "$OUT") \
      >"$LOGS/$RUNID.metrics.log" 2>&1 \
      || fail METRICS-FAIL "$RUNID" "$LOGS/$RUNID.metrics.log"
  fi

  if [ -f "$RESULTS/$SCENE/efficiency.json" ]; then
    echo "$(date -Is) $JOB DONE $RUNID (report exists)" | tee -a "$BATCHLOG"
    continue
  fi
  mkdir -p "$RESULTS/$SCENE"
  echo "$(date -Is) $JOB REPORT $RUNID" | tee -a "$BATCHLOG"
  if python scripts/inria_report.py --method 3duir \
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
