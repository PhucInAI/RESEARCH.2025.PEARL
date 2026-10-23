#!/usr/bin/env bash
#SBATCH --job-name=pearl_rev_job3
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                 # EDIT: pin a GPU type if supported, e.g. gpu:a100:1
#SBATCH --time=60:00:00              # 3 x (~5 h train @10M + ~1 h sweep) + sweeps/reports + margin
#SBATCH --output=assets/outputs/slurm/%x-%j.out
#SBATCH --error=assets/outputs/slurm/%x-%j.err
#
# PEARL revision batch 3/3: everything CURRENT_TASK.md still lists as blocked
# on the cluster — cheap guaranteed work first, long risky runs last.
#   A1  sweep of the never-swept 200K run (2026-03-03_050614, 14 ckpts)
#   A2  efficiency reports over the four published mains' best/ (R2-#6)
#   A3  error maps over the four mains (paper figures)
#   B   rev_b10M_{Curasao,IUI3-RedSea,Panama}: 10M-budget replacements for the
#       OOM'd rev_nobudget_* pair (35.4M / 20.4M gaussians at ~80.8 GB — the
#       failure is kept as evidence). rev_b10M_Curasao also regenerates paper
#       Table 7's unbacked 10M row. ADR-012: --steps-per-save 2000 here (a 10M
#       ckpt is ~6.9 G; 10 saves = ~69 G/run transient vs ~690 G at the frozen
#       protocol's 200) — coarser best-checkpoint selection accepted.
#
#     sbatch scripts/revision_job3.sbatch          # from the repo root
#
# Companion: revision_job2.sbatch may be resubmitted concurrently (idempotent).
# Do NOT resubmit job1 for the nobudget pair — its crash dirs hit the
# incomplete-training SKIP by design. Requires scripts/push_to_cluster.sh to
# have shipped the 200K run dir and the four mains' best/ dirs beforehand.
# Disk: ~230 G transient (3 x 69 G checkpoint sweeps + best copies) — verify
# free space before launching.
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

# --- batch bookkeeping (same layout as job1/job2) ------------------
PAPER=assets/results/paper
LOGS=$PAPER/logs
MEMLOGS=$PAPER/memlogs
mkdir -p "$LOGS" "$MEMLOGS"
BATCHLOG=$PAPER/revision_runs.log
FINAL_CKPT=step-000019999.ckpt
JOB="job3[${SLURM_JOB_ID:-local}]"

gpu_ok() { nvidia-smi -L >/dev/null 2>&1; }

echo "Start at:  $(date)"
echo "Job:       ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Env:       $CONDA_ENV"
nvidia-smi -L || true

# --- A1: sweep the 200K-budget run (Table 7's 200K row) ------------
RUN200K=assets/outputs/unnamed/pearl/2026-03-03_050614
TS200K=$(basename "$RUN200K")
if [ ! -d "$RUN200K" ]; then
  echo "$(date -Is) $JOB MISSING b200K ($RUN200K not pushed — run push_to_cluster.sh)" | tee -a "$BATCHLOG"
else
  SCENE200K=$(python -c "from pearl.run_paths import scene_name; print(scene_name('$RUN200K/config.yml'))")
  if [ -f "assets/results/$SCENE200K/$TS200K/ckpt_sweep/all_results.json" ]; then
    echo "$(date -Is) $JOB DONE b200K (sweep already complete)" | tee -a "$BATCHLOG"
  elif ! gpu_ok; then
    echo "$(date -Is) $JOB GPU-DEAD — skipping b200K sweep" | tee -a "$BATCHLOG"
  else
    echo "$(date -Is) $JOB SWEEP b200K ($SCENE200K/$TS200K)" | tee -a "$BATCHLOG"
    if python scripts/eval_all_ckpts.py --run-dir "$RUN200K" >"$LOGS/b200K.sweep.log" 2>&1; then
      echo "$(date -Is) $JOB DONE b200K" | tee -a "$BATCHLOG"
    else
      echo "$(date -Is) $JOB SWEEP-FAIL b200K" | tee -a "$BATCHLOG"
    fi
  fi
  if [ -f "assets/results/$SCENE200K/$TS200K/best/config.yml" ] \
     && [ ! -f "assets/results/$SCENE200K/$TS200K/analysis/efficiency.json" ] && gpu_ok; then
    echo "$(date -Is) $JOB EFF b200K" | tee -a "$BATCHLOG"
    python scripts/efficiency_report.py \
      --load-config "assets/results/$SCENE200K/$TS200K/best/config.yml" \
      >"$LOGS/eff_b200K.log" 2>&1 \
      || echo "$(date -Is) $JOB EFF-FAIL b200K" | tee -a "$BATCHLOG"
  fi
fi

# --- A2: efficiency reports over the published mains (R2-#6) -------
# Keep this scene|timestamp list in sync with scripts/push_to_cluster.sh.
MAINS=(
  "Curasao|2026-02-26_033350"
  "IUI3-RedSea|2026-02-26_033443"
  "JapaneseGradens-RedSea|2026-02-26_033626"
  "Panama|2026-02-26_082359"
)

for spec in "${MAINS[@]}"; do
  IFS='|' read -r SCENE TS <<<"$spec"
  BEST="assets/results/$SCENE/$TS/best/config.yml"
  if [ ! -f "$BEST" ]; then
    echo "$(date -Is) $JOB MISSING eff_main_$SCENE ($BEST not pushed)" | tee -a "$BATCHLOG"
    continue
  fi
  if [ -f "assets/results/$SCENE/$TS/analysis/efficiency.json" ]; then
    echo "$(date -Is) $JOB SKIP eff_main_$SCENE (report exists)" | tee -a "$BATCHLOG"
    continue
  fi
  if ! gpu_ok; then
    echo "$(date -Is) $JOB GPU-DEAD — aborting efficiency reports at $SCENE" | tee -a "$BATCHLOG"
    break
  fi
  echo "$(date -Is) $JOB EFF main $SCENE ($TS)" | tee -a "$BATCHLOG"
  python scripts/efficiency_report.py --load-config "$BEST" \
    >"$LOGS/eff_main_$SCENE.log" 2>&1 \
    || echo "$(date -Is) $JOB EFF-FAIL main $SCENE" | tee -a "$BATCHLOG"
done

# --- A3: error maps over the mains (paper figures) -----------------
if [ "${PEARL_ERROR_MAPS:-1}" = "1" ]; then
  for spec in "${MAINS[@]}"; do
    IFS='|' read -r SCENE TS <<<"$spec"
    BEST="assets/results/$SCENE/$TS/best/config.yml"
    [ -f "$BEST" ] || continue
    if [ -f "assets/results/$SCENE/$TS/analysis/error_maps/index.json" ]; then
      echo "$(date -Is) $JOB SKIP emap_$SCENE (maps exist)" | tee -a "$BATCHLOG"
      continue
    fi
    if ! gpu_ok; then
      echo "$(date -Is) $JOB GPU-DEAD — aborting error maps at $SCENE" | tee -a "$BATCHLOG"
      break
    fi
    echo "$(date -Is) $JOB EMAP $SCENE ($TS)" | tee -a "$BATCHLOG"
    python scripts/error_map_dump.py --load-config "$BEST" \
      >"$LOGS/emap_$SCENE.log" 2>&1 \
      || echo "$(date -Is) $JOB EMAP-FAIL $SCENE" | tee -a "$BATCHLOG"
  done
fi

# --- B: the three 10M-budget runs (ADR-012) ------------------------
B10M_FLAGS="--pipeline.model.max-gaussians 10000000"
STEPS_PER_SAVE=2000   # ADR-012 deviation: 10M ckpts are ~6.9 G each

RUNS=(
  "rev_b10M_Curasao|Curasao|$B10M_FLAGS"
  "rev_b10M_IUI3-RedSea|IUI3-RedSea|$B10M_FLAGS"
  "rev_b10M_Panama|Panama|$B10M_FLAGS"
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
      --steps-per-save "$STEPS_PER_SAVE" --output-dir assets/outputs --experiment-name "$RUNID" \
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
  else
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
      continue
    fi
  fi

  if [ -f "assets/results/$SCENE/$TS/best/config.yml" ] \
     && [ ! -f "assets/results/$SCENE/$TS/analysis/efficiency.json" ] && gpu_ok; then
    echo "$(date -Is) $JOB EFF $RUNID" | tee -a "$BATCHLOG"
    python scripts/efficiency_report.py \
      --load-config "assets/results/$SCENE/$TS/best/config.yml" \
      >"$LOGS/eff_$RUNID.log" 2>&1 \
      || echo "$(date -Is) $JOB EFF-FAIL $RUNID" | tee -a "$BATCHLOG"
  fi
done
echo "$(date -Is) $JOB BATCH EXITED" | tee -a "$BATCHLOG"
echo "Finished at: $(date)"
