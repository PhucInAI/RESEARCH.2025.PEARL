#!/usr/bin/env bash
#SBATCH --job-name=phase01_step4_b0
#SBATCH --partition=main             # EDIT: cluster partition (README defaults to main)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # EDIT: SAM+CLIP preprocessing can be memory-heavy
#SBATCH --gres=gpu:1                 # EDIT: pin the H100, e.g. gpu:h100:1, if the cluster supports typed gres
#SBATCH --time=72:00:00              # EDIT: covers all thirteen runs sequentially (preprocess + 3 levels + eval each; mip scenes are 185-311 images at ~1300-1560 px)
#SBATCH --output=assets/outputs/slurm/%x-%j.out
#SBATCH --error=assets/outputs/slurm/%x-%j.err
#
# Phase 1 steps 4-6 — LangSplatV2 B0 baseline on the H100 (ADR-015). Runs ALL
# thirteen scenes in one job, sequentially. Launch with no arguments:
#
#     sbatch scripts/slurm/phase01_step4_run_baseline_b0.sbatch
#
# Each scene's geometry (the EXP-9x recovery/reconstruction and EXP-8x
# conversion experiments) must already exist. --teacher-mode preprocess is the explicit approval for SAM+CLIP
# extraction; override with `TEACHER_MODE=reuse sbatch ...` when validated
# language_features/ already exist. Thin wrapper (ADR-009): no logic beyond the
# run list — the driver runs its own strict preflight per scene.
set -euo pipefail

# --- cluster modules: pick the CUDA runtime matching the pip cu130 wheel ---
# module load cuda/13.0 gcc/12       # EDIT to your cluster's module names

# --- paths and environment (ADR-002 / ADR-005) ---
ROOT_DIR="${XVIEW_SPLAT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CONDA_PROFILE="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
CONDA_ENV="${XVIEW_SPLAT_CONDA_ENV:-xviewsplat}"
if [[ ! -f "$CONDA_PROFILE" ]]; then
    echo "Conda profile not found: $CONDA_PROFILE" >&2
    echo "Set CONDA_SH to the correct conda.sh path." >&2
    exit 1
fi
source "$CONDA_PROFILE"
conda activate "$CONDA_ENV"

mkdir -p assets/outputs/slurm

TEACHER_MODE="${TEACHER_MODE:-preprocess}"

echo "Start at:  $(date)"
echo "Job:       ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Env:       $CONDA_ENV   teacher-mode: $TEACHER_MODE"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L || true

# --- the thirteen B0 runs: "b0_number dataset scene rgb_geometry_id ckpt_filename" ---
# b0_number is a bare EXP-NN; the driver stamps the Anywhere-on-Earth (UTC-12)
# date to build the B0 experiment id (EXP-NN-YYYYMMDD). Numbers are globally
# unique so the thirteen same-day runs never resolve to the same directory.
# rgb_geometry_id is the geometry experiment's already-resolved directory name
# under assets/outputs/ — EDIT each YYYYMMDD to the AOE date its step (2b/2d/3)
# produced. Five recovered (rgb-geometry-recovered.pth) + four self-trained
# (chkpnt30000.pth) + four converted pretrained (rgb-geometry-pretrained.pth,
# ADR-016).
RUNS=(
    "EXP-01 lerf_ovs teatime       EXP-90-YYYYMMDD rgb-geometry-recovered.pth"
    "EXP-02 lerf_ovs ramen         EXP-95-YYYYMMDD rgb-geometry-recovered.pth"
    "EXP-03 lerf_ovs figurines     EXP-96-YYYYMMDD rgb-geometry-recovered.pth"
    "EXP-04 lerf_ovs waldo_kitchen EXP-97-YYYYMMDD rgb-geometry-recovered.pth"
    "EXP-07 3d_ovs   lawn          EXP-98-YYYYMMDD rgb-geometry-recovered.pth"
    "EXP-05 3d_ovs   bed           EXP-91-YYYYMMDD chkpnt30000.pth"
    "EXP-06 3d_ovs   bench         EXP-92-YYYYMMDD chkpnt30000.pth"
    "EXP-08 3d_ovs   room          EXP-93-YYYYMMDD chkpnt30000.pth"
    "EXP-09 3d_ovs   sofa          EXP-94-YYYYMMDD chkpnt30000.pth"
    "EXP-10 mip_nerf360_ovs bonsai  EXP-80-YYYYMMDD rgb-geometry-pretrained.pth"
    "EXP-11 mip_nerf360_ovs counter EXP-81-YYYYMMDD rgb-geometry-pretrained.pth"
    "EXP-12 mip_nerf360_ovs garden  EXP-82-YYYYMMDD rgb-geometry-pretrained.pth"
    "EXP-13 mip_nerf360_ovs room    EXP-83-YYYYMMDD rgb-geometry-pretrained.pth"
)

failed=()
for row in "${RUNS[@]}"; do
    read -r b0_id dataset scene rgb_id ckpt <<<"$row"
    echo "==================================================================="
    echo ">>> B0 $b0_id  $dataset/$scene  <-  $rgb_id/$ckpt  ($(date))"
    if python scripts/phase01_step4_run_baseline_b0.py \
        --experiment-id "$b0_id" --dataset "$dataset" --scene "$scene" \
        --rgb-checkpoint "assets/outputs/$rgb_id/model/$ckpt" \
        --model-index 0 --teacher-mode "$TEACHER_MODE"; then
        echo "<<< $b0_id OK"
    else
        echo "<<< $b0_id FAILED (exit $?) — continuing with remaining scenes" >&2
        failed+=("$b0_id $dataset/$scene")
    fi
done

echo "==================================================================="
echo "Finished at: $(date)"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED runs (${#failed[@]}/${#RUNS[@]}):" >&2
    printf '  - %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "All ${#RUNS[@]} B0 runs completed."
