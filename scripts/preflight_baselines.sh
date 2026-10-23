#!/usr/bin/env bash
# ======================================================================
# GPU-node preflight for the baseline batch — one srun before any 24 h
# job goes into the queue:
#
#     srun --partition=main --gres=gpu:1 --cpus-per-task=8 --mem=48G \
#          --time=00:40:00 bash scripts/preflight_baselines.sh [ws|3duir|all]
#
# Both 2026-08-14 failures were import-time bugs that a GPU-less login
# node could not distinguish from a missing driver (KI-010, KI-011), so
# each cost a full job cycle. This runs the same imports and a token
# amount of real work where the batch would, and leaves behind two
# artifacts the batch reuses: the gsplat JIT build and the 3D-UIR depth
# maps. Idempotent; nothing here overwrites an experiment.
# ======================================================================
set -uo pipefail

ROOT_DIR="${PEARL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

WS_ENV="${WS_CONDA_ENV:-wsplat}"
UIR_ENV="${UIR_CONDA_ENV:-uir3d}"
REPO_UIR="${UIR_REPO:-$(dirname "$ROOT_DIR")/3D-UIR}"
DAV2="${DAV2:-$(dirname "$ROOT_DIR")/Depth-Anything-V2}"
VIEWS=assets/data/uir_views
SCENES=(Curasao IUI3-RedSea JapaneseGradens-RedSea Panama)
TARGET="${1:-all}"
FAILED=0

# A non-interactive `srun` shell has no conda on PATH, so the usual
# `conda info --base` probe comes back empty — say so instead of failing
# every check with "conda: command not found".
CONDA_PROFILE="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_PROFILE" ]]; then
    echo "Conda profile not found: $CONDA_PROFILE — set CONDA_SH." >&2
    exit 1
fi
set +u
source "$CONDA_PROFILE"
set -u

step() { echo; echo "=== $* ==="; }
ok()   { echo "PREFLIGHT-OK   $*"; }
bad()  { echo "PREFLIGHT-FAIL $*"; FAILED=$((FAILED + 1)); }

nvidia-smi -L || true

# --- water-splatting: import, then 50 real steps --------------------
preflight_ws() {
    step "[ws] $WS_ENV imports"
    if conda run -n "$WS_ENV" python -c "
import torch, nerfstudio, water_splatting
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))
"; then ok "ws imports"; else bad "ws imports"; return; fi

    # 50 steps exercise what --help cannot: the colmap dataparser, the
    # gsplat JIT build (cached for the batch) and a checkpoint write.
    step "[ws] 50-step training smoke"
    local out=assets/outputs/preflight
    rm -rf "$out"
    if conda run -n "$WS_ENV" ns-train water-splatting --vis tensorboard \
        --output-dir "$out" --experiment-name preflight_ws \
        --max-num-iterations 50 --steps-per-save 50 \
        --data "assets/data/undistorted_seathrunerf_dataset/${SCENES[0]}" \
        colmap --downscale-factor 1 --colmap-path sparse --images-path images \
        2>&1 | tail -12; then
        if ls "$out"/preflight_ws/water-splatting/*/nerfstudio_models/*.ckpt >/dev/null 2>&1; then
            ok "ws 50-step training + checkpoint"
        else
            bad "ws 50-step training (no checkpoint written)"
        fi
    else
        bad "ws 50-step training"
    fi
    rm -rf "$out"
}

# --- 3D-UIR: imports, then the whole depth chain --------------------
preflight_3duir() {
    step "[3duir] $UIR_ENV imports (numpy ABI + CUDA extensions)"
    if conda run -n "$UIR_ENV" python -c "
import numpy as np, cv2, torch
import diff_gaussian_rasterization, simple_knn, fused_ssim, lpips, kornia
import tinycudann   # 3D-UIR imports it at module load; a login-node build fails here
print('numpy', np.__version__, 'cv2', cv2.__version__, 'torch', torch.__version__)
print('numpy->cuda', float(torch.from_numpy(np.ones(3, dtype='float32')).cuda().sum()))
"; then ok "3duir imports"; else bad "3duir imports"; return; fi

    # The depth maps and depth_params.json the batch job expects; both of
    # its guards skip whatever this leaves behind (3D-UIR README steps 3-4).
    for SCENE in "${SCENES[@]}"; do
        mkdir -p "$VIEWS/$SCENE/sparse/0"
        ln -sfn "$ROOT_DIR/assets/data/undistorted_seathrunerf_dataset/$SCENE/images" "$VIEWS/$SCENE/images"
        for BIN in cameras.bin images.bin points3D.bin; do
            ln -sfn "$ROOT_DIR/assets/data/undistorted_seathrunerf_dataset/$SCENE/sparse/$BIN" \
                "$VIEWS/$SCENE/sparse/0/$BIN"
        done

        local n_img n_depth
        n_img=$(ls "$VIEWS/$SCENE/images/" | wc -l)
        n_depth=$(ls "$VIEWS/$SCENE/depths" 2>/dev/null | wc -l)
        step "[3duir] depths $SCENE ($n_depth/$n_img present)"
        if [ "$n_depth" -lt "$n_img" ]; then
            (cd "$DAV2" && conda run -n "$UIR_ENV" python run.py --encoder vitl --pred-only --grayscale \
                --img-path "$ROOT_DIR/$VIEWS/$SCENE/images" \
                --outdir "$ROOT_DIR/$VIEWS/$SCENE/depths") 2>&1 | tail -3
        fi
        n_depth=$(ls "$VIEWS/$SCENE/depths" 2>/dev/null | wc -l)
        [ "$n_depth" -ge "$n_img" ] && ok "depths $SCENE ($n_depth)" || { bad "depths $SCENE ($n_depth/$n_img)"; continue; }

        if [ ! -f "$VIEWS/$SCENE/sparse/0/depth_params.json" ]; then
            (cd "$REPO_UIR" && conda run -n "$UIR_ENV" python utils/make_depth_scale.py \
                --base_dir "$ROOT_DIR/$VIEWS/$SCENE" \
                --depths_dir "$ROOT_DIR/$VIEWS/$SCENE/depths") 2>&1 | tail -3
        fi
        [ -f "$VIEWS/$SCENE/sparse/0/depth_params.json" ] \
            && ok "depth_params.json $SCENE" || bad "depth_params.json $SCENE"
    done

    step "[3duir] train.py CLI"
    (cd "$REPO_UIR" && conda run -n "$UIR_ENV" python train.py -h >/dev/null) \
        && ok "3duir train.py -h" || bad "3duir train.py -h"
}

case "$TARGET" in
    ws)    preflight_ws ;;
    3duir) preflight_3duir ;;
    all)   preflight_ws; preflight_3duir ;;
    *) echo "Usage: bash scripts/preflight_baselines.sh [ws|3duir|all]" >&2; exit 1 ;;
esac

echo
if [ "$FAILED" -eq 0 ]; then
    echo "PREFLIGHT PASSED — safe to sbatch the baseline jobs"
else
    echo "PREFLIGHT FAILED ($FAILED check(s)) — do NOT submit until they pass"
fi
exit "$FAILED"
