#!/usr/bin/env bash
# ======================================================================
# Push the PEARL repo + cluster-needed assets to the SLURM server.
#
#     bash scripts/push_to_cluster.sh                   # code + dataset only
#     DRY_RUN=1 bash scripts/push_to_cluster.sh          # itemized preview
#     PUSH_BASELINES=0 bash scripts/push_to_cluster.sh   # skip the sibling repos
#     PUSH_CHECKPOINTS=1 bash scripts/push_to_cluster.sh # also send our own
#                                                         # trained checkpoints
#
# Default ships ONLY what training needs to start: all code, the two
# undistorted datasets, and (PUSH_BASELINES=1 by default) the baseline
# sibling snapshots — water-splatting, seasplat, 3D-UIR, Depth-Anything-V2
# + its 1.3 G weights, BASELINE_PINS.txt (fetched by fetch_baselines.sh).
# Those are third-party code/weights, not our results, so they push by
# default; set PUSH_BASELINES=0 to send code+dataset alone.
#
# PEARL's OWN trained artifacts — the never-swept 200K run (~2.2 G,
# revision_job3 section A1) and the four published mains' best/ exports
# (~2.7 G, section A2) — are opt-in only (PUSH_CHECKPOINTS=1). Without
# them, job3's A1/A2 sections log MISSING and skip cleanly (by design);
# push them later, once, whenever those sections are actually wanted.
#
# Everything else under assets/ stays local. NEVER uses --delete —
# cluster-side results must survive every push.
# ======================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(dirname "$SCRIPT_DIR")
REMOTE="${PEARL_REMOTE:-H100}"
REMOTE_ROOT="${PEARL_REMOTE_ROOT:-/mnt/data/phucnt3/3D_Reconstruction/RESEARCH.2025.PEARL}"
PUSH_BASELINES="${PUSH_BASELINES:-1}"
PUSH_CHECKPOINTS="${PUSH_CHECKPOINTS:-0}"

OPTS=(-avh --itemize-changes)
if [ "${DRY_RUN:-0}" = "1" ]; then
    OPTS+=(-n)
    echo "DRY RUN — nothing will be transferred"
fi

ssh "$REMOTE" true || { echo "Cannot reach $REMOTE — check ssh config." >&2; exit 1; }

echo "Push $LOCAL_ROOT -> $REMOTE:$REMOTE_ROOT"
echo "  code + dataset : always"
echo "  baselines      : $([ "$PUSH_BASELINES" = 1 ] && echo yes || echo 'no (PUSH_BASELINES=0)')"
echo "  our checkpoints: $([ "$PUSH_CHECKPOINTS" = 1 ] && echo yes || echo 'no (default — set PUSH_CHECKPOINTS=1 to include)')"

# Anchored, ordered rsync filters: code flows by default; under assets/
# only the dataset tree passes (checkpoints are a separate, opt-in step).
FILTERS=(
    --exclude='/.git/'
    --exclude='/.claude/'
    --exclude='/.ruff_cache/'
    --exclude='/.vscode/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='/assets/results.zip'
    --include='/assets/'
    --include='/assets/data/'
    --include='/assets/data/undistorted_seathrunerf_dataset/***'
    --include='/assets/data/undistorted_llff_dataset/***'
    --exclude='/assets/***'
)

rsync "${OPTS[@]}" "${FILTERS[@]}" "$LOCAL_ROOT/" "$REMOTE:$REMOTE_ROOT"

# --- opt-in: our own trained checkpoints (PUSH_CHECKPOINTS=1) ------
if [ "$PUSH_CHECKPOINTS" = 1 ]; then
    # Keep this list in sync with revision_job3.sbatch MAINS.
    MAINS=(
        "Curasao/2026-02-26_033350"
        "IUI3-RedSea/2026-02-26_033443"
        "JapaneseGradens-RedSea/2026-02-26_033626"
        "Panama/2026-02-26_082359"
    )
    RUN200K="assets/outputs/unnamed/pearl/2026-03-03_050614"

    du -sh "$LOCAL_ROOT/$RUN200K" 2>/dev/null || echo "WARN: $RUN200K missing locally"
    for m in "${MAINS[@]}"; do
        du -sh "$LOCAL_ROOT/assets/results/$m/best" 2>/dev/null \
            || echo "WARN: assets/results/$m/best missing locally"
    done

    CKPT_FILTERS=(
        --include='/assets/'
        --include='/assets/outputs/'
        --include='/assets/outputs/unnamed/'
        --include='/assets/outputs/unnamed/pearl/'
        --include="/$RUN200K/***"
        --include='/assets/results/'
    )
    for m in "${MAINS[@]}"; do
        scene=${m%%/*}
        CKPT_FILTERS+=(
            --include="/assets/results/$scene/"
            --include="/assets/results/$m/"
            --include="/assets/results/$m/best/***"
        )
    done
    CKPT_FILTERS+=(--exclude='/assets/***')

    rsync "${OPTS[@]}" "${CKPT_FILTERS[@]}" "$LOCAL_ROOT/" "$REMOTE:$REMOTE_ROOT"
fi

# --- baseline sibling snapshots → remote parent (ADR-013) ----------
if [ "$PUSH_BASELINES" = 1 ]; then
    PARENT=$(dirname "$LOCAL_ROOT")
    REMOTE_PARENT=$(dirname "$REMOTE_ROOT")
    for d in water-splatting seasplat 3D-UIR Depth-Anything-V2; do
        if [ -d "$PARENT/$d" ]; then
            echo "Push $PARENT/$d -> $REMOTE:$REMOTE_PARENT/"
            rsync "${OPTS[@]}" --exclude='__pycache__/' "$PARENT/$d" "$REMOTE:$REMOTE_PARENT/"
        else
            echo "WARN: $PARENT/$d missing locally — run scripts/fetch_baselines.sh first"
        fi
    done
    if [ -f "$PARENT/BASELINE_PINS.txt" ]; then
        rsync "${OPTS[@]}" "$PARENT/BASELINE_PINS.txt" "$REMOTE:$REMOTE_PARENT/"
    fi
fi

echo "Push finished."
