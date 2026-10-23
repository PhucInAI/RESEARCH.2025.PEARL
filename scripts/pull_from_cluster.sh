#!/usr/bin/env bash
# ======================================================================
# Pull cluster run artifacts back into assets/revision/ — never into the
# canonical local assets/results/ tree.
#
#     bash scripts/pull_from_cluster.sh               # everything but weights
#     PULL_CKPTS=1 bash scripts/pull_from_cluster.sh  # weights too (+32 GB)
#     DRY_RUN=1 bash scripts/pull_from_cluster.sh     # itemized preview
#
# Carries the revision sweeps/best exports/analysis JSONs, the run and
# eval configs, the baseline reports (assets/results/baselines/), the
# error-map and figure renders, the batch/mem logs and the slurm
# stdout/stderr — ~210 MB. Model weights are excluded BY PATTERN
# (*.ckpt/*.pth/*.pt/*.ply), not by filename: best.ckpt alone is 32 GB
# across the sweep winners, and a stray step-*.ckpt or an INRIA
# point_cloud.ply under results/ would otherwise slip through. Set
# PULL_CKPTS=1 to take them anyway. NEVER uses --delete.
# ======================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(dirname "$SCRIPT_DIR")
REMOTE="${PEARL_REMOTE:-H100}"
REMOTE_ROOT="${PEARL_REMOTE_ROOT:-/mnt/data/phucnt3/3D_Reconstruction/RESEARCH.2025.PEARL}"

OPTS=(-avh --itemize-changes)
if [ "${DRY_RUN:-0}" = "1" ]; then
    OPTS+=(-n)
    echo "DRY RUN — nothing will be transferred"
fi
if [ "${PULL_CKPTS:-0}" != "1" ]; then
    OPTS+=(--exclude='*.ckpt' --exclude='*.pth' --exclude='*.pt' --exclude='*.ply')
fi

ssh "$REMOTE" true || { echo "Cannot reach $REMOTE — check ssh config." >&2; exit 1; }

mkdir -p "$LOCAL_ROOT/assets/revision/results" "$LOCAL_ROOT/assets/revision/slurm"

echo "Pull $REMOTE:$REMOTE_ROOT/assets/results/ -> assets/revision/results/"
rsync "${OPTS[@]}" \
    "$REMOTE:$REMOTE_ROOT/assets/results/" "$LOCAL_ROOT/assets/revision/results/"

echo "Pull $REMOTE:$REMOTE_ROOT/assets/outputs/slurm/ -> assets/revision/slurm/"
rsync "${OPTS[@]}" \
    "$REMOTE:$REMOTE_ROOT/assets/outputs/slurm/" "$LOCAL_ROOT/assets/revision/slurm/"

echo "Pull finished."
