#!/usr/bin/env bash
# ======================================================================
# Fetch pinned baseline snapshots as SIBLINGS of the PEARL repo:
#
#     bash scripts/fetch_baselines.sh
#
#     <parent>/water-splatting      (+ glm submodule)
#     <parent>/seasplat             (+ dxyang d-g-r fork + glm, simple-knn)
#     <parent>/3D-UIR               (submodules vendored in-tree upstream)
#     <parent>/Depth-Anything-V2    (+ ViT-L weights ~1.3 G, 3D-UIR dep)
#     <parent>/BASELINE_PINS.txt    (commit SHAs of every snapshot)
#
# git is not installed on this machine, so these are codeload tarball
# snapshots at resolved HEAD SHAs — build-identical to clones, minus
# .git metadata (ADR-013 amendment). SIBR_viewers (interactive viewer
# only) is deliberately not fetched. Idempotent: any existing non-empty
# destination is left untouched.
# ======================================================================
# Fail-fast on purpose (matches setup_baseline_envs.sh, not the sbatch
# runners): a half-fetched repo tree is worse than a stopped script.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PEARL_ROOT=$(dirname "$SCRIPT_DIR")
PARENT="${BASELINES_PARENT:-$(dirname "$PEARL_ROOT")}"
PINS="$PARENT/BASELINE_PINS.txt"
DAV2_WEIGHTS_URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

# --- helpers -------------------------------------------------------
json_field() {
    # json_field <key> — read one string field from stdin JSON
    python3 -c "import json,sys; print(json.load(sys.stdin)['$1'])"
}

gh_api() { curl -fsSL "https://api.github.com/$1"; }

head_sha() {
    # head_sha <owner/repo> <branch> — commit SHA of the branch tip
    gh_api "repos/$1/commits/$2" | json_field sha
}

gitlink_sha() {
    # gitlink_sha <owner/repo> <path> <ref> — pinned SHA of a submodule entry
    gh_api "repos/$1/contents/$2?ref=$3" | json_field sha
}

occupied() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

extract_into() {
    # extract_into <tar-url> <dest> — tarball's single top dir becomes dest
    local url="$1" dest="$2" tmp
    tmp=$(mktemp -d)
    curl -fsSL "$url" | tar xz -C "$tmp"
    rmdir "$dest" 2>/dev/null || true   # parent tarballs leave empty submodule dirs
    mv "$tmp"/*/ "$dest"
    rmdir "$tmp"
}

fetch_github() {
    # fetch_github <name> <owner/repo> <sha> <dest>
    local name="$1" repo="$2" sha="$3" dest="$4"
    if occupied "$dest"; then
        echo "FETCH-SKIP $name ($dest exists — left untouched)"
        return
    fi
    echo "FETCH      $name  $repo @ ${sha:0:12}"
    extract_into "https://codeload.github.com/$repo/tar.gz/$sha" "$dest"
    echo "$name $repo $sha" >>"$PINS"
}

fetch_gitlab_inria() {
    # fetch_gitlab_inria <name> <group/project> <sha> <dest>
    local name="$1" proj="$2" sha="$3" dest="$4"
    if occupied "$dest"; then
        echo "FETCH-SKIP $name ($dest exists — left untouched)"
        return
    fi
    echo "FETCH      $name  gitlab.inria.fr/$proj @ ${sha:0:12}"
    extract_into "https://gitlab.inria.fr/$proj/-/archive/$sha/${proj##*/}-$sha.tar.gz" "$dest"
    echo "$name gitlab.inria.fr/$proj $sha" >>"$PINS"
}

# --- water-splatting -----------------------------------------------
WS_DEST="$PARENT/water-splatting"
if occupied "$WS_DEST"; then
    echo "FETCH-SKIP water-splatting (exists)"
else
    WS_SHA=$(head_sha water-splatting/water-splatting main)
    fetch_github water-splatting water-splatting/water-splatting "$WS_SHA" "$WS_DEST"
    GLM_SHA=$(gitlink_sha water-splatting/water-splatting water_splatting/cuda/csrc/third_party/glm "$WS_SHA")
    fetch_github ws-glm g-truc/glm "$GLM_SHA" "$WS_DEST/water_splatting/cuda/csrc/third_party/glm"
fi

# --- seasplat ------------------------------------------------------
SS_DEST="$PARENT/seasplat"
if occupied "$SS_DEST"; then
    echo "FETCH-SKIP seasplat (exists)"
else
    SS_SHA=$(head_sha dxyang/seasplat master)
    fetch_github seasplat dxyang/seasplat "$SS_SHA" "$SS_DEST"
    DGR_SHA=$(gitlink_sha dxyang/seasplat submodules/diff-gaussian-rasterization "$SS_SHA")
    fetch_github seasplat-dgr dxyang/diff-gaussian-rasterization "$DGR_SHA" \
        "$SS_DEST/submodules/diff-gaussian-rasterization"
    DGR_GLM_SHA=$(gitlink_sha dxyang/diff-gaussian-rasterization third_party/glm "$DGR_SHA")
    fetch_github seasplat-dgr-glm g-truc/glm "$DGR_GLM_SHA" \
        "$SS_DEST/submodules/diff-gaussian-rasterization/third_party/glm"
    KNN_SHA=$(gitlink_sha dxyang/seasplat submodules/simple-knn "$SS_SHA")
    fetch_gitlab_inria seasplat-simple-knn bkerbl/simple-knn "$KNN_SHA" \
        "$SS_DEST/submodules/simple-knn"
fi

# --- 3D-UIR (submodules vendored in-tree upstream) -----------------
UIR_DEST="$PARENT/3D-UIR"
if occupied "$UIR_DEST"; then
    echo "FETCH-SKIP 3D-UIR (exists)"
else
    UIR_SHA=$(head_sha bilityniu/3D-UIR main)
    fetch_github 3D-UIR bilityniu/3D-UIR "$UIR_SHA" "$UIR_DEST"
fi
if [ ! -f "$UIR_DEST/submodules/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp" ]; then
    echo "WARN: 3D-UIR d-g-r glm not vendored upstream — fetching g-truc/glm master"
    GLM_FIX_SHA=$(head_sha g-truc/glm master)
    fetch_github 3duir-dgr-glm g-truc/glm "$GLM_FIX_SHA" \
        "$UIR_DEST/submodules/diff-gaussian-rasterization/third_party/glm"
fi

# --- Depth-Anything-V2 + ViT-L weights (3D-UIR depth dep) ----------
DAV2_DEST="$PARENT/Depth-Anything-V2"
if occupied "$DAV2_DEST"; then
    echo "FETCH-SKIP Depth-Anything-V2 (exists)"
else
    DAV2_SHA=$(head_sha DepthAnything/Depth-Anything-V2 main)
    fetch_github Depth-Anything-V2 DepthAnything/Depth-Anything-V2 "$DAV2_SHA" "$DAV2_DEST"
fi
WEIGHTS="$DAV2_DEST/checkpoints/depth_anything_v2_vitl.pth"
if [ -f "$WEIGHTS" ] && [ "$(stat -c%s "$WEIGHTS")" -gt 1000000000 ]; then
    echo "FETCH-SKIP DAv2 weights (exists, $(du -h "$WEIGHTS" | cut -f1))"
else
    echo "FETCH      DAv2 ViT-L weights (~1.3 G)"
    mkdir -p "$DAV2_DEST/checkpoints"
    curl -fL --retry 3 -o "$WEIGHTS" "$DAV2_WEIGHTS_URL"
fi

echo
echo "Pins recorded in $PINS:"
cat "$PINS" 2>/dev/null || echo "(none — everything pre-existed)"
echo "Fetch finished."
