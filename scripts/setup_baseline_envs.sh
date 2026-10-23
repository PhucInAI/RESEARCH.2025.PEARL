#!/usr/bin/env bash
# ======================================================================
# One-time baseline setup — run on the CLUSTER LOGIN node (internet +
# conda required for pip, no GPU needed):
#
#     bash scripts/setup_baseline_envs.sh [ws|seasplat|3duir|all]
#
# Per baseline: an isolated conda env built against the pinned snapshot
# repos that live as SIBLINGS of the PEARL root (fetched locally by
# scripts/fetch_baselines.sh, shipped by scripts/push_to_cluster.sh;
# pins in ../BASELINE_PINS.txt — ADR-013). Idempotent — existing envs
# are reused and finished steps skipped, but a torch whose build no
# longer matches the pin below IS reinstalled over the old one.
#
# Separate envs are mandatory: SeaSplat and 3D-UIR ship different forks
# of the same `diff_gaussian_rasterization` package name, and the ws env
# must not touch the production `pearl` env it mirrors.
#
# Stacks — the target cluster's system CUDA is 12.2 (`/usr/local/cuda`,
# no cuda modulefiles), so every torch CUDA build here is <= 12.2 and
# the extensions compile against that system nvcc with no conda
# cuda-toolkit download (ADR-014). 3D-UIR is the exception: torch
# refuses a CUDA *major* mismatch, so its cu118 stack pulls
# cuda-toolkit 11.8 into its own env.
#   wsplat   CLONE of the production `pearl` env + water-splatting
#   seasplat py3.10  torch 2.1.0+cu121  d-g-r + simple-knn   (upstream README)
#   3duir    py3.10  torch 2.1.0+cu118  d-g-r + simple-knn + fused-ssim + tcnn
# The WS env is a clone, not a rebuild (ADR-015): WS and PEARL share the
# nerfstudio backbone, so one stack keeps the efficiency numbers directly
# comparable — and a from-scratch build is no longer reproducible anyway.
# Its torch must be >= 2.4 (nerfstudio calls `@torch_compile()` at import
# and Dynamo rejects python 3.12 below that — KI-010), yet every torch in
# 2.4–2.6 pins `nvidia-cudnn-cu12==9.1.0.70`, which PyPI and the pytorch
# index have both stopped serving (KI-008 again). The `pearl` env still
# holds that wheel, so cloning it is the only offline-exact route.
# ======================================================================
# Fail-fast on purpose (unlike the sbatch runners' `set -uo`): a
# half-built env is worse than a stopped setup script.
set -euo pipefail

ROOT_DIR="${PEARL_ROOT:-$PWD}"
BASELINES_SRC="${BASELINES_SRC:-$(dirname "$ROOT_DIR")}"
WS_ENV="${WS_ENV:-wsplat}"
PEARL_ENV="${PEARL_CONDA_ENV:-pearl}"
SEASPLAT_ENV="${SEASPLAT_ENV:-seasplat}"
UIR_ENV="${UIR_ENV:-uir3d}"
TARGET="${1:-all}"

# setuptools >= 82 removed `pkg_resources`, which tiny-cuda-nn's setup.py
# and torch 2.1's cpp_extension both import by name — the
# `ModuleNotFoundError: No module named 'pkg_resources'` that killed the
# first H100 setup run (KI-009). 69.5.1 is the last release that also
# still exposes `pkg_resources.packaging`, which torch 2.1 needs.
BUILD_TOOLS=(setuptools==69.5.1 wheel ninja)

# A100 (sm80) plus H100 headroom (sm90); also what lets CUDA extensions
# build on the GPU-less login node, where torch/tcnn autodetection fails.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-80;90}"
export MAX_JOBS="${MAX_JOBS:-8}"

CONDA_PROFILE="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_PROFILE" ]]; then
    echo "Conda profile not found: $CONDA_PROFILE — set CONDA_SH." >&2
    exit 1
fi
# nerfstudio's conda hooks read $ZSH_VERSION unguarded — suspend nounset
set +u
source "$CONDA_PROFILE"
set -u

# --- helpers -------------------------------------------------------
have_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }

require_repo() {
    # require_repo <dir> — the pinned snapshot must have been synced up
    if [ ! -d "$1" ]; then
        echo "Missing $1 — run scripts/fetch_baselines.sh locally, then" >&2
        echo "scripts/push_to_cluster.sh, before this setup." >&2
        exit 1
    fi
}

pin_build_tools() {
    # Pin the from-source build toolchain of the ACTIVE env. Called again
    # after any large pip install that could drag setuptools forward.
    python -m pip install --upgrade pip
    pip install "${BUILD_TOOLS[@]}"
}

find_nvcc() {
    # find_nvcc <cuda major> — echo the CUDA_HOME of an nvcc with that
    # major version, or nothing. torch errors out on a CUDA *major*
    # mismatch against the wheel and only warns on a minor one, so the
    # system 12.2 toolkit is a valid builder for every cu12x stack.
    local want="$1" cand ver
    for cand in "${CUDA_HOME:-}/bin/nvcc" $(command -v nvcc || true) \
                /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        [ -x "$cand" ] || continue
        ver=$("$cand" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
        [ "$ver" = "$want" ] && { echo "${cand%/bin/nvcc}"; return 0; }
    done
    return 0   # "not found" is a normal answer here, not a `set -e` abort
}

ensure_nvcc() {
    # ensure_nvcc <cuda major> <conda label e.g. 12.1.0> — system toolkit
    # first, then a module, then a conda cuda-toolkit (~2 G in the env).
    local want="$1" label="$2" home
    home=$(find_nvcc "$want")
    if [ -z "$home" ] && type module >/dev/null 2>&1; then
        module load "cuda/${label%.*}" 2>/dev/null || true   # EDIT if your module names differ
        home=$(find_nvcc "$want")
    fi
    if [ -z "$home" ]; then
        echo "No CUDA $want toolkit on this node — installing cuda-toolkit $label into the active env"
        conda install -y -c "nvidia/label/cuda-$label" cuda-toolkit
        home=$(find_nvcc "$want")
    fi
    [ -n "$home" ] || { echo "No CUDA $want toolkit — cannot build extensions." >&2; exit 1; }
    export CUDA_HOME="$home"
    export PATH="$home/bin:$PATH"
    export LD_LIBRARY_PATH="$home/lib64:${LD_LIBRARY_PATH:-}"
    nvcc --version | tail -2
}

ensure_torch() {
    # ensure_torch <expected torch.__version__ prefix> <pip args…> —
    # reinstalls when the env carries a different build (the H100 ws env
    # was first built against cu124, which overshoots the system 12.2).
    local want="$1"; shift
    if python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$want') else 1)" 2>/dev/null; then
        echo "TORCH-SKIP $want (already installed)"
    else
        pip install "$@"
    fi
}

fetch_tcnn() {
    # echo the path of a tiny-cuda-nn torch-bindings source tree, cloning it
    # next to the baseline snapshots on first use. A codeload tarball will
    # not do — tcnn needs its git submodules (cutlass, fmt, cmrc).
    local dir="$BASELINES_SRC/tiny-cuda-nn"
    [ -d "$dir" ] || git clone --recursive --depth 1 \
        https://github.com/NVlabs/tiny-cuda-nn.git "$dir" >&2
    echo "$dir/bindings/torch"
}

build_tcnn() {
    # build_tcnn <setup target> — REFUSE to build without a visible driver.
    # A binding compiled on the GPU-less login node links against the libcuda
    # stub and dies at import on the compute node with
    # `undefined symbol: cuMemAddressFree` — which is how a whole baseline
    # wave got to the queue with an unusable env (KI-014).
    if ! nvidia-smi -L >/dev/null 2>&1; then
        echo "No GPU visible — tiny-cuda-nn must be built on a compute node:" >&2
        echo "  srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:30:00 \\" >&2
        echo "       bash scripts/setup_baseline_envs.sh $1" >&2
        exit 1
    fi
    pip install --no-build-isolation --no-deps "$(fetch_tcnn)"
}

patch_cstdint() {
    # patch_cstdint <d-g-r dir> — GCC>=13 needs <cstdint> in the CUDA
    # rasterizer (fix recorded in the 3D-UIR reproduction's note.txt).
    # Guarded insert before the first #include; a no-op when present.
    local f="$1/cuda_rasterizer/rasterizer_impl.cu"
    if [ -f "$f" ] && ! grep -q '#include <cstdint>' "$f"; then
        sed -i '0,/^#include/s//#include <cstdint>\n&/' "$f"
        echo "PATCHED $f (+#include <cstdint>)"
    fi
}

smoke() {
    # smoke <env> <python -c payload> — non-fatal (login nodes may lack
    # libcuda), but ALWAYS print the traceback: discarding it is what let a
    # py3.12 Dynamo crash and a numpy-ABI crash pass as node artifacts and
    # reach the batch queue (KI-010, KI-011).
    local env="$1" payload="$2" out
    if out=$(conda run -n "$env" python -c "$payload" 2>&1); then
        echo "SMOKE-OK   $env"
    else
        echo "SMOKE-WARN $env — import failed on this node:"
        echo "$out" | tail -5 | sed 's/^/           /'
        echo "           re-check: srun --gres=gpu:1 conda run -n $env python -c '$payload'"
    fi
}

check_cli() {
    # check_cli <label> <env> <cmd…> — the method's own CLI must load; a
    # failure prints its error tail for the same reason smoke() does.
    local label="$1" env="$2" out
    shift 2
    if out=$(conda run -n "$env" "$@" 2>&1); then
        echo "SETUP-OK $label"
    else
        echo "SETUP-WARN $label — CLI check failed:"
        echo "$out" | tail -5 | sed 's/^/           /'
    fi
}

# --- water-splatting (env: $WS_ENV) --------------------------------
setup_ws() {
    # A clone of the production `pearl` env, minus the pearl plugin, plus
    # water-splatting — see the header for why this is not a build (ADR-015).
    # tiny-cuda-nn is deliberately absent: the pearl stack does without it
    # and water_splatting never imports it.
    echo "=== [ws] env $WS_ENV = clone of $PEARL_ENV + water-splatting ==="
    if have_env "$WS_ENV"; then
        echo "ENV-SKIP $WS_ENV (exists)"
    else
        have_env "$PEARL_ENV" || {
            echo "No conda env '$PEARL_ENV' to clone — create it per the PEARL" >&2
            echo "README first, or point PEARL_CONDA_ENV at an equivalent env." >&2
            exit 1
        }
        conda create -y --name "$WS_ENV" --clone "$PEARL_ENV"
    fi
    set +u; conda activate "$WS_ENV"; set -u
    # `conda create --clone` re-materialises conda-owned files over the ones
    # pip put in <env>/lib: MKL's core loses `iJIT_NotifyEvent` and libiomp5
    # /libtbb come back as stubs, so `import torch` dies with an undefined
    # symbol or `Intel MKL FATAL ERROR: Cannot load <mkl-loader>` (KI-013).
    # Re-assert the wheels; the versions are the source env's own.
    pip install --force-reinstall --no-deps \
        mkl==2024.0.0 intel-openmp==2024.2.2 intel-cmplr-lib-ur==2024.2.2 tbb==2021.13.1
    # The clone registers PEARL as a nerfstudio method; the baseline must
    # run on upstream code alone. Removing the editable install touches
    # only this env's site-packages, never the PEARL source tree.
    if python -c "import pearl" 2>/dev/null; then pip uninstall -y pearl; fi
    require_repo "$BASELINES_SRC/water-splatting"
    ensure_nvcc 12 12.1.0   # water_splatting ships its own CUDA rasterizer
    python -c "import water_splatting" 2>/dev/null || \
        pip install -e "$BASELINES_SRC/water-splatting" --no-deps --no-build-isolation
    set +u; conda deactivate; set -u
    smoke "$WS_ENV" "import torch, nerfstudio, water_splatting"
    check_cli ws "$WS_ENV" ns-train water-splatting --help
}

# --- seasplat (env: $SEASPLAT_ENV) ---------------------------------
setup_seasplat() {
    echo "=== [seasplat] env $SEASPLAT_ENV + seasplat ==="
    if have_env "$SEASPLAT_ENV"; then
        echo "ENV-SKIP $SEASPLAT_ENV (exists)"
    else
        conda create -y --name "$SEASPLAT_ENV" python=3.10
    fi
    set +u; conda activate "$SEASPLAT_ENV"; set -u
    pin_build_tools
    ensure_torch 2.1.0+cu121 torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121
    require_repo "$BASELINES_SRC/seasplat"
    # Upstream requirements.txt is a full `pip freeze` of the authors' env:
    # it pins `torch==2.0.1+cu118`, a local version that PyPI does not
    # serve and that contradicts the repo's own README torch line, so
    # installing it verbatim aborts the whole file. Install the upstream
    # pins of exactly what seasplat imports instead (deepseecolor and
    # lpipsPyTorch are vendored in the repo, not pip packages).
    pip install numpy==1.26.4 Pillow==9.5.0 plyfile==0.8.1 tqdm==4.66.5 \
        jaxtyping==0.2.33 typeguard==2.13.3 kornia==0.7.3 matplotlib==3.5.3 \
        splines==0.3.0 tensorboard==2.17.1
    ensure_nvcc 12 12.1.0
    patch_cstdint "$BASELINES_SRC/seasplat/submodules/diff-gaussian-rasterization"
    python -c "import diff_gaussian_rasterization" 2>/dev/null || \
        pip install "$BASELINES_SRC/seasplat/submodules/diff-gaussian-rasterization" --no-build-isolation
    python -c "import simple_knn" 2>/dev/null || \
        pip install "$BASELINES_SRC/seasplat/submodules/simple-knn" --no-build-isolation
    set +u; conda deactivate; set -u
    smoke "$SEASPLAT_ENV" "import torch, diff_gaussian_rasterization, simple_knn, kornia, splines"
    check_cli seasplat "$SEASPLAT_ENV" python "$BASELINES_SRC/seasplat/train.py" -h
}

# --- 3D-UIR (env: $UIR_ENV) ----------------------------------------
setup_3duir() {
    echo "=== [3duir] env $UIR_ENV + 3D-UIR ==="
    if have_env "$UIR_ENV"; then
        echo "ENV-SKIP $UIR_ENV (exists)"
    else
        conda create -y --name "$UIR_ENV" python=3.10
    fi
    set +u; conda activate "$UIR_ENV"; set -u
    pin_build_tools
    ensure_torch 2.1.0+cu118 torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118
    require_repo "$BASELINES_SRC/3D-UIR"
    require_repo "$BASELINES_SRC/Depth-Anything-V2"
    # Upstream README names a requirements.txt the repo does not ship and
    # environment.yml lists only four pip deps; train.py additionally
    # needs lpips + kornia (utils/loss_utils.py, utils/general_utils.py)
    # and matplotlib, which Depth-Anything-V2's run.py also imports.
    if [ -f "$BASELINES_SRC/3D-UIR/requirements.txt" ]; then
        pip install -r "$BASELINES_SRC/3D-UIR/requirements.txt"
    else
        pip install tqdm joblib lpips kornia matplotlib scipy
    fi
    # Asserted last so it wins in both branches: torch 2.1 is built against
    # the numpy 1.x C-ABI, and an unpinned resolve installs numpy 2.x plus
    # an opencv/plyfile pair that requires it. The first numpy->tensor
    # crossing then raises `RuntimeError: Numpy is not available`, which is
    # where the whole DAv2 depth pass died (KI-011).
    pip install "numpy==1.26.4" "opencv-python==4.10.0.84" "plyfile==1.0.3"
    ensure_nvcc 11 11.8.0
    patch_cstdint "$BASELINES_SRC/3D-UIR/submodules/diff-gaussian-rasterization"
    python -c "import diff_gaussian_rasterization" 2>/dev/null || \
        pip install "$BASELINES_SRC/3D-UIR/submodules/diff-gaussian-rasterization" --no-build-isolation
    python -c "import simple_knn" 2>/dev/null || \
        pip install "$BASELINES_SRC/3D-UIR/submodules/simple-knn" --no-build-isolation
    python -c "import fused_ssim" 2>/dev/null || \
        pip install "$BASELINES_SRC/3D-UIR/submodules/fused-ssim" --no-build-isolation
    # Mandatory here: scene/appearace_model.py imports tcnn at module load.
    python -c "import tinycudann" 2>/dev/null || build_tcnn 3duir
    set +u; conda deactivate; set -u
    smoke "$UIR_ENV" "import numpy, cv2, torch, diff_gaussian_rasterization, simple_knn, fused_ssim, tinycudann, lpips, kornia; torch.from_numpy(numpy.zeros(1))"
    check_cli 3duir "$UIR_ENV" python "$BASELINES_SRC/3D-UIR/train.py" -h
}

# --- dispatch ------------------------------------------------------
# Subshells so one env's CUDA_HOME/PATH exports never leak into the next.
case "$TARGET" in
    ws)       (setup_ws) ;;
    seasplat) (setup_seasplat) ;;
    3duir)    (setup_3duir) ;;
    all)      (setup_ws); (setup_seasplat); (setup_3duir) ;;
    *) echo "Usage: bash scripts/setup_baseline_envs.sh [ws|seasplat|3duir|all]" >&2; exit 1 ;;
esac

echo "Setup finished for: $TARGET"
