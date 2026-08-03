<p align="center">
    <h1 align="center">PEARL<br/><sub>Photometric-Error and Attenuation-Regularized Learning for Underwater 3D Gaussian Splatting</sub></h1>
  <p align="center">
    Thanh-Phuc Nguyen
    ·
    Hoang-Dieu Vu
    ·
    Nhat-Minh Hoang
    ·
    Kien Do
    ·
    Huy-Hieu Pham
  </p>
  <p align="center">
    Phenikaa University · FPT Smart Cloud · VinUni-Illinois Smart Health Center · GREEN-X, VinUniversity
  </p>
  <p align="center">
    <b><i>Paper under review</i></b>
  </p>
</p>
<p align="center">
  <img alt="PEARL reconstruction of the Curacao SeaThru-NeRF scene" src="pearl_teaser.jpg" />
</p>

<p align="justify">
<strong>PEARL</strong> is a medium-aware training framework for underwater 3D Gaussian Splatting, built on top of
<a href="https://github.com/water-splatting/water-splatting">WaterSplatting</a> and
<a href="https://docs.nerf.studio/">nerfstudio</a>. Underwater imagery breaks two assumptions of standard 3DGS
training: photometric error is dominated by wavelength-dependent attenuation and backscatter rather than geometry,
and the usual densification heuristics misallocate the Gaussian budget to regions the medium — not the scene —
fails to explain. PEARL addresses both with two principles:
</p>

- **Principle A — Budgeted residual-driven capacity allocation.** Per-Gaussian photometric error is accumulated by
  residual attribution (an EMA over compositing-weight-normalized residuals) and combined with screen-space activity
  into a densification score. Splitting (along the principal axis), duplication, and pruning all operate under a hard
  budget `N_max`, with complexity-based seeding at initialization.
- **Principle B — Medium-consistent supervision.** Relative L1/SSIM reconstruction losses, edge-aware reweighting,
  a view-conditioned medium MLP (backscatter color, backscatter density, attenuation), depth-dependent Beer-Lambert
  attenuation, and attenuation regularization (magnitude, smoothness, R≥G≥B ordering).

Both principles are independent toggles (`enable_error_densification`, `enable_budget_aware_split`,
`enable_complexity_init`, `enable_edge_loss`, `enable_depth_attenuation`) for ablations.

## Installation

Tested with Python 3.12, PyTorch 2.6.0 + CUDA 12.4, and nerfstudio 1.1.4 on Ubuntu 24.04 (RTX 3090).

```bash
# Environment
conda create --name pearl -y python=3.12
conda activate pearl
python -m pip install --upgrade pip
pip install "setuptools<82.0.0"

# PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# tiny-cuda-nn (needs a CUDA toolkit matching your driver)
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

# nerfstudio
pip install nerfstudio==1.1.4
ns-install-cli

# PEARL (the glm submodule is required to build the CUDA rasterizer)
git clone --recursive <repository-url>
cd PEARL
pip install -e . --no-build-isolation
```

## Data Preprocessing

We follow WaterSplatting and recompute camera parameters with distortion correction using
[COLMAP](https://github.com/colmap/colmap)'s `image_undistorter` on the
[SeaThru-NeRF](https://sea-thru-nerf.github.io/) dataset:

```bash
colmap image_undistorter \
  --image_path assets/data/SeathruNeRF_dataset/IUI3-RedSea/images_wb \
  --input_path assets/data/SeathruNeRF_dataset/IUI3-RedSea/colmap/sparse/0 \
  --output_path assets/data/undistorted_seathrunerf_dataset/IUI3-RedSea \
  --output_type COLMAP
```

## Training

```bash
ns-train pearl --vis viewer+wandb --output-dir assets/outputs \
  --data assets/data/undistorted_seathrunerf_dataset/IUI3-RedSea \
  colmap --downscale-factor 1 --colmap-path sparse --images-path images
```

Useful flags:

- `--pipeline.model.max-gaussians` — hard budget `N_max` (default 1,000,000).
- `--pipeline.model.enable-error-densification False` etc. — ablate individual PEARL components.
- `ns-train pearl-big ...` — variant without post-densification culling.

The train/test splits follow the SeaThru-NeRF protocol, not the nerfstudio defaults.

## Evaluation

Everything derived from a checkpoint is written under `assets/results/<scene>/<timestamp>/`,
where `<scene>` and `<timestamp>` are read out of the run's `config.yml`. Our own scripts
resolve that destination themselves; only `ns-eval` needs to be told:

```bash
ns-eval --load-config assets/outputs/unnamed/pearl/<timestamp>/config.yml \
  --output-path        assets/results/<scene>/<timestamp>/eval/metrics.json \
  --render-output-path assets/results/<scene>/<timestamp>/eval
```

To sweep every saved checkpoint of a run and report the best one per metric:

```bash
python scripts/eval_all_ckpts.py --run-dir assets/outputs/unnamed/pearl/<timestamp>
# -> assets/results/<scene>/<timestamp>/ckpt_sweep/
#  + assets/results/<scene>/<timestamp>/best/   (winner as best.ckpt + config + metrics)
```

The sweep exports its winner — the PSNR-maximising checkpoint, which is the protocol
the paper reports — to `best/` under step-free names. nerfstudio finds checkpoints only
by their `step-%09d.ckpt` filename, so PEARL ships three thin wrappers that resolve
`best.ckpt` and otherwise pass every flag straight through to nerfstudio:

| PEARL command | equivalent to |
|---|---|
| `pearl-eval` | `ns-eval` |
| `pearl-viewer` | `ns-viewer` |
| `pearl-render` | `ns-render` |

```bash
pearl-eval --load-config assets/results/<scene>/<timestamp>/best/config.yml \
  --output-path assets/results/<scene>/<timestamp>/eval/best_metrics.json
```

Given any other config they behave exactly like the `ns-*` command they wrap, so they
are safe to use everywhere.

Robustness studies from the paper (pose-noise sensitivity, medium-mismatch stress, Gaussian-budget ablation) live in
`scripts/`; camera-path generation and offline rendering utilities live in `tools/`. The
per-run studies take `--load-config` and write into that run's `analysis/` folder; the
cross-run budget ablation lands in `assets/results/paper/`.

```bash
python scripts/pose_noise_sensitivity.py --load-config assets/outputs/unnamed/pearl/<timestamp>/config.yml
python scripts/pose_noise_draw.py        --load-config assets/outputs/unnamed/pearl/<timestamp>/config.yml
```

## Interactive Viewer

```bash
pearl-viewer --load-config assets/outputs/unnamed/pearl/<timestamp>/config.yml
pearl-viewer --load-config assets/results/<scene>/<timestamp>/best/config.yml   # the sweep winner
```

## Rendering Videos

Needs `ffmpeg` on `PATH` (`conda install -c conda-forge ffmpeg`).

```bash
pearl-render camera-path --load-config assets/outputs/unnamed/pearl/<timestamp>/config.yml \
  --camera-path-filename <camera_path.json> \
  --output-path assets/results/<scene>/<timestamp>/renders/pearl.mp4
```

The offline renderers in `tools/` pick that destination up automatically — drop
`--output-path` and they write into the run's own `renders/` folder:

```bash
python tools/render_seathru_camera_paths.py \
  --load-config assets/outputs/unnamed/pearl/<timestamp>/config.yml --camera-path virtual
```

## Citation

The paper is under review; a citation entry will be added upon publication.

```bibtex
@article{nguyen2026pearl,
  title={{PEARL}: Photometric-Error and Attenuation-Regularized Learning for Underwater {3D} Gaussian Splatting},
  author={Nguyen, Thanh-Phuc and Vu, Hoang-Dieu and Hoang, Nhat-Minh and Do, Kien and Pham, Huy-Hieu},
  year={2026},
  note={Under review}
}
```

## Acknowledgements

This codebase is built upon [WaterSplatting](https://github.com/water-splatting/water-splatting)
(Li et al., 3DV 2025), which combines 3DGS with volume rendering for underwater scenes, and inherits its
gsplat-derived CUDA rasterizer. We thank the authors for releasing their code under Apache-2.0.
We also thank the authors of [SeaThru-NeRF](https://sea-thru-nerf.github.io/) for the dataset.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Original WaterSplatting code
Copyright 2024 Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre Elsig and Jonas Kulhanek;
PEARL modifications Copyright 2026 Thanh-Phuc Nguyen et al.
