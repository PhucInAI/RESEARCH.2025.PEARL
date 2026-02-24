# ruff: noqa: E741
# Copyright 2024 Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre Elsig and Jonas KulhanekS. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Python package for combining 3DGS with volume rendering to enable water/fog modeling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from ._torch_impl import quat_to_rotmat
from .project_gaussians import project_gaussians
from .rasterize import rasterize_gaussians
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects
from .sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal
import torch.nn.functional as F

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import SHEncoding


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


@dataclass
class PearlModelConfig(ModelConfig):
    """PEARL model config."""

    _target: Type = field(default_factory=lambda: PearlModel)
    num_steps: int = 15000
    """Number of steps to train the model"""
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    num_downscales: int = 1
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.005
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_alpha_thresh_post: float = 0.005
    """threshold of opacity for post culling gaussians"""
    reset_alpha_thresh: float = 0.5
    """threshold of opacity for resetting alpha"""
    cull_scale_thresh: float = 10.
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    zero_medium: bool = False
    """If True, zero out the medium field"""
    reset_alpha_every: int = 5
    """Every this many refinement steps, reset the alpha"""
    abs_grad_densification: bool = True
    """If True, use absolute gradient for densification"""
    densify_grad_thresh: float = 0.0004
    """threshold of positional gradient norm for densifying gaussians (0.0004, 0.0008)"""
    densify_size_thresh: float = 0.001
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    clip_thresh: float = 0.01
    """minimum depth threshold"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 0
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.25
    """weight of ssim loss"""
    main_loss: Literal["l1", "reg_l1", "reg_l2"] = "reg_l1"
    """main loss to use"""
    ssim_loss: Literal["reg_ssim", "ssim"] = "reg_ssim"
    """ssim loss to use"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    num_layers_medium: int = 3
    """Number of hidden layers for medium MLP."""
    hidden_dim_medium: int = 256
    """Dimension of hidden layers for medium MLP."""
    medium_density_bias: float = 0.0
    """Bias for medium density (sigma_bs and sigma_attn)."""
    mlp_type: Literal["tcnn", "torch"] = "tcnn"
    """Type of MLP to use for medium MLP."""

    # ------------------------------------------------------------------
    # PEARL modules (independent toggles)
    # ------------------------------------------------------------------

    enable_error_densification: bool = True
    """Module-1: use per-Gaussian photometric error (from image residuals) to guide densification."""

    enable_budget_aware_split: bool = True
    """Module-2: enable budget-aware densification (max_gaussians cap) and optional steepest-direction splitting."""

    enable_complexity_init: bool = True
    """Module-3: enable complexity-based / depth-guided extra Gaussian seeding at initialization."""

    enable_edge_loss: bool = True
    """Module-4: enable edge-aware image loss weighting (edge-preserving supervision)."""

    enable_depth_attenuation: bool = True
    """Module-5: enable depth-dependent color attenuation on object colors (RGB Beer-Lambert style)."""

    # ---- Training schedules ----
    depth_attenuation_warmup_steps: int = 2000
    """Disable depth attenuation for the first N training steps."""

    edge_loss_warmup_steps: int = 2000
    """Linearly ramp edge_loss_weight from 0 to its configured value over this many steps."""

    densify_error_warmup_steps: int = 2000
    """Linearly anneal effective densify_error_thresh from 2× to 1× its value over this many steps."""

    # ------------------------------------------------------------------
    # New configuration parameters for research-based improvements
    #
    densify_error_thresh: float = 0.0020
    """Threshold on per‑Gaussian photometric error for densification. Error above this triggers splits/dups."""

    max_gaussians: Optional[int] = 1000000
    """Maximum allowed number of Gaussians N_max. Density control projects split/dup operations to satisfy the cap (Eq. 15) and prunes by utility if still above the cap (Eq. 17)."""

    edge_loss_weight: float = 0.05
    """Weight for edge‑aware loss; if >0, errors near edges are upweighted by (1 + edge_loss_weight * gradient magnitude)."""

    use_depth_attenuation: bool = True
    """If True, apply a depth‑dependent color attenuation model to object colors before rasterization."""

    attenuation_coeff_init: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial attenuation coefficients (R,G,B) for depth‑dependent color attenuation. Only used if use_depth_attenuation=True."""

    # Complexity-based / depth-guided initialization
    complexity_init_factor: float = 0.0
    """
    Fraction of additional Gaussians (relative to initial count) to seed
    at initialization in locally complex regions. 0.0 disables this path.
    """

    complexity_init_k: int = 8
    """k for k-NN used to estimate local geometric complexity at init."""

    complexity_depth_weight: float = 0.0
    """
    If > 0, bias complexity scores toward points that are further along
    the scene radius (approximate depth); useful if far regions tend to
    be under-covered.
    """

    # Steepest-direction splitting
    use_steepest_split: bool = True
    """
    If True, split Gaussians along their principal axis (largest scale
    in world space) instead of random offsets in the ellipsoid.
    """

    # ------------------------------------------------------------------
    # Principle A: hard-budget density control (Eq. 11-17)
    # ------------------------------------------------------------------

    densify_h_thresh: float = 0.0008
    """Threshold on densification score h_i (Eq. 13) to form split/dup candidates (Eq. 16).

    Practical note: set this on the same scale as densify_grad_thresh; by default we mirror the 3DGS gradient threshold.
    """

    lambda_E: float = 1.0
    """λ_E in Eq. (13), controlling how strongly persistent error biases densification."""

    lambda_U: float = 1.0
    """λ_U in Eq. (17), controlling how strongly persistent error biases utility pruning when above N_max."""

    error_ema_eta: float = 0.95
    """EMA coefficient η for persistent error (Eq. 12). Larger = slower updates."""

    error_attr_every: int = 5
    """Update residual-attributed error stats every this many training steps (performance control)."""

    error_attr_num_pixels: int = 4096
    """Number of pixels |Ω_t| sampled for residual attribution when updating E_i^(t)."""

    error_attr_topk: int = 32
    """Max number of Gaussians per pixel used for approximate compositing weights in residual attribution."""

    error_attr_alpha_thresh: float = 1e-4
    """Ignore very small per-pixel alpha contributions when attributing residuals."""

    # ------------------------------------------------------------------
    # Principle B: medium-consistent supervision (Eq. 24-28)
    # ------------------------------------------------------------------

    enable_attn_regularization: bool = True
    """If True, apply attenuation regularization L_attn (magnitude/smoothness/ordering)."""

    attn_loss_weight: float = 0.01
    """λ_attn in Eq. (28)."""

    attn_prior_beta: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Weak prior β_0 (dataset mean) in L_attn-mag."""

    attn_lambda_beta: float = 1.0
    attn_lambda_smooth: float = 1.0
    attn_lambda_ord: float = 1.0



class PearlModel(Model):
    """
    Args:
        config: PEARL configuration to instantiate model
    """

    config: PearlModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)
        # Storage for last predicted and ground truth images (used for error-based densification / debugging)
        self.last_pred_image: Optional[torch.Tensor] = None
        self.last_gt_image: Optional[torch.Tensor] = None

        # Per-Gaussian persistent error statistics (Eq. 12 in the paper)
        self.gaussian_error_ema: Optional[torch.Tensor] = None
        self.gaussian_activity: Optional[torch.Tensor] = None  # g_i, updated from screen-space grads

        # Cached projection/rasterization inputs for residual attribution (updated in get_outputs)
        self._proj_conics: Optional[torch.Tensor] = None
        self._proj_depths: Optional[torch.Tensor] = None
        self._proj_opacity: Optional[torch.Tensor] = None
        self._proj_num_tiles_hit: Optional[torch.Tensor] = None
        self._proj_block_width: Optional[int] = None
        self._proj_hw: Optional[Tuple[int, int]] = None
        self._proj_gauss_indices: Optional[torch.Tensor] = None  # maps projected tensors -> full gaussian indices

        self._dir_cache = {}  # key -> [H,W,3] unit dirs in camera coords

    def _get_dirs_cam_unit(self, H, W, fx, fy, cx, cy, device, dtype):
        key = (H, W, float(fx), float(fy), float(cx), float(cy), str(device), str(dtype))
        if key in self._dir_cache:
            return self._dir_cache[key]

        y = torch.arange(H, device=device, dtype=dtype)
        x = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")          # (H,W)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        dirs = torch.stack([xx, yy, torch.ones_like(xx)], -1) # (H,W,3)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
        self._dir_cache[key] = dirs
        return dirs

    def populate_modules(self):
        # initialize the medium MLP
        self.direction_encoding = SHEncoding(levels=4, implementation="tcnn")
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()
        # Medium MLP
        # Extract number of layers and hidden dimension from config (handles simple int values)
        num_layers_medium = int(self.config.num_layers_medium)
        hidden_dim_medium = int(self.config.hidden_dim_medium)
        # Medium density bias is a scalar float controlling initial density; store it on model
        self.medium_density_bias = float(self.config.medium_density_bias)
        # ------------------------Medium network------------------------
        # Medium MLP
        if num_layers_medium > 1:
            self.medium_mlp = MLP(
                in_dim=self.direction_encoding.get_out_dim(),
                num_layers=num_layers_medium,
                layer_width=hidden_dim_medium,
                out_dim=9,
                activation=nn.Sigmoid(),
                out_activation=None,
                implementation=self.config.mlp_type,
            )
        else:
            self.medium_mlp = nn.Linear(self.direction_encoding.get_out_dim(), 9)
            self.config.mlp_type = "torch"

        # ------------------------ Gaussian initialization ------------------------
        # Base means: SFM points if provided, otherwise random in a cube
        if self.seed_points is not None and not self.config.random_init:
            base_means = self.seed_points[0]  # [N0, 3]
        else:
            base_means = (
                (torch.rand((self.config.num_random, 3)) - 0.5)
                * self.config.random_scale
            )

        # Optionally: complexity-based / depth-guided extra seeding at initialization
        means_tensor = base_means
        if (
            self.config.enable_complexity_init
            and self.config.complexity_init_factor > 0.0
            and means_tensor.shape[0] > self.config.complexity_init_k
        ):
            # k-NN distances on the base set (CPU for sklearn)
            with torch.no_grad():
                dists_np, _ = self.k_nearest_sklearn(
                    means_tensor.detach().cpu(), self.config.complexity_init_k
                )
            dists = torch.from_numpy(dists_np).to(self.device)  # [N0, K]
            local_scale = dists.mean(dim=-1)  # [N0]

            # Complexity ~ inverse local scale (denser regions are "complex")
            complexity = 1.0 / (local_scale + 1e-6)

            # Optional depth bias: approximate depth by radius from origin
            if self.config.complexity_depth_weight > 0.0:
                depth = means_tensor.norm(dim=-1)  # [N0]
                depth_norm = depth / (depth.mean() + 1e-6)
                complexity = complexity * (
                    1.0 + self.config.complexity_depth_weight * depth_norm
                )

            # How many extra points to seed
            n_extra = int(self.config.complexity_init_factor * means_tensor.shape[0])
            n_extra = max(0, min(n_extra, means_tensor.shape[0]))
            if n_extra > 0:
                topk_idx = torch.topk(complexity, k=n_extra, largest=True).indices  # [n_extra]

                # Jitter new points around complex ones with std ~ local_scale
                jitter_std = 0.5 * local_scale[topk_idx].unsqueeze(-1)  # [n_extra, 1]
                noise = torch.randn_like(means_tensor[topk_idx]) * jitter_std
                extra_means = means_tensor[topk_idx] + noise

                means_tensor = torch.cat([means_tensor, extra_means], dim=0)

        # Register as Parameter
        means = torch.nn.Parameter(means_tensor)
        self.xys_grad_norm = None
        self.max_2Dsize = None

        # k-NN based scale initialization on the (possibly) augmented set
        dists_np, _ = self.k_nearest_sklearn(means.detach().cpu(), 3)
        distances = torch.from_numpy(dists_np)  # [N, 3]
        # Global scale initialization is identical to original code
        scale_init = torch.log(
            torch.clamp(torch.tensor(distances.mean()), min=1e-2)
        )
        scales = torch.nn.Parameter(
            torch.ones((means.shape[0], 3)) * scale_init
        )
        num_points = means.shape[0]
        quats = random_quat_tensor(means.shape[0])
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": torch.nn.Parameter(quats),
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )
        # Initialize depth attenuation coefficients (always created to keep optimizer groups stable).
        # If depth attenuation is disabled (Module-5 off), we do not apply it in rendering.
        init_coeffs = torch.tensor(
            self.config.attenuation_coeff_init, dtype=torch.float32
        )
        # Always register attenuation_coeff as a Parameter so the optimizer group is never empty.
        # When depth attenuation is disabled (Module-5 off), this parameter is simply unused in rendering
        # and will therefore receive no gradients.
        self.attenuation_coeff = torch.nn.Parameter(init_coeffs)

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    # NOTE: medium_mlp and direction_encoding are normal nn.Modules defined
    # in populate_modules(). We intentionally do not override them as
    # properties that access gauss_params, since they are not part of
    # the Gaussian parameter set. Leaving them as regular attributes
    # allows them to be optimized separately and referenced directly.

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = self.config.num_steps
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        # if self.step >= self.config.stop_split_at:
        #     return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            if self.config.abs_grad_densification:
                assert self.xys_grad_abs is not None
                grads = self.xys_grad_abs.detach().norm(dim=-1)
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.depths_accum = self.depths
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]
                self.depths_accum[visible_mask] = self.depths[visible_mask] + self.depths_accum[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def _attenuation_regularization(self, beta: torch.Tensor) -> torch.Tensor:
        """Attenuation regularization (page 7): L_attn-mag + L_attn-smooth + L_attn-ord.

        Note: The paper defines smoothness on a viewing-sphere neighborhood. Here we use an image-grid neighborhood
        over the discretized ray directions (equivalent to a local neighborhood in direction space for small FoV).
        """
        beta0 = torch.tensor(
            getattr(self.config, 'attn_prior_beta', (0.0, 0.0, 0.0)),
            device=beta.device,
            dtype=beta.dtype,
        ).view(1, 1, 3)

        l_mag = ((beta - beta0) ** 2).sum(dim=-1).mean()

        # 4-neighbor smoothness on the ray-direction grid
        dy = (beta[1:, :, :] - beta[:-1, :, :]) ** 2
        dx = (beta[:, 1:, :] - beta[:, :-1, :]) ** 2
        l_smooth = dy.sum(dim=-1).mean() + dx.sum(dim=-1).mean()

        # Enforce common underwater absorption ordering: beta_R >= beta_G >= beta_B
        l_ord = (F.relu(beta[..., 1] - beta[..., 0]) + F.relu(beta[..., 2] - beta[..., 1])).mean()

        lam_beta = float(getattr(self.config, 'attn_lambda_beta', 1.0))
        lam_smooth = float(getattr(self.config, 'attn_lambda_smooth', 1.0))
        lam_ord = float(getattr(self.config, 'attn_lambda_ord', 1.0))
        return lam_beta * l_mag + lam_smooth * l_smooth + lam_ord * l_ord


    @torch.no_grad()
    def _update_gaussian_error_ema(self, err_img: torch.Tensor) -> None:
        """Update per-Gaussian error EMA using residual attribution (Eq. 11-12).

        err_img: (H,W) scalar L1 residual per pixel.
        """
        if self._proj_conics is None or self._proj_depths is None or self._proj_opacity is None or self._proj_num_tiles_hit is None:
            return
        if self._proj_block_width is None or self._proj_hw is None:
            return

        H, W = err_img.shape
        block_width = int(self._proj_block_width)
        tile_bounds = ((W + block_width - 1) // block_width, (H + block_width - 1) // block_width, 1)

        # Ensure persistent buffers exist for the full Gaussian set
        if self.gaussian_error_ema is None or self.gaussian_error_ema.shape[0] != self.num_points:
            new_ema = torch.zeros((self.num_points,), device=self.device, dtype=torch.float32)
            if self.gaussian_error_ema is not None:
                n_old = min(int(self.gaussian_error_ema.shape[0]), int(self.num_points))
                new_ema[:n_old] = self.gaussian_error_ema[:n_old]
            self.gaussian_error_ema = new_ema

        # Map projected indices to the full Gaussian indices
        num_proj = int(self._proj_conics.shape[0])
        if self._proj_gauss_indices is None:
            full_idx_map = torch.arange(num_proj, device=self.device, dtype=torch.long)
        else:
            full_idx_map = self._proj_gauss_indices.to(self.device).long()

        # Sample pixels Ω_t proportional to residual magnitude
        n_samp = int(getattr(self.config, 'error_attr_num_pixels', 2048))
        n_samp = max(1, min(n_samp, H * W))
        flat_err = err_img.reshape(-1)
        probs = (flat_err + 1e-6)
        probs = probs / probs.sum()
        pix = torch.multinomial(probs, n_samp, replacement=False)
        ys = (pix // W).long()
        xs = (pix % W).long()

        # Build tile bins (same mapping as rasterizer)
        num_intersects, cum_tiles_hit = compute_cumulative_intersects(self._proj_num_tiles_hit.int())
        if num_intersects < 1:
            return
        (_, _, _, gaussian_ids_sorted, tile_bins) = bin_and_sort_gaussians(
            num_proj,
            num_intersects,
            self.xys.detach(),
            self._proj_depths.detach(),
            self.radii.detach(),
            cum_tiles_hit,
            tile_bounds,
            block_width,
        )

        gaussian_ids_sorted = gaussian_ids_sorted.squeeze(-1).long()
        tile_bins_cpu = tile_bins.long().detach()

        xys = self.xys.detach()
        conics = self._proj_conics.detach()
        depths = self._proj_depths.detach().squeeze(-1)
        opacity = self._proj_opacity.detach().squeeze(-1)

        topk = int(getattr(self.config, 'error_attr_topk', 16))
        alpha_thresh = float(getattr(self.config, 'error_attr_alpha_thresh', 1e-4))

        num_acc = torch.zeros_like(self.gaussian_error_ema)
        den_acc = torch.zeros_like(self.gaussian_error_ema)

        for y, x in zip(ys.tolist(), xs.tolist()):
            tile_x = x // block_width
            tile_y = y // block_width
            tile_id = tile_y * tile_bounds[0] + tile_x
            if tile_id < 0 or tile_id >= tile_bins_cpu.shape[0]:
                continue
            start, end = tile_bins_cpu[tile_id]
            if end <= start:
                continue
            cand = gaussian_ids_sorted[int(start) : int(end)]
            if cand.numel() == 0:
                continue

            xy = xys[cand]
            dx = xy[:, 0] - float(x)
            dy = xy[:, 1] - float(y)
            c = conics[cand]
            q = c[:, 0] * dx * dx + 2.0 * c[:, 1] * dx * dy + c[:, 2] * dy * dy
            a = opacity[cand] * torch.exp(-0.5 * q)
            m = a > alpha_thresh
            if not bool(m.any()):
                continue
            cand = cand[m]
            a = a[m]

            if cand.numel() > topk:
                a, idx = torch.topk(a, k=topk, largest=True)
                cand = cand[idx]

            d = depths[cand]
            order = torch.argsort(d)
            cand = cand[order]
            a = a[order].clamp(0.0, 0.999)

            one_minus = (1.0 - a)
            T_prev = torch.cumprod(torch.cat([a.new_ones(1), one_minus[:-1]]), dim=0)
            w = a * T_prev
            sumw = w.sum() + 1e-6
            r = w / sumw

            e = err_img[y, x]
            full_cand = full_idx_map[cand]
            num_acc.index_add_(0, full_cand, r * e)
            den_acc.index_add_(0, full_cand, r)

        mask = den_acc > 0
        if not bool(mask.any()):
            return
        E_inst = torch.zeros_like(self.gaussian_error_ema)
        E_inst[mask] = num_acc[mask] / (den_acc[mask] + 1e-6)

        eta = float(getattr(self.config, 'error_ema_eta', 0.95))
        self.gaussian_error_ema[mask] = eta * self.gaussian_error_ema[mask] + (1.0 - eta) * E_inst[mask]


    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and (self.step % reset_interval > self.num_train_data + self.config.refine_every)
            )
            deleted_masks = []
            if do_densification:
                # Activity statistic g_i from screen-space gradients (used in h_i and utility u_i).
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                g_i = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                g_i = g_i.squeeze().detach().float()

                # Ensure persistent buffers exist and have current size
                if self.gaussian_activity is None or self.gaussian_activity.shape[0] != self.num_points:
                    self.gaussian_activity = torch.zeros((self.num_points,), device=self.device, dtype=torch.float32)
                self.gaussian_activity[:] = g_i

                if self.gaussian_error_ema is None or self.gaussian_error_ema.shape[0] != self.num_points:
                    new_ema = torch.zeros((self.num_points,), device=self.device, dtype=torch.float32)
                    if self.gaussian_error_ema is not None:
                        n_old = min(int(self.gaussian_error_ema.shape[0]), int(self.num_points))
                        new_ema[:n_old] = self.gaussian_error_ema[:n_old]
                    self.gaussian_error_ema = new_ema

                Ebar = self.gaussian_error_ema
                active = (self.vis_counts > 0).squeeze()
                Ebar_mean = Ebar[active].mean() if bool(active.any()) else Ebar.mean()

                # Densification score h_i (Eq. 13): combines activity and persistent error
                lamE = float(getattr(self.config, 'lambda_E', 1.0))
                h_i = g_i * (1.0 + lamE * (Ebar / (Ebar_mean + 1e-6)))

                # Candidate sets (Eq. 16)
                s1 = torch.exp(self.scales).max(dim=-1).values.squeeze()
                h_mask = h_i > float(getattr(self.config, 'densify_h_thresh', self.config.densify_grad_thresh))

                splits = (s1 > self.config.densify_size_thresh).squeeze() & h_mask
                dups = (s1 <= self.config.densify_size_thresh).squeeze() & h_mask
                if self.step < self.config.stop_screen_size_at:
                    splits = splits | ((self.max_2Dsize > self.config.split_screen_size).squeeze() & h_mask)

                # Budget projection (Eq. 15): sort by h_i and take largest feasible prefix
                nsamps = int(self.config.n_split_samples)
                if self.config.enable_budget_aware_split and self.config.max_gaussians is not None:
                    budget = max(0, int(self.config.max_gaussians) - int(self.num_points))
                    cand_mask = splits | dups
                    cand_idx = torch.where(cand_mask)[0]
                    if cand_idx.numel() > 0 and budget > 0:
                        order = torch.argsort(h_i[cand_idx], descending=True)
                        selected = torch.zeros_like(cand_mask, dtype=torch.bool)
                        for j in order.tolist():
                            i = int(cand_idx[j].item())
                            cost = nsamps if bool(splits[i].item()) else 1
                            if cost <= budget:
                                selected[i] = True
                                budget -= cost
                            else:
                                break
                            if budget <= 0:
                                break
                        splits = splits & selected
                        dups = dups & selected
                    else:
                        splits = torch.zeros_like(splits, dtype=torch.bool)
                        dups = torch.zeros_like(dups, dtype=torch.bool)

                # Perform splitting and duplicating
                split_params = self.split_gaussians(splits, nsamps)
                dup_params = self.dup_gaussians(dups)

                # Append new Gaussians to parameter tensors
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                # Extend persistent stats for new Gaussians (copy parent stats)
                new_E_split = Ebar[splits].repeat(nsamps) if int(splits.sum()) > 0 else Ebar.new_empty((0,))
                new_E_dup = Ebar[dups] if int(dups.sum()) > 0 else Ebar.new_empty((0,))
                self.gaussian_error_ema = torch.cat([Ebar, new_E_split, new_E_dup], dim=0)

                new_g_split = g_i[splits].repeat(nsamps) if int(splits.sum()) > 0 else g_i.new_empty((0,))
                new_g_dup = g_i[dups] if int(dups.sum()) > 0 else g_i.new_empty((0,))
                self.gaussian_activity = torch.cat([g_i, new_g_split, new_g_dup], dim=0)

                # Extend screen-size statistics for new Gaussians
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params['scales'][:, 0]),
                        torch.zeros_like(dup_params['scales'][:, 0]),
                    ],
                    dim=0,
                )

                # Update optimizer state for new Gaussians
                split_idcs = torch.where(splits)[0]
                if split_idcs.numel() > 0:
                    self.dup_in_all_optim(optimizers, split_idcs, nsamps)
                dup_idcs = torch.where(dups)[0]
                if dup_idcs.numel() > 0:
                    self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # Cull original Gaussians after splitting
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )
                deleted_masks.append(self.cull_gaussians(splits_mask))

                # Budget-aware pruning (Eq. 17) if still above cap after standard pruning
                if self.config.enable_budget_aware_split and self.config.max_gaussians is not None and self.num_points > int(self.config.max_gaussians):
                    k_remove = int(self.num_points - int(self.config.max_gaussians))
                    Ebar2 = self.gaussian_error_ema if self.gaussian_error_ema is not None else torch.zeros((self.num_points,), device=self.device)
                    g2 = self.gaussian_activity if self.gaussian_activity is not None else torch.ones_like(Ebar2)
                    Emean2 = Ebar2.mean()
                    lamU = float(getattr(self.config, 'lambda_U', 1.0))
                    util = g2 * (1.0 + lamU * (Ebar2 / (Emean2 + 1e-6)))
                    low = torch.topk(util, k=k_remove, largest=False).indices
                    extra = torch.zeros((self.num_points,), device=self.device, dtype=torch.bool)
                    extra[low] = True
                    deleted_masks.append(self.cull_gaussians(extra))
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_masks = [self.cull_gaussians()]
            else:
                deleted_masks = []

            if deleted_masks:
                for dm in deleted_masks:
                    if dm is not None:
                        self.remove_from_all_optim(optimizers, dm)

                # reset the exp of optimizer
                for key in ["medium_mlp"]:
                    if key not in optimizers.optimizers:
                        continue
                    optim = optimizers.optimizers[key]
                    if not optim.param_groups or not optim.param_groups[0]["params"]:
                        continue
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state.get(param, {})
                    if "exp_avg" in param_state:
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])


            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                # Reset value is set to be reset_alpha_thresh
                reset_value = self.config.reset_alpha_thresh
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.depths_accum = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        if self.step < self.config.stop_split_at:
            cull_alpha_thresh = self.config.cull_alpha_thresh
        else:
            cull_alpha_thresh = self.config.cull_alpha_thresh_post
        culls = (torch.sigmoid(self.opacities) < cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        # Keep auxiliary per-Gaussian statistics in sync with topology changes
        if self.gaussian_error_ema is not None:
            self.gaussian_error_ema = self.gaussian_error_ema[~culls]
        if self.gaussian_activity is not None:
            self.gaussian_activity = self.gaussian_activity[~culls]

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int):
        """
        Split Gaussians that are too large.

        If Module-2 is enabled and config.use_steepest_split is True, we split along the principal axis
        (largest scale in world space). Otherwise we fall back to the original
        random-direction splitting.
        """
        n_splits = int(split_mask.sum().item())
        CONSOLE.log(
            f"Splitting {n_splits}/{self.num_points} gaussians "
            f"({split_mask.sum().item() / max(self.num_points, 1):.4f})"
        )

        if n_splits == 0:
            # Return empty tensors with correct shapes for all parameters
            out: Dict[str, torch.Tensor] = {}
            for name, param in self.gauss_params.items():
                out[name] = param.new_empty((0,) + param.shape[1:])
            return out

        size_fac = 1.6  # same shrink factor as original paper

        if bool(self.config.enable_budget_aware_split) and getattr(self.config, "use_steepest_split", True):
            # ---------------- principal-axis ("steepest") splitting ----------------
            base_means = self.means[split_mask]                         # [N, 3]
            base_scales_exp = torch.exp(self.scales[split_mask])        # [N, 3]

            # Shrink scales (original behavior) and update originals in-place
            new_scales_exp = base_scales_exp / size_fac                 # [N, 3]
            self.scales[split_mask] = torch.log(new_scales_exp)

            # World-space principal axis = column of rotation matrix with largest scale
            quats = self.quats[split_mask]
            quats = quats / quats.norm(dim=-1, keepdim=True)            # [N, 4]
            rots = quat_to_rotmat(quats)                                # [N, 3, 3]

            axis_idx = torch.argmax(new_scales_exp, dim=-1)             # [N]
            # gather the principal axis direction from rotation matrices
            idx = axis_idx.view(-1, 1, 1).expand(-1, 3, 1)              # [N, 3, 1]
            dir_world = torch.gather(rots, dim=2, index=idx).squeeze(-1)  # [N, 3]
            dir_world = dir_world / (dir_world.norm(dim=-1, keepdim=True) + 1e-9)

            axis_scale = new_scales_exp[
                torch.arange(n_splits, device=self.device), axis_idx
            ]  # [N]

            # Repeat per original Gaussian samps times (same pattern as original code)
            base_means_rep = base_means.repeat(samps, 1)                # [samps*N, 3]
            dir_rep = dir_world.repeat(samps, 1)                         # [samps*N, 3]
            axis_scale_rep = axis_scale.repeat(samps)                    # [samps*N]

            # 1D offsets along principal axis
            offsets_1d = torch.randn(samps * n_splits, device=self.device) * axis_scale_rep * 0.5
            new_means = base_means_rep + offsets_1d.unsqueeze(-1) * dir_rep   # [samps*N, 3]

            new_scales = torch.log(new_scales_exp).repeat(samps, 1)      # [samps*N, 3]
            new_quats = self.quats[split_mask].repeat(samps, 1)         # [samps*N, 4]

        else:
            # ---------------- original random-direction splitting ----------------
            centered_samples = torch.randn(
                (samps * n_splits, 3), device=self.device
            )  # [samps*N, 3] axis-aligned samples

            scaled_samples = (
                torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
            )  # [samps*N, 3]

            quats = self.quats[split_mask]
            quats = quats / quats.norm(dim=-1, keepdim=True)            # [N, 4]
            rots = quat_to_rotmat(quats.repeat(samps, 1))               # [samps*N, 3, 3]
            rotated_samples = torch.bmm(rots, scaled_samples.unsqueeze(-1)).squeeze(-1)

            new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)

            new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(
                samps, 1
            )
            self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
            new_quats = self.quats[split_mask].repeat(samps, 1)

        # Features / opacities same as original code
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        new_opacities = self.opacities[split_mask].repeat(samps, 1)

        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, *([1] * (param.dim() - 1)))
        return out


    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers."""
        gps = self.get_gaussian_param_groups()
        gps["medium_mlp"] = list(self.medium_mlp.parameters())

        # SH direction encoding has no trainable parameters in the tcnn implementation; do not register an optimizer group.

        # Always expose attenuation_coeff as a group; when Module-5 is disabled it is unused and thus gets no grads.
        gps["attenuation_coeff"] = [self.attenuation_coeff]

        # Drop any empty groups defensively (prevents torch optimizer errors).
        gps = {
            k: v
            for k, v in gps.items()
            if v is not None and len(v) > 0 and all(p is not None for p in v)
        }
        return gps


    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def get_outputs(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        self.last_fx = camera.fx.item()
        self.last_fy = camera.fy.item()

        # Medium
        # Encode directions
        # y = torch.linspace(0., H, H, device=self.device)
        # x = torch.linspace(0., W, W, device=self.device)
        # yy, xx = torch.meshgrid(y, x)
        # yy = (yy - cy) / camera.fy.item()
        # xx = (xx - cx) / camera.fx.item()
        # directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        # norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
        # directions = directions / norms
        directions = self._get_dirs_cam_unit(H, W, camera.fx.item(), camera.fy.item(), cx, cy, device=self.device, dtype=R.dtype)
        directions = directions @ R.T

        # directions_flat = directions.view(-1, 3)
        # directions_encoded = self.direction_encoding(directions_flat)
        # outputs_shape = directions.shape[:-1]

        # # Medium MLP forward pass
        # if self.config.mlp_type == "tcnn":
        #     medium_base_out = self.medium_mlp(directions_encoded)
        # else:
        #     medium_base_out = self.medium_mlp(directions_encoded.float())

        # # different activations for different outputs
        # medium_rgb = (
        #     self.colour_activation(medium_base_out[..., :3])
        #     .view(*outputs_shape, -1)
        #     .to(directions)
        # )
        # medium_bs = (
        #     self.sigma_activation(medium_base_out[..., 3:6] + self.medium_density_bias)
        #     .view(*outputs_shape, -1)
        #     .to(directions)
        # )
        # medium_attn = (
        #     self.sigma_activation(medium_base_out[..., 6:] + self.medium_density_bias)
        #     .view(*outputs_shape, -1)
        #     .to(directions)
        # )

        outputs_shape = directions.shape[:-1]
        flat = directions.reshape(-1, 3)
        N = flat.shape[0]
        chunk = int(getattr(self.config, "medium_chunk_size", 65536))

        use_amp = (
            (not self.training)
            and bool(getattr(self.config, "eval_use_autocast", True))
            and flat.is_cuda
        )

        # Store in fp32 to keep downstream kernels/ops safe; peak is reduced by chunking.
        medium_rgb_flat  = torch.empty((N, 3), device=self.device, dtype=torch.float32)
        medium_bs_flat   = torch.empty((N, 3), device=self.device, dtype=torch.float32)
        medium_attn_flat = torch.empty((N, 3), device=self.device, dtype=torch.float32)

        amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)
        with amp_ctx:
            for s in range(0, N, chunk):
                e = min(N, s + chunk)
                enc = self.direction_encoding(flat[s:e])

                if self.config.mlp_type == "tcnn":
                    base = self.medium_mlp(enc)
                else:
                    base = self.medium_mlp(enc.float())

                rgb  = self.colour_activation(base[..., :3])
                bs   = self.sigma_activation(base[..., 3:6] + self.medium_density_bias)
                attn = self.sigma_activation(base[..., 6:] + self.medium_density_bias)

                medium_rgb_flat[s:e]  = rgb.float()
                medium_bs_flat[s:e]   = bs.float()
                medium_attn_flat[s:e] = attn.float()

        medium_rgb  = medium_rgb_flat.view(*outputs_shape, 3).to(directions)
        medium_bs   = medium_bs_flat.view(*outputs_shape, 3).to(directions)
        medium_attn = medium_attn_flat.view(*outputs_shape, 3).to(directions)

        if self.config.zero_medium:
            medium_rgb = torch.zeros_like(medium_rgb)
            medium_bs = torch.zeros_like(medium_bs)
            medium_attn = torch.zeros_like(medium_attn)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = medium_rgb
                depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb,
                        "rgb_object": torch.zeros_like(rgb), "rgb_medium": medium_rgb, "pred_image": rgb,
                        "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn, "medium_attn_raw": medium_attn}
        else:
            crop_ids = None

        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
            clip_thresh=self.config.clip_thresh,
        )  # type: ignore

        self.depths = depths.detach()

        # Cache projection buffers for residual attribution (used by error-driven densification)
        self._proj_conics = conics.detach()
        self._proj_depths = depths.detach()
        self._proj_num_tiles_hit = num_tiles_hit.detach()
        self._proj_block_width = BLOCK_WIDTH
        self._proj_hw = (H, W)
        if crop_ids is not None and crop_ids.sum() != 0:
            self._proj_gauss_indices = torch.where(crop_ids)[0].detach()
        else:
            self._proj_gauss_indices = None

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = medium_rgb
            depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb,
                    "rgb_object": torch.zeros_like(rgb), "rgb_clear": torch.zeros_like(rgb), "rgb_clear_clamp": torch.zeros_like(rgb), "rgb_medium": medium_rgb, "pred_image": rgb,
                    "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn, "medium_attn_raw": medium_attn}

        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            # Camera position in world coordinates for this (single) view
            cam_pos = camera.camera_to_worlds[0, :3, 3].detach()  # (3,)
            viewdirs = means_crop.detach() - cam_pos[None, :]     # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])
        # Optional global scaling of the direction-conditioned attenuation field β(v).
        # This keeps the model aligned with Eq. (25) while allowing a conservative warmup.
        medium_attn_eff = medium_attn
        apply_depth_attn = (
            self.config.enable_depth_attenuation
            and self.config.use_depth_attenuation
            and self.attenuation_coeff is not None
            and (not self.training or self.step >= self.config.depth_attenuation_warmup_steps)
        )
        if apply_depth_attn:
            coeff = torch.nn.functional.softplus(self.attenuation_coeff).to(medium_attn_eff)
            medium_attn_eff = medium_attn_eff * coeff.view(1, 1, 3)

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        # Cache the opacity actually used by the rasterizer (depends on rasterize_mode)
        self._proj_opacity = opacities.detach()

        self.xys_grad_abs = torch.zeros_like(self.xys)

        rgb_object, rgb_clear, rgb_medium, depth_im, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            self.xys_grad_abs,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            medium_rgb,
            medium_bs,
            medium_attn_eff,
            H,
            W,
            BLOCK_WIDTH,
            background=medium_rgb,
            return_alpha=True,
            step=self.step,
        )  # type: ignore

        rgb = rgb_object + rgb_medium
        rgb_clear_clamp = torch.clamp(rgb_clear, 0., 1.)
        rgb_clear = rgb_clear / (rgb_clear + 1.)

        depth_im = depth_im[..., None]
        alpha = alpha[..., None]
        depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": medium_rgb,
                "rgb_object": rgb_object, "rgb_clear": rgb_clear, "rgb_clear_clamp": rgb_clear_clamp, "rgb_medium": rgb_medium, "pred_image": rgb,
                "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn_eff, "medium_attn_raw": medium_attn}  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            # alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return image[..., :3]
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        for i in range(3):
            # 3 channels
            metrics_dict[f"medium_attn_{i}"] = outputs["medium_attn"][:, :, i].mean()
            metrics_dict[f"medium_bs_{i}"] = outputs["medium_bs"][:, :, i].mean()
            metrics_dict[f"medium_rgb_{i}"] = outputs["medium_rgb"][:, :, i].mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["pred_image"]

        # Save last predicted and GT images for photometric error-based densification
        # We detach here to avoid storing computation graph
        if self.training and self.config.enable_error_densification:
            self.last_pred_image = pred_img.detach()
            self.last_gt_image = gt_img.detach()

        # Apply mask if present (for GT and predicted images)
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        # Compute reconstruction error per pixel
        if self.config.main_loss == "l1":
            per_pixel_error = torch.abs(gt_img - pred_img)
            recon_loss = per_pixel_error.mean()
        elif self.config.main_loss == "reg_l1":
            per_pixel_error = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3))
            recon_loss = per_pixel_error.mean()
        else:
            per_pixel_error = ((pred_img - gt_img) / (pred_img.detach() + 1e-3)) ** 2
            recon_loss = per_pixel_error.mean()
        # Edge-aware reweighting (Eq. 23-24): use Sobel magnitude on grayscale GT, normalized to [0,1].
        per_pixel_l1 = torch.abs(gt_img - pred_img).mean(dim=-1)  # (H,W)
        edge_loss = gt_img.new_tensor(0.0)
        effective_edge_weight = 0.0
        if self.config.enable_edge_loss and self.config.edge_loss_weight > 0:
            effective_edge_weight = float(self.config.edge_loss_weight)
            if self.training and self.config.edge_loss_warmup_steps > 0:
                t = min(1.0, float(self.step) / float(self.config.edge_loss_warmup_steps))
                effective_edge_weight *= t

            if effective_edge_weight > 0.0:
                gray = gt_img.mean(dim=-1, keepdim=True)
                sobel_x = torch.tensor(
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    dtype=torch.float32,
                    device=gt_img.device,
                ).view(1, 1, 3, 3)
                sobel_y = torch.tensor(
                    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                    dtype=torch.float32,
                    device=gt_img.device,
                ).view(1, 1, 3, 3)

                gray_nchw = gray.permute(2, 0, 1).unsqueeze(0)  # (1,1,H,W)
                grad_x = torch.nn.functional.conv2d(gray_nchw, sobel_x, padding=1)
                grad_y = torch.nn.functional.conv2d(gray_nchw, sobel_y, padding=1)
                grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(0).squeeze(0)  # (H,W)

                gmin = grad_mag.amin()
                gmax = grad_mag.amax()
                g_tilde = (grad_mag - gmin) / (gmax - gmin + 1e-6)
                weight_map = 1.0 + g_tilde
                edge_loss = (per_pixel_l1 * weight_map).mean()

        # SSIM component
        if self.config.ssim_loss != "ssim":
            simloss = 1 - self.ssim(
                (gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
                (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
            )
        else:
            simloss = 1 - self.ssim(
                gt_img.permute(2, 0, 1)[None, ...],
                pred_img.permute(2, 0, 1)[None, ...],
            )
        base_loss = (1 - self.config.ssim_lambda) * recon_loss + self.config.ssim_lambda * simloss

        # Attenuation regularization (Eq. 28 with attenuation terms)
        attn_reg = gt_img.new_tensor(0.0)
        if bool(getattr(self.config, 'enable_attn_regularization', True)) and float(getattr(self.config, 'attn_loss_weight', 0.0)) > 0.0:
            beta = outputs.get('medium_attn', None)
            if beta is not None:
                attn_reg = self._attenuation_regularization(beta)

        main_loss = base_loss + effective_edge_weight * edge_loss + float(getattr(self.config, 'attn_loss_weight', 0.0)) * attn_reg

        # Update per-Gaussian persistent error stats (Eq. 11-12) for error-driven densification.
        if self.training and self.config.enable_error_densification:
            try:
                every = int(getattr(self.config, 'error_attr_every', 1))
                if every <= 1 or (self.step % every == 0):
                    self._update_gaussian_error_ema(per_pixel_l1.detach())
            except Exception:
                pass

        return {
            'main_loss': main_loss,
            'recon_loss': recon_loss,
            'ssim_loss': simloss,
            'edge_loss': edge_loss,
            'attn_reg': attn_reg,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])

        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)

        # d = self._get_downscale_factor()
        # if d > 1:
        #     # torchvision can be slow to import, so we do it lazily.
        #     import torchvision.transforms.functional as TF

        #     newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
        #     predicted_rgb = TF.resize(predicted_rgb.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        # else:
        #     predicted_rgb = predicted_rgb
        # --- Robust eval resizing ---
        # GT and rendered images must have identical H,W for TorchMetrics (PSNR/SSIM/LPIPS).
        # Some pipelines may provide downscaled GT images but full-res cameras.
        if predicted_rgb.shape[0] != gt_rgb.shape[0] or predicted_rgb.shape[1] != gt_rgb.shape[1]:
            import torchvision.transforms.functional as TF
            H, W = gt_rgb.shape[0], gt_rgb.shape[1]
            predicted_rgb = TF.resize(
                predicted_rgb.permute(2, 0, 1), [H, W], antialias=True
            ).permute(1, 2, 0)

            # Also resize visualization outputs to keep logging consistent.
            def _resize_hw(x: torch.Tensor) -> torch.Tensor:
                if not torch.is_tensor(x):
                    return x
                if x.ndim == 3 and (x.shape[0] != H or x.shape[1] != W):
                    return TF.resize(x.permute(2, 0, 1), [H, W], antialias=True).permute(1, 2, 0)
                if x.ndim == 2 and (x.shape[0] != H or x.shape[1] != W):
                    return TF.resize(x[None, ...], [H, W], antialias=True)[0]
                return x

            outputs["rgb_medium"] = _resize_hw(outputs["rgb_medium"])
            outputs["rgb_object"] = _resize_hw(outputs["rgb_object"])
            outputs["depth"]      = _resize_hw(outputs["depth"])
            outputs["rgb"]        = _resize_hw(outputs["rgb"])
            outputs["rgb_clear"]  = _resize_hw(outputs["rgb_clear"])

        output_gt_rgb = gt_rgb.cpu()

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"gt": output_gt_rgb, "rgb_medium": outputs["rgb_medium"], "rgb_object": outputs["rgb_object"], "depth": outputs["depth"], "rgb": outputs["rgb"], "rgb_clear": outputs["rgb_clear"]}
        # images_dict = {
        #     "gt": output_gt_rgb,
        #     "rgb_medium": outputs["rgb_medium"].detach().cpu(),
        #     "rgb_object": outputs["rgb_object"].detach().cpu(),
        #     "depth": outputs["depth"].detach().cpu(),
        #     "rgb": outputs["rgb"].detach().cpu(),
        #     "rgb_clear": outputs["rgb_clear"].detach().cpu(),
        # }
        return metrics_dict, images_dict
