"""Nerfstudio method specifications for PEARL"""


from dataclasses import dataclass

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from pearl.ai_logger import aiLogger
from pearl.config_yaml import anchor_locations, dump_config_text
from pearl.pearl import PearlModelConfig

# Published-protocol schedule (the four 1M main runs trained 20000 steps with
# scheduler max_steps 20000; paper Table 2's T=10K is a typo — KI-004).
NUM_STEPS = 20000


# ========================================================================
# Trainer: nerfstudio's, saving a readable config.yml
# ========================================================================
@dataclass
class PearlTrainerConfig(TrainerConfig):
    """TrainerConfig whose saved config.yml carries short tags and repo-relative paths."""

    def save_config(self) -> None:
        """Save config to base directory, in the PEARL config.yml format."""
        base_dir = self.get_base_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        anchor_locations(self)
        config_yaml_path = base_dir / "config.yml"
        config_yaml_path.write_text(dump_config_text(self), "utf8")
        aiLogger.info(f"saved config to {config_yaml_path}")


# ========================================================================
# Base method: ns-train pearl
# ========================================================================
pearl_method = MethodSpecification(
    config=PearlTrainerConfig(
        method_name="pearl",
        steps_per_eval_image=1000,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=NUM_STEPS,
        mixed_precision=False,
        save_only_latest_checkpoint=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            ),
            model=PearlModelConfig(
                num_steps=NUM_STEPS,
                main_loss="reg_l1",
                ssim_loss="reg_ssim",
                zero_medium=False,
                attn_loss_weight=0.01,
                error_attr_every=10,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=NUM_STEPS,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0025,
                    max_steps=NUM_STEPS,
                ),
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0025 / 20,
                    max_steps=NUM_STEPS,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.05,
                    max_steps=NUM_STEPS,
                ),
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.005,
                    max_steps=NUM_STEPS,
                ),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.001,
                    max_steps=NUM_STEPS,
                ),
            },
            "medium_mlp": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=NUM_STEPS,
                ),
            },
            # Depth attenuation coefficients: low LR to avoid instability.
            "attenuation_coeff": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=NUM_STEPS,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="PEARL (Photometric-Error and Attenuation-Regularized Learning) for underwater scenes.",
)


# ========================================================================
# Big variant: ns-train pearl-big
# ========================================================================
pearl_method_big = MethodSpecification(
    config=PearlTrainerConfig(
        method_name="pearl-big",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=NUM_STEPS,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            ),
            model=PearlModelConfig(
                num_steps=NUM_STEPS,
                continue_cull_post_densification=False,
                attn_loss_weight=0.01,
                error_attr_every=10,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=NUM_STEPS,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "medium_mlp": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.5e-4, max_steps=NUM_STEPS,
                ),
            },
            # Depth attenuation coefficients: low LR to avoid instability.
            "attenuation_coeff": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=NUM_STEPS,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="PEARL (big) for underwater scenes.",
)
