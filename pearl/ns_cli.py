"""PEARL-aware ns-viewer / ns-render / ns-eval: load best/ exports directly."""


from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import torch
import yaml
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts import eval as ns_eval
from nerfstudio.scripts import render as ns_render
from nerfstudio.scripts.viewer import run_viewer as ns_viewer
from nerfstudio.utils.eval_utils import eval_setup

from pearl.ai_logger import aiLogger


# ========================================================================
# Checkpoint unpickling
# ========================================================================
# torch 2.6 flipped torch.load to weights_only=True; nerfstudio still calls it
# bare, so every run checkpoint fails on the numpy scalars its metadata carries.
# Allowlisting those three globals fixes the delegated nerfstudio path too, and
# keeps the safe unpickler on — unlike weights_only=False.
torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])


# ========================================================================
# Checkpoint resolution
# ========================================================================
# nerfstudio finds checkpoints only through the step-%09d.ckpt filename pattern
# (eval_utils.eval_load_checkpoint), so the step-free ADR-008 export is invisible
# to it: assets/results/<scene>/<ts>/best/config.yml would silently fall back to
# the run's *last* step instead of the PSNR-winning one. Loading best.ckpt by
# hand is what makes that config a runnable target.
BEST_CKPT_NAME = "best.ckpt"


def pearl_eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """eval_setup that loads best.ckpt when the config sits in a best/ export."""
    config_path = Path(config_path)
    best_ckpt = config_path.parent / BEST_CKPT_NAME
    if not best_ckpt.exists():
        return eval_setup(config_path, eval_num_rays_per_chunk, test_mode, update_config_callback)

    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk
    if update_config_callback is not None:
        config = update_config_callback(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    pipeline.eval()

    loaded_state = torch.load(best_ckpt, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    aiLogger.info(f"loaded best checkpoint {best_ckpt}")
    return config, pipeline, best_ckpt, loaded_state["step"]


# ========================================================================
# Console scripts
# ========================================================================
# Each nerfstudio script bound eval_setup at import time, so the wrappers rebind
# that module attribute and then hand over to the stock tyro CLI — every
# subcommand and flag of ns-viewer / ns-render / ns-eval stays available.
def viewer_entrypoint() -> None:
    """Entrypoint for pearl-viewer."""
    ns_viewer.eval_setup = pearl_eval_setup
    ns_viewer.entrypoint()


def render_entrypoint() -> None:
    """Entrypoint for pearl-render."""
    ns_render.eval_setup = pearl_eval_setup
    ns_render.entrypoint()


def eval_entrypoint() -> None:
    """Entrypoint for pearl-eval."""
    ns_eval.eval_setup = pearl_eval_setup
    ns_eval.entrypoint()
