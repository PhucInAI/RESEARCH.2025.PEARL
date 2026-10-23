"""Stock ns-eval plus the KI-007 torch-2.6 allowlist — pearl-free."""


# ========================================================================
# KI-007 checkpoint unpickling
# ========================================================================
# Replicated from pearl/ns_cli.py: torch 2.6 flipped torch.load to
# weights_only=True; nerfstudio still calls it bare, so run checkpoints
# fail on the numpy scalars their metadata carries. This shim exists
# because the baseline `ws` conda env does not install the pearl package
# (ADR-013) — it emits nothing of its own, so no logger is needed.
import numpy as np
import torch

torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])

from nerfstudio.scripts import eval as ns_eval  # noqa: E402


if __name__ == "__main__":
    ns_eval.entrypoint()
