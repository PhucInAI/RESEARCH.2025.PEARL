"""Torch-side helper functions.

This file exists to keep the public API stable and to centralize small math utilities.
"""

from __future__ import annotations

import torch


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion(s) [w, x, y, z] to rotation matrix/matrices.

    Args:
        quat: (..., 4)

    Returns:
        (..., 3, 3)
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quat shape (...,4), got {tuple(quat.shape)}")
    q = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz

    rot = torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )
    return rot


# These forward wrappers are provided for backwards compatibility with older imports.
# The authoritative implementations are in project_gaussians.py / rasterize.py.

def project_gaussians_forward(*args, **kwargs):
    import pearl.cuda as _C
    return _C.project_gaussians_forward(*args, **kwargs)


def rasterize_gaussians_forward(*args, **kwargs):
    import pearl.cuda as _C
    # Prefer the RGB path by default.
    return _C.rasterize_forward(*args, **kwargs)
