#!/usr/bin/env python3
"""Render checkpoint-based gaussian and image companion videos from a Nerfstudio camera path.

Notes:
- The main output at ``--output-path`` is the sharper, more point-cloud-like render.
  It keeps only the last gaussian hit per pixel shell, so it looks crisper and sparser.
- A sibling dense-gaussian video is also written next to it.
  That version accumulates many gaussian hits per pixel, so it looks fuller, softer,
  and more 3DGS-like.
- By default, a second sibling video built from the nearest real dataset images is also
  written to show what image-only coverage looks like along the same camera path.
- All outputs use the same camera path, frame count, resolution, and fps.
  Only the rendering source and compositing style change.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

BACKGROUND_COLOR = np.array([6, 10, 14], dtype=np.uint8)
MAX_GAUSSIAN_RADIUS = 6
GAUSSIAN_RADIUS_SCALE = 1.8


@dataclass(frozen=True)
class GaussianCloud:
    means: Any
    rgb: Any
    opacity: Any
    scale: Any


@dataclass(frozen=True)
class ProjectedGaussians:
    pixels_x: Any
    pixels_y: Any
    radii: Any
    opacity: Any
    rgb: Any


@dataclass(frozen=True)
class TrainingImageLibrary:
    image_paths: list[Path]
    positions: np.ndarray
    forwards: np.ndarray
    ups: np.ndarray
    position_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-config",
        type=Path,
        required=True,
        help="Path to the Nerfstudio config.yml file.",
    )
    parser.add_argument(
        "--camera-path",
        required=True,
        help="Camera path JSON file, or one of: actual, virtual, limited.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to the primary custom render output. A sibling dense-gaussian video is also written.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Optional output video fps. Defaults to the fps stored in the camera-path JSON.",
    )
    parser.add_argument(
        "--generate-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write a companion video from the nearest real dataset images along the camera path.",
    )
    return parser.parse_args()


def parse_dataset_path_from_config(config_path: Path) -> Path | None:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "data: !!python/object/apply:pathlib.PosixPath":
            continue
        parts = []
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].startswith("      - "):
            parts.append(lines[cursor].split("- ", 1)[1])
            cursor += 1
        if parts:
            return Path(*parts)
    return None


def resolve_camera_path(camera_path_arg: str, load_config_path: Path) -> Path:
    keyword = camera_path_arg.strip().lower()
    if keyword not in {"actual", "virtual", "limited"}:
        return Path(camera_path_arg)

    dataset_path = parse_dataset_path_from_config(load_config_path)
    if dataset_path is None:
        raise SystemExit(f"Could not find dataset path inside config: {load_config_path}")

    filename = {
        "actual": "trajectory.json",
        "virtual": "virtual_trajectory.json",
        "limited": "limited_trajectory.json",
    }[keyword]
    return dataset_path / "camera_paths" / filename


def resolve_checkpoint_path(load_config_path: Path) -> Path:
    model_dir = load_config_path.parent / "nerfstudio_models"
    if not model_dir.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {model_dir}")

    checkpoints = sorted(model_dir.glob("step-*.ckpt"))
    if not checkpoints:
        raise SystemExit(f"No checkpoint files found in: {model_dir}")

    def step_number(path: Path) -> int:
        return int(path.stem.split("-", 1)[1])

    return max(checkpoints, key=step_number)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def sh0_to_rgb(features_dc: np.ndarray) -> np.ndarray:
    c0 = 0.28209479177387814
    return np.clip(features_dc * c0 + 0.5, 0.0, 1.0)


def require_torch():
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "Torch is required to read checkpoint gaussians. Activate the pearl environment first."
        ) from error
    return torch


def select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint_gaussians(checkpoint_path: Path, device) -> GaussianCloud:
    torch = require_torch()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["pipeline"]

    means = state["_model.gauss_params.means"].detach().to(dtype=torch.float32)
    scales = torch.exp(state["_model.gauss_params.scales"].detach().to(dtype=torch.float32))
    features_dc = state["_model.gauss_params.features_dc"].detach().to(dtype=torch.float32)
    opacities = torch.sigmoid(
        state["_model.gauss_params.opacities"].detach().to(dtype=torch.float32).squeeze(-1)
    )

    c0 = 0.28209479177387814
    rgb = torch.clamp(features_dc * c0 + 0.5, 0.0, 1.0)
    scale = torch.maximum(torch.amax(scales, dim=1), 0.5 * torch.mean(scales, dim=1))

    valid = torch.isfinite(means).all(dim=1)
    valid &= torch.isfinite(rgb).all(dim=1)
    valid &= torch.isfinite(opacities)
    valid &= torch.isfinite(scale)
    valid &= opacities > 0.02
    valid &= scale > 1e-5

    means = means[valid]
    rgb = rgb[valid]
    opacities = opacities[valid]
    scale = scale[valid]
    if int(means.shape[0]) == 0:
        raise ValueError(f"No usable gaussians found in checkpoint: {checkpoint_path}")

    lower = torch.quantile(means, 0.002, dim=0)
    upper = torch.quantile(means, 0.998, dim=0)
    box_mask = ((means >= lower) & (means <= upper)).all(dim=1)
    if int(box_mask.sum().item()) >= int(means.shape[0]) // 2:
        means = means[box_mask]
        rgb = rgb[box_mask]
        opacities = opacities[box_mask]
        scale = scale[box_mask]

    return GaussianCloud(
        means=means.to(device=device, non_blocking=True),
        rgb=rgb.to(device=device, non_blocking=True),
        opacity=opacities.to(device=device, non_blocking=True),
        scale=scale.to(device=device, non_blocking=True),
    )


def load_camera_path(camera_path_path: Path) -> dict:
    with camera_path_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def derive_dense_output_path(output_path: Path) -> Path:
    if output_path.stem == "custom":
        return output_path.with_name(f"dense_gaussian{output_path.suffix}")
    return output_path.with_name(f"{output_path.stem}_dense_gaussian{output_path.suffix}")


def derive_image_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_images{output_path.suffix}")


def load_training_image_library(load_config_path: Path) -> TrainingImageLibrary:
    dataset_path = parse_dataset_path_from_config(load_config_path)
    if dataset_path is None:
        raise SystemExit(f"Could not find dataset path inside config: {load_config_path}")

    from generate_seathru_camera_paths import (
        colmap_image_to_camera_to_world,
        compute_dataparser_transform_and_scale,
        convert_camera_to_world_opencv_to_opengl,
        read_images_binary,
        transform_pose_to_nerfstudio_space,
    )

    images = read_images_binary(dataset_path / "sparse" / "images.bin")
    ordered_images = sorted(images.values(), key=lambda image: image.name)
    if not ordered_images:
        raise ValueError(f"No registered images found in {dataset_path / 'sparse' / 'images.bin'}")

    opengl_poses = np.stack(
        [convert_camera_to_world_opencv_to_opengl(colmap_image_to_camera_to_world(image)) for image in ordered_images],
        axis=0,
    )
    dataparser_transform, dataparser_scale = compute_dataparser_transform_and_scale(opengl_poses)

    poses = []
    image_paths: list[Path] = []
    for image in ordered_images:
        image_path = dataset_path / "images" / image.name
        if not image_path.is_file():
            continue
        pose_opencv = colmap_image_to_camera_to_world(image)
        poses.append(transform_pose_to_nerfstudio_space(pose_opencv, dataparser_transform, dataparser_scale))
        image_paths.append(image_path)

    if not poses:
        raise ValueError(f"No training images were found under {dataset_path / 'images'}")

    poses_array = np.stack(poses, axis=0)
    positions = poses_array[:, :3, 3]
    forwards = -poses_array[:, :3, 2]
    ups = poses_array[:, :3, 1]
    position_scale = max(
        float(np.percentile(np.linalg.norm(positions - np.median(positions, axis=0), axis=1), 70.0)),
        1e-3,
    )

    return TrainingImageLibrary(
        image_paths=image_paths,
        positions=positions,
        forwards=forwards,
        ups=ups,
        position_scale=position_scale,
    )


def select_nearest_training_image_index(target_pose: np.ndarray, training_images: TrainingImageLibrary) -> int:
    target_position = target_pose[:3, 3]
    target_forward = -target_pose[:3, 2]
    target_up = target_pose[:3, 1]

    position_cost = np.linalg.norm(training_images.positions - target_position, axis=1) / training_images.position_scale
    forward_cost = 1.0 - np.clip(training_images.forwards @ target_forward, -1.0, 1.0)
    up_cost = 1.0 - np.clip(training_images.ups @ target_up, -1.0, 1.0)
    score = position_cost + 0.75 * forward_cost + 0.25 * up_cost
    return int(np.argmin(score))


def load_resized_training_image(
    image_index: int,
    training_images: TrainingImageLibrary,
    width: int,
    height: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    if image_index not in cache:
        with Image.open(training_images.image_paths[image_index]) as image:
            rgb_image = image.convert("RGB")
            if rgb_image.size != (width, height):
                rgb_image = rgb_image.resize((width, height), Image.LANCZOS)
            cache[image_index] = np.array(rgb_image, dtype=np.uint8)
    return cache[image_index]


def render_image_frame(
    camera_to_world: np.ndarray,
    training_images: TrainingImageLibrary,
    width: int,
    height: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    image_index = select_nearest_training_image_index(camera_to_world, training_images)
    return load_resized_training_image(image_index, training_images, width, height, cache)


def project_gaussians(
    gaussian_cloud: GaussianCloud,
    camera_to_world: np.ndarray,
    width: int,
    height: int,
    vertical_fov_degrees: float,
) -> ProjectedGaussians:
    torch = require_torch()
    device = gaussian_cloud.means.device
    camera_to_world_tensor = torch.as_tensor(camera_to_world, dtype=torch.float32, device=device)
    rotation = camera_to_world_tensor[:3, :3]
    translation = camera_to_world_tensor[:3, 3]
    camera_space = (gaussian_cloud.means - translation) @ rotation
    depth = -camera_space[:, 2]

    valid = depth > 1e-4
    if not bool(valid.any()):
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        return ProjectedGaussians(
            pixels_x=empty_i32,
            pixels_y=empty_i32,
            radii=empty_i32,
            opacity=torch.empty((0,), dtype=torch.float32, device=device),
            rgb=torch.empty((0, 3), dtype=torch.float32, device=device),
        )

    camera_space = camera_space[valid]
    depth = depth[valid]
    opacity = gaussian_cloud.opacity[valid]
    scale = gaussian_cloud.scale[valid]
    rgb = gaussian_cloud.rgb[valid]

    fy = height / (2.0 * math.tan(math.radians(vertical_fov_degrees) * 0.5))
    fx = fy
    cx = width * 0.5
    cy = height * 0.5

    pixels_x = torch.round(fx * (camera_space[:, 0] / depth) + cx).to(dtype=torch.int32)
    pixels_y = torch.round(cy - fy * (camera_space[:, 1] / depth)).to(dtype=torch.int32)
    radii = torch.clamp(
        torch.round(fx * scale / depth * GAUSSIAN_RADIUS_SCALE),
        1,
        MAX_GAUSSIAN_RADIUS,
    ).to(dtype=torch.int32)

    in_bounds = (pixels_x >= -MAX_GAUSSIAN_RADIUS) & (pixels_x < width + MAX_GAUSSIAN_RADIUS)
    in_bounds &= (pixels_y >= -MAX_GAUSSIAN_RADIUS) & (pixels_y < height + MAX_GAUSSIAN_RADIUS)
    if not bool(in_bounds.any()):
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        return ProjectedGaussians(
            pixels_x=empty_i32,
            pixels_y=empty_i32,
            radii=empty_i32,
            opacity=torch.empty((0,), dtype=torch.float32, device=device),
            rgb=torch.empty((0, 3), dtype=torch.float32, device=device),
        )

    pixels_x = pixels_x[in_bounds]
    pixels_y = pixels_y[in_bounds]
    radii = radii[in_bounds]
    depth = depth[in_bounds]
    opacity = opacity[in_bounds]
    rgb = rgb[in_bounds]

    sort_order = torch.argsort(depth, descending=True)
    return ProjectedGaussians(
        pixels_x=pixels_x[sort_order],
        pixels_y=pixels_y[sort_order],
        radii=radii[sort_order],
        opacity=opacity[sort_order],
        rgb=rgb[sort_order],
    )


def precompute_disk_offsets(max_radius: int) -> list[tuple[int, int, int, float]]:
    offsets = []
    for dx in range(-max_radius, max_radius + 1):
        for dy in range(-max_radius, max_radius + 1):
            distance = math.sqrt(dx * dx + dy * dy)
            required_radius = int(math.ceil(distance))
            if required_radius <= max_radius:
                sigma = max(0.85, 0.55 * max(required_radius, 1))
                shell_weight = math.exp(-(distance * distance) / (2.0 * sigma * sigma))
                offsets.append((dx, dy, required_radius, shell_weight))
    offsets.sort(key=lambda item: (-item[2], -(abs(item[0]) + abs(item[1]))))
    return offsets


DISK_OFFSETS = precompute_disk_offsets(MAX_GAUSSIAN_RADIUS)


def keep_last_unique(indices):
    torch = require_torch()
    if int(indices.numel()) == 0:
        return torch.empty((0,), dtype=torch.int64, device=indices.device)
    order = torch.argsort(indices, stable=True)
    sorted_indices = indices[order]
    keep = torch.empty(sorted_indices.shape[0], dtype=torch.bool, device=indices.device)
    keep[-1] = True
    keep[:-1] = sorted_indices[:-1] != sorted_indices[1:]
    return order[keep]


def render_custom_frame(projected: ProjectedGaussians, width: int, height: int) -> np.ndarray:
    torch = require_torch()
    device = projected.pixels_x.device

    frame = torch.empty((height * width, 3), dtype=torch.float32, device=device)
    background = torch.tensor(BACKGROUND_COLOR, dtype=torch.float32, device=device)
    frame[:] = background
    if int(projected.pixels_x.numel()) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8) + BACKGROUND_COLOR

    for dx, dy, required_radius, shell_weight in DISK_OFFSETS:
        shell_mask = projected.radii >= required_radius
        if not bool(shell_mask.any()):
            continue

        pixel_x = projected.pixels_x[shell_mask] + dx
        pixel_y = projected.pixels_y[shell_mask] + dy
        valid = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
        if not bool(valid.any()):
            continue

        shell_rgb = projected.rgb[shell_mask][valid]
        flat_indices = pixel_y[valid] * width + pixel_x[valid]
        chosen = keep_last_unique(flat_indices)
        chosen_indices = flat_indices[chosen].to(dtype=torch.int64)
        chosen_alpha = torch.clamp(projected.opacity[shell_mask][valid][chosen] * shell_weight, 0.0, 1.0)
        chosen_rgb = shell_rgb[chosen] * 255.0
        blended = background[None, :] * (1.0 - chosen_alpha[:, None]) + chosen_rgb * chosen_alpha[:, None]
        frame.index_copy_(0, chosen_indices, blended)

    frame_image = frame.reshape(height, width, 3).clamp_(0.0, 255.0).to(dtype=torch.uint8).cpu().numpy()
    image = Image.fromarray(frame_image)
    soft_glow = Image.blend(image, image.filter(ImageFilter.GaussianBlur(radius=0.8)), alpha=0.18)
    return np.array(soft_glow, dtype=np.uint8)


def render_dense_gaussian_frame(projected: ProjectedGaussians, width: int, height: int) -> np.ndarray:
    torch = require_torch()
    device = projected.pixels_x.device
    background = torch.tensor(BACKGROUND_COLOR, dtype=torch.float32, device=device)
    if int(projected.pixels_x.numel()) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8) + BACKGROUND_COLOR

    color_accum = torch.zeros((height * width, 3), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((height * width,), dtype=torch.float32, device=device)

    # Accumulate many gaussian-center hits per pixel to create a denser, more splat-like companion video.
    for dx, dy, required_radius, shell_weight in DISK_OFFSETS:
        shell_mask = projected.radii >= required_radius
        if not bool(shell_mask.any()):
            continue

        pixel_x = projected.pixels_x[shell_mask] + dx
        pixel_y = projected.pixels_y[shell_mask] + dy
        valid = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
        if not bool(valid.any()):
            continue

        flat_indices = (pixel_y[valid] * width + pixel_x[valid]).to(dtype=torch.int64)
        shell_rgb = projected.rgb[shell_mask][valid]
        shell_opacity = projected.opacity[shell_mask][valid]
        shell_radii = projected.radii[shell_mask][valid].to(dtype=torch.float32)

        weights = shell_opacity * shell_weight * (0.75 + 0.25 * torch.sqrt(shell_radii))
        color_accum.index_add_(0, flat_indices, shell_rgb * weights[:, None])
        weight_accum.index_add_(0, flat_indices, weights)

    frame = torch.empty((height * width, 3), dtype=torch.float32, device=device)
    frame[:] = background

    valid_pixels = weight_accum > 1e-6
    if bool(valid_pixels.any()):
        foreground = torch.zeros_like(frame)
        foreground[valid_pixels] = color_accum[valid_pixels] / weight_accum[valid_pixels, None]
        foreground *= 255.0
        alpha = torch.clamp(1.0 - torch.exp(-weight_accum * 0.9), 0.0, 0.985)
        frame = background[None, :] * (1.0 - alpha[:, None]) + foreground * alpha[:, None]

    frame_image = frame.reshape(height, width, 3).clamp_(0.0, 255.0).to(dtype=torch.uint8).cpu().numpy()
    image = Image.fromarray(frame_image)
    soft_glow = Image.blend(image, image.filter(ImageFilter.GaussianBlur(radius=1.25)), alpha=0.32)
    return np.array(soft_glow, dtype=np.uint8)


def start_ffmpeg_process(width: int, height: int, fps: float, output_path: Path) -> subprocess.Popen:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)


def render_outputs(
    gaussian_cloud: GaussianCloud,
    load_config_path: Path,
    camera_path_data: dict,
    output_path: Path,
    output_fps: float | None = None,
    generate_images: bool = True,
) -> tuple[Path, Path | None]:
    width = int(camera_path_data["render_width"])
    height = int(camera_path_data["render_height"])
    fps = float(output_fps if output_fps is not None else camera_path_data.get("fps", 30.0))
    frames = camera_path_data["camera_path"]
    if not frames:
        raise ValueError("Camera path contains no frames.")
    if fps <= 0.0:
        raise ValueError(f"FPS must be positive, got {fps}")

    dense_output_path = derive_dense_output_path(output_path)
    image_output_path = derive_image_output_path(output_path) if generate_images else None
    custom_process = start_ffmpeg_process(width=width, height=height, fps=fps, output_path=output_path)
    dense_process = start_ffmpeg_process(width=width, height=height, fps=fps, output_path=dense_output_path)
    image_process = (
        start_ffmpeg_process(width=width, height=height, fps=fps, output_path=image_output_path)
        if image_output_path is not None
        else None
    )
    training_images = load_training_image_library(load_config_path) if generate_images else None
    image_cache: dict[int, np.ndarray] = {}
    try:
        for frame in frames:
            camera_to_world = np.array(frame["camera_to_world"], dtype=np.float64).reshape(4, 4)
            vertical_fov = float(frame["fov"])
            projected = project_gaussians(
                gaussian_cloud=gaussian_cloud,
                camera_to_world=camera_to_world,
                width=width,
                height=height,
                vertical_fov_degrees=vertical_fov,
            )
            custom_image = render_custom_frame(projected=projected, width=width, height=height)
            dense_image = render_dense_gaussian_frame(projected=projected, width=width, height=height)

            assert custom_process.stdin is not None
            custom_process.stdin.write(custom_image.tobytes())
            assert dense_process.stdin is not None
            dense_process.stdin.write(dense_image.tobytes())
            if image_process is not None and training_images is not None:
                image_frame = render_image_frame(
                    camera_to_world=camera_to_world,
                    training_images=training_images,
                    width=width,
                    height=height,
                    cache=image_cache,
                )
                assert image_process.stdin is not None
                image_process.stdin.write(image_frame.tobytes())
    finally:
        processes = [custom_process, dense_process]
        if image_process is not None:
            processes.append(image_process)

        for process in processes:
            if process.stdin is not None:
                process.stdin.close()
        labeled_processes = [(custom_process, str(output_path)), (dense_process, str(dense_output_path))]
        if image_process is not None and image_output_path is not None:
            labeled_processes.append((image_process, str(image_output_path)))
        for process, label in labeled_processes:
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for {label} with exit code {process.returncode}")

    return dense_output_path, image_output_path


def main() -> None:
    args = parse_args()
    if not args.load_config.is_file():
        raise SystemExit(f"Config file does not exist: {args.load_config}")

    torch = require_torch()
    device = select_device(torch)
    checkpoint_path = resolve_checkpoint_path(args.load_config)
    camera_path = resolve_camera_path(args.camera_path, args.load_config)
    if not camera_path.is_file():
        raise SystemExit(f"Camera path file does not exist: {camera_path}")

    gaussian_cloud = load_checkpoint_gaussians(checkpoint_path, device=device)
    camera_path_data = load_camera_path(camera_path)
    device_label = f"{device.type}:{device.index}" if device.index is not None else device.type
    print(f"Using device: {device_label}")
    if device.type == "cuda":
        print(f"CUDA GPU: {torch.cuda.get_device_name(device)}")
    print(f"Rendering dense-gaussian video from {checkpoint_path.name} with {gaussian_cloud.means.shape[0]} gaussians")
    dense_output_path = derive_dense_output_path(args.output_path)
    print(f"Camera path: {camera_path} -> {args.output_path}")
    print(f"Dense-gaussian companion output: {dense_output_path}")
    if args.generate_images:
        print(f"Image companion output: {derive_image_output_path(args.output_path)}")
    if args.fps is not None:
        print(f"Output FPS override: {args.fps}")
    with torch.inference_mode():
        render_outputs(
            gaussian_cloud,
            args.load_config,
            camera_path_data,
            args.output_path,
            output_fps=args.fps,
            generate_images=args.generate_images,
        )


if __name__ == "__main__":
    main()
