#!/usr/bin/env python3
"""Render a sparse-to-dense 3DGS progress video from a Nerfstudio camera path.

The main output shows a staged reconstruction story:
- sparse COLMAP points
- an early 3DGS subset
- a denser 3DGS subset
- the full dense gaussian model

It accepts the same inputs as ``render_seathru_camera_paths.py`` so it can be dropped
into the same workflow.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pearl.ai_logger import aiLogger

from generate_seathru_camera_paths import (
    colmap_image_to_camera_to_world,
    compute_dataparser_transform_and_scale,
    convert_camera_to_world_opencv_to_opengl,
    read_images_binary,
)
from render_seathru_camera_paths import (
    GaussianCloud,
    derive_image_output_path,
    load_camera_path,
    load_checkpoint_gaussians,
    load_training_image_library,
    parse_dataset_path_from_config,
    project_gaussians,
    render_custom_frame,
    render_dense_gaussian_frame,
    render_image_frame,
    require_torch,
    resolve_camera_path,
    resolve_checkpoint_path,
    select_device,
    start_ffmpeg_process,
)

PROGRESS_SEED = 20260328
SPARSE_OPACITY = 0.95
SPARSE_SCALE_DIVISOR = 450.0
EARLY_STAGE_FRACTION = 0.05
MID_STAGE_FRACTION = 0.20


@dataclass(frozen=True)
class SparsePointCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    error: np.ndarray
    track_length: np.ndarray


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
        help="Path to the sparse-to-dense progress video output.",
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


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_sparse_points3d_binary(path_to_model_file: Path) -> SparsePointCloud:
    xyz = []
    rgb = []
    error = []
    track_length = []
    with path_to_model_file.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            _point_id, x, y, z, red, green, blue, point_error = read_next_bytes(fid, 43, "QdddBBBd")
            point_track_length = int(read_next_bytes(fid, 8, "Q")[0])
            fid.seek(8 * point_track_length, 1)
            xyz.append((x, y, z))
            rgb.append((red, green, blue))
            error.append(point_error)
            track_length.append(point_track_length)

    return SparsePointCloud(
        xyz=np.array(xyz, dtype=np.float64),
        rgb=np.array(rgb, dtype=np.uint8),
        error=np.array(error, dtype=np.float64),
        track_length=np.array(track_length, dtype=np.int32),
    )


def transform_points_to_nerfstudio_space(
    xyz: np.ndarray,
    dataparser_transform: np.ndarray,
    dataparser_scale: float,
) -> np.ndarray:
    homogeneous = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    transformed = (dataparser_transform @ homogeneous.T).T[:, :3]
    transformed *= dataparser_scale
    return transformed


def filter_sparse_point_cloud(point_cloud: SparsePointCloud) -> tuple[np.ndarray, np.ndarray]:
    mask = np.all(np.isfinite(point_cloud.xyz), axis=1)
    finite_error = point_cloud.error[np.isfinite(point_cloud.error)]
    if finite_error.size > 0:
        mask &= point_cloud.error <= np.percentile(finite_error, 90.0)
    positive_tracks = point_cloud.track_length[point_cloud.track_length > 0]
    if positive_tracks.size > 0:
        mask &= point_cloud.track_length >= max(2, int(np.percentile(positive_tracks, 25.0)))

    filtered_xyz = point_cloud.xyz[mask]
    filtered_rgb = point_cloud.rgb[mask]
    if len(filtered_xyz) < 256:
        mask = np.all(np.isfinite(point_cloud.xyz), axis=1)
        filtered_xyz = point_cloud.xyz[mask]
        filtered_rgb = point_cloud.rgb[mask]
    if len(filtered_xyz) == 0:
        raise ValueError("Sparse COLMAP point cloud is empty after filtering.")

    lower = np.percentile(filtered_xyz, 5.0, axis=0)
    upper = np.percentile(filtered_xyz, 95.0, axis=0)
    box_mask = np.all((filtered_xyz >= lower) & (filtered_xyz <= upper), axis=1)
    if int(box_mask.sum()) >= 256:
        filtered_xyz = filtered_xyz[box_mask]
        filtered_rgb = filtered_rgb[box_mask]

    return filtered_xyz, filtered_rgb


def load_sparse_gaussian_cloud(load_config_path: Path, device) -> GaussianCloud:
    dataset_path = parse_dataset_path_from_config(load_config_path)
    if dataset_path is None:
        raise SystemExit(f"Could not find dataset path inside config: {load_config_path}")

    sparse_points = read_sparse_points3d_binary(dataset_path / "sparse" / "points3D.bin")
    images = read_images_binary(dataset_path / "sparse" / "images.bin")
    ordered_images = sorted(images.values(), key=lambda image: image.name)
    if not ordered_images:
        raise ValueError(f"No registered images found in {dataset_path / 'sparse' / 'images.bin'}")

    opengl_poses = np.stack(
        [convert_camera_to_world_opencv_to_opengl(colmap_image_to_camera_to_world(image)) for image in ordered_images],
        axis=0,
    )
    dataparser_transform, dataparser_scale = compute_dataparser_transform_and_scale(opengl_poses)

    transformed_xyz = transform_points_to_nerfstudio_space(
        sparse_points.xyz,
        dataparser_transform=dataparser_transform,
        dataparser_scale=dataparser_scale,
    )
    filtered_xyz, filtered_rgb = filter_sparse_point_cloud(
        SparsePointCloud(
            xyz=transformed_xyz,
            rgb=sparse_points.rgb,
            error=sparse_points.error,
            track_length=sparse_points.track_length,
        )
    )

    scene_scale = max(
        float(np.percentile(np.linalg.norm(filtered_xyz - np.median(filtered_xyz, axis=0), axis=1), 80.0)),
        0.05,
    )
    sparse_scale = max(scene_scale / SPARSE_SCALE_DIVISOR, 0.002)

    torch = require_torch()
    return GaussianCloud(
        means=torch.as_tensor(filtered_xyz, dtype=torch.float32, device=device),
        rgb=torch.as_tensor(filtered_rgb.astype(np.float32) / 255.0, dtype=torch.float32, device=device),
        opacity=torch.full((len(filtered_xyz),), SPARSE_OPACITY, dtype=torch.float32, device=device),
        scale=torch.full((len(filtered_xyz),), sparse_scale, dtype=torch.float32, device=device),
    )


def subset_gaussian_cloud(gaussian_cloud: GaussianCloud, indices) -> GaussianCloud:
    return GaussianCloud(
        means=gaussian_cloud.means[indices],
        rgb=gaussian_cloud.rgb[indices],
        opacity=gaussian_cloud.opacity[indices],
        scale=gaussian_cloud.scale[indices],
    )


def build_dense_stage_clouds(gaussian_cloud: GaussianCloud) -> tuple[GaussianCloud, GaussianCloud, GaussianCloud]:
    torch = require_torch()
    total_count = int(gaussian_cloud.means.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROGRESS_SEED)
    permutation = torch.randperm(total_count, generator=generator).to(device=gaussian_cloud.means.device)

    early_count = max(1, int(round(total_count * EARLY_STAGE_FRACTION)))
    mid_count = max(early_count + 1, int(round(total_count * MID_STAGE_FRACTION)))
    early_indices = permutation[:early_count]
    mid_indices = permutation[:mid_count]
    return (
        subset_gaussian_cloud(gaussian_cloud, early_indices),
        subset_gaussian_cloud(gaussian_cloud, mid_indices),
        gaussian_cloud,
    )


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0
    x = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def blend_frames(frame_a: np.ndarray, frame_b: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return frame_a
    if alpha >= 1.0:
        return frame_b
    blended = (1.0 - alpha) * frame_a.astype(np.float32) + alpha * frame_b.astype(np.float32)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def annotate_frame(frame: np.ndarray, label: str, progress: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    draw.rounded_rectangle((20, 20, 250, 62), radius=12, fill=(10, 16, 22))
    draw.text((34, 32), label, fill=(230, 238, 245))

    bar_left = 20
    bar_right = width - 20
    bar_top = height - 28
    bar_bottom = height - 16
    draw.rounded_rectangle((bar_left, bar_top, bar_right, bar_bottom), radius=6, fill=(18, 24, 30))
    filled_right = bar_left + int(round((bar_right - bar_left) * np.clip(progress, 0.0, 1.0)))
    draw.rounded_rectangle((bar_left, bar_top, filled_right, bar_bottom), radius=6, fill=(74, 189, 255))
    return np.array(image, dtype=np.uint8)


def render_sparse_frame(
    sparse_cloud: GaussianCloud,
    camera_to_world: np.ndarray,
    width: int,
    height: int,
    vertical_fov: float,
) -> np.ndarray:
    sparse_projection = project_gaussians(
        gaussian_cloud=sparse_cloud,
        camera_to_world=camera_to_world,
        width=width,
        height=height,
        vertical_fov_degrees=vertical_fov,
    )
    return render_custom_frame(projected=sparse_projection, width=width, height=height)


def render_dense_stage_frame(
    dense_cloud: GaussianCloud,
    camera_to_world: np.ndarray,
    width: int,
    height: int,
    vertical_fov: float,
) -> np.ndarray:
    dense_projection = project_gaussians(
        gaussian_cloud=dense_cloud,
        camera_to_world=camera_to_world,
        width=width,
        height=height,
        vertical_fov_degrees=vertical_fov,
    )
    return render_dense_gaussian_frame(projected=dense_projection, width=width, height=height)


def render_progress_frame(
    frame_index: int,
    frame_count: int,
    sparse_cloud: GaussianCloud,
    early_dense_cloud: GaussianCloud,
    mid_dense_cloud: GaussianCloud,
    full_dense_cloud: GaussianCloud,
    camera_to_world: np.ndarray,
    width: int,
    height: int,
    vertical_fov: float,
) -> np.ndarray:
    progress = 1.0 if frame_count <= 1 else frame_index / float(frame_count - 1)

    if progress <= 0.20:
        frame = render_sparse_frame(sparse_cloud, camera_to_world, width, height, vertical_fov)
        label = "Sparse COLMAP"
    elif progress <= 0.45:
        alpha = smoothstep(0.20, 0.45, progress)
        sparse_frame = render_sparse_frame(sparse_cloud, camera_to_world, width, height, vertical_fov)
        early_frame = render_dense_stage_frame(early_dense_cloud, camera_to_world, width, height, vertical_fov)
        frame = blend_frames(sparse_frame, early_frame, alpha)
        label = "3DGS Early"
    elif progress <= 0.75:
        alpha = smoothstep(0.45, 0.75, progress)
        early_frame = render_dense_stage_frame(early_dense_cloud, camera_to_world, width, height, vertical_fov)
        mid_frame = render_dense_stage_frame(mid_dense_cloud, camera_to_world, width, height, vertical_fov)
        frame = blend_frames(early_frame, mid_frame, alpha)
        label = "3DGS Growing"
    else:
        alpha = smoothstep(0.75, 1.0, progress)
        mid_frame = render_dense_stage_frame(mid_dense_cloud, camera_to_world, width, height, vertical_fov)
        full_frame = render_dense_stage_frame(full_dense_cloud, camera_to_world, width, height, vertical_fov)
        frame = blend_frames(mid_frame, full_frame, alpha)
        label = "3DGS Dense"

    return annotate_frame(frame, label=label, progress=progress)


def render_progress_outputs(
    sparse_cloud: GaussianCloud,
    full_dense_cloud: GaussianCloud,
    load_config_path: Path,
    camera_path_data: dict,
    output_path: Path,
    output_fps: float | None = None,
    generate_images: bool = True,
) -> Path | None:
    width = int(camera_path_data["render_width"])
    height = int(camera_path_data["render_height"])
    fps = float(output_fps if output_fps is not None else camera_path_data.get("fps", 30.0))
    frames = camera_path_data["camera_path"]
    if not frames:
        raise ValueError("Camera path contains no frames.")
    if fps <= 0.0:
        raise ValueError(f"FPS must be positive, got {fps}")

    early_dense_cloud, mid_dense_cloud, full_dense_cloud = build_dense_stage_clouds(full_dense_cloud)
    progress_process = start_ffmpeg_process(width=width, height=height, fps=fps, output_path=output_path)
    image_output_path = derive_image_output_path(output_path) if generate_images else None
    image_process = (
        start_ffmpeg_process(width=width, height=height, fps=fps, output_path=image_output_path)
        if image_output_path is not None
        else None
    )
    training_images = load_training_image_library(load_config_path) if generate_images else None
    image_cache: dict[int, np.ndarray] = {}

    try:
        for frame_index, frame in enumerate(frames):
            camera_to_world = np.array(frame["camera_to_world"], dtype=np.float64).reshape(4, 4)
            vertical_fov = float(frame["fov"])
            progress_frame = render_progress_frame(
                frame_index=frame_index,
                frame_count=len(frames),
                sparse_cloud=sparse_cloud,
                early_dense_cloud=early_dense_cloud,
                mid_dense_cloud=mid_dense_cloud,
                full_dense_cloud=full_dense_cloud,
                camera_to_world=camera_to_world,
                width=width,
                height=height,
                vertical_fov=vertical_fov,
            )
            assert progress_process.stdin is not None
            progress_process.stdin.write(progress_frame.tobytes())

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
        processes = [progress_process]
        if image_process is not None:
            processes.append(image_process)
        for process in processes:
            if process.stdin is not None:
                process.stdin.close()

        labeled_processes = [(progress_process, str(output_path))]
        if image_process is not None and image_output_path is not None:
            labeled_processes.append((image_process, str(image_output_path)))
        for process, label in labeled_processes:
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for {label} with exit code {process.returncode}")

    return image_output_path


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

    sparse_cloud = load_sparse_gaussian_cloud(args.load_config, device=device)
    dense_cloud = load_checkpoint_gaussians(checkpoint_path, device=device)
    camera_path_data = load_camera_path(camera_path)

    device_label = f"{device.type}:{device.index}" if device.index is not None else device.type
    aiLogger.info(f"Using device: {device_label}")
    if device.type == "cuda":
        aiLogger.info(f"CUDA GPU: {torch.cuda.get_device_name(device)}")
    aiLogger.info(
        "Rendering sparse-to-dense progress video "
        f"with {sparse_cloud.means.shape[0]} sparse points and {dense_cloud.means.shape[0]} dense gaussians"
    )
    aiLogger.info(f"Camera path: {camera_path} -> {args.output_path}")
    if args.generate_images:
        aiLogger.info(f"Image companion output: {derive_image_output_path(args.output_path)}")
    if args.fps is not None:
        aiLogger.info(f"Output FPS override: {args.fps}")

    with torch.inference_mode():
        render_progress_outputs(
            sparse_cloud=sparse_cloud,
            full_dense_cloud=dense_cloud,
            load_config_path=args.load_config,
            camera_path_data=camera_path_data,
            output_path=args.output_path,
            output_fps=args.fps,
            generate_images=args.generate_images,
        )


if __name__ == "__main__":
    main()
