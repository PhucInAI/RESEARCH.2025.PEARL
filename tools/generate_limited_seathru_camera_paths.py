#!/usr/bin/env python3
"""Generate limited-angle, sparse, slightly jittered camera paths for SeaThru scenes.

These paths are meant to visualize the "images alone are limited" story:
- only part of the original capture arc is kept
- keyframes are reduced to create incomplete coverage
- small pose / fov jitter is added to make the path feel less idealized
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from pearl.ai_logger import aiLogger

from generate_seathru_camera_paths import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_FPS,
    build_viewer_camera_path,
    colmap_image_to_camera_to_world,
    compute_dataparser_transform_and_scale,
    convert_camera_to_world_opencv_to_opengl,
    get_intrinsics,
    interpolate_path,
    iter_target_scenes,
    read_cameras_binary,
    read_images_binary,
    transform_pose_to_nerfstudio_space,
    vertical_fov_degrees,
)

DEFAULT_OUTPUT_NAME = "limited_trajectory.json"
LIMITED_CAPTURE_RATIO = 0.45
TARGET_KEYFRAME_RATIO = 0.45
MAX_LIMITED_KEYFRAMES = 9
LIMITED_FRAMES_PER_TRANSITION = 18
TRANSLATION_JITTER_RATIO = 0.015
ROTATION_JITTER_STD_DEGREES = np.array([0.8, 1.2, 0.8], dtype=np.float64)
FOV_JITTER_STD_DEGREES = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing SeaThru scenes.",
    )
    return parser.parse_args()


def deterministic_seed(scene_name: str) -> int:
    seed = 0
    for character in scene_name.encode("utf-8"):
        seed = (seed * 131 + int(character)) % (2**32)
    return seed


def select_limited_window(origins: np.ndarray, window_size: int) -> np.ndarray:
    if len(origins) <= window_size:
        return np.arange(len(origins), dtype=np.int32)

    scene_center = np.median(origins, axis=0)
    azimuths = np.unwrap(np.arctan2(origins[:, 1] - scene_center[1], origins[:, 0] - scene_center[0]))

    best_start = 0
    best_score: tuple[float, float] | None = None
    for start_index in range(len(origins) - window_size + 1):
        stop_index = start_index + window_size
        window_azimuths = azimuths[start_index:stop_index]
        window_origins = origins[start_index:stop_index]

        azimuth_span = float(np.ptp(window_azimuths))
        compactness = float(np.sum(np.linalg.norm(window_origins - np.median(window_origins, axis=0), axis=1)))
        score = (azimuth_span, compactness)
        if best_score is None or score < best_score:
            best_start = start_index
            best_score = score

    return np.arange(best_start, best_start + window_size, dtype=np.int32)


def select_sparse_keyframes(frame_count: int) -> np.ndarray:
    if frame_count <= 2:
        return np.arange(frame_count, dtype=np.int32)

    target_count = max(5, min(MAX_LIMITED_KEYFRAMES, int(round(frame_count * TARGET_KEYFRAME_RATIO))))
    indices = np.unique(np.round(np.linspace(0, frame_count - 1, target_count)).astype(np.int32))

    # Drop one interior keyframe when possible so the interpolated path clearly shows missing coverage.
    if len(indices) >= 8:
        drop_index = len(indices) // 2
        indices = np.delete(indices, drop_index)

    return indices


def jitter_keyframes(
    poses: np.ndarray,
    fovs: np.ndarray,
    scene_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(poses) <= 2:
        return poses.copy(), fovs.copy()

    rng = np.random.default_rng(deterministic_seed(scene_name))
    jittered_poses = poses.copy()
    jittered_fovs = fovs.copy()

    origins = poses[:, :3, 3]
    scene_scale = max(float(np.percentile(np.linalg.norm(origins - np.median(origins, axis=0), axis=1), 75.0)), 0.05)
    translation_std = TRANSLATION_JITTER_RATIO * scene_scale

    for pose_index in range(1, len(poses) - 1):
        pose = poses[pose_index].copy()
        translation_noise = rng.normal(scale=translation_std, size=3)
        translation_noise[2] *= 0.6
        pose[:3, 3] += translation_noise

        rotation_noise = Rotation.from_euler(
            "xyz",
            rng.normal(scale=ROTATION_JITTER_STD_DEGREES, size=3),
            degrees=True,
        ).as_matrix()
        pose[:3, :3] = Rotation.from_matrix(pose[:3, :3] @ rotation_noise).as_matrix()
        jittered_poses[pose_index] = pose
        jittered_fovs[pose_index] = float(np.clip(fovs[pose_index] + rng.normal(scale=FOV_JITTER_STD_DEGREES), 35.0, 80.0))

    return jittered_poses, jittered_fovs


def generate_scene_limited_camera_path(scene_dir: Path) -> Path:
    cameras = read_cameras_binary(scene_dir / "sparse" / "cameras.bin")
    images = read_images_binary(scene_dir / "sparse" / "images.bin")
    ordered_images = sorted(images.values(), key=lambda image: image.name)
    if not ordered_images:
        raise ValueError(f"No registered images found in {scene_dir / 'sparse' / 'images.bin'}")

    opengl_poses = np.stack(
        [convert_camera_to_world_opencv_to_opengl(colmap_image_to_camera_to_world(image)) for image in ordered_images],
        axis=0,
    )
    dataparser_transform, dataparser_scale = compute_dataparser_transform_and_scale(opengl_poses)

    poses = []
    fovs = []
    width = None
    height = None

    for image in ordered_images:
        camera = cameras[image.camera_id]
        _fx, fy, _cx, _cy = get_intrinsics(camera)
        pose_opencv = colmap_image_to_camera_to_world(image)
        poses.append(transform_pose_to_nerfstudio_space(pose_opencv, dataparser_transform, dataparser_scale))
        fovs.append(vertical_fov_degrees(camera.height, fy))
        width = camera.width
        height = camera.height

    assert width is not None and height is not None

    poses_array = np.stack(poses, axis=0)
    fovs_array = np.array(fovs, dtype=np.float64)

    window_size = min(len(poses_array), max(6, int(round(len(poses_array) * LIMITED_CAPTURE_RATIO))))
    limited_window_indices = select_limited_window(poses_array[:, :3, 3], window_size=window_size)
    limited_poses = poses_array[limited_window_indices]
    limited_fovs = fovs_array[limited_window_indices]

    sparse_keyframe_indices = select_sparse_keyframes(len(limited_poses))
    keyframe_poses = limited_poses[sparse_keyframe_indices]
    keyframe_fovs = limited_fovs[sparse_keyframe_indices]
    keyframe_poses, keyframe_fovs = jitter_keyframes(keyframe_poses, keyframe_fovs, scene_name=scene_dir.name)

    dense_poses, dense_fovs = interpolate_path(
        keyframe_poses,
        keyframe_fovs,
        LIMITED_FRAMES_PER_TRANSITION,
    )

    viewer_path = build_viewer_camera_path(
        poses_keyframes=keyframe_poses,
        fovs_keyframes=keyframe_fovs,
        poses_dense=dense_poses,
        fovs_dense=dense_fovs,
        width=width,
        height=height,
        fps=DEFAULT_FPS,
        frames_per_transition=LIMITED_FRAMES_PER_TRANSITION,
    )

    output_dir = scene_dir / "camera_paths"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DEFAULT_OUTPUT_NAME
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(viewer_path, file, indent=2)
        file.write("\n")
    return output_path


def main() -> None:
    args = parse_args()
    if not args.dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {args.dataset_root}")

    scene_dirs = iter_target_scenes(args.dataset_root)
    if not scene_dirs:
        raise SystemExit("No scene directories found.")

    for scene_dir in scene_dirs:
        output_path = generate_scene_limited_camera_path(scene_dir)
        aiLogger.info(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
