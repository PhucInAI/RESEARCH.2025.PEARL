#!/usr/bin/env python3
"""Generate polished virtual camera paths for SeaThru scenes."""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from generate_seathru_camera_paths import (
    DEFAULT_FPS,
    build_viewer_camera_path,
    colmap_image_to_camera_to_world,
    compute_dataparser_transform_and_scale,
    convert_camera_to_world_opencv_to_opengl,
    get_intrinsics,
    iter_target_scenes,
    normalize_vector,
    read_cameras_binary,
    read_images_binary,
    transform_pose_to_nerfstudio_space,
    vertical_fov_degrees,
)

DEFAULT_OUTPUT_NAME = "virtual_trajectory.json"
DEFAULT_DURATION_SECONDS = 8.0
DEFAULT_DENSE_FRAMES = int(DEFAULT_FPS * DEFAULT_DURATION_SECONDS)
DEFAULT_SPLINE_SAMPLES = 1024


@dataclass(frozen=True)
class PointCloud:
    xyz: np.ndarray
    error: np.ndarray
    track_length: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/undistorted_seathrunerf_dataset"),
        help="Root directory containing SeaThru scenes.",
    )
    return parser.parse_args()


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3d_binary(path_to_model_file: Path) -> PointCloud:
    xyz = []
    error = []
    track_length = []
    with path_to_model_file.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            _point_id, x, y, z, _r, _g, _b, point_error = read_next_bytes(fid, 43, "QdddBBBd")
            point_track_length = int(read_next_bytes(fid, 8, "Q")[0])
            fid.seek(8 * point_track_length, 1)
            xyz.append((x, y, z))
            error.append(point_error)
            track_length.append(point_track_length)

    return PointCloud(
        xyz=np.array(xyz, dtype=np.float64),
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


def filter_point_cloud(point_cloud: PointCloud) -> np.ndarray:
    xyz = point_cloud.xyz
    if len(xyz) == 0:
        raise ValueError("Point cloud is empty.")

    mask = np.all(np.isfinite(xyz), axis=1)
    if np.any(np.isfinite(point_cloud.error)):
        mask &= point_cloud.error <= np.percentile(point_cloud.error[np.isfinite(point_cloud.error)], 90.0)
    if np.any(point_cloud.track_length > 0):
        mask &= point_cloud.track_length >= max(2, int(np.percentile(point_cloud.track_length, 25.0)))

    filtered = xyz[mask]
    if len(filtered) < 256:
        filtered = xyz[np.all(np.isfinite(xyz), axis=1)]

    lower = np.percentile(filtered, 5.0, axis=0)
    upper = np.percentile(filtered, 95.0, axis=0)
    box_mask = np.all((filtered >= lower) & (filtered <= upper), axis=1)
    if int(box_mask.sum()) >= 256:
        filtered = filtered[box_mask]

    return filtered


def compute_ray_focus(camera_origins: np.ndarray, camera_directions: np.ndarray) -> np.ndarray:
    system_matrix = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)

    for origin, direction in zip(camera_origins, camera_directions):
        direction = normalize_vector(direction)
        projection = identity - np.outer(direction, direction)
        system_matrix += projection
        rhs += projection @ origin

    system_matrix += 1e-6 * identity
    return np.linalg.solve(system_matrix, rhs)


def remove_camera_outliers(camera_poses: np.ndarray) -> np.ndarray:
    origins = camera_poses[:, :3, 3]
    median_origin = np.median(origins, axis=0)
    distances = np.linalg.norm(origins - median_origin, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    threshold = median_distance + 3.5 * 1.4826 * max(mad, 1e-6)
    mask = distances <= threshold
    if int(mask.sum()) < max(4, len(camera_poses) // 2):
        return camera_poses
    return camera_poses[mask]


def reduce_redundancy(points: np.ndarray, min_step: float) -> np.ndarray:
    if len(points) <= 2:
        return points

    kept = [points[0]]
    last_point = points[0]
    for point in points[1:-1]:
        if np.linalg.norm(point - last_point) >= min_step:
            kept.append(point)
            last_point = point
    kept.append(points[-1])
    return np.stack(kept, axis=0)


def smooth_polyline(points: np.ndarray, passes: int = 2) -> np.ndarray:
    if len(points) < 3:
        return points.copy()

    smoothed = points.copy()
    for _ in range(passes):
        updated = smoothed.copy()
        updated[1:-1] = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
        smoothed = updated
    return smoothed


def normalize_xy(vector_xy: np.ndarray) -> np.ndarray:
    vector = np.array([vector_xy[0], vector_xy[1], 0.0], dtype=np.float64)
    return normalize_vector(vector)


def orthogonal_xy(direction: np.ndarray) -> np.ndarray:
    return np.array([-direction[1], direction[0], 0.0], dtype=np.float64)


def clamp_vector(vector: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(vector, lower), upper)


def look_at_pose(position: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    forward = normalize_vector(target - position)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
    right = normalize_vector(right)
    up = normalize_vector(np.cross(right, forward))

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = position
    return pose


def sample_constant_speed_spline(
    anchor_positions: np.ndarray,
    anchor_targets: np.ndarray,
    anchor_fovs: np.ndarray,
    frame_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(anchor_positions) < 2:
        raise ValueError("Need at least two anchor positions for spline sampling.")

    anchor_times = np.linspace(0.0, 1.0, len(anchor_positions), dtype=np.float64)
    dense_times = np.linspace(0.0, 1.0, max(DEFAULT_SPLINE_SAMPLES, frame_count * 4), dtype=np.float64)

    position_spline = CubicSpline(anchor_times, anchor_positions, axis=0, bc_type="clamped")
    target_spline = CubicSpline(anchor_times, anchor_targets, axis=0, bc_type="clamped")
    fov_spline = CubicSpline(anchor_times, anchor_fovs, axis=0, bc_type="clamped")

    dense_positions = position_spline(dense_times)
    segment_lengths = np.linalg.norm(np.diff(dense_positions, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if cumulative_lengths[-1] <= 1e-8:
        sample_times = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    else:
        target_lengths = np.linspace(0.0, cumulative_lengths[-1], frame_count, dtype=np.float64)
        sample_times = np.interp(target_lengths, cumulative_lengths, dense_times)

    positions = position_spline(sample_times)
    targets = target_spline(sample_times)
    fovs = np.clip(fov_spline(sample_times), 35.0, 75.0)
    return positions, targets, fovs


def compute_virtual_path_anchors(
    camera_poses: np.ndarray,
    point_cloud_xyz: np.ndarray,
    median_fov: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_origins = camera_poses[:, :3, 3]
    camera_directions = -camera_poses[:, :3, 2]

    point_lower = np.percentile(point_cloud_xyz, 5.0, axis=0)
    point_upper = np.percentile(point_cloud_xyz, 95.0, axis=0)
    point_center = np.median(point_cloud_xyz, axis=0)
    ray_focus = clamp_vector(compute_ray_focus(camera_origins, camera_directions), point_lower, point_upper)
    focus_center = 0.7 * point_center + 0.3 * ray_focus
    focus_center[2] = 0.5 * focus_center[2] + 0.5 * np.percentile(point_cloud_xyz[:, 2], 60.0)

    cleaned_camera_poses = remove_camera_outliers(camera_poses)
    cleaned_origins = cleaned_camera_poses[:, :3, 3]
    min_step = max(0.05, 0.06 * np.percentile(np.linalg.norm(cleaned_origins - focus_center, axis=1), 50.0))
    representative_origins = smooth_polyline(reduce_redundancy(cleaned_origins, min_step=min_step))

    start_hint = normalize_xy(representative_origins[0, :2] - focus_center[:2])
    end_hint = normalize_xy(representative_origins[-1, :2] - focus_center[:2])
    mean_hint = representative_origins[:, :2] - focus_center[:2]
    mean_hint = normalize_xy(mean_hint.mean(axis=0))
    hero_direction = normalize_vector(0.55 * mean_hint + 0.45 * end_hint)
    side_direction = orthogonal_xy(hero_direction)

    start_side_projection = float(np.mean((representative_origins[: max(1, len(representative_origins) // 3), :2] - focus_center[:2]) @ side_direction[:2]))
    start_side_sign = 1.0 if start_side_projection >= 0.0 else -1.0

    start_direction = normalize_vector(0.5 * start_hint + 0.5 * (hero_direction + 0.7 * start_side_sign * side_direction))
    arc_direction = normalize_vector(0.55 * start_direction + 0.45 * hero_direction)
    approach_direction = normalize_vector(hero_direction + 0.15 * start_side_sign * side_direction)
    sweep_in_direction = normalize_vector(hero_direction + 0.10 * start_side_sign * side_direction)
    sweep_out_direction = normalize_vector(hero_direction - 0.12 * start_side_sign * side_direction)

    camera_radii_xy = np.linalg.norm(cleaned_origins[:, :2] - focus_center[:2], axis=1)
    point_radii_xy = np.linalg.norm(point_cloud_xyz[:, :2] - focus_center[:2], axis=1)
    hero_radius = max(
        np.percentile(point_radii_xy, 75.0),
        0.7 * np.percentile(point_radii_xy, 85.0) + 0.3 * np.percentile(camera_radii_xy, 25.0),
    )
    wide_radius = max(np.percentile(camera_radii_xy, 85.0), hero_radius * 1.35)
    arc_radius = 0.7 * wide_radius + 0.3 * hero_radius
    approach_radius = 1.12 * hero_radius
    sweep_radius = 1.03 * hero_radius

    camera_height_offsets = cleaned_origins[:, 2] - focus_center[2]
    point_height_extent = np.percentile(point_cloud_xyz[:, 2], 85.0) - np.percentile(point_cloud_xyz[:, 2], 15.0)
    hero_height = np.percentile(camera_height_offsets, 45.0)
    wide_height = max(np.percentile(camera_height_offsets, 75.0), hero_height + 0.08 * max(point_height_extent, 1.0))
    approach_height = 0.55 * wide_height + 0.45 * hero_height

    wide_target = focus_center
    hero_target = 0.65 * focus_center + 0.35 * ray_focus
    hero_target = clamp_vector(hero_target, point_lower, point_upper)

    anchor_positions = np.stack(
        [
            focus_center + wide_radius * start_direction + np.array([0.0, 0.0, wide_height], dtype=np.float64),
            focus_center + arc_radius * arc_direction + np.array([0.0, 0.0, wide_height], dtype=np.float64),
            focus_center + approach_radius * approach_direction + np.array([0.0, 0.0, approach_height], dtype=np.float64),
            focus_center + sweep_radius * sweep_in_direction + np.array([0.0, 0.0, hero_height], dtype=np.float64),
            focus_center + sweep_radius * sweep_out_direction + np.array([0.0, 0.0, hero_height], dtype=np.float64),
            focus_center + hero_radius * hero_direction + np.array([0.0, 0.0, hero_height], dtype=np.float64),
        ],
        axis=0,
    )
    anchor_targets = np.stack(
        [
            wide_target,
            wide_target,
            0.8 * wide_target + 0.2 * hero_target,
            0.5 * wide_target + 0.5 * hero_target,
            0.3 * wide_target + 0.7 * hero_target,
            hero_target,
        ],
        axis=0,
    )
    anchor_fovs = np.array(
        [
            min(median_fov + 6.0, 70.0),
            min(median_fov + 4.0, 68.0),
            median_fov + 1.0,
            median_fov,
            max(median_fov - 1.5, 40.0),
            max(median_fov - 2.0, 40.0),
        ],
        dtype=np.float64,
    )

    anchor_poses = np.stack(
        [look_at_pose(position, target, world_up) for position, target in zip(anchor_positions, anchor_targets)],
        axis=0,
    )
    return anchor_poses, anchor_targets, anchor_fovs


def generate_scene_virtual_camera_path(scene_dir: Path) -> Path:
    cameras = read_cameras_binary(scene_dir / "sparse" / "cameras.bin")
    images = read_images_binary(scene_dir / "sparse" / "images.bin")
    point_cloud = read_points3d_binary(scene_dir / "sparse" / "points3D.bin")
    ordered_images = sorted(images.values(), key=lambda image: image.name)
    if not ordered_images:
        raise ValueError(f"No registered images found in {scene_dir / 'sparse' / 'images.bin'}")

    opengl_poses = np.stack(
        [convert_camera_to_world_opencv_to_opengl(colmap_image_to_camera_to_world(image)) for image in ordered_images],
        axis=0,
    )
    dataparser_transform, dataparser_scale = compute_dataparser_transform_and_scale(opengl_poses)

    camera_poses = []
    fovs = []
    width = None
    height = None

    for image in ordered_images:
        camera = cameras[image.camera_id]
        _fx, fy, _cx, _cy = get_intrinsics(camera)
        pose_opencv = colmap_image_to_camera_to_world(image)
        camera_poses.append(transform_pose_to_nerfstudio_space(pose_opencv, dataparser_transform, dataparser_scale))
        fovs.append(vertical_fov_degrees(camera.height, fy))
        width = camera.width
        height = camera.height

    assert width is not None and height is not None

    camera_poses_array = np.stack(camera_poses, axis=0)
    point_cloud_xyz = transform_points_to_nerfstudio_space(
        point_cloud.xyz,
        dataparser_transform=dataparser_transform,
        dataparser_scale=dataparser_scale,
    )
    filtered_point_cloud_xyz = filter_point_cloud(
        PointCloud(
            xyz=point_cloud_xyz,
            error=point_cloud.error,
            track_length=point_cloud.track_length,
        )
    )

    anchor_poses, anchor_targets, anchor_fovs = compute_virtual_path_anchors(
        camera_poses=camera_poses_array,
        point_cloud_xyz=filtered_point_cloud_xyz,
        median_fov=float(np.median(np.array(fovs, dtype=np.float64))),
    )

    anchor_positions = anchor_poses[:, :3, 3]
    dense_positions, dense_targets, dense_fovs = sample_constant_speed_spline(
        anchor_positions=anchor_positions,
        anchor_targets=anchor_targets,
        anchor_fovs=anchor_fovs,
        frame_count=DEFAULT_DENSE_FRAMES,
    )
    dense_poses = np.stack(
        [
            look_at_pose(position, target, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            for position, target in zip(dense_positions, dense_targets)
        ],
        axis=0,
    )

    frames_per_transition = max(1, int(round((len(dense_poses) - 1) / max(1, len(anchor_poses) - 1))))
    viewer_path = build_viewer_camera_path(
        poses_keyframes=anchor_poses,
        fovs_keyframes=anchor_fovs,
        poses_dense=dense_poses,
        fovs_dense=dense_fovs,
        width=width,
        height=height,
        fps=DEFAULT_FPS,
        frames_per_transition=frames_per_transition,
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
        output_path = generate_scene_virtual_camera_path(scene_dir)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
