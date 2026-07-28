#!/usr/bin/env python3
"""Generate Nerfstudio viewer / ns-render camera paths directly from SeaThru datasets."""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pearl.ai_logger import aiLogger


@dataclass(frozen=True)
class CameraModel:
    model_id: int
    model_name: str
    num_params: int


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    def qvec2rotmat(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)


CAMERA_MODELS: Tuple[CameraModel, ...] = (
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
)

CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in CAMERA_MODELS}
DEFAULT_FPS = 30.0
DEFAULT_FRAMES_PER_TRANSITION = 12
DEFAULT_OUTPUT_NAME = "trajectory.json"


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


def read_cameras_binary(path_to_model_file: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with path_to_model_file.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = int(camera_properties[0])
            model_id = int(camera_properties[1])
            width = int(camera_properties[2])
            height = int(camera_properties[3])
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = np.array(
                read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params),
                dtype=np.float64,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=CAMERA_MODEL_IDS[model_id].model_name,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_images_binary(path_to_model_file: Path) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with path_to_model_file.open("rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = int(binary_image_properties[0])
            qvec = np.array(binary_image_properties[1:5], dtype=np.float64)
            tvec = np.array(binary_image_properties[5:8], dtype=np.float64)
            camera_id = int(binary_image_properties[8])

            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            name = image_name.decode("utf-8")

            num_points2d = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            fid.seek(24 * num_points2d, 1)

            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name)
    return images


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float64,
    )


def get_intrinsics(camera: Camera) -> Tuple[float, float, float, float]:
    model = camera.model
    params = camera.params
    if model in {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
        "FOV",
    }:
        focal = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
        return focal, focal, cx, cy
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    raise ValueError(f"Unsupported COLMAP camera model: {model}")


def vertical_fov_degrees(height: int, fy: float) -> float:
    return math.degrees(2.0 * math.atan(height / (2.0 * fy)))


def colmap_image_to_camera_to_world(image: Image) -> np.ndarray:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = image.qvec2rotmat()
    world_to_camera[:3, 3] = image.tvec
    return np.linalg.inv(world_to_camera)


def convert_camera_to_world_opencv_to_opengl(camera_to_world_opencv: np.ndarray) -> np.ndarray:
    converted = camera_to_world_opencv.copy()
    converted[:3, 1:3] *= -1.0
    return converted


def apply_colmap_world_coordinate_convention(camera_to_world_opengl: np.ndarray) -> np.ndarray:
    transformed = camera_to_world_opengl[np.array([0, 2, 1, 3]), :].copy()
    transformed[2, :] *= -1.0
    return transformed


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def rotation_matrix_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    source = normalize_vector(a.astype(np.float64))
    target = normalize_vector(b.astype(np.float64))
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if np.allclose(np.abs(source), axis):
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cross = normalize_vector(np.cross(source, axis))
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * (skew @ skew)

    axis = cross / cross_norm
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    theta = math.acos(dot)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def compute_dataparser_transform_and_scale(camera_to_worlds_opengl: np.ndarray) -> Tuple[np.ndarray, float]:
    saved_space_poses = np.stack(
        [apply_colmap_world_coordinate_convention(pose) for pose in camera_to_worlds_opengl],
        axis=0,
    )
    origins = saved_space_poses[:, :3, 3]
    translation = origins.mean(axis=0)
    up = normalize_vector(saved_space_poses[:, :3, 1].mean(axis=0))
    rotation = rotation_matrix_between(up, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = rotation @ -translation

    oriented_poses = transform @ saved_space_poses
    max_translation = float(np.max(np.abs(oriented_poses[:, :3, 3])))
    if max_translation == 0.0:
        raise ValueError("Cannot scale poses with zero translation extent.")
    scale = 1.0 / max_translation

    applied_transform = np.eye(4, dtype=np.float64)
    applied_transform = applied_transform[np.array([0, 2, 1, 3]), :]
    applied_transform[2, :] *= -1.0

    dataparser_transform = transform @ applied_transform
    return dataparser_transform, scale


def transform_pose_to_nerfstudio_space(
    camera_to_world_opencv: np.ndarray,
    dataparser_transform: np.ndarray,
    dataparser_scale: float,
) -> np.ndarray:
    converted = convert_camera_to_world_opencv_to_opengl(camera_to_world_opencv)
    transformed = dataparser_transform @ converted
    transformed[:3, 3] *= dataparser_scale
    return transformed


def interpolate_path(
    poses: np.ndarray,
    fovs: np.ndarray,
    frames_per_transition: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(poses) == 1:
        return poses.copy(), fovs.copy()

    key_times = np.arange(len(poses), dtype=np.float64)
    frame_count = (len(poses) - 1) * frames_per_transition + 1
    sample_times = np.linspace(0.0, len(poses) - 1, frame_count, dtype=np.float64)

    rotations = Rotation.from_matrix(poses[:, :3, :3])
    slerp = Slerp(key_times, rotations)
    dense_rotations = slerp(sample_times).as_matrix()

    dense_translations = np.column_stack(
        [
            np.interp(sample_times, key_times, poses[:, 0, 3]),
            np.interp(sample_times, key_times, poses[:, 1, 3]),
            np.interp(sample_times, key_times, poses[:, 2, 3]),
        ]
    )
    dense_fovs = np.interp(sample_times, key_times, fovs)

    dense_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], frame_count, axis=0)
    dense_poses[:, :3, :3] = dense_rotations
    dense_poses[:, :3, 3] = dense_translations
    return dense_poses, dense_fovs


def flatten_matrix(matrix: np.ndarray) -> List[float]:
    return [float(value) for value in matrix.reshape(-1)]


def build_viewer_camera_path(
    poses_keyframes: np.ndarray,
    fovs_keyframes: np.ndarray,
    poses_dense: np.ndarray,
    fovs_dense: np.ndarray,
    width: int,
    height: int,
    fps: float,
    frames_per_transition: int,
) -> Dict[str, object]:
    aspect = float(width) / float(height)
    seconds = 0.0 if len(poses_dense) == 0 else len(poses_dense) / fps
    default_transition = 0.0 if len(poses_keyframes) <= 1 else frames_per_transition / fps

    return {
        "default_fov": float(fovs_keyframes[0]),
        "default_transition_sec": float(default_transition),
        "keyframes": [
            {
                "matrix": flatten_matrix(pose),
                "fov": float(fov),
                "aspect": aspect,
                "override_transition_enabled": False,
                "override_transition_sec": None,
            }
            for pose, fov in zip(poses_keyframes, fovs_keyframes)
        ],
        "camera_type": "perspective",
        "render_height": height,
        "render_width": width,
        "fps": float(fps),
        "seconds": float(seconds),
        "is_cycle": False,
        "smoothness_value": 0.0,
        "camera_path": [
            {
                "camera_to_world": flatten_matrix(pose),
                "fov": float(fov),
                "aspect": aspect,
            }
            for pose, fov in zip(poses_dense, fovs_dense)
        ],
    }


def iter_target_scenes(dataset_root: Path) -> List[Path]:
    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def generate_scene_camera_path(scene_dir: Path) -> Path:
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

    keyframe_poses: List[np.ndarray] = []
    keyframe_fovs: List[float] = []
    width = None
    height = None

    for image in ordered_images:
        camera = cameras[image.camera_id]
        _fx, fy, _cx, _cy = get_intrinsics(camera)
        pose_opencv = colmap_image_to_camera_to_world(image)
        pose_nerfstudio = transform_pose_to_nerfstudio_space(pose_opencv, dataparser_transform, dataparser_scale)

        keyframe_poses.append(pose_nerfstudio)
        keyframe_fovs.append(vertical_fov_degrees(camera.height, fy))
        width = camera.width
        height = camera.height

    assert width is not None and height is not None

    keyframe_poses_array = np.stack(keyframe_poses, axis=0)
    keyframe_fovs_array = np.array(keyframe_fovs, dtype=np.float64)
    dense_poses, dense_fovs = interpolate_path(
        keyframe_poses_array,
        keyframe_fovs_array,
        DEFAULT_FRAMES_PER_TRANSITION,
    )

    viewer_path = build_viewer_camera_path(
        poses_keyframes=keyframe_poses_array,
        fovs_keyframes=keyframe_fovs_array,
        poses_dense=dense_poses,
        fovs_dense=dense_fovs,
        width=width,
        height=height,
        fps=DEFAULT_FPS,
        frames_per_transition=DEFAULT_FRAMES_PER_TRANSITION,
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
        output_path = generate_scene_camera_path(scene_dir=scene_dir)
        aiLogger.info(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
