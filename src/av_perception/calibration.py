"""Camera-LiDAR calibration primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 1280
    height: int = 720
    fx: float = 850.0
    fy: float = 850.0
    cx: float = 640.0
    cy: float = 360.0

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Extrinsic:
    rotation: Rotation
    translation: np.ndarray

    def as_params(self) -> np.ndarray:
        return np.concatenate([self.rotation.as_rotvec(), self.translation.astype(np.float64)])

    @staticmethod
    def from_params(params: np.ndarray) -> "Extrinsic":
        return Extrinsic(Rotation.from_rotvec(params[:3]), params[3:6].astype(np.float64))


@dataclass
class CalibrationProblem:
    intrinsics: CameraIntrinsics
    lidar_points: np.ndarray
    image_points: np.ndarray
    truth: Extrinsic
    initial: Extrinsic
    noise_px: float


def nominal_lidar_to_camera() -> Rotation:
    # LiDAR: x forward, y left, z up. Camera: x right, y down, z forward.
    matrix = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    return Rotation.from_matrix(matrix)


def make_truth_extrinsic() -> Extrinsic:
    base = nominal_lidar_to_camera()
    mount_error = Rotation.from_euler("zyx", [1.2, -0.6, 0.4], degrees=True)
    translation = np.array([0.08, -0.18, -0.34], dtype=np.float64)
    return Extrinsic(mount_error * base, translation)


def perturb_extrinsic(extrinsic: Extrinsic) -> Extrinsic:
    perturb = Rotation.from_euler("zyx", [4.5, -2.8, 2.2], degrees=True)
    translation = extrinsic.translation + np.array([0.42, -0.26, 0.18], dtype=np.float64)
    return Extrinsic(perturb * extrinsic.rotation, translation)


def generate_calibration_points(seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    boards = [
        (16.0, -4.5, 1.25),
        (22.0, 3.8, 1.35),
        (31.0, -1.5, 1.55),
        (42.0, 5.5, 1.2),
    ]
    points: list[np.ndarray] = []
    for x, y, z in boards:
        xs = x + rng.normal(0.0, 0.05, size=24)
        ys = y + rng.uniform(-1.1, 1.1, size=24)
        zs = z + rng.uniform(-0.85, 0.85, size=24)
        points.append(np.column_stack([xs, ys, zs]))

    vehicle_corners = np.array(
        [
            [26.0, -8.0, 0.2],
            [26.0, -5.8, 0.2],
            [30.5, -8.0, 0.2],
            [30.5, -5.8, 0.2],
            [26.0, -8.0, 1.8],
            [26.0, -5.8, 1.8],
            [30.5, -8.0, 1.8],
            [30.5, -5.8, 1.8],
        ],
        dtype=np.float64,
    )
    points.append(vehicle_corners)
    return np.concatenate(points, axis=0).astype(np.float64)


def project_points(
    lidar_points: np.ndarray,
    extrinsic: Extrinsic,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    cam = extrinsic.rotation.apply(lidar_points) + extrinsic.translation[None, :]
    depth = cam[:, 2]
    uv = np.column_stack(
        [
            intrinsics.fx * cam[:, 0] / depth + intrinsics.cx,
            intrinsics.fy * cam[:, 1] / depth + intrinsics.cy,
        ]
    )
    return uv, depth


def visible_mask(uv: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    return (
        (depth > 1.0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < intrinsics.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < intrinsics.height)
    )


def make_problem(seed: int = 13, noise_px: float = 1.2) -> CalibrationProblem:
    intrinsics = CameraIntrinsics()
    truth = make_truth_extrinsic()
    initial = perturb_extrinsic(truth)
    points = generate_calibration_points(seed)
    uv, depth = project_points(points, truth, intrinsics)
    keep = visible_mask(uv, depth, intrinsics)
    points = points[keep]
    uv = uv[keep]
    rng = np.random.default_rng(seed + 1000)
    noisy_uv = uv + rng.normal(0.0, noise_px, size=uv.shape)
    return CalibrationProblem(intrinsics, points, noisy_uv, truth, initial, noise_px)


def reprojection_residuals(
    params: np.ndarray,
    lidar_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    extrinsic = Extrinsic.from_params(params)
    projected, depth = project_points(lidar_points, extrinsic, intrinsics)
    residual = projected - image_points
    invalid = depth <= 0.2
    if invalid.any():
        residual[invalid] += 1e4
    return residual.reshape(-1)


def optimize_extrinsic(problem: CalibrationProblem) -> Extrinsic:
    result = least_squares(
        reprojection_residuals,
        problem.initial.as_params(),
        args=(problem.lidar_points, problem.image_points, problem.intrinsics),
        loss="huber",
        f_scale=3.0,
        max_nfev=200,
    )
    return Extrinsic.from_params(result.x)


def reprojection_rmse(problem: CalibrationProblem, extrinsic: Extrinsic) -> float:
    projected, _ = project_points(problem.lidar_points, extrinsic, problem.intrinsics)
    residual = projected - problem.image_points
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def extrinsic_errors(truth: Extrinsic, candidate: Extrinsic) -> dict[str, float]:
    delta = candidate.rotation * truth.rotation.inv()
    return {
        "rotation_error_deg": float(np.linalg.norm(delta.as_rotvec()) * 180.0 / np.pi),
        "translation_error_m": float(np.linalg.norm(candidate.translation - truth.translation)),
    }
