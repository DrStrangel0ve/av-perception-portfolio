"""Synthetic BEV driving scenes for perception pipeline testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Box2D


@dataclass(frozen=True)
class AgentSpec:
    track_id: int
    start_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    length: float = 4.5
    width: float = 1.9
    yaw: float = 0.0
    points_per_frame: int = 220
    label: str = "vehicle"


@dataclass(frozen=True)
class ScenarioConfig:
    num_frames: int = 80
    dt: float = 0.1
    seed: int = 7
    x_range: tuple[float, float] = (-35.0, 55.0)
    y_range: tuple[float, float] = (-28.0, 28.0)
    clutter_points: int = 900
    lidar_noise_std: float = 0.08


@dataclass
class Frame:
    frame_id: int
    timestamp_s: float
    points: np.ndarray
    boxes: list[Box2D]


def default_agents() -> list[AgentSpec]:
    return [
        AgentSpec(1, (-18.0, -3.0), (5.8, 0.08), yaw=0.0),
        AgentSpec(2, (32.0, 5.5), (-4.0, -0.05), yaw=np.pi),
        AgentSpec(3, (3.0, -23.0), (0.7, 4.7), length=4.2, width=1.8, yaw=np.pi / 2),
        AgentSpec(4, (18.0, 16.0), (0.0, 0.0), length=4.8, width=2.0, yaw=-0.2),
        AgentSpec(5, (-8.0, 18.0), (3.8, -2.7), length=4.4, width=1.9, yaw=-0.62),
    ]


def _sample_box_points(box: Box2D, count: int, rng: np.random.Generator) -> np.ndarray:
    half_l = box.length * 0.5
    half_w = box.width * 0.5
    side = rng.integers(0, 4, size=count)
    u = rng.uniform(-1.0, 1.0, size=count)
    local = np.zeros((count, 2), dtype=np.float32)
    local[side == 0] = np.column_stack([np.full((side == 0).sum(), half_l), u[side == 0] * half_w])
    local[side == 1] = np.column_stack([np.full((side == 1).sum(), -half_l), u[side == 1] * half_w])
    local[side == 2] = np.column_stack([u[side == 2] * half_l, np.full((side == 2).sum(), half_w)])
    local[side == 3] = np.column_stack([u[side == 3] * half_l, np.full((side == 3).sum(), -half_w)])

    c = np.cos(box.yaw)
    s = np.sin(box.yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = local @ rot.T + box.center
    z = rng.uniform(0.2, 1.8, size=(count, 1)).astype(np.float32)
    intensity = rng.uniform(0.35, 1.0, size=(count, 1)).astype(np.float32)
    return np.column_stack([xy, z, intensity]).astype(np.float32)


def generate_sequence(
    config: ScenarioConfig = ScenarioConfig(),
    agents: list[AgentSpec] | None = None,
) -> list[Frame]:
    rng = np.random.default_rng(config.seed)
    agents = agents or default_agents()
    frames: list[Frame] = []

    for frame_id in range(config.num_frames):
        t = frame_id * config.dt
        boxes: list[Box2D] = []
        point_sets: list[np.ndarray] = []

        for agent in agents:
            x = agent.start_xy[0] + agent.velocity_xy[0] * t
            y = agent.start_xy[1] + agent.velocity_xy[1] * t
            if not (config.x_range[0] < x < config.x_range[1] and config.y_range[0] < y < config.y_range[1]):
                continue
            yaw = agent.yaw
            if abs(agent.velocity_xy[0]) + abs(agent.velocity_xy[1]) > 0.2:
                yaw = float(np.arctan2(agent.velocity_xy[1], agent.velocity_xy[0]))
            box = Box2D(x, y, agent.length, agent.width, yaw, track_id=agent.track_id, label=agent.label)
            boxes.append(box)
            points = _sample_box_points(box, agent.points_per_frame, rng)
            points[:, :2] += rng.normal(0.0, config.lidar_noise_std, size=(len(points), 2)).astype(np.float32)
            point_sets.append(points)

        clutter_xy = np.column_stack(
            [
                rng.uniform(config.x_range[0], config.x_range[1], config.clutter_points),
                rng.uniform(config.y_range[0], config.y_range[1], config.clutter_points),
            ]
        )
        clutter_z = rng.uniform(-0.05, 0.25, size=(config.clutter_points, 1))
        clutter_i = rng.uniform(0.02, 0.18, size=(config.clutter_points, 1))
        point_sets.append(np.column_stack([clutter_xy, clutter_z, clutter_i]).astype(np.float32))

        points = np.concatenate(point_sets, axis=0)
        frames.append(Frame(frame_id, t, points, boxes))

    return frames


def save_sequence_npz(frames: list[Frame], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        boxes = np.array(
            [
                [box.track_id or -1, box.x, box.y, box.length, box.width, box.yaw]
                for box in frame.boxes
            ],
            dtype=np.float32,
        )
        np.savez_compressed(
            output / f"frame_{frame.frame_id:04d}.npz",
            points=frame.points,
            boxes=boxes,
            timestamp_s=np.array([frame.timestamp_s], dtype=np.float32),
        )
