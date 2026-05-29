"""Simple BEV clustering detector for LiDAR-style point clouds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Box2D


@dataclass(frozen=True)
class DetectorConfig:
    x_range: tuple[float, float] = (-35.0, 55.0)
    y_range: tuple[float, float] = (-28.0, 28.0)
    min_points: int = 35
    cluster_radius_m: float = 0.85
    min_height_m: float = 0.15
    min_box_length: float = 1.0
    max_box_length: float = 9.5
    min_box_width: float = 0.7
    max_box_width: float = 5.8


class BevClusterDetector:
    def __init__(self, config: DetectorConfig = DetectorConfig()) -> None:
        self.config = config

    def detect(self, points: np.ndarray) -> list[Box2D]:
        cfg = self.config
        keep = (
            (points[:, 0] >= cfg.x_range[0])
            & (points[:, 0] <= cfg.x_range[1])
            & (points[:, 1] >= cfg.y_range[0])
            & (points[:, 1] <= cfg.y_range[1])
            & (points[:, 2] >= cfg.min_height_m)
        )
        obj_points = points[keep]
        if len(obj_points) == 0:
            return []

        components = self._point_components(obj_points[:, :2], cfg.cluster_radius_m)

        detections: list[Box2D] = []
        for indices in components:
            component_points = obj_points[indices]
            if len(component_points) < cfg.min_points:
                continue

            min_xy = component_points[:, :2].min(axis=0)
            max_xy = component_points[:, :2].max(axis=0)
            dims = max_xy - min_xy
            length = float(max(dims[0], dims[1]))
            width_box = float(min(dims[0], dims[1]))
            if not (cfg.min_box_length <= length <= cfg.max_box_length):
                continue
            if not (cfg.min_box_width <= width_box <= cfg.max_box_width):
                continue
            score = float(min(1.0, len(component_points) / 220.0))
            center = (min_xy + max_xy) * 0.5
            detections.append(
                Box2D(
                    x=float(center[0]),
                    y=float(center[1]),
                    length=float(dims[0]),
                    width=float(dims[1]),
                    yaw=0.0,
                    score=score,
                )
            )

        return sorted(detections, key=lambda box: box.score, reverse=True)

    @staticmethod
    def _point_components(xy: np.ndarray, radius: float) -> list[np.ndarray]:
        parent = np.arange(len(xy), dtype=np.int32)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        tree = cKDTree(xy)
        for a, b in tree.query_pairs(radius):
            union(int(a), int(b))

        groups: dict[int, list[int]] = {}
        for index in range(len(xy)):
            root = find(index)
            groups.setdefault(root, []).append(index)
        return [np.asarray(indices, dtype=np.int32) for indices in groups.values()]
