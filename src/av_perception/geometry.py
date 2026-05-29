"""Geometry helpers for BEV boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box2D:
    x: float
    y: float
    length: float
    width: float
    yaw: float = 0.0
    score: float = 1.0
    track_id: int | None = None
    label: str = "vehicle"

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    @property
    def area(self) -> float:
        return float(max(self.length, 0.0) * max(self.width, 0.0))

    def corners(self) -> np.ndarray:
        half_l = self.length * 0.5
        half_w = self.width * 0.5
        local = np.array(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ],
            dtype=np.float32,
        )
        c = np.cos(self.yaw)
        s = np.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        return local @ rot.T + self.center

    def aabb(self) -> tuple[float, float, float, float]:
        corners = self.corners()
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    def center_distance(self, other: "Box2D") -> float:
        return float(np.linalg.norm(self.center - other.center))


def aabb_iou(a: Box2D, b: Box2D) -> float:
    ax0, ay0, ax1, ay1 = a.aabb()
    bx0, by0, bx1, by1 = b.aabb()
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)
