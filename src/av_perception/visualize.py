"""BEV visualizations for perception experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geometry import Box2D
from .synthetic import Frame


class BevCanvas:
    def __init__(
        self,
        x_range: tuple[float, float] = (-35.0, 55.0),
        y_range: tuple[float, float] = (-28.0, 28.0),
        pixels_per_meter: float = 8.0,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.pixels_per_meter = pixels_per_meter
        self.width = int((x_range[1] - x_range[0]) * pixels_per_meter)
        self.height = int((y_range[1] - y_range[0]) * pixels_per_meter)

    def xy_to_px(self, xy: np.ndarray) -> np.ndarray:
        x = (xy[:, 0] - self.x_range[0]) * self.pixels_per_meter
        y = (self.y_range[1] - xy[:, 1]) * self.pixels_per_meter
        return np.column_stack([x, y]).astype(np.float32)

    def render(
        self,
        frame: Frame,
        detections: list[Box2D],
        tracks: list[Box2D],
        output: Path,
    ) -> None:
        image = Image.new("RGB", (self.width, self.height), (13, 18, 24))
        draw = ImageDraw.Draw(image, "RGBA")

        points = frame.points
        keep = (
            (points[:, 0] >= self.x_range[0])
            & (points[:, 0] <= self.x_range[1])
            & (points[:, 1] >= self.y_range[0])
            & (points[:, 1] <= self.y_range[1])
        )
        px = self.xy_to_px(points[keep, :2])
        for x, y in px[::2]:
            image.putpixel((int(x), int(y)), (125, 134, 145))

        self._draw_ego(draw)
        for box in frame.boxes:
            self._draw_box(draw, box, (45, 220, 120, 220), width=2)
        for box in detections:
            self._draw_box(draw, box, (255, 198, 64, 210), width=1)
        for box in tracks:
            self._draw_box(draw, box, (80, 168, 255, 235), width=3)
            center = self.xy_to_px(np.array([[box.x, box.y]], dtype=np.float32))[0]
            draw.text((float(center[0]) + 4, float(center[1]) + 4), str(box.track_id), fill=(170, 215, 255, 255))

        draw.text(
            (12, 10),
            f"frame {frame.frame_id:03d}  green=gt  yellow=det  blue=track",
            fill=(225, 232, 240, 255),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)

    def _draw_ego(self, draw: ImageDraw.ImageDraw) -> None:
        ego = Box2D(0.0, 0.0, 4.8, 2.1, 0.0)
        self._draw_box(draw, ego, (230, 80, 80, 220), width=2)

    def _draw_box(self, draw: ImageDraw.ImageDraw, box: Box2D, fill: tuple[int, int, int, int], width: int) -> None:
        corners = self.xy_to_px(box.corners())
        points = [(float(x), float(y)) for x, y in corners]
        for i in range(len(points)):
            draw.line([points[i], points[(i + 1) % len(points)]], fill=fill, width=width)
        nose = self.xy_to_px(np.array([[box.x + np.cos(box.yaw) * box.length * 0.5, box.y]], dtype=np.float32))[0]
        center = self.xy_to_px(np.array([[box.x, box.y]], dtype=np.float32))[0]
        draw.line([(float(center[0]), float(center[1])), (float(nose[0]), float(nose[1]))], fill=fill, width=width)


def write_gif(image_paths: list[Path], output: Path, duration_ms: int = 90) -> None:
    frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in image_paths]
    if not frames:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
