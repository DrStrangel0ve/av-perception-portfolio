#!/usr/bin/env python3
"""Run a synthetic camera-LiDAR extrinsic calibration experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from av_perception.calibration import (  # noqa: E402
    Extrinsic,
    extrinsic_errors,
    make_problem,
    optimize_extrinsic,
    project_points,
    reprojection_rmse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--noise-px", type=float, default=1.2)
    parser.add_argument("--output", type=Path, default=Path("results/calibration_lab_baseline"))
    return parser.parse_args()


def draw_overlay(problem, optimized: Extrinsic, output: Path) -> None:
    intr = problem.intrinsics
    image = Image.new("RGB", (intr.width, intr.height), (17, 22, 29))
    draw = ImageDraw.Draw(image, "RGBA")

    for y in range(0, intr.height, 40):
        draw.line([(0, y), (intr.width, y)], fill=(38, 48, 60, 120), width=1)
    for x in range(0, intr.width, 40):
        draw.line([(x, 0), (x, intr.height)], fill=(38, 48, 60, 120), width=1)

    initial_uv, _ = project_points(problem.lidar_points, problem.initial, intr)
    opt_uv, _ = project_points(problem.lidar_points, optimized, intr)
    obs_uv = problem.image_points

    def dot(uv: np.ndarray, color: tuple[int, int, int, int], radius: int) -> None:
        for x, y in uv:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    dot(initial_uv, (240, 80, 80, 160), 4)
    dot(obs_uv, (245, 230, 120, 210), 3)
    dot(opt_uv, (70, 220, 135, 210), 3)

    for before, after in zip(initial_uv, opt_uv):
        draw.line([(float(before[0]), float(before[1])), (float(after[0]), float(after[1]))], fill=(120, 160, 190, 60))

    draw.text((16, 14), "red=perturbed  yellow=observed  green=optimized", fill=(230, 236, 242))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def draw_residual_chart(problem, optimized: Extrinsic, output: Path) -> None:
    width, height = 760, 420
    image = Image.new("RGB", (width, height), (17, 22, 29))
    draw = ImageDraw.Draw(image, "RGBA")
    initial_uv, _ = project_points(problem.lidar_points, problem.initial, problem.intrinsics)
    opt_uv, _ = project_points(problem.lidar_points, optimized, problem.intrinsics)
    init_err = np.linalg.norm(initial_uv - problem.image_points, axis=1)
    opt_err = np.linalg.norm(opt_uv - problem.image_points, axis=1)
    bins = np.linspace(0, max(float(init_err.max()), 1.0), 18)
    init_hist, _ = np.histogram(init_err, bins=bins)
    opt_hist, _ = np.histogram(opt_err, bins=bins)
    max_count = max(int(init_hist.max()), int(opt_hist.max()), 1)
    chart_x0, chart_y0 = 70, 50
    chart_w, chart_h = 650, 300
    draw.rectangle((chart_x0, chart_y0, chart_x0 + chart_w, chart_y0 + chart_h), outline=(80, 92, 110), width=1)
    bar_w = chart_w / len(init_hist)
    for i, (a, b) in enumerate(zip(init_hist, opt_hist)):
        x = chart_x0 + i * bar_w
        ah = chart_h * a / max_count
        bh = chart_h * b / max_count
        draw.rectangle((x + 2, chart_y0 + chart_h - ah, x + bar_w * 0.45, chart_y0 + chart_h), fill=(240, 80, 80, 180))
        draw.rectangle((x + bar_w * 0.52, chart_y0 + chart_h - bh, x + bar_w - 2, chart_y0 + chart_h), fill=(70, 220, 135, 180))
    draw.text((24, 14), "Reprojection residual histogram, red=initial, green=optimized", fill=(230, 236, 242))
    draw.text((chart_x0, chart_y0 + chart_h + 18), "pixel error bins", fill=(190, 200, 210))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def write_metrics(metrics: dict[str, float], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    lines = ["# Camera-LiDAR Calibration Metrics", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in metrics.items():
        lines.append(f"| `{key}` | {value:.6g} |")
    lines.append("")
    (output / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    problem = make_problem(seed=args.seed, noise_px=args.noise_px)
    optimized = optimize_extrinsic(problem)

    initial_errors = extrinsic_errors(problem.truth, problem.initial)
    optimized_errors = extrinsic_errors(problem.truth, optimized)
    metrics = {
        "num_correspondences": float(len(problem.lidar_points)),
        "noise_px": float(args.noise_px),
        "initial_rmse_px": reprojection_rmse(problem, problem.initial),
        "optimized_rmse_px": reprojection_rmse(problem, optimized),
        "initial_rotation_error_deg": initial_errors["rotation_error_deg"],
        "optimized_rotation_error_deg": optimized_errors["rotation_error_deg"],
        "initial_translation_error_m": initial_errors["translation_error_m"],
        "optimized_translation_error_m": optimized_errors["translation_error_m"],
    }
    write_metrics(metrics, args.output)
    draw_overlay(problem, optimized, args.output / "projection_overlay.png")
    draw_residual_chart(problem, optimized, args.output / "residual_histogram.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
