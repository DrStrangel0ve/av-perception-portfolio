#!/usr/bin/env python3
"""Run the synthetic BEV detection and tracking benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from av_perception.detector import BevClusterDetector, DetectorConfig
from av_perception.metrics import detection_metrics, tracking_metrics
from av_perception.synthetic import ScenarioConfig, generate_sequence, save_sequence_npz
from av_perception.tracker import MultiObjectTracker, TrackerConfig
from av_perception.visualize import BevCanvas, write_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/bev_tracking_baseline"))
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--render-every", type=int, default=2)
    return parser.parse_args()


def write_metrics(metrics: dict[str, float], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "metrics.json"
    csv_path = output / "metrics.csv"
    md_path = output / "metrics.md"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    lines = [
        "# BEV Tracking Baseline Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| `{key}` | {value:.6g} |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    scenario = ScenarioConfig(num_frames=args.frames, seed=args.seed)
    frames = generate_sequence(scenario)
    if args.save_dataset:
        save_sequence_npz(frames, output / "sequence_npz")

    detector = BevClusterDetector(DetectorConfig())
    tracker = MultiObjectTracker(TrackerConfig(dt=scenario.dt))
    canvas = BevCanvas()

    detections_by_frame = []
    tracks_by_frame = []
    truth_by_frame = []
    rendered_paths: list[Path] = []

    for frame in frames:
        detections = detector.detect(frame.points)
        tracks = tracker.step(detections)
        detections_by_frame.append(detections)
        tracks_by_frame.append(tracks)
        truth_by_frame.append(frame.boxes)

        if frame.frame_id % args.render_every == 0:
            render_path = output / "frames" / f"bev_{frame.frame_id:04d}.png"
            canvas.render(frame, detections, tracks, render_path)
            rendered_paths.append(render_path)

    metrics = {
        **detection_metrics(detections_by_frame, truth_by_frame),
        **tracking_metrics(tracks_by_frame, truth_by_frame),
        "frames": float(len(frames)),
        "seed": float(args.seed),
    }
    write_metrics(metrics, output)
    write_gif(rendered_paths, output / "bev_tracking.gif")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
