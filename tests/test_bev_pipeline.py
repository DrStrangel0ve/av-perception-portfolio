from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from av_perception.detector import BevClusterDetector
from av_perception.metrics import detection_metrics
from av_perception.synthetic import ScenarioConfig, generate_sequence
from av_perception.tracker import MultiObjectTracker


def test_bev_pipeline_smoke() -> None:
    frames = generate_sequence(ScenarioConfig(num_frames=8, seed=11))
    detector = BevClusterDetector()
    tracker = MultiObjectTracker()
    detections = []
    tracks = []
    truth = []
    for frame in frames:
        frame_dets = detector.detect(frame.points)
        detections.append(frame_dets)
        tracks.append(tracker.step(frame_dets))
        truth.append(frame.boxes)

    metrics = detection_metrics(detections, truth)
    assert metrics["precision"] > 0.8
    assert metrics["recall"] > 0.5
    assert any(tracks)
