"""Detection and tracking metrics for BEV perception experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import Box2D


@dataclass
class DetectionFrameResult:
    true_positive: int
    false_positive: int
    false_negative: int
    center_errors: list[float]


def match_boxes(
    predicted: list[Box2D],
    truth: list[Box2D],
    max_distance_m: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not predicted or not truth:
        return [], list(range(len(predicted))), list(range(len(truth)))

    costs = np.zeros((len(predicted), len(truth)), dtype=np.float32)
    for i, pred in enumerate(predicted):
        for j, gt in enumerate(truth):
            costs[i, j] = pred.center_distance(gt)
    rows, cols = linear_sum_assignment(costs)

    matches: list[tuple[int, int, float]] = []
    unmatched_pred = set(range(len(predicted)))
    unmatched_gt = set(range(len(truth)))
    for row, col in zip(rows, cols):
        distance = float(costs[row, col])
        if distance > max_distance_m:
            continue
        matches.append((int(row), int(col), distance))
        unmatched_pred.discard(int(row))
        unmatched_gt.discard(int(col))
    return matches, sorted(unmatched_pred), sorted(unmatched_gt)


def detection_metrics(
    detections_by_frame: list[list[Box2D]],
    truth_by_frame: list[list[Box2D]],
    max_distance_m: float = 2.0,
) -> dict[str, float]:
    tp = fp = fn = 0
    center_errors: list[float] = []
    for detections, truth in zip(detections_by_frame, truth_by_frame):
        matches, unmatched_det, unmatched_gt = match_boxes(detections, truth, max_distance_m)
        tp += len(matches)
        fp += len(unmatched_det)
        fn += len(unmatched_gt)
        center_errors.extend(distance for _, _, distance in matches)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_center_error_m": float(np.mean(center_errors)) if center_errors else float("nan"),
    }


def tracking_metrics(
    tracks_by_frame: list[list[Box2D]],
    truth_by_frame: list[list[Box2D]],
    max_distance_m: float = 2.5,
) -> dict[str, float]:
    tp = fp = fn = id_switches = 0
    center_errors: list[float] = []
    gt_to_track: dict[int, int] = {}

    for tracks, truth in zip(tracks_by_frame, truth_by_frame):
        matches, unmatched_tracks, unmatched_gt = match_boxes(tracks, truth, max_distance_m)
        tp += len(matches)
        fp += len(unmatched_tracks)
        fn += len(unmatched_gt)
        for track_index, gt_index, distance in matches:
            gt_id = truth[gt_index].track_id
            track_id = tracks[track_index].track_id
            if gt_id is not None and track_id is not None:
                previous = gt_to_track.get(gt_id)
                if previous is not None and previous != track_id:
                    id_switches += 1
                gt_to_track[gt_id] = track_id
            center_errors.append(distance)

    total_gt = sum(len(frame) for frame in truth_by_frame)
    mota = 1.0 - (fn + fp + id_switches) / total_gt if total_gt else 0.0
    motp = float(np.mean(center_errors)) if center_errors else float("nan")
    return {
        "track_tp": float(tp),
        "track_fp": float(fp),
        "track_fn": float(fn),
        "id_switches": float(id_switches),
        "mota": mota,
        "motp_center_error_m": motp,
        "mostly_tracked_gt": float(len(gt_to_track)),
    }
