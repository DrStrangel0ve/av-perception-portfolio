"""Constant-velocity multi-object tracker for BEV detections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import Box2D


@dataclass(frozen=True)
class TrackerConfig:
    dt: float = 0.1
    max_match_distance_m: float = 4.0
    max_age: int = 6
    min_hits: int = 2


@dataclass
class Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    length: float
    width: float
    age: int = 0
    hits: int = 1
    missed: int = 0

    def to_box(self) -> Box2D:
        return Box2D(
            x=float(self.state[0]),
            y=float(self.state[1]),
            length=float(self.length),
            width=float(self.width),
            score=1.0,
            track_id=self.track_id,
        )


class MultiObjectTracker:
    def __init__(self, config: TrackerConfig = TrackerConfig()) -> None:
        self.config = config
        self.next_id = 1
        self.tracks: list[Track] = []

    def step(self, detections: list[Box2D]) -> list[Box2D]:
        for track in self.tracks:
            self._predict(track)

        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        for track_index, det_index in matches:
            self._update(self.tracks[track_index], detections[det_index])

        for track_index in unmatched_tracks:
            track = self.tracks[track_index]
            track.missed += 1
            track.age += 1

        for det_index in unmatched_dets:
            self._start_track(detections[det_index])

        self.tracks = [track for track in self.tracks if track.missed <= self.config.max_age]
        return [
            track.to_box()
            for track in self.tracks
            if track.hits >= self.config.min_hits and track.missed == 0
        ]

    def _predict(self, track: Track) -> None:
        dt = self.config.dt
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        q = np.diag([0.08, 0.08, 0.5, 0.5]).astype(np.float32)
        track.state = f @ track.state
        track.covariance = f @ track.covariance @ f.T + q

    def _update(self, track: Track, detection: Box2D) -> None:
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        r = np.diag([0.25, 0.25]).astype(np.float32)
        z = detection.center
        innovation = z - h @ track.state
        s = h @ track.covariance @ h.T + r
        k = track.covariance @ h.T @ np.linalg.inv(s)
        track.state = track.state + k @ innovation
        track.covariance = (np.eye(4, dtype=np.float32) - k @ h) @ track.covariance
        track.length = 0.85 * track.length + 0.15 * detection.length
        track.width = 0.85 * track.width + 0.15 * detection.width
        track.hits += 1
        track.age += 1
        track.missed = 0

    def _start_track(self, detection: Box2D) -> None:
        state = np.array([detection.x, detection.y, 0.0, 0.0], dtype=np.float32)
        covariance = np.diag([0.6, 0.6, 8.0, 8.0]).astype(np.float32)
        self.tracks.append(
            Track(
                track_id=self.next_id,
                state=state,
                covariance=covariance,
                length=detection.length,
                width=detection.width,
            )
        )
        self.next_id += 1

    def _associate(self, detections: list[Box2D]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        costs = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(self.tracks):
            track_center = track.state[:2]
            for j, det in enumerate(detections):
                costs[i, j] = np.linalg.norm(track_center - det.center)

        rows, cols = linear_sum_assignment(costs)
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        for row, col in zip(rows, cols):
            if costs[row, col] > self.config.max_match_distance_m:
                continue
            matches.append((int(row), int(col)))
            unmatched_tracks.discard(int(row))
            unmatched_dets.discard(int(col))
        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)
