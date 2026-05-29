# Project Plan

## Project 1: BEV Detection And Tracking

Status: baseline complete.

- Synthetic LiDAR-style sequence generator.
- KD-tree point-clustering detector.
- Constant-velocity multi-object tracker.
- Detection precision/recall/F1 and tracking MOTA/MOTP.
- BEV GIF visualization.

Next improvements:

- scenario-tagged metrics for crossing, parked, and close actors;
- oriented-box fitting instead of axis-aligned boxes;
- learned detector baseline;
- track confidence and lifecycle analysis.

## Project 2: Camera-LiDAR Calibration Lab

Status: baseline complete.

Goal: show understanding of coordinate frames, projection, transforms, and optimization.

- Generate camera intrinsics/extrinsics and 3D calibration points. Done.
- Render projected LiDAR/correspondence overlays. Done.
- Perturb extrinsics. Done.
- Optimize full 6DoF extrinsic from 2D-3D correspondences. Done.
- Report reprojection error before/after. Done.

Baseline:

- 104 correspondences with 1.2 px observation noise.
- Reprojection RMSE: 49.58 px to 1.58 px.
- Rotation error: 5.70 deg to 0.049 deg.
- Translation error: 0.526 m to 0.011 m.

Next improvements:

- robust outlier correspondences;
- rolling-shutter or timestamp offset simulation;
- multiple camera rigs;
- calibration quality gates that fail unsafe extrinsics.

## Project 3: Occupancy Forecasting

Goal: make a small version of occupancy/scene understanding work.

- Rasterize temporal BEV occupancy.
- Predict future occupied cells at 0.5s and 1.0s.
- Compare persistence, constant-velocity, and learned BEV CNN baselines.
- Report IoU, precision/recall, and rare-event slices.

## Project 4: Rare Event Mining

Goal: demonstrate evaluation discipline.

- Generate cut-ins, occlusions, crossing traffic, and parked-vehicle door-zone analogs.
- Run detector/tracker across scenario families.
- Produce per-scenario scorecards and failure clips.

## Project 5: Open Dataset Adapter

Goal: bridge from controlled synthetic tests to public autonomous-driving data.

- Add a common `Frame` contract for real datasets.
- Implement KITTI tracking adapter first.
- Add nuScenes mini or Waymo Open Dataset support when storage allows.
- Preserve the same metrics and visualizations.
