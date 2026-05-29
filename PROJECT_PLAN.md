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

Goal: show understanding of coordinate frames, projection, transforms, and optimization.

- Generate camera intrinsics/extrinsics and 3D boxes.
- Render projected LiDAR/box overlays.
- Perturb extrinsics.
- Optimize yaw/pitch/roll/translation from 2D-3D correspondences.
- Report reprojection error before/after.

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
