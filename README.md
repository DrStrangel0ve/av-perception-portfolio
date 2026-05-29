# AV Perception Portfolio

Autonomous vehicle perception projects aimed at Waymo/Zoox-style internship signals: BEV scene understanding, object detection, tracking, occupancy, calibration, robustness, and evaluation.

The first project is intentionally small and runnable on a laptop. It builds a complete BEV LiDAR-style perception loop:

1. Generate synthetic driving scenes with moving/crossing/parked actors.
2. Produce LiDAR-like point clouds and ground-truth 2D BEV boxes.
3. Detect objects with a KD-tree point-clustering baseline.
4. Track detections with a constant-velocity Kalman tracker.
5. Report detection and tracking metrics.
6. Render BEV frames and an animated GIF.

This is not a SOTA detector. It is a clean baseline that makes failure modes visible before adding learned detectors or real open datasets.

The second project is a camera-LiDAR extrinsic calibration lab. It generates 3D LiDAR calibration targets, projects them into a synthetic camera with a known transform, perturbs that transform, and recovers the extrinsic by minimizing reprojection error.

## BEV Detection And Tracking

![BEV tracking demo](results/bev_tracking_baseline/bev_tracking.gif)

| Metric | Value |
| --- | ---: |
| Detection precision | 0.98125 |
| Detection recall | 0.785 |
| Detection F1 | 0.872222 |
| Mean detection center error | 0.170 m |
| Tracking MOTA | 0.7475 |
| Tracking MOTP center error | 0.203 m |
| ID switches | 12 |
| Frames | 80 |

The main failure mode is actor interaction: diagonal and crossing vehicles can temporarily merge or split in point clustering, causing missed detections and ID switches. That is a useful baseline because real AV systems care deeply about rare, close, and partially occluded actors.

## Camera-LiDAR Calibration

![Calibration projection overlay](results/calibration_lab_baseline/projection_overlay.png)

![Calibration residual histogram](results/calibration_lab_baseline/residual_histogram.png)

| Metric | Value |
| --- | ---: |
| Correspondences | 104 |
| Observation noise | 1.2 px |
| Initial reprojection RMSE | 49.580 px |
| Optimized reprojection RMSE | 1.584 px |
| Initial rotation error | 5.695 deg |
| Optimized rotation error | 0.049 deg |
| Initial translation error | 0.526 m |
| Optimized translation error | 0.011 m |

This project is deliberately geometry-heavy. It shows the coordinate-frame discipline behind multimodal perception: LiDAR points in ego coordinates, camera intrinsics, camera extrinsics, 3D-to-2D projection, noisy correspondences, nonlinear least-squares optimization, and before/after reprojection metrics.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/run_bev_tracking_demo.py --frames 80 --seed 7 --save-dataset
python scripts/run_calibration_lab.py --seed 13 --noise-px 1.2
```

Outputs land in `results/bev_tracking_baseline/`:

- `metrics.json`, `metrics.csv`, `metrics.md`
- `bev_tracking.gif`
- rendered BEV PNG frames
- optional compressed synthetic sequence files

Calibration outputs land in `results/calibration_lab_baseline/`:

- `metrics.json`, `metrics.csv`, `metrics.md`
- `projection_overlay.png`
- `residual_histogram.png`

## Repo Shape

- `src/av_perception/calibration.py`: camera intrinsics, extrinsics, projection, calibration optimization, and error metrics.
- `src/av_perception/synthetic.py`: synthetic driving scene and LiDAR-style point generation.
- `src/av_perception/detector.py`: KD-tree BEV clustering detector.
- `src/av_perception/tracker.py`: constant-velocity multi-object tracker.
- `src/av_perception/metrics.py`: detection and tracking metrics.
- `src/av_perception/visualize.py`: BEV frame and GIF renderer.
- `scripts/run_bev_tracking_demo.py`: end-to-end benchmark runner.
- `scripts/run_calibration_lab.py`: camera-LiDAR calibration benchmark runner.
- `results/bev_tracking_baseline/`: committed baseline metrics and visualization.
- `results/calibration_lab_baseline/`: committed calibration metrics and overlays.

## Perception Roadmap

1. BEV detection and tracking baseline. Done.
2. Camera-LiDAR calibration lab. Done.
3. Occupancy grid forecasting: predict short-horizon occupied cells from temporal LiDAR BEV history.
4. Rare-event stress testing: generate cut-ins, occlusions, parked vehicles, close crossings, and adverse clutter; report per-scenario metrics.
5. Open dataset adapters: add KITTI/nuScenes/Waymo Open Dataset conversion into this repo's common frame contract.
6. Learned baseline: swap clustering for a tiny PointPillars-style or BEV CNN detector and compare against the classical baseline.

## Why This Is Internship-Relevant

Waymo and Zoox perception roles emphasize ML-driven perception, scene understanding, occupancy, rare events, optimization, and robust evaluation. This repo is designed to grow along those axes while keeping every result reproducible and inspectable.

## License

MIT for code and generated synthetic assets.
