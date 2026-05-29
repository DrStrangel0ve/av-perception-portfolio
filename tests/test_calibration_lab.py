from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from av_perception.calibration import (
    extrinsic_errors,
    make_problem,
    optimize_extrinsic,
    reprojection_rmse,
)


def test_calibration_optimization_improves_extrinsic() -> None:
    problem = make_problem(seed=3, noise_px=1.0)
    optimized = optimize_extrinsic(problem)

    initial_rmse = reprojection_rmse(problem, problem.initial)
    optimized_rmse = reprojection_rmse(problem, optimized)
    initial_error = extrinsic_errors(problem.truth, problem.initial)
    optimized_error = extrinsic_errors(problem.truth, optimized)

    assert optimized_rmse < initial_rmse * 0.1
    assert optimized_error["rotation_error_deg"] < initial_error["rotation_error_deg"]
    assert optimized_error["translation_error_m"] < initial_error["translation_error_m"]
