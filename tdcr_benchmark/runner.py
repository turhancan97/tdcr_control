from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from tdcr_benchmark.config import RobotConfig, load_config
from tdcr_benchmark.models import (
    ConstantCurvatureModel,
    CosseratRodModel,
    PiecewiseConstantCurvatureModel,
    PseudoRigidBodyModel,
    SubsegmentCosseratRodModel,
)
from tdcr_benchmark.models.base import ModelResult
from tdcr_benchmark.output import plot_result, write_result


def run_benchmark(
    config_path: str | Path | None = None,
    output_root: str | Path = "outputs",
    tensions: np.ndarray | None = None,
    timestamped: bool = True,
) -> tuple[Path, list[ModelResult]]:
    config = load_config(config_path)
    if tensions is not None:
        config = replace_tensions(config, tensions)

    run_dir = Path(output_root)
    if timestamped:
        run_dir = run_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ConstantCurvatureModel(),
        PiecewiseConstantCurvatureModel(),
        CosseratRodModel(),
        SubsegmentCosseratRodModel(),
        PseudoRigidBodyModel(),
    ]
    results: list[ModelResult] = []
    for model in models:
        result = model.forward_kinematics(config)
        results.append(result)
        write_result(run_dir, result)
        plot_result(run_dir, result, config)

    return run_dir, results


def replace_tensions(config: RobotConfig, tensions: np.ndarray) -> RobotConfig:
    from dataclasses import replace

    array = np.asarray(tensions, dtype=float).reshape(-1)
    if array.size != 6:
        raise ValueError("--tensions must provide exactly six values.")
    return replace(config, tensions=array)
