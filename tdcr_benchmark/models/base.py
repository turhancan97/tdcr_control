from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdcr_benchmark.config import RobotConfig


@dataclass
class ModelResult:
    model_name: str
    backbone_s: np.ndarray
    backbone_positions: np.ndarray
    tip_pose: np.ndarray
    disk_frames: np.ndarray | None = None
    solver_success: bool = True
    message: str = ""


class TDCRModel:
    name = "base"

    def forward_kinematics(self, config: RobotConfig) -> ModelResult:
        raise NotImplementedError


class PendingModel(TDCRModel):
    def forward_kinematics(self, config: RobotConfig) -> ModelResult:
        raise NotImplementedError(f"{self.name} is scaffolded but not implemented in the first Python milestone.")
