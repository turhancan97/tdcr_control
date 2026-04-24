from __future__ import annotations

import numpy as np

from tdcr_benchmark.config import RobotConfig
from tdcr_benchmark.math_utils import EPS, constant_curvature_transform
from tdcr_benchmark.models.base import ModelResult, TDCRModel


class ConstantCurvatureModel(TDCRModel):
    name = "cc"

    def forward_kinematics(self, config: RobotConfig) -> ModelResult:
        l_d = self.tendon_lengths_from_displacements(config)
        var_cc = self.solve_configuration(l_d, config.number_disks, config.r_disk)
        T1_cc, T2_cc = self.construct_tdcr_cc(var_cc)

        positions_1 = T1_cc[:, :3, 3]
        positions_2 = T2_cc[:, :3, 3]
        backbone_positions = np.vstack([positions_1, positions_2])
        s1 = np.linspace(0.0, var_cc[2, 0], T1_cc.shape[0])
        s2 = var_cc[2, 0] + np.linspace(0.0, var_cc[2, 1], T2_cc.shape[0])
        backbone_s = np.concatenate([s1, s2])

        return ModelResult(
            model_name=self.name,
            backbone_s=backbone_s,
            backbone_positions=backbone_positions,
            tip_pose=T2_cc[-1],
            disk_frames=self.disk_frames(var_cc, config.number_disks),
        )

    @staticmethod
    def tendon_lengths_from_displacements(config: RobotConfig) -> np.ndarray:
        q = config.cc_tendon_displacements
        return np.array(
            [
                config.segment_lengths[0] - q[:3],
                config.segment_lengths[1] - (q[3:] - q[:3]),
            ],
            dtype=float,
        )

    @staticmethod
    def solve_configuration(l_d: np.ndarray, n: np.ndarray, r_disk: float) -> np.ndarray:
        kappa1, phi1, l1 = configuration(l_d[0], int(n[0]), r_disk)
        kappa2, phi2, l2 = configuration(l_d[1], int(n[1]), r_disk)
        return np.array([[kappa1, kappa2], [phi1, phi2], [l1, l2]], dtype=float)

    @staticmethod
    def construct_tdcr_cc(var_cc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kappa1, kappa2 = var_cc[0]
        phi1, phi2 = var_cc[1]
        l1, l2 = var_cc[2]
        T1_cc, _ = constant_curvature_transform(kappa1, phi1, l1, sect_points=50)
        T2, _ = constant_curvature_transform(kappa2, phi2 - phi1, l2, sect_points=50)
        T1_tip = T1_cc[-1]
        T2_cc = np.array([T1_tip @ T for T in T2])
        return T1_cc, T2_cc

    @staticmethod
    def disk_frames(var_cc: np.ndarray, n: np.ndarray) -> np.ndarray:
        kappa1, kappa2 = var_cc[0]
        phi1, phi2 = var_cc[1]
        l1, l2 = var_cc[2]
        T1, _ = constant_curvature_transform(kappa1, phi1, l1, sect_points=int(n[0]) + 1)
        T2_local, _ = constant_curvature_transform(kappa2, phi2 - phi1, l2, sect_points=int(n[1]) + 1)
        T2 = np.array([T1[-1] @ T for T in T2_local])
        return np.concatenate([T1, T2[1:]], axis=0)


def configuration(l_t: np.ndarray, n: int, d: float) -> tuple[float, float, float]:
    temp_sq = l_t[0] ** 2 + l_t[1] ** 2 + l_t[2] ** 2 - l_t[0] * l_t[1] - l_t[0] * l_t[2] - l_t[1] * l_t[2]
    if temp_sq <= EPS:
        return 0.0, 0.0, float(l_t[0])
    temp = float(np.sqrt(temp_sq))
    total = float(l_t[0] + l_t[1] + l_t[2])
    kappa = (2.0 * temp) / (d * total)
    phi = float(np.arctan2(np.sqrt(3.0) * (l_t[1] + l_t[2] - 2.0 * l_t[0]), 3.0 * (l_t[2] - l_t[1])))
    asin_arg = np.clip(temp / (3.0 * n * d), -1.0, 1.0)
    length = ((n * d * total) / temp) * np.arcsin(asin_arg)
    return float(kappa), phi, float(length)
