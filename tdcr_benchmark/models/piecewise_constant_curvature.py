from __future__ import annotations

import numpy as np
from scipy.optimize import root

from tdcr_benchmark.config import RobotConfig
from tdcr_benchmark.math_utils import EPS, ccsub_transform, cross_columns
from tdcr_benchmark.models.base import ModelResult, TDCRModel


class PiecewiseConstantCurvatureModel(TDCRModel):
    name = "ccsub"

    def forward_kinematics(self, config: RobotConfig) -> ModelResult:
        var0 = 0.01 * np.ones(3 * config.n_disk)
        solve = root(lambda var: self.residual(var, config), var0, method="hybr", options={"xtol": 1e-7, "maxfev": 5000})
        var = solve.x
        backbone_s, backbone_positions = self.construct_tdcr_ccsub(config.subsegment_lengths, config.n_disk, var)
        tip_pose = ccsub_transform(var, config.subsegment_lengths, config.n_disk, 0)
        return ModelResult(
            model_name=self.name,
            backbone_s=backbone_s,
            backbone_positions=backbone_positions,
            tip_pose=tip_pose,
            disk_frames=self.disk_frames(config.subsegment_lengths, config.n_disk, var),
            solver_success=bool(solve.success),
            message=str(solve.message),
        )

    @staticmethod
    def residual(var: np.ndarray, config: RobotConfig) -> np.ndarray:
        n = config.number_disks
        n_disk = config.n_disk
        lengths = config.subsegment_lengths
        F = config.tensions
        p_tendon = config.p_tendon
        nt = len(F) // len(n)
        Fdisk = np.vstack(
            [
                np.tile(F, (n[0], 1)),
                np.hstack([np.zeros((n[1], nt)), np.tile(F[nt:], (n[1], 1))]),
            ]
        )

        res = np.zeros(n_disk * 3)
        F_prev = np.zeros(4)
        M_prev = np.zeros(3)

        for ss_i in range(n_disk, 0, -1):
            idx = ss_i - 1
            T_i = ccsub_transform(var, lengths, ss_i, ss_i - 1)
            beta, gamma, epsi = var[3 * idx : 3 * idx + 3]
            k = float(np.hypot(beta, gamma))
            phi = float(np.arctan2(gamma, beta)) if k >= EPS else 0.0
            theta = lengths[idx] * k

            p_ti = T_i @ p_tendon
            p_i = T_i @ np.array([0.0, 0.0, 0.0, 1.0])
            norm_ct1 = np.linalg.norm(-p_ti[:3, :] + p_tendon[:3, :], axis=0)
            norm_ct1 = np.maximum(norm_ct1, EPS)
            zi = T_i[:, 2]

            if ss_i < n_disk:
                T_i1 = ccsub_transform(var, lengths, ss_i + 1, ss_i - 1)
                p_ti1 = T_i1 @ p_tendon
                norm_ct2 = np.linalg.norm(-p_ti[:3, :] + p_ti1[:3, :], axis=0)
                norm_ct2 = np.maximum(norm_ct2, EPS)
                F_rel = (
                    ((-p_ti + p_tendon) / norm_ct1) * Fdisk[idx]
                    + ((-p_ti + p_ti1) / norm_ct2) * Fdisk[idx + 1]
                )
                if ss_i == n[0]:
                    distal = F_rel[:, 3:6] - np.outer(zi, zi @ F_rel[:, 3:6])
                    F_rel = np.hstack([F_rel[:, 0:3], distal])
                else:
                    F_rel = F_rel - np.outer(zi, zi @ F_rel)
            else:
                F_rel = ((-p_ti + p_tendon) / norm_ct1) * Fdisk[idx]
                F_prev = np.zeros(4)
                M_prev = np.zeros(3)

            M_rel = cross_columns(p_ti[:3, :], F_rel[:3, :])

            if ss_i == n_disk:
                Rt = ccsub_transform(var, lengths, ss_i, 0)
                Fex = np.linalg.solve(Rt, config.external_force)
                Mex = np.linalg.solve(Rt[:3, :3], config.external_moment[:3]) + np.cross(p_i[:3], Fex[:3])
            else:
                Fex = np.zeros(4)
                Mex = np.zeros(3)

            Rg = ccsub_transform(var, lengths, ss_i - 1, 0)
            Fg_disk = config.disk_weight[idx] * np.linalg.solve(Rg, np.array([0.0, 0.0, -1.0, 0.0]))
            Fg_bb = config.backbone_weight[idx] * np.linalg.solve(Rg, np.array([0.0, 0.0, -1.0, 0.0]))
            if k < EPS:
                p_g = np.array([0.0, 0.0, lengths[idx] / 2.0])
            else:
                g_cog = np.sin(theta / 2.0) / (theta / 2.0) if abs(theta) >= EPS else 1.0
                p_g = (1.0 / k) * np.array(
                    [
                        (1.0 - g_cog * np.cos(theta / 2.0)) * np.cos(phi),
                        (1.0 - g_cog * np.cos(theta / 2.0)) * np.sin(phi),
                        g_cog * np.sin(theta / 2.0),
                    ]
                )

            F_prev_trans = T_i @ F_prev
            M_prev_trans = (T_i @ np.array([M_prev[0], M_prev[1], M_prev[2], 0.0]))[:3]

            F_net = F_rel.sum(axis=1) + F_prev_trans + Fg_disk + Fg_bb + Fex
            M_net = (
                M_rel.sum(axis=1)
                + M_prev_trans
                + np.cross(p_i[:3], F_prev_trans[:3])
                + np.cross(p_i[:3], Fg_disk[:3])
                + np.cross(p_g, Fg_bb[:3])
                + Mex
            )

            M_bend = np.array([-gamma * config.youngs_modulus * config.moment_of_inertia, beta * config.youngs_modulus * config.moment_of_inertia, 0.0])
            M_tor = (T_i @ np.array([0.0, 0.0, 2.0 * config.moment_of_inertia * config.shear_modulus * epsi / lengths[idx], 0.0]))[:3]
            res[3 * idx : 3 * idx + 3] = M_bend + M_tor - M_net
            F_prev = F_net
            M_prev = M_net

        return res

    @staticmethod
    def construct_tdcr_ccsub(lengths: np.ndarray, n_disk: int, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        all_s: list[np.ndarray] = []
        all_p: list[np.ndarray] = []
        s_offset = 0.0
        p_last = np.zeros(3)

        for ss_i in range(1, n_disk + 1):
            idx = ss_i - 1
            beta, gamma, _ = var[3 * idx : 3 * idx + 3]
            k = float(np.hypot(beta, gamma))
            phi = float(np.arctan2(gamma, beta)) if k >= EPS else 0.0
            si = np.linspace(0.0, lengths[idx], 30)
            if k < EPS:
                p_i = np.vstack([np.zeros_like(si), np.zeros_like(si), si])
            else:
                r = 1.0 / k
                p_i = r * np.vstack(
                    [
                        (1.0 - np.cos(si / r)) * np.cos(phi),
                        (1.0 - np.cos(si / r)) * np.sin(phi),
                        np.sin(si / r),
                    ]
                )
            R_i = ccsub_transform(var, lengths, ss_i - 1, 0)[:3, :3]
            p_plot = (R_i @ p_i).T + p_last
            all_s.append(s_offset + si)
            all_p.append(p_plot)
            s_offset += lengths[idx]
            p_last = p_plot[-1]

        return np.concatenate(all_s), np.vstack(all_p)

    @staticmethod
    def disk_frames(lengths: np.ndarray, n_disk: int, var: np.ndarray) -> np.ndarray:
        return np.array([ccsub_transform(var, lengths, disk_i, 0) for disk_i in range(n_disk + 1)])
