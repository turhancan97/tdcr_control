from __future__ import annotations

import numpy as np
from scipy.optimize import root

from tdcr_benchmark.config import RobotConfig
from tdcr_benchmark.math_utils import EPS, cross_columns, rot_y, rot_z, transform
from tdcr_benchmark.models.base import ModelResult, TDCRModel


class PseudoRigidBodyModel(TDCRModel):
    name = "prbm"

    def __init__(self, nrb: int = 4):
        self.nrb = nrb
        self.gamma = np.array([0.125, 0.35, 0.388, 0.136], dtype=float)
        self.gamma = self.gamma / self.gamma.sum()

    def forward_kinematics(self, config: RobotConfig) -> ModelResult:
        phi0 = 0.0
        var0 = np.tile(np.concatenate([[phi0], np.zeros(self.nrb)]), config.n_disk)
        solve = root(lambda var: self.residual(var, config), var0, method="hybr", options={"xtol": 1e-7, "maxfev": 5000})
        var = solve.x
        backbone_s, backbone_positions = self.construct_tdcr_prbm(config.subsegment_lengths, config.n_disk, var)
        tip_pose, _ = self.trans_mat_prbm(var, config.subsegment_lengths, config.n_disk, 0)
        return ModelResult(
            model_name=self.name,
            backbone_s=backbone_s,
            backbone_positions=backbone_positions,
            tip_pose=tip_pose,
            disk_frames=self.disk_frames(config.subsegment_lengths, config.n_disk, var),
            solver_success=bool(solve.success),
            message=str(solve.message),
        )

    def residual(self, var: np.ndarray, config: RobotConfig) -> np.ndarray:
        n = config.number_disks
        n_disk = config.n_disk
        nrb = self.nrb
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
        res = np.zeros(n_disk * (nrb + 1))
        F_prev = np.zeros(3)
        M_prev = np.zeros(3)

        for ss_i in range(n_disk, 0, -1):
            idx = ss_i - 1
            T_i, Trb = self.trans_mat_prbm(var, lengths, ss_i, ss_i - 1)
            base = (nrb + 1) * idx
            theta = var[base : base + 3]
            phi = var[base + 3]
            epsi = var[base + 4]
            ni = np.array([np.cos(phi + np.pi / 2.0), np.sin(phi + np.pi / 2.0), 0.0])

            p_ti = T_i @ p_tendon
            p_i = T_i @ np.array([0.0, 0.0, 0.0, 1.0])
            norm_ct1 = np.maximum(np.linalg.norm(-p_ti[:3, :] + p_tendon[:3, :], axis=0), EPS)
            Pi = np.column_stack([Trb[k][:3, 3] for k in range(nrb)])
            zi = T_i[:, 2]

            if ss_i < n_disk:
                T_i2, _ = self.trans_mat_prbm(var, lengths, ss_i + 1, ss_i - 1)
                p_ti2 = T_i2 @ p_tendon
                norm_ct2 = np.maximum(np.linalg.norm(-p_ti[:3, :] + p_ti2[:3, :], axis=0), EPS)
                Fi = ((p_tendon - p_ti) / norm_ct1) * Fdisk[idx] + ((p_ti2 - p_ti) / norm_ct2) * Fdisk[idx + 1]
                if ss_i == n[0]:
                    distal = Fi[:, 3:6] - np.outer(zi, zi @ Fi[:, 3:6])
                    Fi = np.hstack([Fi[:, 0:3], distal])
                else:
                    Fi = Fi - np.outer(zi, zi @ Fi)
            else:
                Fi = ((p_tendon - p_ti) / norm_ct1) * Fdisk[idx]

            Mi = cross_columns(p_ti[:3, :] - Pi[:, -1, None], Fi[:3, :])

            Rt, _ = self.trans_mat_prbm(var, lengths, ss_i - 1, 0)
            Fex = np.linalg.solve(Rt, config.external_force)
            R_ex, _ = self.trans_mat_prbm(var, lengths, n_disk, ss_i - 1)
            p_ex = R_ex[:3, 3]
            Mex = np.linalg.solve(Rt[:3, :3], config.external_moment[:3]) - np.cross(Pi[:, -1] - p_ex, Fex[:3])

            if ss_i < n_disk:
                Ftot = T_i[:3, :3] @ F_prev + Fi[:3, :].sum(axis=1)
                Mtot = T_i[:3, :3] @ M_prev + np.cross(T_i2[:3, 3] - Pi[:, -1], T_i[:3, :3] @ F_prev) + Mi.sum(axis=1)
            else:
                Ftot = Fi[:3, :].sum(axis=1)
                Mtot = Mi.sum(axis=1)

            K = np.array([3.25, 2.84, 2.95]) * config.youngs_modulus * config.moment_of_inertia / lengths[idx]
            for k in range(nrb - 1):
                Rb = Trb[k + 1][:3, :3]
                Mnetb = Rb.T @ (np.cross(Pi[:, -1] - Pi[:, k], Ftot + Fex[:3]) + Mtot + Mex)
                res[base + k] = K[k] * theta[k] - Mnetb[1]

            Mnet = np.cross(p_i[:3], Ftot + Fex[:3]) + Mtot + Mex
            Mphi = Mnet.copy()
            Mphi[2] = 0.0
            res[base + 3] = ni @ Mphi - np.linalg.norm(Mphi)
            Mepsi = T_i[:3, :3].T @ Mnet
            res[base + 4] = Mepsi[2] - 2.0 * config.shear_modulus * config.moment_of_inertia / lengths[idx] * epsi
            F_prev = Ftot
            M_prev = Mtot

        return res

    def trans_mat_prbm(self, var: np.ndarray, lengths: np.ndarray, q: int, p: int) -> tuple[np.ndarray, list[np.ndarray]]:
        if q < p:
            raise ValueError("q must be greater than or equal to p.")
        T = np.eye(4)
        Trb: list[np.ndarray] = []
        nrb = self.nrb
        for iter_idx in range(p, q):
            base = (nrb + 1) * iter_idx
            theta = var[base : base + 3]
            phi = var[base + 3]
            epsi = var[base + 4]
            R_phi = transform(rot_z(phi), np.zeros(3))
            R_phi_epsi = transform(rot_z(epsi - phi), np.zeros(3))
            Ti = R_phi @ transform(np.eye(3), np.array([0.0, 0.0, self.gamma[0] * lengths[iter_idx]]))
            Trb.append(T @ Ti)
            for k in range(nrb - 1):
                joint = np.eye(4)
                joint[:3, :3] = rot_y(theta[k])
                joint[:3, 3] = np.array([
                    self.gamma[k + 1] * lengths[iter_idx] * np.sin(theta[k]),
                    0.0,
                    self.gamma[k + 1] * lengths[iter_idx] * np.cos(theta[k]),
                ])
                Ti = Ti @ joint
                Trb.append(T @ Ti)
            T = T @ Ti @ R_phi_epsi
        return T, Trb

    def construct_tdcr_prbm(self, lengths: np.ndarray, n_disk: int, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, Trb = self.trans_mat_prbm(var, lengths, n_disk, 0)
        points = [np.zeros(3)]
        s_values = [0.0]
        cursor = 0.0
        for k in range(n_disk):
            for t in range(self.nrb):
                points.append(Trb[self.nrb * k + t][:3, 3])
                cursor += self.gamma[t] * lengths[k]
                s_values.append(cursor)
        return np.array(s_values), np.vstack(points)

    def disk_frames(self, lengths: np.ndarray, n_disk: int, var: np.ndarray) -> np.ndarray:
        return np.array([self.trans_mat_prbm(var, lengths, disk_i, 0)[0] for disk_i in range(n_disk + 1)])
