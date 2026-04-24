from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from tdcr_benchmark.config import RobotConfig
from tdcr_benchmark.math_utils import lie
from tdcr_benchmark.models.base import ModelResult, TDCRModel
from tdcr_benchmark.models.cosserat_rod import _tip_pose_from_state
from tdcr_benchmark.models.vc_common import disk_frames_from_states, get_stiffness, row_to_rotation, setup_vc_params


class SubsegmentCosseratRodModel(TDCRModel):
    name = "vcref"

    def forward_kinematics(self, config: RobotConfig) -> ModelResult:
        param = setup_vc_params(config)
        init_guess = np.tile(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), config.n_disk)
        solve = root(lambda guess: self._residual(guess, config, param)[0], init_guess, method="hybr", options={"xtol": 1e-5, "maxfev": 5000})
        _, y_total = self._residual(solve.x, config, param)
        return ModelResult(
            model_name=self.name,
            backbone_s=y_total[:, 18],
            backbone_positions=y_total[:, :3],
            tip_pose=_tip_pose_from_state(y_total[-1]),
            disk_frames=disk_frames_from_states(y_total, config),
            solver_success=bool(solve.success),
            message=str(solve.message),
        )

    def _residual(self, input_guess: np.ndarray, config: RobotConfig, param):
        n_disk = config.n_disk
        break_points = np.concatenate(
            [
                np.linspace(0.0, config.segment_lengths[0], config.number_disks[0] + 1),
                np.linspace(config.segment_lengths[0] + config.segment_lengths[1] / config.number_disks[1], config.segment_lengths.sum(), config.number_disks[1]),
            ]
        )
        kbt, kse = get_stiffness(1, param)
        m_states = np.zeros((n_disk + 1, 19))
        init_state = np.concatenate([np.zeros(3), np.eye(3).reshape(9), input_guess[:6], [0.0]])
        m_states[0] = init_state
        y_total_parts = []
        res_mat = np.zeros(6 * n_disk)

        for k in range(1, n_disk + 1):
            y = self._run_ivp(init_state, k, break_points, kbt, kse)
            y_total_parts.append(y)
            m_states[k] = y[-1]
            R_c = row_to_rotation(y[-1])
            v_c = y[-1, 12:15]
            u_c = y[-1, 15:18]
            if k < n_disk:
                v_n = input_guess[6 * k : 6 * k + 3]
                u_n = input_guess[6 * k + 3 : 6 * k + 6]
            else:
                v_n = np.array([0.0, 0.0, 1.0])
                u_n = np.zeros(3)
            n_c = R_c @ kse @ (v_c - np.array([0.0, 0.0, 1.0]))
            m_c = R_c @ kbt @ u_c
            n_n = R_c @ kse @ (v_n - np.array([0.0, 0.0, 1.0]))
            m_n = R_c @ kbt @ u_n
            res_mat[6 * (k - 1) : 6 * (k - 1) + 3] = n_c - n_n
            res_mat[6 * (k - 1) + 3 : 6 * (k - 1) + 6] = m_c - m_n
            if k < n_disk:
                init_state = np.concatenate([y[-1, :12], input_guess[6 * k : 6 * k + 6], [y[-1, 18]]])

        self._add_tendon_disk_loads(res_mat, m_states, config, param)
        return res_mat, np.vstack(y_total_parts)

    def _run_ivp(self, y_init: np.ndarray, k: int, break_points: np.ndarray, kbt: np.ndarray, kse: np.ndarray) -> np.ndarray:
        sol = solve_ivp(
            lambda s, y: self._deriv(y, kbt, kse),
            (break_points[k - 1], break_points[k]),
            y_init,
            atol=1e-6,
            rtol=1e-3,
            first_step=0.005,
        )
        return sol.y.T

    @staticmethod
    def _deriv(y: np.ndarray, kbt: np.ndarray, kse: np.ndarray) -> np.ndarray:
        u = y[15:18]
        v = y[12:15]
        R = y[3:12].reshape(3, 3)
        v_dot = -np.linalg.solve(kse, lie(u) @ kse @ (v - np.array([0.0, 0.0, 1.0])))
        u_dot = -np.linalg.solve(kbt, lie(u) @ kbt @ u + lie(v) @ kse @ (v - np.array([0.0, 0.0, 1.0])))
        return np.concatenate([R @ v, (R @ lie(u)).reshape(9), v_dot, u_dot, [1.0]])

    @staticmethod
    def _add_tendon_disk_loads(res_mat: np.ndarray, m_states: np.ndarray, config: RobotConfig, param) -> None:
        n = config.number_disks
        nt = len(config.tensions) // len(n)
        Fdisk = np.vstack(
            [
                np.tile(config.tensions, (n[0], 1)),
                np.hstack([np.zeros((n[1], nt)), np.tile(config.tensions[nt:], (n[1], 1))]),
            ]
        )
        r = np.column_stack([param.r, param.r])
        n_disk = config.n_disk
        for k in range(1, n_disk + 1):
            y_c = m_states[k]
            p_c = y_c[:3]
            R_c = row_to_rotation(y_c)
            y_p = m_states[k - 1]
            p_p = y_p[:3]
            R_p = row_to_rotation(y_p)
            tendon_pos_cur = R_c @ r + p_c[:, None]
            tendon_pos_prev = R_p @ r + p_p[:, None]
            if k < n_disk:
                y_n = m_states[k + 1]
                tendon_pos_next = row_to_rotation(y_n) @ r + y_n[:3, None]
            zi = R_c[:, 2]
            F_tendon = np.zeros(3)
            M_tendon = np.zeros(3)
            if k == n_disk:
                for m in range(6):
                    dir1 = tendon_pos_prev[:, m] - tendon_pos_cur[:, m]
                    force1 = Fdisk[k - 1, m] * dir1 / max(np.linalg.norm(dir1), 1e-9)
                    F_tendon += force1
                    M_tendon += np.cross(tendon_pos_cur[:, m] - p_c, force1)
                res_mat[6 * (k - 1) : 6 * (k - 1) + 3] -= F_tendon + config.external_force[:3]
                res_mat[6 * (k - 1) + 3 : 6 * (k - 1) + 6] -= M_tendon + config.external_moment[:3]
            elif k == n[0]:
                for m in range(6):
                    dir1 = tendon_pos_prev[:, m] - tendon_pos_cur[:, m]
                    dir2 = tendon_pos_next[:, m] - tendon_pos_cur[:, m]
                    force1 = Fdisk[k - 1, m] * dir1 / max(np.linalg.norm(dir1), 1e-9) + Fdisk[k, m] * dir2 / max(np.linalg.norm(dir2), 1e-9)
                    if m > 2:
                        force1 = force1 - (zi @ force1) * zi
                    F_tendon += force1
                    M_tendon += np.cross(tendon_pos_cur[:, m] - p_c, force1)
                res_mat[6 * (k - 1) : 6 * (k - 1) + 3] -= F_tendon
                res_mat[6 * (k - 1) + 3 : 6 * (k - 1) + 6] -= M_tendon
            else:
                for m in range(6):
                    dir1 = tendon_pos_prev[:, m] - tendon_pos_cur[:, m]
                    dir2 = tendon_pos_next[:, m] - tendon_pos_cur[:, m]
                    force1 = Fdisk[k - 1, m] * dir1 / max(np.linalg.norm(dir1), 1e-9) + Fdisk[k - 1, m] * dir2 / max(np.linalg.norm(dir2), 1e-9)
                    force1 = force1 - (zi @ force1) * zi
                    F_tendon += force1
                    M_tendon += np.cross(tendon_pos_cur[:, m] - p_c, force1)
                res_mat[6 * (k - 1) : 6 * (k - 1) + 3] -= F_tendon
                res_mat[6 * (k - 1) + 3 : 6 * (k - 1) + 6] -= M_tendon
