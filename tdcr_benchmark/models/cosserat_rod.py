from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from tdcr_benchmark.config import RobotConfig
from tdcr_benchmark.math_utils import lie
from tdcr_benchmark.models.base import ModelResult, TDCRModel
from tdcr_benchmark.models.vc_common import boundcond, disk_frames_from_states, get_stiffness, intermedquant, row_to_rotation, setup_vc_params


class CosseratRodModel(TDCRModel):
    name = "vc"

    def forward_kinematics(self, config: RobotConfig) -> ModelResult:
        param = setup_vc_params(config)
        Fc = np.column_stack([config.tensions[:3], config.tensions[3:]])
        init_guess = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        solve = root(lambda guess: self._residual(guess, Fc, config.external_force[:3], config.segment_lengths, param), init_guess, method="hybr", options={"xtol": 1e-7, "maxfev": 1000})
        y = self._run_ivp(solve.x, Fc, config.segment_lengths, param, config.external_force[:3])
        return ModelResult(
            model_name=self.name,
            backbone_s=y[:, 18],
            backbone_positions=y[:, :3],
            tip_pose=_tip_pose_from_state(y[-1]),
            disk_frames=disk_frames_from_states(y, config),
            solver_success=bool(solve.success),
            message=str(solve.message),
        )

    def _residual(self, init_guess: np.ndarray, F: np.ndarray, Ftex: np.ndarray, L: np.ndarray, param) -> np.ndarray:
        y = self._run_ivp(init_guess, F, L, param, Ftex)
        res, _, _ = boundcond(len(y) - 1, F[:, 1], Ftex, 2, y, param.r, param)
        return res

    def _run_ivp(self, init_guess: np.ndarray, F: np.ndarray, L: np.ndarray, param, Ftex: np.ndarray) -> np.ndarray:
        y0 = np.concatenate([np.zeros(3), np.eye(3).reshape(9), init_guess[:6], [0.0]])
        y_rows = [y0]
        break_points = [0.0, float(L[0]), float(L.sum())]
        for k in (1, 2):
            sol = solve_ivp(
                lambda s, y: self._deriv(y, k, F, param),
                (break_points[k - 1], break_points[k]),
                y_rows[-1],
                atol=1e-6,
                rtol=1e-6,
                first_step=0.005,
            )
            segment = sol.y.T
            y_rows.extend(segment[1:])
            if k < 2:
                y_curr = y_rows[-1]
                R = row_to_rotation(y_curr)
                v = y_curr[12:15]
                u = y_curr[15:18]
                kbt, kse = get_stiffness(k, param)
                y_array = np.vstack(y_rows)
                _, F_sigma, L_sigma = boundcond(len(y_array) - 1, F[:, 0], Ftex, 2, y_array, param.r, param)
                v_new = v - np.linalg.solve(kse, R.T @ F_sigma)
                u_new = u - np.linalg.solve(kbt, R.T @ L_sigma)
                y_rows.append(np.concatenate([y_curr[:12], v_new, u_new, [y_curr[18]]]))
        return _unique_rows(np.vstack(y_rows))

    def _deriv(self, y: np.ndarray, k: int, F: np.ndarray, param) -> np.ndarray:
        u = y[15:18]
        v = y[12:15]
        R = y[3:12].reshape(3, 3)
        kbt, kse = get_stiffness(k, param)
        eg = np.array([0.0, 0.0, -1.0])
        fe = (param.q_tube[k - 1 :].sum() + param.q_disks[k - 1]) * eg
        le = np.zeros(3)
        A, B, G, H, c, d = intermedquant(u, v, R, param.r, kse, kbt, F, fe, le, k)
        vu_dot = np.linalg.solve(np.block([[kse + A, G], [B, kbt + H]]), np.concatenate([d, c]))
        p_dot = R @ v
        R_dot = R @ lie(u)
        return np.concatenate([p_dot, R_dot.reshape(9), vu_dot, [1.0]])


def _tip_pose_from_state(yrow: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = yrow[3:12].reshape(3, 3)
    T[:3, 3] = yrow[:3]
    return T


def _unique_rows(rows: np.ndarray) -> np.ndarray:
    _, idx = np.unique(rows, axis=0, return_index=True)
    return rows[np.sort(idx)]
