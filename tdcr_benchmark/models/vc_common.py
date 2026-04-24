from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdcr_benchmark.config import RobotConfig
from tdcr_benchmark.math_utils import EPS, lie


@dataclass(frozen=True)
class VCParams:
    kbt1: np.ndarray
    kbt2: np.ndarray
    kse1: np.ndarray
    kse2: np.ndarray
    q_tube: np.ndarray
    q_disks: np.ndarray
    r: np.ndarray


def setup_vc_params(config: RobotConfig) -> VCParams:
    E = config.youngs_modulus
    nu = config.poissons_ratio
    G = config.shear_modulus
    ro1 = 0.0
    ri1 = 0.0
    ro2 = config.outer_radius
    ri2 = config.inner_radius
    I1 = 0.25 * np.pi * (ro1**4 - ri1**4)
    I2 = 0.25 * np.pi * (ro2**4 - ri2**4)
    A1 = np.pi * (ro1**2 - ri1**2)
    A2 = np.pi * (ro2**2 - ri2**2)
    kbt1 = np.diag([E * I1, E * I1, G * 2.0 * I1])
    kbt2 = np.diag([E * I2, E * I2, G * 2.0 * I2])
    kse1 = np.diag([G * A1, G * A1, E * A1])
    kse2 = np.diag([G * A2, G * A2, E * A2])

    # MATLAB setup_param_vc multiplies m2 by 9.81. main.m passes m_bb(1),
    # which is already zero when gravity is disabled.
    q_tube = np.array([0.0, config.backbone_weight[0] * 9.81 if config.backbone_weight.size else 0.0])
    q_disks = (config.n_disk * config.disk_weight[0] * 9.81) * np.ones_like(config.subsegment_lengths) / config.subsegment_lengths
    return VCParams(kbt1=kbt1, kbt2=kbt2, kse1=kse1, kse2=kse2, q_tube=q_tube, q_disks=q_disks, r=config.p_tendon[:3, :3])


def get_stiffness(k: int, param: VCParams) -> tuple[np.ndarray, np.ndarray]:
    if k == 1:
        return param.kbt1 + param.kbt2, param.kse1 + param.kse2
    if k == 2:
        return param.kbt2, param.kse2
    raise ValueError("VC section index must be 1 or 2.")


def row_to_rotation(y: np.ndarray) -> np.ndarray:
    return np.asarray(y[3:12], dtype=float).reshape(3, 3)


def state_to_pose_rows(y: np.ndarray) -> np.ndarray:
    return np.column_stack([y[:, 3:6], y[:, 6:9], y[:, 9:12], y[:, 0:3]])


def state_to_transform(yrow: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = row_to_rotation(yrow)
    T[:3, 3] = yrow[:3]
    return T


def disk_frames_from_states(y: np.ndarray, config: RobotConfig) -> np.ndarray:
    disk_s = np.concatenate(
        [
            np.linspace(0.0, config.segment_lengths[0], config.number_disks[0] + 1),
            np.linspace(
                config.segment_lengths[0] + config.segment_lengths[1] / config.number_disks[1],
                config.segment_lengths.sum(),
                config.number_disks[1],
            ),
        ]
    )
    frames = []
    for s_value in disk_s:
        idx = int(np.argmin(np.abs(y[:, 18] - s_value)))
        frames.append(state_to_transform(y[idx]))
    return np.array(frames)


def tendon_path_dot(u: np.ndarray, v: np.ndarray, ri: np.ndarray) -> np.ndarray:
    return lie(u) @ ri + v


def tendon_loads_at_state(yrow: np.ndarray, tau: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v_sigma = yrow[12:15]
    u_sigma = yrow[15:18]
    R_sigma = row_to_rotation(yrow)
    F_sigma = np.zeros(3)
    L_sigma = np.zeros(3)
    for i in range(3):
        p_dot = R_sigma @ tendon_path_dot(u_sigma, v_sigma, r[:, i])
        norm = max(float(np.linalg.norm(p_dot)), EPS)
        force = -(tau[i] / norm) * p_dot
        F_sigma += force
        L_sigma += -tau[i] * lie(R_sigma @ r[:, i]) @ ((1.0 / norm) * p_dot)
    return F_sigma, L_sigma


def boundcond(sigma_ind: int, tau: np.ndarray, F: np.ndarray, sect: int, y: np.ndarray, r: np.ndarray, param: VCParams):
    y_sigma = y[sigma_ind]
    F_sigma, L_sigma = tendon_loads_at_state(y_sigma, tau, r)

    if sect < 2:
        y_minus = y[sigma_ind - 1]
        y_plus = y[sigma_ind + 1]
        R_minus = row_to_rotation(y_minus)
        R_plus = row_to_rotation(y_plus)
        kbt, kse = get_stiffness(sect, param)
        n_minus = R_minus @ kse @ (y_minus[12:15] - np.array([0.0, 0.0, 1.0]))
        m_minus = R_minus @ kbt @ y_minus[15:18]
        kbt, kse = get_stiffness(sect + 1, param)
        n_plus = R_plus @ kse @ (y_plus[12:15] - np.array([0.0, 0.0, 1.0]))
        m_plus = R_plus @ kbt @ y_plus[15:18]
        res = np.concatenate([n_minus - n_plus - F_sigma, m_minus - m_plus - L_sigma])
    else:
        R_sigma = row_to_rotation(y_sigma)
        kbt, kse = get_stiffness(sect, param)
        n_sigma = R_sigma @ kse @ (y_sigma[12:15] - np.array([0.0, 0.0, 1.0]))
        m_sigma = R_sigma @ kbt @ y_sigma[15:18]
        res = np.concatenate([n_sigma - F_sigma - F, m_sigma - L_sigma])

    return res, F_sigma, L_sigma


def _single_tendon_quant(u: np.ndarray, v: np.ndarray, ri: np.ndarray, tau: float):
    pib_dot = tendon_path_dot(u, v, ri)
    norm = max(float(np.linalg.norm(pib_dot)), EPS)
    Lp = lie(pib_dot)
    Lr = lie(ri)
    A = -(tau / norm**3) * (Lp @ Lp)
    B = Lr @ A
    G = -(A @ Lr)
    H = -(B @ Lr)
    a = A @ lie(u) @ pib_dot
    b = Lr @ a
    return A, B, G, H, a, b


def intermedquant(u: np.ndarray, v: np.ndarray, R: np.ndarray, r: np.ndarray, Kse: np.ndarray, Kbt: np.ndarray, tau_stell: np.ndarray, fe: np.ndarray, le: np.ndarray, k: int):
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    G = np.zeros((3, 3))
    H = np.zeros((3, 3))
    a = np.zeros(3)
    b = np.zeros(3)

    sections = [1]
    if k == 1:
        sections.append(0)

    for section in sections:
        for i in range(3):
            Ai, Bi, Gi, Hi, ai, bi = _single_tendon_quant(u, v, r[:, i], tau_stell[i, section])
            A += Ai
            B += Bi
            G += Gi
            H += Hi
            a += ai
            b += bi

    c = -lie(u) @ Kbt @ u - lie(v) @ Kse @ (v - np.array([0.0, 0.0, 1.0])) - R.T @ le - b
    d = -lie(u) @ Kse @ (v - np.array([0.0, 0.0, 1.0])) - R.T @ fe - a
    return A, B, G, H, c, d
