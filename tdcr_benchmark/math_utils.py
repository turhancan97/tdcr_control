from __future__ import annotations

import numpy as np


EPS = 1e-9


def lie(xvec: np.ndarray) -> np.ndarray:
    x = np.asarray(xvec, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -x[2], x[1]],
            [x[2], 0.0, -x[0]],
            [-x[1], x[0], 0.0],
        ]
    )


def cross_columns(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.cross(np.asarray(u, dtype=float).T, np.asarray(v, dtype=float).T).T


def rot_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return T


def ccsub_local_transform(beta: float, gamma: float, epsi: float, length: float) -> np.ndarray:
    k = float(np.hypot(beta, gamma))
    if k < EPS:
        R = rot_z(epsi)
        p_i = np.array([0.0, 0.0, length])
        return transform(R, p_i)

    phi = float(np.arctan2(gamma, beta))
    theta = k * length
    p_i = np.array(
        [
            (1.0 - np.cos(theta)) * np.cos(phi) / k,
            (1.0 - np.cos(theta)) * np.sin(phi) / k,
            np.sin(theta) / k,
        ]
    )
    R = rot_z(phi) @ rot_y(theta) @ rot_z(epsi - phi)
    return transform(R, p_i)


def ccsub_transform(var: np.ndarray, lengths: np.ndarray, q: int, p: int) -> np.ndarray:
    if q < p:
        raise ValueError("q must be greater than or equal to p.")
    T = np.eye(4)
    for idx in range(p, q):
        beta, gamma, epsi = var[3 * idx : 3 * idx + 3]
        T = T @ ccsub_local_transform(beta, gamma, epsi, float(lengths[idx]))
    return T


def constant_curvature_transform(kappa: float, phi: float, length: float, sect_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    si = np.linspace(0.0, length, sect_points)
    transforms = np.zeros((sect_points, 4, 4))
    c_p = np.cos(phi)
    s_p = np.sin(phi)
    for i, s in enumerate(si):
        c_ks = np.cos(kappa * s)
        s_ks = np.sin(kappa * s)
        if abs(kappa) < EPS:
            position = np.array([0.0, 0.0, s])
        else:
            position = np.array([c_p * (1.0 - c_ks) / kappa, s_p * (1.0 - c_ks) / kappa, s_ks / kappa])
        R = np.array(
            [
                [c_p * c_ks, -s_p, c_p * s_ks],
                [s_p * c_ks, c_p, s_p * s_ks],
                [-s_ks, 0.0, c_ks],
            ]
        )
        transforms[i] = transform(R, position)
    return transforms, si

