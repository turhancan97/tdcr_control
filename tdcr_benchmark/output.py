from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from tdcr_benchmark.config import RobotConfig
from tdcr_benchmark.models.base import ModelResult


def write_result(output_dir: Path, result: ModelResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_backbone_csv(output_dir / f"{result.model_name}_backbone.csv", result)
    np.savetxt(output_dir / f"{result.model_name}_tip_pose.csv", result.tip_pose, delimiter=",")


def write_backbone_csv(path: Path, result: ModelResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["s", "x", "y", "z"])
        for s, point in zip(result.backbone_s, result.backbone_positions):
            writer.writerow([f"{s:.12g}", f"{point[0]:.12g}", f"{point[1]:.12g}", f"{point[2]:.12g}"])


def plot_result(output_dir: Path, result: ModelResult, config: RobotConfig) -> None:
    plot_static_result(output_dir, result, config)
    plot_interactive_result(output_dir, result, config)


def plot_static_result(output_dir: Path, result: ModelResult, config: RobotConfig) -> None:
    numpy_major = int(np.__version__.split(".", maxsplit=1)[0])
    if numpy_major >= 2:
        warnings.warn(
            "Skipping static PNG plot because this environment uses NumPy 2.x with Matplotlib built for NumPy 1.x.",
            RuntimeWarning,
        )
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Skipping static PNG plot because Matplotlib is unavailable: {exc}", RuntimeWarning)
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = result.backbone_positions
    ax.plot(p[:, 0], p[:, 1], p[:, 2], linewidth=3)
    if result.disk_frames is not None:
        _plot_static_disks_and_tendons(ax, result.disk_frames, config)
    lmax = float(config.segment_lengths.sum())
    ax.set_xlim(-lmax, lmax)
    ax.set_ylim(-lmax, lmax)
    ax.set_zlim(0.0, lmax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(result.model_name)
    ax.view_init(elev=0.0, azim=90.0)
    fig.tight_layout()
    fig.savefig(output_dir / f"{result.model_name}.png", dpi=160)
    plt.close(fig)


def plot_interactive_result(output_dir: Path, result: ModelResult, config: RobotConfig) -> None:
    p = result.backbone_positions
    lmax = float(config.segment_lengths.sum())
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=p[:, 0],
                y=p[:, 1],
                z=p[:, 2],
                mode="lines",
                line={"width": 8},
                name=result.model_name,
            )
        ]
    )
    if result.disk_frames is not None:
        _add_interactive_disks_and_tendons(fig, result.disk_frames, config)
    fig.update_layout(
        title=result.model_name,
        scene={
            "xaxis": {"title": "x (m)", "range": [-lmax, lmax]},
            "yaxis": {"title": "y (m)", "range": [-lmax, lmax]},
            "zaxis": {"title": "z (m)", "range": [0.0, lmax]},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    fig.write_html(output_dir / f"{result.model_name}.html", include_plotlyjs=True, full_html=True)


def _plot_static_disks_and_tendons(ax, disk_frames: np.ndarray, config: RobotConfig) -> None:
    tendon_points = _tendon_points(disk_frames, config)
    for tendon_idx in range(tendon_points.shape[1]):
        pts = tendon_points[:, tendon_idx, :]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="red", linewidth=0.8, alpha=0.65)

    theta = np.linspace(0.0, 2.0 * np.pi, 72)
    disk_radius = config.r_disk + 0.002
    circle_local = np.vstack([disk_radius * np.cos(theta), disk_radius * np.sin(theta), np.zeros_like(theta)])
    for frame in disk_frames[1:]:
        circle = (frame[:3, :3] @ circle_local).T + frame[:3, 3]
        ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="#00a6a6", linewidth=0.8, alpha=0.65)


def _add_interactive_disks_and_tendons(fig: go.Figure, disk_frames: np.ndarray, config: RobotConfig) -> None:
    tendon_points = _tendon_points(disk_frames, config)
    for tendon_idx in range(tendon_points.shape[1]):
        pts = tendon_points[:, tendon_idx, :]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="lines",
                line={"color": "red", "width": 3},
                name=f"tendon {tendon_idx + 1}",
                opacity=0.65,
                showlegend=tendon_idx == 0,
                legendgroup="tendons",
            )
        )

    disk_x: list[float | None] = []
    disk_y: list[float | None] = []
    disk_z: list[float | None] = []
    theta = np.linspace(0.0, 2.0 * np.pi, 72)
    disk_radius = config.r_disk + 0.002
    circle_local = np.vstack([disk_radius * np.cos(theta), disk_radius * np.sin(theta), np.zeros_like(theta)])
    for frame in disk_frames[1:]:
        circle = (frame[:3, :3] @ circle_local).T + frame[:3, 3]
        disk_x.extend(circle[:, 0].tolist() + [None])
        disk_y.extend(circle[:, 1].tolist() + [None])
        disk_z.extend(circle[:, 2].tolist() + [None])
    fig.add_trace(
        go.Scatter3d(
            x=disk_x,
            y=disk_y,
            z=disk_z,
            mode="lines",
            line={"color": "#00a6a6", "width": 2},
            name="disks",
            opacity=0.8,
        )
    )


def _tendon_points(disk_frames: np.ndarray, config: RobotConfig) -> np.ndarray:
    tendon_local = config.p_tendon[:3, :]
    points = []
    for frame in disk_frames:
        points.append((frame[:3, :3] @ tendon_local + frame[:3, 3:4]).T)
    return np.array(points)
