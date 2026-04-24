from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import least_squares

from tdcr_benchmark.config import RobotConfig, default_config_path, load_config
from tdcr_benchmark.models.constant_curvature import ConstantCurvatureModel


@dataclass(frozen=True)
class AnimationFrame:
    index: int
    target_tip: np.ndarray
    tendon_commands: np.ndarray
    actual_tip: np.ndarray
    solve_success: bool
    solve_cost: float
    model_result: object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a TDCR tip-trajectory animation using the CC model.")
    parser.add_argument("--config", default=str(default_config_path()), help="Path to config.yaml.")
    parser.add_argument("--output-root", default="outputs", help="Directory where timestamped animation outputs are written.")
    parser.add_argument("--no-timestamp", action="store_true", help="Write directly into --output-root instead of a timestamped subdirectory.")
    parser.add_argument("--frames", type=int, default=120, help="Number of trajectory samples/frames.")
    parser.add_argument("--trajectory", choices=("line", "circle"), default="circle", help="Trajectory type in task space.")
    parser.add_argument("--plane-z", type=float, default=0.35, help="Fixed Z plane for trajectory targets.")
    parser.add_argument("--line-start-x", type=float, default=0.034)
    parser.add_argument("--line-start-y", type=float, default=0.00)
    parser.add_argument("--line-end-x", type=float, default=0.22)
    parser.add_argument("--line-end-y", type=float, default=0.00)
    parser.add_argument("--circle-center-x", type=float, default=0.00)
    parser.add_argument("--circle-center-y", type=float, default=0.00)
    parser.add_argument("--circle-radius", type=float, default=0.2)
    parser.add_argument("--ik-max-iter", type=int, default=150, help="Maximum evaluations per frame for IK.")
    parser.add_argument("--ik-tol", type=float, default=1e-5, help="Tolerance for IK least-squares solve.")
    parser.add_argument("--tendon-min", type=float, default=-0.03, help="Lower bound for tendon command values.")
    parser.add_argument("--tendon-max", type=float, default=0.03, help="Upper bound for tendon command values.")
    return parser


def generate_line_trajectory(frames: int, start_xy: np.ndarray, end_xy: np.ndarray, plane_z: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, frames)
    xy = start_xy[None, :] + (end_xy - start_xy)[None, :] * t[:, None]
    return np.column_stack([xy, np.full(frames, plane_z, dtype=float)])


def generate_circle_trajectory(frames: int, center_xy: np.ndarray, radius: float, plane_z: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    x = center_xy[0] + radius * np.cos(theta)
    y = center_xy[1] + radius * np.sin(theta)
    z = np.full(frames, plane_z, dtype=float)
    return np.column_stack([x, y, z])


def solve_cc_ik_sequence(
    config: RobotConfig,
    targets: np.ndarray,
    tendon_min: float,
    tendon_max: float,
    ik_max_iter: int,
    ik_tol: float,
) -> list[AnimationFrame]:
    model = ConstantCurvatureModel()
    lower = np.full(6, tendon_min, dtype=float)
    upper = np.full(6, tendon_max, dtype=float)
    current = np.clip(np.zeros(6, dtype=float), lower, upper)
    frames: list[AnimationFrame] = []

    for idx, target in enumerate(targets):
        def objective(q: np.ndarray) -> np.ndarray:
            cfg = replace(config, cc_tendon_displacements=np.asarray(q, dtype=float))
            result = model.forward_kinematics(cfg)
            return result.tip_pose[:3, 3] - target

        solve = least_squares(
            objective,
            x0=current,
            bounds=(lower, upper),
            max_nfev=ik_max_iter,
            xtol=ik_tol,
            ftol=ik_tol,
            gtol=ik_tol,
        )
        q_sol = np.clip(solve.x, lower, upper)
        cfg = replace(config, cc_tendon_displacements=q_sol)
        result = model.forward_kinematics(cfg)
        current = q_sol
        frames.append(
            AnimationFrame(
                index=idx,
                target_tip=np.asarray(target, dtype=float),
                tendon_commands=q_sol,
                actual_tip=result.tip_pose[:3, 3].copy(),
                solve_success=bool(solve.success),
                solve_cost=float(solve.cost),
                model_result=result,
            )
        )
    return frames


def _tendon_points(disk_frames: np.ndarray, config: RobotConfig) -> np.ndarray:
    tendon_local = config.p_tendon[:3, :]
    return np.array([(frame[:3, :3] @ tendon_local + frame[:3, 3:4]).T for frame in disk_frames])


def _frame_traces(model_result, config: RobotConfig, include_legend: bool) -> list[go.Scatter3d]:
    traces: list[go.Scatter3d] = []
    p = model_result.backbone_positions
    traces.append(
        go.Scatter3d(
            x=p[:, 0],
            y=p[:, 1],
            z=p[:, 2],
            mode="lines",
            line={"width": 8, "color": "#1f77b4"},
            name="backbone",
            showlegend=include_legend,
            legendgroup="backbone",
        )
    )

    if model_result.disk_frames is not None:
        tendon_points = _tendon_points(model_result.disk_frames, config)
        for tendon_idx in range(tendon_points.shape[1]):
            pts = tendon_points[:, tendon_idx, :]
            traces.append(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    line={"color": "red", "width": 3},
                    name=f"tendon {tendon_idx + 1}",
                    showlegend=include_legend and tendon_idx == 0,
                    legendgroup="tendons",
                    opacity=0.65,
                )
            )

        disk_x: list[float | None] = []
        disk_y: list[float | None] = []
        disk_z: list[float | None] = []
        theta = np.linspace(0.0, 2.0 * np.pi, 72)
        disk_radius = config.r_disk + 0.002
        circle_local = np.vstack([disk_radius * np.cos(theta), disk_radius * np.sin(theta), np.zeros_like(theta)])
        for frame in model_result.disk_frames[1:]:
            circle = (frame[:3, :3] @ circle_local).T + frame[:3, 3]
            disk_x.extend(circle[:, 0].tolist() + [None])
            disk_y.extend(circle[:, 1].tolist() + [None])
            disk_z.extend(circle[:, 2].tolist() + [None])
        traces.append(
            go.Scatter3d(
                x=disk_x,
                y=disk_y,
                z=disk_z,
                mode="lines",
                line={"color": "#00a6a6", "width": 2},
                name="disks",
                showlegend=include_legend,
                legendgroup="disks",
                opacity=0.8,
            )
        )
    return traces




def _trajectory_traces(targets: np.ndarray, actual: np.ndarray, current_index: int, include_legend: bool) -> list[go.Scatter3d]:
    current_actual = actual[: current_index + 1]
    current_target = targets[current_index]
    current_tip = actual[current_index]
    return [
        go.Scatter3d(
            x=targets[:, 0],
            y=targets[:, 1],
            z=targets[:, 2],
            mode="lines",
            line={"color": "#2ca02c", "width": 4, "dash": "dash"},
            name="target trajectory",
            showlegend=include_legend,
            legendgroup="trajectory",
        ),
        go.Scatter3d(
            x=current_actual[:, 0],
            y=current_actual[:, 1],
            z=current_actual[:, 2],
            mode="lines",
            line={"color": "#ff7f0e", "width": 5},
            name="actual trajectory",
            showlegend=include_legend,
            legendgroup="trajectory",
        ),
        go.Scatter3d(
            x=[current_target[0]],
            y=[current_target[1]],
            z=[current_target[2]],
            mode="markers",
            marker={"color": "#2ca02c", "size": 5, "symbol": "diamond"},
            name="target tip",
            showlegend=include_legend,
            legendgroup="trajectory",
        ),
        go.Scatter3d(
            x=[current_tip[0]],
            y=[current_tip[1]],
            z=[current_tip[2]],
            mode="markers",
            marker={"color": "#ff7f0e", "size": 5},
            name="actual tip",
            showlegend=include_legend,
            legendgroup="trajectory",
        ),
    ]

def write_animation_html(output_dir: Path, frames: list[AnimationFrame], config: RobotConfig) -> None:
    if not frames:
        raise ValueError("No frames to render.")
    lmax = float(config.segment_lengths.sum())
    targets = np.array([frame.target_tip for frame in frames], dtype=float)
    actual = np.array([frame.actual_tip for frame in frames], dtype=float)
    first_traces = _frame_traces(frames[0].model_result, config, include_legend=True) + _trajectory_traces(targets, actual, 0, include_legend=True)
    figure_frames = [
        go.Frame(
            data=_frame_traces(frame.model_result, config, include_legend=False) + _trajectory_traces(targets, actual, frame.index, include_legend=False),
            name=str(frame.index),
        )
        for frame in frames
    ]

    fig = go.Figure(data=first_traces, frames=figure_frames)
    fig.update_layout(
        title="cc trajectory animation",
        scene={
            "xaxis": {"title": "x (m)", "range": [-lmax, lmax]},
            "yaxis": {"title": "y (m)", "range": [-lmax, lmax]},
            "zaxis": {"title": "z (m)", "range": [0.0, lmax]},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {"label": str(frame.index), "method": "animate", "args": [[str(frame.index)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
                    for frame in frames
                ],
            }
        ],
    )
    fig.write_html(output_dir / "animation.html", include_plotlyjs=True, full_html=True)


def write_timeseries_csv(output_dir: Path, frames: list[AnimationFrame]) -> None:
    with (output_dir / "trajectory_targets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "z"])
        for frame in frames:
            writer.writerow([frame.index, *frame.target_tip.tolist()])

    with (output_dir / "trajectory_tip_actual.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "z", "solve_success", "solve_cost"])
        for frame in frames:
            writer.writerow([frame.index, *frame.actual_tip.tolist(), int(frame.solve_success), frame.solve_cost])

    with (output_dir / "trajectory_tendon_commands.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "q1", "q2", "q3", "q4", "q5", "q6"])
        for frame in frames:
            writer.writerow([frame.index, *frame.tendon_commands.tolist()])


def run_animation(
    config_path: str | Path | None = None,
    output_root: str | Path = "outputs",
    timestamped: bool = True,
    frames: int = 120,
    trajectory: str = "line",
    plane_z: float = 0.32,
    line_start_x: float = -0.03,
    line_start_y: float = 0.00,
    line_end_x: float = 0.03,
    line_end_y: float = 0.00,
    circle_center_x: float = 0.00,
    circle_center_y: float = 0.00,
    circle_radius: float = 0.03,
    ik_max_iter: int = 150,
    ik_tol: float = 1e-5,
    tendon_min: float = -0.03,
    tendon_max: float = 0.03,
) -> tuple[Path, list[AnimationFrame]]:
    if frames <= 1:
        raise ValueError("--frames must be greater than 1.")
    if tendon_min >= tendon_max:
        raise ValueError("--tendon-min must be less than --tendon-max.")
    config = load_config(config_path)
    run_dir = Path(output_root)
    if timestamped:
        run_dir = run_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if trajectory == "line":
        targets = generate_line_trajectory(
            frames=frames,
            start_xy=np.array([line_start_x, line_start_y], dtype=float),
            end_xy=np.array([line_end_x, line_end_y], dtype=float),
            plane_z=plane_z,
        )
    elif trajectory == "circle":
        targets = generate_circle_trajectory(
            frames=frames,
            center_xy=np.array([circle_center_x, circle_center_y], dtype=float),
            radius=float(circle_radius),
            plane_z=plane_z,
        )
    else:
        raise ValueError(f"Unsupported trajectory type: {trajectory}")

    solved_frames = solve_cc_ik_sequence(
        config=config,
        targets=targets,
        tendon_min=tendon_min,
        tendon_max=tendon_max,
        ik_max_iter=ik_max_iter,
        ik_tol=ik_tol,
    )
    write_animation_html(run_dir, solved_frames, config)
    write_timeseries_csv(run_dir, solved_frames)
    return run_dir, solved_frames


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir, solved_frames = run_animation(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        timestamped=not args.no_timestamp,
        frames=args.frames,
        trajectory=args.trajectory,
        plane_z=args.plane_z,
        line_start_x=args.line_start_x,
        line_start_y=args.line_start_y,
        line_end_x=args.line_end_x,
        line_end_y=args.line_end_y,
        circle_center_x=args.circle_center_x,
        circle_center_y=args.circle_center_y,
        circle_radius=args.circle_radius,
        ik_max_iter=args.ik_max_iter,
        ik_tol=args.ik_tol,
        tendon_min=args.tendon_min,
        tendon_max=args.tendon_max,
    )
    success_count = sum(1 for frame in solved_frames if frame.solve_success)
    print(f"Wrote animation outputs to {run_dir}")
    print(f"Frames solved successfully: {success_count}/{len(solved_frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

