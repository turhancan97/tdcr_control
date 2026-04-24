from pathlib import Path

import numpy as np

from tdcr_benchmark.animate import (
    generate_circle_trajectory,
    generate_line_trajectory,
    run_animation,
    solve_cc_ik_sequence,
)
from tdcr_benchmark.config import load_config


def test_generate_line_trajectory_shape_and_plane():
    traj = generate_line_trajectory(
        frames=5,
        start_xy=np.array([-0.02, -0.01]),
        end_xy=np.array([0.02, 0.01]),
        plane_z=0.31,
    )
    assert traj.shape == (5, 3)
    assert np.allclose(traj[:, 2], 0.31)
    np.testing.assert_allclose(traj[0], [-0.02, -0.01, 0.31])
    np.testing.assert_allclose(traj[-1], [0.02, 0.01, 0.31])


def test_generate_circle_trajectory_shape_and_plane():
    traj = generate_circle_trajectory(
        frames=8,
        center_xy=np.array([0.01, -0.02]),
        radius=0.03,
        plane_z=0.29,
    )
    assert traj.shape == (8, 3)
    assert np.allclose(traj[:, 2], 0.29)
    r = np.linalg.norm(traj[:, :2] - np.array([0.01, -0.02]), axis=1)
    np.testing.assert_allclose(r, 0.03, atol=1e-12)


def test_solve_cc_ik_sequence_bounds_and_status():
    config = load_config()
    targets = generate_line_trajectory(
        frames=4,
        start_xy=np.array([-0.01, 0.0]),
        end_xy=np.array([0.01, 0.0]),
        plane_z=0.32,
    )
    frames = solve_cc_ik_sequence(
        config=config,
        targets=targets,
        tendon_min=-0.03,
        tendon_max=0.03,
        ik_max_iter=80,
        ik_tol=1e-4,
    )
    assert len(frames) == 4
    for frame in frames:
        assert frame.tendon_commands.shape == (6,)
        assert np.all(frame.tendon_commands >= -0.03 - 1e-10)
        assert np.all(frame.tendon_commands <= 0.03 + 1e-10)
        assert frame.actual_tip.shape == (3,)
        assert isinstance(frame.solve_success, bool)


def test_animate_cli_outputs(tmp_path: Path):
    output_dir, frames = run_animation(
        output_root=tmp_path,
        timestamped=False,
        frames=6,
        trajectory="line",
        plane_z=0.32,
        ik_max_iter=40,
        ik_tol=1e-4,
    )
    assert output_dir == tmp_path
    assert len(frames) == 6
    assert (tmp_path / "animation.html").exists()
    assert (tmp_path / "trajectory_targets.csv").exists()
    assert (tmp_path / "trajectory_tip_actual.csv").exists()
    assert (tmp_path / "trajectory_tendon_commands.csv").exists()


def test_short_regression_mean_tracking_error():
    output_dir, frames = run_animation(
        output_root=Path("/tmp/tdcr_animate_regression"),
        timestamped=False,
        frames=8,
        trajectory="line",
        plane_z=0.32,
        ik_max_iter=60,
        ik_tol=1e-4,
    )
    del output_dir
    target = np.array([frame.target_tip for frame in frames])
    actual = np.array([frame.actual_tip for frame in frames])
    mean_err = np.linalg.norm(actual - target, axis=1).mean()
    assert mean_err < 0.08
