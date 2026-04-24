from pathlib import Path

import numpy as np

from tdcr_benchmark.runner import run_benchmark


def test_benchmark_matches_python_reference_outputs(tmp_path: Path):
    run_benchmark(output_root=tmp_path, timestamped=False)
    reference_dir = Path(__file__).resolve().parents[1] / "reference_outputs"

    for model in ("cc", "ccsub", "vc", "vcref", "prbm"):
        actual_tip = np.loadtxt(tmp_path / f"{model}_tip_pose.csv", delimiter=",")
        reference_tip = np.loadtxt(reference_dir / f"{model}_tip_pose.csv", delimiter=",")
        np.testing.assert_allclose(actual_tip, reference_tip, rtol=1e-7, atol=1e-9)

        actual_backbone = np.loadtxt(tmp_path / f"{model}_backbone.csv", delimiter=",", skiprows=1)
        reference_backbone = np.loadtxt(reference_dir / f"{model}_backbone.csv", delimiter=",", skiprows=1)
        np.testing.assert_allclose(actual_backbone, reference_backbone, rtol=1e-7, atol=1e-9)
