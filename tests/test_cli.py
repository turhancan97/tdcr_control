from pathlib import Path

from tdcr_benchmark.cli import main


def test_cli_writes_outputs(tmp_path: Path):
    exit_code = main(["--output-root", str(tmp_path), "--no-timestamp"])
    assert exit_code == 0
    for model in ("cc", "ccsub", "vc", "vcref", "prbm"):
        assert (tmp_path / f"{model}_backbone.csv").exists()
        assert (tmp_path / f"{model}_tip_pose.csv").exists()
        assert (tmp_path / f"{model}.html").exists()
