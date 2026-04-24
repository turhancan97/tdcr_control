from tdcr_benchmark.config import load_config
from tdcr_benchmark.models.piecewise_constant_curvature import PiecewiseConstantCurvatureModel


def test_ccsub_forward_kinematics_output_shape():
    config = load_config()
    result = PiecewiseConstantCurvatureModel().forward_kinematics(config)
    assert result.backbone_positions.shape == (600, 3)
    assert result.tip_pose.shape == (4, 4)
    assert result.solver_success

