import numpy as np

from tdcr_benchmark.config import load_config
from tdcr_benchmark.models.constant_curvature import ConstantCurvatureModel, configuration


def test_configuration_equal_tendon_lengths_is_straight():
    kappa, phi, length = configuration(np.array([0.2, 0.2, 0.2]), 10, 0.01)
    assert kappa == 0.0
    assert phi == 0.0
    assert length == 0.2


def test_constant_curvature_forward_kinematics_default_shape():
    config = load_config()
    result = ConstantCurvatureModel().forward_kinematics(config)
    assert result.backbone_positions.shape == (100, 3)
    np.testing.assert_allclose(result.tip_pose[:3, 3], [0.0, 0.0, 0.4], atol=1e-12)

