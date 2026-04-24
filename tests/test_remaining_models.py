from tdcr_benchmark.config import load_config
from tdcr_benchmark.models import CosseratRodModel, PseudoRigidBodyModel, SubsegmentCosseratRodModel


def test_remaining_models_forward_kinematics():
    config = load_config()
    for model in (CosseratRodModel(), SubsegmentCosseratRodModel(), PseudoRigidBodyModel()):
        result = model.forward_kinematics(config)
        assert result.solver_success, result.message
        assert result.tip_pose.shape == (4, 4)
        assert result.backbone_positions.shape[1] == 3
