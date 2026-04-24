from tdcr_benchmark.config import load_config


def test_load_default_config():
    config = load_config()
    assert config.number_disks.tolist() == [10, 10]
    assert config.n_disk == 20
    assert config.p_tendon.shape == (4, 6)
    assert config.tensions.tolist() == [8.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert config.cc_tendon_displacements.shape == (6,)

