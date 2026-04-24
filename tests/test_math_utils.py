import numpy as np

from tdcr_benchmark.math_utils import ccsub_transform, lie


def test_lie_matrix_cross_product_equivalence():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(lie(x) @ y, np.cross(x, y))


def test_ccsub_transform_zero_curvature_advances_along_z():
    var = np.zeros(3)
    T = ccsub_transform(var, np.array([0.2]), 1, 0)
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 0.2], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)

