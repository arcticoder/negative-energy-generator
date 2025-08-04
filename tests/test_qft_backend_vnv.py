import numpy as np
import pytest
from simulation.qft_backend import PhysicsCore


def test_build_toy_ansatz_shape_and_values():
    """
    V&V: PhysicsCore.build_toy_ansatz should produce a 4x4 tensor with correct shape and values.
    """
    x = np.linspace(-1, 1, 3)
    core = PhysicsCore(grid=(x, x, x), dx=x[1] - x[0])
    T = core.build_toy_ansatz({'alpha': 2.0, 'beta': 0.0})
    # Expect shape (4,4,3,3,3)
    assert T.shape == (4, 4, 3, 3, 3)
    # Only T[0,0] should be nonzero and equal to alpha (2.0)
    assert np.allclose(T[0, 0], 2.0)
    # All other tensor components should be zero
    for i in range(4):
        for j in range(4):
            if (i, j) != (0, 0):
                assert np.allclose(T[i, j], 0.0)


def test_local_energy_density_and_find_negative():
    """
    V&V: PhysicsCore.local_energy_density should return T00 for default observer,
    and find_negative should correctly flag negative density entries.
    """
    x = np.array([0.0, 1.0])
    core = PhysicsCore(grid=(x, x, x), dx=1.0)
    # Create custom stress-energy tensor with known T00 profile
    T = np.zeros((4, 4, 2, 2, 2))
    T00 = np.array([[[0.0, -2.0], [3.0, 0.0]], [[1.0, -1.0], [2.0, -3.0]]])
    T[0, 0] = T00
    rho = core.local_energy_density(T)
    # local energy density should equal T00 for u=(1,0,0,0)
    assert rho.shape == (2, 2, 2)
    assert np.allclose(rho, T00)
    # Mask negative entries
    mask = core.find_negative(rho)
    expected_neg_count = np.sum(T00 < 0)
    assert mask.dtype == bool
    assert np.sum(mask) == expected_neg_count
