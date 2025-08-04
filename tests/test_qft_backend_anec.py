import numpy as np
from simulation.qft_backend import PhysicsCore

def test_compute_anec_and_check_anec_positive():
    """
    compute_anec should correctly integrate a positive density array,
    and check_anec should return False for non-negative integral.
    """
    # Simple rho: constant 2.0 over grid
    N = 5
    dx = 0.1
    rho = np.full(N, 2.0)
    # Use PhysicsCore to compute
    core = PhysicsCore(grid=(np.linspace(0,1,N),)*3, dx=dx)
    anec = core.compute_anec(rho, dx)
    assert np.isclose(anec, 2.0 * N * dx)
    assert not core.check_anec(rho, dx)


def test_compute_anec_and_check_anec_negative():
    """
    compute_anec should integrate a negative density array,
    and check_anec should return True for negative integral.
    """
    N = 4
    dx = 0.5
    rho = np.array([-1.0, -2.0, -3.0, -4.0])
    core = PhysicsCore(grid=(np.linspace(0,1,N),)*3, dx=dx)
    anec = core.compute_anec(rho, dx)
    expected = rho.sum() * dx
    assert np.isclose(anec, expected)
    assert core.check_anec(rho, dx)
