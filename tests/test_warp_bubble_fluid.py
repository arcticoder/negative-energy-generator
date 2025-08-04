import numpy as np
from simulation.dark_fluid import generate_warp_bubble_fluid

def test_warp_bubble_fluid_basic():
    N = 101
    dx = 0.1
    rho0 = -1.5
    R = 2.0
    sigma = 0.5
    x, rho = generate_warp_bubble_fluid(N, dx, rho0, R, sigma)
    # Check shapes
    assert isinstance(x, np.ndarray) and x.shape == (N,)
    assert isinstance(rho, np.ndarray) and rho.shape == (N,)
    # Maximum density magnitude should equal rho0 at shell
    assert np.isclose(rho.max(), rho0, atol=1e-6)
    # All values should be non-positive
    assert np.all(rho <= 0)
    # Values near x=0 should be near zero (no shell center)
    center_idx = N // 2
    assert abs(rho[center_idx]) < abs(rho0)
