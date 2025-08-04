import numpy as np
from simulation.dark_fluid import generate_phantom_dark_fluid

def test_generate_phantom_dark_fluid_basic():
    N = 5
    dx = 0.2
    rho0 = -2.0
    w = -1.2
    x, rho, p = generate_phantom_dark_fluid(N, dx, rho0, w)
    # Spatial grid should have correct shape and spacing
    assert isinstance(x, np.ndarray) and x.shape == (N,)
    assert np.isclose(x[1] - x[0], dx)
    # Density should be constant rho0
    assert np.allclose(rho, rho0)
    # Pressure should be w * rho0
    assert np.allclose(p, rho0 * w)
