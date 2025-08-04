import numpy as np
from simulation.dark_fluid import generate_alcubierre_fluid

def test_generate_alcubierre_fluid_basic():
    N = 101
    dx = 0.1
    rho0 = -1.0
    R = 2.0
    sigma = 0.5
    x, rho = generate_alcubierre_fluid(N, dx, rho0, R, sigma)
    # Check shapes and spacing
    assert isinstance(x, np.ndarray) and x.shape == (N,)
    assert isinstance(rho, np.ndarray) and rho.shape == (N,)
    assert np.isclose(x[1] - x[0], dx)
    # f(r) shape ensures max |rho| equals rho0
    assert np.isclose(np.max(np.abs(rho)), abs(rho0), atol=1e-6)
    # density is non-positive everywhere
    assert np.all(rho <= 0)
    # near center (r=0), f(0) ~ [tanh(R/sigma)-tanh(-R/sigma)]/(2*tanh(R/sigma)) ~1
    center_idx = N // 2
    assert rho[center_idx] == pytest.approx(rho0)
