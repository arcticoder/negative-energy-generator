import numpy as np
from simulation.dark_fluid import generate_vacuum_fluctuation_fluid

def test_generate_vacuum_fluctuation_fluid_basic():
    N = 100
    dx = 0.05
    amp = 1e-2
    corr_len = 0.2
    seed = 123
    x, rho = generate_vacuum_fluctuation_fluid(N, dx, amplitude=amp, corr_len=corr_len, seed=seed)
    # Check shapes and spacing
    assert isinstance(x, np.ndarray) and x.shape == (N,)
    assert isinstance(rho, np.ndarray) and rho.shape == (N,)
    assert np.isclose(x[1] - x[0], dx)
    # Values should be finite and vary
    assert np.all(np.isfinite(rho))
    assert np.std(rho) > 0
