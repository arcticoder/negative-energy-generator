import numpy as np
from simulation.dark_fluid import generate_phase_transition_fluid

def test_generate_phase_transition_fluid_basic():
    N = 51
    dx = 0.1
    rho_core = -2.0
    rho_env = -0.5
    R_core = 2.0
    width = 0.5
    x, rho = generate_phase_transition_fluid(N, dx, rho_core, rho_env, R_core, width)
    # Check shapes and spacing
    assert isinstance(x, np.ndarray) and x.shape == (N,)
    assert isinstance(rho, np.ndarray) and rho.shape == (N,)
    assert np.isclose(x[1] - x[0], dx)
    # Core region values near center should be ~rho_core
    center_idx = N//2
    assert np.isclose(rho[center_idx], rho_core, atol=1e-6)
    # Far region values near edges should be ~rho_env
    assert np.isclose(rho[0], rho_env, atol=1e-6)
