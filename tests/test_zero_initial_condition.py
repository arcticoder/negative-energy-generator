import sys
import pathlib
# Ensure 'src' on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import numpy as np
from simulation.lattice_qft import solve_klein_gordon  # type: ignore

def test_zero_initial_condition():
    N = 20
    dx = 0.1
    dt = 0.01
    steps = 50
    alpha = 1.0
    beta = 0.3

    phi, phi_dt = solve_klein_gordon(N, dx, dt, steps, alpha, beta)
    # With zero initial conditions, solution remains zero for homogeneous equation
    assert np.allclose(phi, 0), "Field should remain zero for zero initial conditions"
    assert np.allclose(phi_dt, 0), "Field derivative should remain zero for zero initial conditions"
