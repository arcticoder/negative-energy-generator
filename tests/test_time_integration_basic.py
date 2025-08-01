import sys
import pathlib
# Ensure local 'src' directory is on the import path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import numpy as np
from simulation.lattice_qft import solve_klein_gordon  # type: ignore

def test_solve_klein_gordon_shapes_and_values():
    N = 50
    dx = 0.1
    dt = 0.01
    steps = 10
    alpha = 0.0  # massless
    beta = 0.0  # no damping

    phi, phi_dt = solve_klein_gordon(N, dx, dt, steps, alpha, beta)

    # Check output shapes
    assert phi.shape == (N,), "Field array phi has incorrect shape"
    assert phi_dt.shape == (N,), "Field derivative phi_dt has incorrect shape"

    # Values should be finite numbers
    assert np.all(np.isfinite(phi)), "Field values contain non-finite entries"
    assert np.all(np.isfinite(phi_dt)), "Field derivative contains non-finite entries"
