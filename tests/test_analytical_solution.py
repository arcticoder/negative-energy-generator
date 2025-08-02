import sys
import pathlib
# Ensure 'src' directory is on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from simulation.lattice_qft import solve_klein_gordon  # type: ignore

def test_analytical_solution_massless():
    # Analytical solution for massless case: phi(x,t)=sin(kx)*cos(omega t) with omega=k
    N = 20
    dx = 1.0
    L = N * dx
    # wave number k = 2*pi / L
    k = 2 * np.pi / L
    dt = 0.1
    # half period T/2 = pi / omega = pi / k = L/2
    half_period = np.pi / k
    steps = int(half_period / dt)

    # initial condition phi_init(x) = sin(kx), phi_dt_init = 0
    x = np.arange(N) * dx
    phi_init = np.sin(k * x)
    phi_dt_init = np.zeros(N)

    # evolve for half period: should invert sign of phi
    phi_final, _ = solve_klein_gordon(N, dx, dt, steps, alpha=0.0, beta=0.0,
                                      phi_init=phi_init, phi_dt_init=phi_dt_init)

    # Expect phi_final ~ -phi_init within tolerance
    assert np.allclose(phi_final, -phi_init, atol=1e-1), \
        f"phi_final not inverted: max diff {np.max(np.abs(phi_final + phi_init)):.3f}"
