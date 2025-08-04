import numpy as np
import pytest
from simulation.lattice_qft import solve_klein_gordon


def test_dynamic_evolution_discretization_accuracy():
    """
    Validate dynamic evolution discretization against analytical solution for a standing wave.
    Initial condition: phi(x,0)=sin(x), phi_t(x,0)=0, analytic solution phi=sin(x)*cos(t).
    """
    # Setup parameters
    N = 100
    L = 2 * np.pi
    dx = L / N
    dt = 0.01
    steps = 50
    alpha = 0.0  # massless
    beta = 0.0   # no damping

    # Spatial grid
    x = np.linspace(0, L, N, endpoint=False)
    phi_init = np.sin(x)
    phi_dt_init = np.zeros(N)

    # Evolve and record states
    phi_final, phi_dt_final, phi_hist, phi_dt_hist = solve_klein_gordon(
        N, dx, dt, steps, alpha, beta, phi_init, phi_dt_init, record_states=True
    )

    # Compare a few time steps to analytic solution
    times = np.arange(steps + 1) * dt
    for idx in [0, steps//2, steps]:
        analytic = np.sin(x) * np.cos(times[idx])
        numerical = phi_hist[idx]
        err = np.max(np.abs(numerical - analytic))
        assert err < 1e-2, f"Max error {err:.3e} exceeds tolerance at step {idx}"
