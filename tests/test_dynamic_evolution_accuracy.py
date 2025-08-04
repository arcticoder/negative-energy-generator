import numpy as np
import pytest
from simulation.lattice_qft import solve_klein_gordon, compute_energy_density


def test_dynamic_evolution_energy_drift():
    """
    Test that the total energy drift over dynamic evolution remains within acceptable tolerance.
    """
    N = 100
    dx = 1.0
    dt = 0.05  # reduced time step for accuracy
    steps = 100
    alpha = 0.0
    beta = 0.0

    # initial sine wave
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    phi_init = np.sin(x)
    phi_dt_init = np.zeros(N)

    # evolve and record states
    phi, phi_dt, phi_hist, phi_dt_hist = solve_klein_gordon(
        N, dx, dt, steps, alpha, beta, phi_init, phi_dt_init, record_states=True
    )
    # compute energies
    energies = np.array([
        np.sum(compute_energy_density(phi_t, phi_dt_t, dx))
        for phi_t, phi_dt_t in zip(phi_hist, phi_dt_hist)
    ])

    initial = energies[0]
    final = energies[-1]
    drift = abs(final - initial) / initial
    # Expect drift to be below 0.005 (0.5%)
    assert drift < 0.005, f"Energy drift too large: {drift:.5f}"
