import numpy as np
import pytest

from simulation.lattice_qft import solve_klein_gordon, compute_energy_density

def test_dynamic_energy_conservation():
    """
    Test that the total energy is conserved over time for undamped (beta=0) Klein-Gordon evolution.
    """
    N = 100
    dx = 1.0
    dt = 0.1
    steps = 50
    alpha = 0.0  # massless case
    beta = 0.0   # no damping

    # initial condition: single-mode sine wave
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    phi_init = np.sin(x)
    phi_dt_init = np.zeros(N)

    # evolve and record states
    phi, phi_dt, phi_hist, phi_dt_hist = solve_klein_gordon(
        N, dx, dt, steps, alpha, beta, phi_init, phi_dt_init, record_states=True
    )

    # compute total energy at each time step
    total_energies = []
    for phi_t, phi_dt_t in zip(phi_hist, phi_dt_hist):
        rho = compute_energy_density(phi_t, phi_dt_t, dx)
        total_energies.append(np.sum(rho))

    E0 = total_energies[0]
    energies = np.array(total_energies)

    # relative variation should be within tolerance
    rel_variation = np.abs(energies - E0) / E0
    # Allow small drift due to numerical dispersion
    tol = 2e-3
    assert np.all(rel_variation < tol), f"Energy not conserved: max variation {rel_variation.max():.6f} exceeds tol={tol}"
