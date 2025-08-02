import sys
import pathlib
# Ensure 'src' on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from simulation.lattice_qft import solve_klein_gordon, compute_energy_density  # type: ignore


def test_energy_conservation():
    # Test energy conservation for massless, undamped evolution
    N = 100
    dx = 0.1
    dt = 0.005
    steps = 200
    alpha = 0.0  # no mass term
    beta = 0.0   # no damping

    # Initial sine wave
    x = np.arange(N) * dx
    phi_init = np.sin(2 * np.pi * x / (N * dx))
    phi_dt_init = np.zeros(N)

    # Evolve field
    phi_final, phi_dt_final = solve_klein_gordon(
        N, dx, dt, steps, alpha, beta,
        phi_init=phi_init, phi_dt_init=phi_dt_init
    )

    # Compute energy
    rho_init = compute_energy_density(phi_init, phi_dt_init, dx)
    rho_final = compute_energy_density(phi_final, phi_dt_final, dx)

    E_init = rho_init.sum()
    E_final = rho_final.sum()

    # Energy should be conserved within tolerance
    assert pytest.approx(E_init, rel=1e-2, abs=1e-6) == E_final
