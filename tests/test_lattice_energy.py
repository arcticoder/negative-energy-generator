import numpy as np
import pytest
from simulation.lattice_qft import solve_klein_gordon, compute_energy_density


def test_compute_energy_density_zero_field():
    """
    Zero field and zero time derivative should produce zero energy density everywhere.
    """
    phi = np.zeros(10)
    phi_dt = np.zeros(10)
    rho = compute_energy_density(phi, phi_dt)
    assert rho.shape == phi.shape
    assert np.allclose(rho, 0)


def test_solve_klein_gordon_basic():
    """
    The placeholder solve_klein_gordon should return zero arrays of correct shape.
    """
    N, dx, dt, steps, alpha, beta = 10, 0.1, 0.01, 1, 0.5, 0.5
    phi, phi_dt = solve_klein_gordon(N, dx, dt, steps, alpha, beta)
    assert phi.shape == (N,)
    assert phi_dt.shape == (N,)
    assert np.allclose(phi, 0)
    assert np.allclose(phi_dt, 0)
