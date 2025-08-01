import numpy as np
import pytest
from simulation.lattice_qft import compute_energy_density


def test_laplacian_accuracy_for_sine_wave():
    """
    Test that the discrete Laplacian on a sine wave approximates the analytical second derivative (-sin(x)).
    """
    # Set up a sine wave on a periodic domain [0, 2π)
    N = 100
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    phi = np.sin(x)
    phi_dt = np.zeros(N)

    # Compute discrete second derivative
    dx = L / N
    lap = (np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)) / (dx**2)

    # Analytical second derivative of sin(x) is -sin(x)
    analytic = -phi

    # Check accuracy within tolerance
    assert lap.shape == analytic.shape
    assert np.allclose(lap, analytic, atol=1e-2), "Discrete Laplacian deviates from analytic second derivative beyond tolerance"
