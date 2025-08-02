"""
Semiclassical backreaction module for 1+1D models: solves toy metric evolution under exotic stress-energy inputs.
"""
import numpy as np

# Physical constant (set to 1 for natural units)
G_CONST = 1.0


def solve_semiclassical_metric(lattice_x, T00, dt=0.01, steps=100, G=G_CONST):
    """
    Evolve metric perturbation h(t, x) under semiclassical gravity equation in 1+1D:
      ∂^2_t h = 8πG T00(x)
    Uses leapfrog integration with periodic boundary conditions in x.

    Args:
        lattice_x (np.ndarray): spatial grid points, shape (N,)
        T00 (np.ndarray): energy density on grid, shape (N,)
        dt (float): time step size
        steps (int): number of time steps to simulate
        G (float): gravitational constant

    Returns:
        h (np.ndarray): final metric perturbation array, shape (N,)
        h_history (np.ndarray): history of h at each time step, shape (steps+1, N)
    """
    N = lattice_x.size
    # Initialize metric perturbations
    h_prev = np.zeros(N)
    h = np.zeros(N)

    # Storage for history
    history = np.zeros((steps + 1, N))
    history[0] = h.copy()

    # Precompute source term
    source = 8 * np.pi * G * T00
    for n in range(1, steps + 1):
        # No spatial derivative in this toy model
        # Leapfrog integration: h_next = 2*h - h_prev + dt^2 * source
        h_next = 2 * h - h_prev + dt**2 * source

        # Update prev and current
        h_prev, h = h, h_next
        history[n] = h.copy()

    return h, history
