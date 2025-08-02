"""
1+1D lattice QFT solver for real scalar fields (finite-difference Klein–Gordon)
"""
import numpy as np

def solve_klein_gordon(N, dx, dt, steps, alpha, beta, phi_init=None, phi_dt_init=None):
    """
    Solve the Klein-Gordon equation on a 1D lattice with periodic boundaries.
    Optional initial conditions phi_init and phi_dt_init can be provided.
    Returns final field phi and its time derivative phi_dt.
    """
    # Initialize fields
    if phi_init is not None:
        phi = phi_init.copy()
    else:
        phi = np.zeros(N)
    # Initialize previous field for leapfrog
    if phi_dt_init is not None:
        # phi_prev = phi(t) - dt * phi_dt(t)
        phi_prev = phi - dt * phi_dt_init
    else:
        phi_prev = np.zeros(N)

    if steps == 0:
        # Return initial derivative or zeros
        return phi, (phi_dt_init.copy() if phi_dt_init is not None else np.zeros(N))

    # Precompute coefficient
    coeff = dt**2
    for _ in range(1, steps + 1):
        # Discrete Laplacian
        laplacian = (np.roll(phi, -1) - 2*phi + np.roll(phi, 1)) / (dx**2)
        # Mass term
        mass_term = alpha * phi
        # Leapfrog integration
        phi_next = 2*phi - phi_prev + coeff * (laplacian - mass_term)
        # Optional damping (beta) -- simple friction term
        if beta != 0:
            phi_next -= beta * dt * (phi - phi_prev)
        # Step forward
        phi_prev, phi = phi, phi_next

    # Compute time derivative (phi_t ≈ (phi - phi_prev) / dt)
    phi_dt = (phi - phi_prev) / dt
    return phi, phi_dt


def compute_energy_density(phi, phi_dt, dx=1.0):
    """
    Compute energy density ρ = ½ (φ̇² + (∇φ)²) on lattice.
    Accepts spatial step dx to normalize gradient.
    """
    # Spatial gradient with periodic boundary (forward difference)
    grad_phi = (np.roll(phi, -1) - phi) / dx
    rho = 0.5 * (phi_dt**2 + grad_phi**2)
    return rho
