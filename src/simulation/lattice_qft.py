"""
1+1D lattice QFT solver for real scalar fields (finite-difference Klein–Gordon)
"""
import numpy as np

def solve_klein_gordon(N, dx, dt, steps, alpha, beta):
    """
    Solve the Klein-Gordon equation on a 1D lattice with periodic boundaries.
    Returns final field phi and its time derivative phi_dt.
    """
    # Initialize fields
    phi_prev = np.zeros(N)
    phi = np.zeros(N)
    # First time derivative phi_dt will be computed at the end

    if steps == 0:
        return phi, np.zeros(N)

    # Precompute coefficient
    coeff = dt**2
    for n in range(1, steps + 1):
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


def compute_energy_density(phi, phi_dt):
    """
    Compute energy density ρ = ½ (φ̇² + (∇φ)²) on lattice.
    """
    # Spatial gradient with periodic boundary
    grad_phi = np.roll(phi, -1) - phi
    rho = 0.5 * (phi_dt**2 + grad_phi**2)
    return rho
