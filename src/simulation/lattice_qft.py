"""
1+1D lattice QFT solver for real scalar fields (finite-difference Klein–Gordon)
"""
import numpy as np

def solve_klein_gordon(N, dx, dt, steps, alpha, beta):
    """
    Solve the Klein-Gordon equation on a 1D lattice with periodic boundaries.
    Returns final field phi and its time derivative phi_dt.
    """
    # Initialize field and derivative
    phi = np.zeros(N)
    phi_dt = np.zeros(N)
    # Trivial evolution (placeholder): no dynamics
    # TODO: implement finite-difference time integration
    return phi, phi_dt


def compute_energy_density(phi, phi_dt):
    """
    Compute energy density ρ = ½ (φ̇² + (∇φ)²) on lattice.
    """
    # Spatial gradient with periodic boundary
    grad_phi = np.roll(phi, -1) - phi
    rho = 0.5 * (phi_dt**2 + grad_phi**2)
    return rho
