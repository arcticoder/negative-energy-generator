"""
Dark Fluid Models for Negative Energy Generator
"""
import numpy as np

def generate_negative_mass_fluid(N, dx, rho0=-1.0):
    """
    Generate a constant negative-mass fluid density profile on a 1D grid.

    Args:
        N (int): number of grid points
        dx (float): spatial step size
        rho0 (float): negative density amplitude (default: -1.0)

    Returns:
        x (np.ndarray): spatial grid points of length N
        rho (np.ndarray): density array of length N, values = rho0
    """
    # Spatial grid
    x = np.linspace(0, N*dx, N, endpoint=False)
    # Constant negative density
    rho = np.full(N, rho0, dtype=float)
    return x, rho
