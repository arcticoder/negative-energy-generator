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

def generate_phantom_dark_fluid(N, dx, rho0=-1.0, w=-1.5):
    """
    Generate a dark fluid with phantom equation-of-state parameter w (p = w * rho).

    Args:
        N (int): number of grid points
        dx (float): spatial step size
        rho0 (float): baseline fluid density (default: -1.0)
        w (float): equation-of-state parameter (w < -1 for phantom fluids)

    Returns:
        x (np.ndarray): spatial grid points of length N
        rho (np.ndarray): density array of length N, values = rho0
        p (np.ndarray): pressure array of length N, p = w * rho
    """
    x = np.linspace(0, N*dx, N, endpoint=False)
    rho = np.full(N, rho0, dtype=float)
    p = rho * w
    return x, rho, p
  
def generate_warp_bubble_fluid(N, dx, rho0=-1.0, R=1.0, sigma=0.2):
    """
    Generate a warp bubble dark fluid density profile using a Gaussian shell centered at zero.

    Args:
        N (int): number of grid points
        dx (float): spatial step size
        rho0 (float): amplitude of the negative density in the bubble region
        R (float): radius of the bubble shell
        sigma (float): width of the bubble shell

    Returns:
        x (np.ndarray): spatial grid points centered at zero, length N
        rho (np.ndarray): density profile of length N
    """
    # Spatial grid centered at zero
    x = np.linspace(- (N//2)*dx, (N//2)*dx, N)
    # Gaussian shell around radius R
    shell = np.exp(- (np.abs(x) - R)**2 / (2 * sigma**2))
    rho = rho0 * shell
    return x, rho
  
def generate_vacuum_fluctuation_fluid(N, dx, amplitude=1e-3, corr_len=1.0, seed=None):
    """
    Generate a dark fluid density profile from simulated quantum vacuum fluctuations.

    Args:
        N (int): number of grid points
        dx (float): spatial step size
        amplitude (float): scale of fluctuations
        corr_len (float): correlation length for smoothing
        seed (int, optional): random seed

    Returns:
        x (np.ndarray): spatial grid points length N
        rho (np.ndarray): smoothed fluctuation profile length N
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=amplitude, size=N)
    # Simple box smoothing for correlation
    window_size = max(1, int(corr_len / dx))
    window = np.ones(window_size) / window_size
    rho = np.convolve(noise, window, mode='same')
    x = np.linspace(0, N*dx, N, endpoint=False)
    return x, rho

def generate_phase_transition_fluid(N, dx, rho_core=-1.0, rho_env=-0.1, R_core=1.0, width=0.5):
    """
    Generate a dark fluid density profile with a phase transition between core and environment.

    Args:
        N (int): number of grid points
        dx (float): spatial step size
        rho_core (float): density inside core region
        rho_env (float): density in environment outside the transition
        R_core (float): radius of core region
        width (float): width of transition region

    Returns:
        x (np.ndarray): spatial grid points centered at zero, length N
        rho (np.ndarray): density profile length N
    """
    x = np.linspace(- (N//2)*dx, (N//2)*dx, N)
    # Smooth transition using logistic function based on radius
    r = np.abs(x)
    # logistic transition around R_core
    t = 1 / (1 + np.exp((r - R_core) / width))
    # interpolate between env and core: core inside, env outside
    rho = rho_env + (rho_core - rho_env) * t
    return x, rho
   
def generate_superfluid_dark_fluid(N, dx, rho0=-1.0, k=1.0):
    """
    Generate a superfluid dark matter density profile as a sine wave pattern.

    Args:
        N (int): number of grid points
        dx (float): spatial step size
        rho0 (float): amplitude of density oscillation (default: -1.0)
        k (float): wave number of oscillation

    Returns:
        x (np.ndarray): spatial grid points centered at zero, length N
        rho (np.ndarray): density profile length N, values = rho0 * sin(k*x)
    """
    # Spatial grid centered at zero
    x = np.linspace(- (N//2) * dx, (N//2) * dx, N)
    # Sine-wave pattern
    rho = rho0 * np.sin(k * x)
    return x, rho
