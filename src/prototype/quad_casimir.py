# src/prototype/quad_casimir.py
"""
Quadrature integration for dynamic Casimir effect calculations.
"""
import numpy as np
from scipy.integrate import quad

hbar, c = 1.054e-34, 3e8

def casimir_energy_instantaneous(d):
    """Instantaneous Casimir energy for gap d."""
    return - (np.pi**2 * hbar * c) / (720 * d**3)

def avg_casimir_gauss(d0, A, ω, n=64):
    """
    Average Casimir energy over oscillation period using Gaussian quadrature.
    
    d(t) = d0 + A*sin(ωt)
    
    Returns time-averaged energy density.
    """
    def integrand(t):
        d_t = d0 + A * np.sin(ω * t)
        # Avoid division by zero
        if d_t <= 0:
            return 0
        return casimir_energy_instantaneous(d_t)
    
    # Integrate over one period
    period = 2 * np.pi / ω
    
    # Use scipy.integrate.quad for robust integration
    try:
        result, error = quad(integrand, 0, period, limit=n)
        return result / period  # Time average
    except:
        # Fallback to simple numerical integration
        t_points = np.linspace(0, period, n)
        energies = [integrand(t) for t in t_points]
        return np.mean(energies)

def avg_casimir_simpson(d0, A, ω, n=128):
    """Alternative implementation using Simpson's rule."""
    period = 2 * np.pi / ω
    t_points = np.linspace(0, period, n)
    
    energies = []
    for t in t_points:
        d_t = d0 + A * np.sin(ω * t)
        if d_t > 0:
            energies.append(casimir_energy_instantaneous(d_t))
        else:
            energies.append(0)  # Avoid unphysical negative gaps
    
    # Simpson's rule integration
    from scipy.integrate import simpson
    return simpson(energies, t_points) / period
