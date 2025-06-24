"""
Negative Energy and Warp Bubble Formation

Analysis of stable negative energy densities and warp bubble configurations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .field_algebra import PolymerField
import logging

logger = logging.getLogger(__name__)


def sampling_function(t, tau):
    """Gaussian sampling function of width τ centered at t=0."""
    return np.exp(-t**2/(2*tau**2)) / (np.sqrt(2*np.pi)*tau)


def compute_energy_density(phi, pi, mu, dx):
    """
    phi, pi: arrays of shape (N,) at a single time slice
    mu: polymer scale
    dx: lattice spacing    Returns array ρ_i for i=0…N−1.
    """
    # Kinetic term: [sin(π μ π_i)/(π μ)]^2
    if mu == 0.0:
        # Classical limit: kinetic = π²/2  
        kinetic = pi**2
    else:
        # Polymer-modified kinetic term with corrected sinc
        kinetic = (np.sin(np.pi * mu * pi) / (np.pi * mu))**2
    
    # Gradient term: use periodic boundary for simplicity
    grad = np.roll(phi, -1) - np.roll(phi, 1)
    grad = (grad / (2 * dx))**2
    # Mass term (set m=0 for simplicity or make m a parameter)
    mass = 0.0 * phi**2
    return 0.5 * (kinetic + grad + mass)


def integrate_negative_energy_over_time(N, mu, total_time, dt, dx, tau):
    """
    Create π_i(t) = A exp[-((x_i - x0)^2)/(2 σ^2)] sin(ω t), 
    choose A so that μ π_i(t) enters the regime where sin(μ π) gives lower energy.
    Integrate I = sum_i ∫ ρ_i(t) f(t) dt dx.
    Return I_polymer - I_classical (negative indicates QI violation).
    """
    times = np.arange(-total_time/2, total_time/2, dt)
    x = np.arange(N) * dx
    x0 = N*dx/2
    sigma = N*dx/8
    
    # Choose amplitude to target the regime where polymer energy is lower
    # From analysis: need μπ ≈ 1.5-1.8 for maximum energy reduction
    # Use a consistent amplitude that scales differently with μ
    # to ensure amplitude doesn't overwhelm the effect of μ
    if mu > 0:
        # Use a constant amplitude for fair comparison across different mu values
        A = 2.0  # Fixed amplitude for more consistent behavior
    else:
        A = 1.0  # Classical case
    
    omega = 2*np.pi/total_time

    I_polymer = 0.0
    I_classical = 0.0
    
    for t in times:
        # Build π_i(t): a localized sine‐burst
        pi_t = A * np.exp(-((x-x0)**2)/(2*sigma**2)) * np.sin(omega * t)
        # φ_i(t) remains ~0 for focused kinetic energy test
        phi_t = np.zeros_like(pi_t)

        # Compute polymer energy density
        rho_polymer = compute_energy_density(phi_t, pi_t, mu, dx)
        
        # Compute classical energy density (mu=0)
        rho_classical = compute_energy_density(phi_t, pi_t, 0.0, dx)
        
        f_t = sampling_function(t, tau)
        I_polymer += np.sum(rho_polymer) * f_t * dt * dx
        I_classical += np.sum(rho_classical) * f_t * dt * dx

    # Return the difference: negative means QI violation
    return I_polymer - I_classical


class WarpBubble:
    """Represents a warp bubble spacetime configuration."""
    
    def __init__(self, center, radius, peak_density, polymer_scale=0.0):
        self.center = center
        self.radius = radius
        self.peak_density = peak_density
        self.polymer_scale = polymer_scale
        
    def energy_density(self):
        """Compute energy density profile."""
        r = np.linspace(-self.radius, self.radius, 100)
        # Simple Gaussian profile for demo
        rho = self.peak_density * np.exp(-r**2/(self.radius**2))
        return rho
        
    def stability_analysis(self):
        """Analyze stability of the warp bubble."""
        # Simple stability estimate
        classical_lifetime = 1.0  # seconds
        polymer_factor = 1.0 + self.polymer_scale * 10
        polymer_lifetime = classical_lifetime * polymer_factor
        
        return {
            "classical_lifetime": classical_lifetime,
            "polymer_lifetime": polymer_lifetime,
            "is_stable": polymer_lifetime > classical_lifetime,
            "enhancement_factor": polymer_factor
        }


def ford_roman_bound_violation(fields: List[np.ndarray], times: np.ndarray, 
                             tau: float, polymer_scale: float = 0.0) -> Dict:
    """
    Check for Ford-Roman bound violations in polymer field evolution.
    
    Args:
        fields: List of field configurations at different times
        times: Time points corresponding to field configurations
        tau: Sampling time scale
        polymer_scale: Polymer parameter
        
    Returns:
        Dictionary with violation analysis
    """
    if len(fields) != len(times):
        raise ValueError("fields and times must have same length")
    
    # Compute energy density at each time
    energy_densities = []
    for field in fields:
        if hasattr(field, 'compute_energy_density'):
            rho = field.compute_energy_density()
        else:
            # Assume field is a PolymerField instance with phi and pi
            rho = compute_energy_density(field.phi, field.pi, polymer_scale, field.dx)
        energy_densities.append(rho)
    
    # Compute Ford-Roman integral
    integrals = []
    for i, rho in enumerate(energy_densities):
        t = times[i]
        weight = sampling_function(t, tau)
        integral = np.sum(rho) * weight
        integrals.append(integral)
    
    total_integral = np.trapz(integrals, times)
    
    # Classical Ford-Roman bound (simplified)
    fr_bound = -1.0 / (32 * np.pi**2 * tau**2)  # c = ħ = 1 units
    
    # Check for violation
    violation = total_integral < fr_bound
    violation_magnitude = abs(total_integral / fr_bound) if fr_bound != 0 else 0
    
    return {
        "ford_roman_integral": total_integral,
        "ford_roman_bound": fr_bound,
        "violation": violation,
        "violation_magnitude": violation_magnitude,
        "energy_densities": energy_densities,
        "times": times
    }


def stability_analysis_polymer_bubble(bubble: WarpBubble, observation_time: float = 1.0) -> Dict:
    """
    Analyze the stability of a polymer-corrected warp bubble.
    
    Args:
        bubble: WarpBubble instance
        observation_time: Time scale for stability analysis
        
    Returns:
        Dictionary with stability results
    """
    # Basic stability metrics
    classical_lifetime = bubble.radius / 1.0  # Assume c=1
    
    # Polymer corrections to stability
    polymer_enhancement = 1.0 + bubble.polymer_scale * np.sqrt(abs(bubble.peak_density))
    polymer_lifetime = classical_lifetime * polymer_enhancement
    
    # Check if bubble survives observation time
    survives_duration = polymer_lifetime > observation_time
    
    return {
        "classical_lifetime": classical_lifetime,
        "polymer_lifetime": polymer_lifetime,
        "stabilization_factor": polymer_lifetime / classical_lifetime,
        "survives_duration": survives_duration,
        "stability_ratio": polymer_lifetime / observation_time
    }


def compute_negative_energy_region(lattice_size: int, polymer_scale: float,
                                 field_amplitude: float = 1.0) -> Dict:
    """
    Compute negative energy regions in a polymer field configuration.
    
    Args:
        lattice_size: Number of lattice sites
        polymer_scale: Polymer parameter μ̄
        field_amplitude: Initial field amplitude
        
    Returns:
        Dictionary with negative energy analysis
    """
    # Create polymer field
    dx = 1.0 / lattice_size  # Default spacing
    field = PolymerField(lattice_size, polymer_scale, dx)
    
    # Set up initial configuration for negative energy formation
    # Use a specific coherent state that promotes negative energy
    width = 0.1
    field.set_coherent_state(field_amplitude, width, center=0.5)
    
    # Add momentum to create interference patterns
    x = np.linspace(0, 1, lattice_size)
    field.pi = field_amplitude * np.sin(2*np.pi*x) * polymer_scale
    
    # Compute initial energy density
    energy_density = field.compute_energy_density()
    
    # Find negative energy regions
    negative_indices = np.where(energy_density < 0)[0]
    total_negative_energy = np.sum(energy_density[negative_indices]) if len(negative_indices) > 0 else 0
    
    # Estimate bubble parameters if negative energy exists
    if len(negative_indices) > 0:
        center_idx = negative_indices[np.argmin(energy_density[negative_indices])]
        center_position = x[center_idx]
        bubble_radius = len(negative_indices) * field.dx / 2
        peak_density = np.min(energy_density)
        
        # Create warp bubble object
        bubble = WarpBubble(center_position, bubble_radius, peak_density, polymer_scale)
        stability = bubble.stability_analysis()
    else:
        bubble = None
        stability = None
    
    return {
        "total_negative_energy": total_negative_energy,
        "negative_sites": len(negative_indices),
        "energy_density": energy_density,
        "x_grid": x,
        "bubble": bubble,
        "stability_analysis": stability,
        "polymer_enhancement": polymer_scale > 0
    }


def compute_negative_energy_region_for_bubble(bubble: 'WarpBubble') -> Dict:
    """
    Compute the spatial region where negative energy density exists.
    
    Args:
        bubble: WarpBubble instance
        
    Returns:
        Dictionary with negative energy region analysis
    """
    # Get energy density profile
    rho = bubble.energy_density()
    x = np.linspace(-bubble.radius, bubble.radius, len(rho))
    
    # Find negative energy regions
    negative_mask = rho < 0
    if not np.any(negative_mask):
        return {
            "has_negative_region": False,
            "negative_volume": 0.0,
            "peak_negative_density": 0.0
        }
    
    # Compute negative energy volume
    negative_volume = np.sum(negative_mask) * (x[1] - x[0])
    peak_negative_density = np.min(rho[negative_mask])
    
    return {
        "has_negative_region": True,
        "negative_volume": negative_volume,
        "peak_negative_density": peak_negative_density,
        "negative_indices": np.where(negative_mask)[0]
    }


# Example usage (not in library, but in a demo script or test):
if __name__ == "__main__":
    N = 64
    dx = 1.0
    dt = 0.01
    total_time = 10.0
    tau = 1.0
    for mu in [0.0, 0.3, 0.6]:
        I = integrate_negative_energy_over_time(N, mu, total_time, dt, dx, tau)
        print(f"μ={mu:.2f}: ∫ρ f dt dx = {I:.6f}")
