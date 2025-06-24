#!/usr/bin/env python3
"""
Dynamic Casimir Cavity
======================

Time-varying boundary cavity for photon production and negative energy generation.

Math: d(t) = d₀ + A sin(ωt) => E̅_C = (ω/2π) ∫₀^(2π/ω) -π²ℏc/(720[d(t)]⁴) dt

Next steps:
1. Build a micro-cavity with variable gap actuator (MEMS or optical pump)
2. Drive at GHz–THz (< τ ~ 1 ps) and lock-in detect sideband photon flux
"""

import numpy as np
from scipy.integrate import quad

class DynamicCasimirCavity:
    """
    Dynamic Casimir Cavity with time-varying boundary.
    
    Math: d(t) = d₀ + A sin(ωt) => E̅_C = (ω/2π) ∫₀^(2π/ω) -π²ℏc/(720[d(t)]⁴) dt
    """
    
    def __init__(self, d0: float, omega: float, amplitude: float):
        """
        Initialize dynamic Casimir cavity.
        
        Args:
            d0: mean gap [m]
            omega: angular freq [rad/s]
            amplitude: oscillation amplitude [m]
        """
        self.d0 = d0
        self.omega = omega
        self.amplitude = amplitude
        self.ħ = 1.054571817e-34  # Planck constant [J⋅s]
        self.c = 2.99792458e8     # Speed of light [m/s]
    
    def gap_function(self, t: float) -> float:
        """Gap as function of time: d(t) = d₀ + A sin(ωt)."""
        return self.d0 + self.amplitude * np.sin(self.omega * t)
    
    def instantaneous_energy_density(self, t: float) -> float:
        """Instantaneous Casimir energy density at time t."""
        d_t = self.gap_function(t)
        return -np.pi**2 * self.ħ * self.c / (720 * d_t**4)
    
    def calculate_time_averaged_energy(self) -> float:
        """
        Calculate time-averaged Casimir energy per unit area [J/m²].
        
        Returns:
            Time-averaged energy density [J/m²]
        """
        period = 2 * np.pi / self.omega
        
        def integrand(t):
            d_t = self.gap_function(t)
            return -np.pi**2 * self.ħ * self.c / (720 * d_t**4)
        
        integral, _ = quad(integrand, 0, period)
        return (self.omega / (2 * np.pi)) * integral
    
    def calculate_photon_production(self) -> float:
        """
        Estimate photon production rate from dynamic Casimir effect.
        
        Returns:
            Photon production rate [s⁻¹]
        """
        # Simplified estimate based on modulation depth and frequency
        modulation_depth = self.amplitude / self.d0
        
        # Production rate scales with (A/d₀)² and ω
        # Rough estimate: N_photons ~ (A/d₀)² × (ω/ω_cavity)³
        omega_cavity = np.pi * self.c / (2 * self.d0)  # Fundamental cavity mode
        
        if self.omega < omega_cavity:
            rate = (modulation_depth**2) * (self.omega / omega_cavity)**3 * 1e12
        else:
            rate = (modulation_depth**2) * 1e12
        
        return rate
    
    def get_modulation_parameters(self) -> dict:
        """Get modulation parameters and derived quantities."""
        period = 2 * np.pi / self.omega
        frequency_hz = self.omega / (2 * np.pi)
        modulation_depth = self.amplitude / self.d0
        
        return {
            'period': period,
            'frequency_hz': frequency_hz,
            'modulation_depth': modulation_depth,
            'max_gap': self.d0 + self.amplitude,
            'min_gap': self.d0 - self.amplitude
        }

def dynamic_casimir_energy(d0: float, A: float, ω: float, n_steps: int = 1000) -> float:
    """
    Standalone function for dynamic Casimir energy calculation.
    
    Args:
        d0: mean gap [m]
        A: oscillation amplitude [m] 
        ω: angular freq [rad/s]
        n_steps: number of integration steps
        
    Returns:
        time-averaged Casimir energy per unit area [J/m²]
    """
    ħ = 1.054571817e-34
    c = 2.99792458e8
    
    # Numerical integration
    ts = np.linspace(0, 2*np.pi/ω, n_steps)
    ds = d0 + A * np.sin(ω * ts)
    ρ = -np.pi**2 * ħ * c / (720 * ds**4)
    
    return float((ω/(2*np.pi)) * np.trapz(ρ, ts))

# Legacy compatibility functions
def average_dynamic_energy(d0, A, ω):
    """Legacy function for dynamic energy calculation."""
    return dynamic_casimir_energy(d0, A, ω)

def sweep_dynamic(d0_range, A_range, ω_range):
    """Legacy function for parameter sweep."""
    best = (0, None)
    for d0 in np.linspace(*d0_range, 10):
        for A in np.linspace(*A_range, 10):
            for ω in np.linspace(*ω_range, 10):
                E = average_dynamic_energy(d0, A, ω)
                if E < best[0]:
                    best = (E, (d0, A, ω))
    return best

def dynamic_casimir_demonstration():
    """Demonstrate dynamic Casimir cavity for negative energy generation."""
    
    print("⚡ DYNAMIC CASIMIR CAVITY DEMONSTRATION")
    print("=" * 40)
    print()
    
    # Example configuration
    d0 = 1e-6        # 1 μm mean gap
    omega = 1e12     # 1 THz modulation
    amplitude = 0.1 * d0  # 10% modulation depth
    
    print(f"Mean gap: {d0 * 1e6:.1f} μm")
    print(f"Modulation frequency: {omega / 1e12:.1f} THz")
    print(f"Amplitude: {amplitude * 1e9:.1f} nm ({amplitude/d0*100:.1f}%)")
    print()
    
    # Create cavity
    cavity = DynamicCasimirCavity(d0, omega, amplitude)
    
    # Get modulation parameters
    params = cavity.get_modulation_parameters()
    print("Modulation parameters:")
    print(f"  Period: {params['period'] * 1e12:.2f} ps")
    print(f"  Max gap: {params['max_gap'] * 1e6:.2f} μm")
    print(f"  Min gap: {params['min_gap'] * 1e6:.2f} μm")
    print(f"  Modulation depth: {params['modulation_depth'] * 100:.1f}%")
    print()
    
    # Calculate energies
    time_averaged_energy = cavity.calculate_time_averaged_energy()
    photon_rate = cavity.calculate_photon_production()
    
    print(f"Time-averaged energy density: {time_averaged_energy:.3e} J/m²")
    print(f"Photon production rate: {photon_rate:.3e} s⁻¹")
    
    # Power estimation
    energy_per_photon = 1.055e-34 * omega  # ℏω
    power = photon_rate * energy_per_photon
    print(f"Estimated power output: {power:.3e} W")
    print()
    
    # Assessment
    if photon_rate > 1e9:
        print("✅ High photon production rate achieved")
    elif photon_rate > 1e6:
        print("⚠️ Moderate photon production - consider optimization")
    else:
        print("❌ Low photon production - needs parameter tuning")
    
    return {
        'cavity': cavity,
        'd0': d0,
        'omega': omega,
        'amplitude': amplitude,
        'time_averaged_energy': time_averaged_energy,
        'photon_rate': photon_rate,
        'power': power,
        'parameters': params
    }

if __name__ == "__main__":
    dynamic_casimir_demonstration()
