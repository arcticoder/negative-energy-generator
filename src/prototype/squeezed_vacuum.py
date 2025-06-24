#!/usr/bin/env python3
"""
Squeezed Vacuum Source
=====================

Parametric vacuum state generation for negative energy density.

Math: ρ_sq = -Σⱼ (ℏωⱼ)/(2Vⱼ) sinh(2rⱼ)

Next steps:
1. Set up an OPO to generate r ≈ 1.5–3
2. Inject into a superconducting high-Q cavity
3. Map out ⟨T₀₀⟩ via homodyne tomography
"""

import numpy as np
from typing import Sequence
from scipy.optimize import minimize_scalar

class SqueezedVacuumSource:
    """
    Squeezed Vacuum Source for negative energy generation.
    
    Math: ρ_sq = -Σⱼ (ℏωⱼ)/(2Vⱼ) sinh(2rⱼ)
    """
    
    def __init__(self, pump_power: float, nonlinearity: float, cavity_finesse: float):
        """
        Initialize squeezed vacuum source.
        
        Args:
            pump_power: pump power [W]
            nonlinearity: χ(2) nonlinearity coefficient
            cavity_finesse: cavity finesse
        """
        self.pump_power = pump_power
        self.nonlinearity = nonlinearity
        self.cavity_finesse = cavity_finesse
        self.ħ = 1.054571817e-34  # Planck constant [J⋅s]
        
        # Default mode parameters
        self.omegas = [2 * np.pi * 1e14]  # 100 THz optical mode
        self.volumes = [1e-15]     # femtoliter mode volume
        self.squeeze_parameters = [self._calculate_squeeze_parameter()]
    
    def _calculate_squeeze_parameter(self) -> float:
        """Calculate squeeze parameter from system parameters."""
        # Simplified model: r ∝ √(P_pump × χ(2) × F)
        # This is a rough approximation
        r = np.sqrt(self.pump_power * self.nonlinearity * self.cavity_finesse / 1e-9)
        return min(r, 3.0)  # Cap at reasonable value
    
    def calculate_squeezing(self) -> float:
        """
        Calculate squeezing factor in dB.
        
        Returns:
            Squeezing factor [dB]
        """
        r = self.squeeze_parameters[0]
        # Squeezing in dB: S_dB = -10 log₁₀(e^(-2r))
        return 20 * r / np.log(10)
    
    def calculate_energy_density(self) -> float:
        """
        Calculate total squeezed vacuum energy density.
        
        Returns:
            Energy density [J/m³]
        """
        total_energy = 0
        for omega, r, V in zip(self.omegas, self.squeeze_parameters, self.volumes):
            energy_density = -self.ħ * omega / (2 * V) * np.sinh(2 * r)
            total_energy += energy_density
        
        return total_energy
    
    def add_mode(self, omega: float, volume: float, squeeze_param: float = None):
        """Add a squeezed mode to the source."""
        self.omegas.append(omega)
        self.volumes.append(volume)
        
        if squeeze_param is None:
            squeeze_param = self._calculate_squeeze_parameter()
        
        self.squeeze_parameters.append(squeeze_param)
    
    def optimize_squeezing(self, omega: float, volume: float, r_max: float = 3.0) -> tuple:
        """
        Optimize squeezing parameter for maximum energy density.
        
        Args:
            omega: mode frequency [rad/s]
            volume: mode volume [m³]
            r_max: maximum squeeze parameter
            
        Returns:
            (optimal_r, optimal_energy_density)
        """
        def objective(r):
            return -(-self.ħ * omega / (2 * volume) * np.sinh(2 * r))
        
        result = minimize_scalar(objective, bounds=(0, r_max), method='bounded')
        return result.x, -result.fun
    
    def get_system_parameters(self) -> dict:
        """Get system parameters and derived quantities."""
        squeezing_db = self.calculate_squeezing()
        noise_reduction = 10**(squeezing_db / 10)
        
        return {
            'pump_power_mW': self.pump_power * 1e3,
            'nonlinearity': self.nonlinearity,
            'finesse': self.cavity_finesse,
            'squeeze_param': self.squeeze_parameters[0],
            'squeezing_dB': squeezing_db,
            'noise_reduction': noise_reduction,
            'num_modes': len(self.omegas)
        }

def squeezed_vacuum_energy(omegas: Sequence[float],
                          r: Sequence[float],
                          volumes: Sequence[float]) -> float:
    """
    Standalone function for squeezed vacuum energy calculation.
    
    Args:
        omegas: mode freqs [rad/s]
        r: squeeze parameters  
        volumes: mode volumes [m³]
        
    Returns:
        total squeezed-vacuum energy density [J/m³]
    """
    ħ = 1.054571817e-34
    ρs = [-ħ*ω/(2*V)*np.sinh(2*ri) for ω,ri,V in zip(omegas, r, volumes)]
    return float(sum(ρs))

# Legacy compatibility functions
def squeezed_density(omega, r, V):
    """Legacy function for squeezed vacuum density."""
    hbar = 1.054e-34
    return - (hbar*omega)/(2*V) * np.sinh(2*r)

def optimize_single_mode(omega, V, r_max=3.0):
    """Legacy optimization function."""
    f = lambda r: -squeezed_density(omega, r, V)  # want most negative
    res = minimize_scalar(f, bounds=(0, r_max), method='bounded')
    return res.x, squeezed_density(omega, res.x, V)

def squeezed_vacuum_demonstration():
    """Demonstrate squeezed vacuum source for negative energy generation."""
    
    print("🌊 SQUEEZED VACUUM SOURCE DEMONSTRATION")
    print("=" * 40)
    print()
    
    # Example configuration
    pump_power = 1e-3      # 1 mW pump
    nonlinearity = 1e-12   # χ(2) coefficient  
    cavity_finesse = 1000  # High-Q cavity
    
    print(f"Pump power: {pump_power * 1e3:.1f} mW")
    print(f"Nonlinearity χ(2): {nonlinearity:.1e}")
    print(f"Cavity finesse: {cavity_finesse}")
    print()
    
    # Create source
    source = SqueezedVacuumSource(pump_power, nonlinearity, cavity_finesse)
    
    # Get system parameters
    params = source.get_system_parameters()
    print("System parameters:")
    print(f"  Squeeze parameter r: {params['squeeze_param']:.2f}")
    print(f"  Squeezing: {params['squeezing_dB']:.1f} dB")
    print(f"  Noise reduction: {params['noise_reduction']:.1f}×")
    print(f"  Number of modes: {params['num_modes']}")
    print()
    
    # Calculate energy density
    energy_density = source.calculate_energy_density()
    print(f"Energy density: {energy_density:.3e} J/m³")
    
    # Convert to J/m² for comparison (assume 1 μm thickness)
    thickness = 1e-6  # 1 μm
    energy_per_area = energy_density * thickness
    print(f"Energy per unit area: {energy_per_area:.3e} J/m²")
    print()
    
    # Optimization example
    omega_opt = 2 * np.pi * 1e14  # 100 THz
    volume_opt = 1e-15            # femtoliter
    optimal_r, optimal_energy = source.optimize_squeezing(omega_opt, volume_opt)
    print(f"Optimization example:")
    print(f"  Optimal r: {optimal_r:.2f}")
    print(f"  Optimal energy density: {optimal_energy:.3e} J/m³")
    print()
    
    # Assessment
    squeezing_db = source.calculate_squeezing()
    if squeezing_db > 10:
        print("✅ Strong squeezing achieved")
    elif squeezing_db > 3:
        print("⚠️ Moderate squeezing - consider enhancement")  
    else:
        print("❌ Weak squeezing - parameter optimization needed")
    
    return {
        'source': source,
        'pump_power': pump_power,
        'nonlinearity': nonlinearity,
        'cavity_finesse': cavity_finesse,
        'energy_density': energy_density,
        'energy_per_area': energy_per_area,
        'squeezing_dB': squeezing_db,
        'parameters': params
    }

if __name__ == "__main__":
    squeezed_vacuum_demonstration()
