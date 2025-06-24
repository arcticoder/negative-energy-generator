#!/usr/bin/env python3
"""
Casimir Array Demonstrator
==========================

Multi-gap Casimir array for negative energy generation.

Math: ρ_C(d_i) = -π²ℏc/(720 d_i⁴) => E_C = Σ_i ρ_C(d_i) d_i ≈ -Σ_i π²ℏc/(720 d_i³)

Next steps:
1. 3D-print or litho-etch a 1 cm² multi-gap array (e.g. 5–10 nm gaps)
2. Mount on piezo stages for ±0.1 nm tuning
3. Probe with an AFM/near-field tip to map the local force/energy density
"""

import numpy as np
from scipy.optimize import minimize

class CasimirArrayDemonstrator:
    """
    Casimir Array Demonstrator for negative energy generation.
    
    Math: ρ_C(d_i) = -π²ℏc/(720 d_i⁴) => E_C = Σ_i ρ_C(d_i) d_i ≈ -Σ_i π²ℏc/(720 d_i³)
    """
    
    def __init__(self, gaps: np.ndarray):
        """
        Initialize Casimir array with specified gaps.
        
        Args:
            gaps: array of plate separations d_i [m]
        """
        self.gaps = np.array(gaps)
        self.ħ = 1.054571817e-34  # Planck constant [J⋅s]
        self.c = 2.99792458e8     # Speed of light [m/s]
    
    def calculate_energy_density(self) -> float:
        """
        Calculate total Casimir energy per unit area [J/m²].
        
        Returns:
            Total Casimir energy per unit area [J/m²]
        """
        # Energy density: ρ = -π²ℏc/(720 d⁴)
        ρ = -np.pi**2 * self.ħ * self.c / (720 * self.gaps**4)
        # E_C per unit area: integrate energy density × gap thickness
        return float(np.sum(ρ * self.gaps))
    
    def get_individual_energies(self) -> np.ndarray:
        """Get energy contribution from each gap."""
        ρ = -np.pi**2 * self.ħ * self.c / (720 * self.gaps**4)
        return ρ * self.gaps
    
    def optimize_gaps(self, target_energy: float = -1e-5) -> np.ndarray:
        """
        Suggest optimized gap configuration for target energy.
        
        Args:
            target_energy: Target total energy density [J/m²]
            
        Returns:
            Optimized gap array
        """
        # Simple optimization: scale gaps to approach target
        current_energy = self.calculate_energy_density()
        if current_energy == 0:
            return self.gaps
        
        scale_factor = (target_energy / current_energy)**(1/3)
        optimized_gaps = self.gaps * scale_factor
        
        # Ensure gaps remain in reasonable range (1-100 nm)
        optimized_gaps = np.clip(optimized_gaps, 1e-9, 100e-9)
        
        return optimized_gaps

def casimir_array_energy(gaps: np.ndarray) -> float:
    """
    Standalone function for Casimir array energy calculation.
    
    Args:
        gaps: array of plate separations d_i [m]
        
    Returns:
        total Casimir energy per unit area [J/m²]
    """
    ħ = 1.054571817e-34
    c = 2.99792458e8
    ρ = -np.pi**2 * ħ * c / (720 * gaps**4)
    # E_C per unit area: integrate energy density × gap thickness
    return float(np.sum(ρ * gaps))

# Legacy compatibility functions
def casimir_density(d):
    """Legacy function for energy density calculation."""
    hbar, c = 1.054e-34, 3e8
    return - (np.pi**2 * hbar * c) / (720 * d**4)

def total_array_energy(ds):
    """Legacy function for total array energy."""
    # energy per unit area (J/m²)
    return np.sum(casimir_density(ds) * ds)

def optimize_casimir(N, d_min, d_max):
    """Legacy optimization function."""
    x0 = np.ones(N) * (d_min + d_max)/2
    bounds = [(d_min, d_max)]*N
    res = minimize(lambda d: -total_array_energy(d), x0, bounds=bounds)
    return res.x, total_array_energy(res.x)

def casimir_array_demonstration():
    """Demonstrate multi-gap Casimir array for negative energy generation."""
    
    print("🔬 CASIMIR ARRAY DEMONSTRATION")
    print("=" * 32)
    print()
    
    # Example configuration
    gaps = np.array([6e-9, 7e-9, 8e-9, 7e-9, 6e-9])  # 5 gaps in nm
    
    print(f"Array configuration: {len(gaps)} gaps")
    print(f"Gap sizes: {gaps * 1e9} nm")
    print()
    
    # Create demonstrator
    demonstrator = CasimirArrayDemonstrator(gaps)
    
    # Calculate energy densities
    energy_densities = demonstrator.get_individual_energies()
    total_energy = demonstrator.calculate_energy_density()
    
    print("Individual gap energies:")
    for i, (gap, energy) in enumerate(zip(gaps, energy_densities)):
        print(f"  Gap {i+1} ({gap*1e9:.1f} nm): {energy:.3e} J/m²")
    
    print()
    print(f"Total energy density: {total_energy:.3e} J/m²")
    print()
    
    # Energy per unit area (1 cm²)
    area = 1e-4  # 1 cm² in m²
    total_energy_per_area = total_energy * area
    print(f"Energy per cm²: {total_energy_per_area:.3e} J")
    
    # Optimization suggestion
    optimized_gaps = demonstrator.optimize_gaps(-1e-5)
    print(f"Suggested optimization: {optimized_gaps * 1e9} nm")
    
    # Assessment
    if abs(total_energy) > 1e-6:
        print("✅ Substantial negative energy density achieved")
    elif abs(total_energy) > 1e-9:
        print("⚠️ Modest negative energy - consider optimization")
    else:
        print("❌ Low energy density - needs enhancement")
    
    return {
        'demonstrator': demonstrator,
        'gaps': gaps,
        'energy_densities': energy_densities,
        'total_energy': total_energy,
        'energy_per_cm2': total_energy_per_area,
        'optimized_gaps': optimized_gaps
    }

if __name__ == "__main__":
    casimir_array_demonstration()
