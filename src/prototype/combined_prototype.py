#!/usr/bin/env python3
"""
Combined Prototype Integration
=============================

Unified vacuum generator integrating all four negative energy sources.

Components:
1. Casimir Array Demonstrator
2. Dynamic Casimir Cavity
3. Squeezed Vacuum Source  
4. Metamaterial Enhancement

Usage: Combined system for maximum negative energy generation.
"""

import numpy as np
from casimir_array import casimir_array_energy, CasimirArrayDemonstrator
from dynamic_casimir import dynamic_casimir_energy, DynamicCasimirCavity
from squeezed_vacuum import squeezed_vacuum_energy, SqueezedVacuumSource
from metamaterial import metamaterial_casimir_energy, MetamaterialEnhancer

class UnifiedVacuumGenerator:
    """
    Unified Vacuum Generator integrating all four negative energy sources.
    
    Combines:
    - Casimir arrays
    - Dynamic Casimir cavities
    - Squeezed vacuum states
    - Metamaterial enhancement
    """
    
    def __init__(self):
        """Initialize unified vacuum generator."""
        self.casimir_array = None
        self.dynamic_cavity = None
        self.squeezed_source = None
        self.metamaterial = None
        
        # Default test configuration
        self._setup_default_configuration()
    
    def _setup_default_configuration(self):
        """Set up default test configuration."""
        # Casimir array with optimized gaps
        gaps = np.array([6e-9, 7e-9, 8e-9, 7e-9, 6e-9])
        self.casimir_array = CasimirArrayDemonstrator(gaps)
        
        # Dynamic Casimir cavity
        d0 = 1e-6         # 1 μm mean gap
        omega = 1e12      # 1 THz modulation
        amplitude = 0.1 * d0  # 10% modulation
        self.dynamic_cavity = DynamicCasimirCavity(d0, omega, amplitude)
        
        # Squeezed vacuum source  
        pump_power = 1e-3      # 1 mW
        nonlinearity = 1e-12   # χ(2)
        finesse = 1000         # High-Q cavity
        self.squeezed_source = SqueezedVacuumSource(pump_power, nonlinearity, finesse)
        
        # Metamaterial enhancement
        epsilon_r = -2.5   # Negative permittivity
        mu_r = -1.8       # Negative permeability
        loss = 0.01       # 1% loss
        self.metamaterial = MetamaterialEnhancer(epsilon_r, mu_r, loss)
    
    def calculate_total_energy(self) -> float:
        """
        Calculate total combined energy density.
        
        Returns:
            Total energy density [J/m²]
        """
        total_energy = 0
        
        # Casimir array contribution
        if self.casimir_array:
            E_casimir = self.casimir_array.calculate_energy_density()
            total_energy += E_casimir
        
        # Dynamic Casimir contribution
        if self.dynamic_cavity:
            E_dynamic = self.dynamic_cavity.calculate_time_averaged_energy()
            total_energy += E_dynamic
        
        # Squeezed vacuum contribution (convert J/m³ to J/m² assuming 1 μm thickness)
        if self.squeezed_source:
            E_squeezed = self.squeezed_source.calculate_energy_density()
            thickness = 1e-6  # 1 μm thickness assumption
            total_energy += E_squeezed * thickness
        
        # Metamaterial enhancement (multiply Casimir by enhancement factor)
        if self.metamaterial and self.casimir_array:
            enhancement = self.metamaterial.calculate_enhancement()
            E_enhanced = self.casimir_array.calculate_energy_density() * (enhancement - 1)
            total_energy += E_enhanced
        
        return total_energy
    
    def estimate_power_output(self) -> float:
        """
        Estimate total power output.
        
        Returns:
            Power output [W]
        """
        total_power = 0
        
        # Dynamic Casimir photon production
        if self.dynamic_cavity:
            photon_rate = self.dynamic_cavity.calculate_photon_production()
            energy_per_photon = 1.055e-34 * self.dynamic_cavity.omega
            total_power += photon_rate * energy_per_photon
        
        # Squeezed vacuum power (rough estimate)
        if self.squeezed_source:
            total_power += self.squeezed_source.pump_power * 0.01  # 1% conversion efficiency
        
        return total_power
    
    def get_component_contributions(self) -> dict:
        """Get individual component energy contributions."""
        contributions = {}
        
        if self.casimir_array:
            contributions['casimir'] = self.casimir_array.calculate_energy_density()
        
        if self.dynamic_cavity:
            contributions['dynamic'] = self.dynamic_cavity.calculate_time_averaged_energy()
        
        if self.squeezed_source:
            E_squeezed = self.squeezed_source.calculate_energy_density()
            contributions['squeezed'] = E_squeezed * 1e-6  # Convert to J/m²
        
        if self.metamaterial and self.casimir_array:
            enhancement = self.metamaterial.calculate_enhancement()
            base_energy = self.casimir_array.calculate_energy_density()
            contributions['metamaterial'] = base_energy * (enhancement - 1)
        
        return contributions
    
    def optimize_combined_system(self) -> dict:
        """Optimize the combined system for maximum energy density."""
        # This is a simplified optimization - in practice would use
        # sophisticated multi-objective optimization
        
        current_energy = self.calculate_total_energy()
        contributions = self.get_component_contributions()
        
        recommendations = {}
        
        # Identify largest contributor
        largest_component = max(contributions, key=lambda k: abs(contributions[k]))
        
        recommendations['primary_focus'] = largest_component
        recommendations['current_energy'] = current_energy
        recommendations['contributions'] = contributions
        
        # Component-specific recommendations
        if abs(contributions.get('casimir', 0)) > abs(current_energy) * 0.3:
            recommendations['casimir'] = "Optimize gap configuration for smaller gaps"
        
        if abs(contributions.get('dynamic', 0)) > abs(current_energy) * 0.3:
            recommendations['dynamic'] = "Increase modulation frequency and depth"
        
        if abs(contributions.get('squeezed', 0)) > abs(current_energy) * 0.1:
            recommendations['squeezed'] = "Increase squeeze parameter and mode count"
        
        if abs(contributions.get('metamaterial', 0)) > abs(current_energy) * 0.2:
            recommendations['metamaterial'] = "Optimize for stronger negative index"
        
        return recommendations

# Legacy compatibility functions
def optimize_casimir(N, d_min, d_max):
    """Legacy function for Casimir optimization."""
    from casimir_array import optimize_casimir as legacy_optimize
    return legacy_optimize(N, d_min, d_max)

def sweep_dynamic(d0_range, A_range, ω_range):
    """Legacy function for dynamic sweep."""
    from dynamic_casimir import sweep_dynamic as legacy_sweep
    return legacy_sweep(d0_range, A_range, ω_range)

def optimize_single_mode(omega, V, r_max=3.0):
    """Legacy function for squeezed vacuum optimization."""
    from squeezed_vacuum import optimize_single_mode as legacy_optimize
    return legacy_optimize(omega, V, r_max)

def optimize_meta(ds, eps_bounds):
    """Legacy function for metamaterial optimization."""
    from metamaterial import optimize_meta as legacy_optimize
    return legacy_optimize(ds, eps_bounds)

def combined_prototype_demonstration():
    """Demonstrate unified vacuum generator with all components."""
    
    print("🔗 UNIFIED VACUUM GENERATOR DEMONSTRATION")
    print("=" * 42)
    print()
    
    # Create unified generator
    generator = UnifiedVacuumGenerator()
    
    print("Components initialized:")
    print("✅ Casimir Array Demonstrator")
    print("✅ Dynamic Casimir Cavity")
    print("✅ Squeezed Vacuum Source")
    print("✅ Metamaterial Enhancement")
    print()
    
    # Calculate individual contributions
    contributions = generator.get_component_contributions()
    print("Individual contributions:")
    for component, energy in contributions.items():
        print(f"  {component.capitalize()}: {energy:.3e} J/m²")
    print()
    
    # Calculate total energy and power
    total_energy = generator.calculate_total_energy()
    total_power = generator.estimate_power_output()
    
    print(f"Total energy density: {total_energy:.3e} J/m²")
    print(f"Total power output: {total_power:.3e} W")
    print()
    
    # Energy per unit area (1 cm²)
    area = 1e-4  # 1 cm²
    energy_per_cm2 = total_energy * area
    print(f"Energy per cm²: {energy_per_cm2:.3e} J")
    print()
    
    # Optimization recommendations
    optimization = generator.optimize_combined_system()
    print("System optimization:")
    print(f"  Primary contributor: {optimization['primary_focus']}")
    for component, recommendation in optimization.items():
        if component.endswith('_focus') or component in ['current_energy', 'contributions']:
            continue
        print(f"  {component}: {recommendation}")
    print()
    
    # Performance assessment
    if abs(total_energy) > 1e-5:
        print("🚀 EXCELLENT: Strong negative energy generation")
    elif abs(total_energy) > 1e-6:
        print("✅ GOOD: Substantial negative energy achieved")
    elif abs(total_energy) > 1e-7:
        print("⚠️ MODERATE: Measurable but limited generation")
    else:
        print("❌ LOW: Minimal negative energy - needs enhancement")
    
    return {
        'generator': generator,
        'total_energy': total_energy,
        'total_power': total_power,
        'contributions': contributions,
        'energy_per_cm2': energy_per_cm2,
        'optimization': optimization
    }

# Legacy demo for compatibility
N = 5

if __name__ == "__main__":
    # Run new demonstration
    combined_prototype_demonstration()
    
    print("\n" + "="*42)
    print("LEGACY COMPATIBILITY TEST")
    print("="*42)
    
    # Run legacy calculations for compatibility
    try:
        ds_opt, E_array = optimize_casimir(N, 5e-9, 1e-8)
        E_dyn, (d0,A,ω) = sweep_dynamic((5e-9,1e-8),(0,5e-9),(1e9,1e12))
        r_opt, E_sq = optimize_single_mode(2*np.pi*1e14, 1e-12)
        eps_opt, E_meta = optimize_meta(ds_opt, (1e-4,1))
        
        print("Casimir array:", E_array)
        print("Dynamic Casimir:", E_dyn)
        print("Squeezed vac:", E_sq)
        print("Metamaterial:", E_meta)
    except Exception as e:
        print(f"Legacy compatibility test failed: {e}")
