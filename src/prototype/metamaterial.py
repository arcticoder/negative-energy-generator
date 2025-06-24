#!/usr/bin/env python3
"""
Metamaterial Enhancement
=======================

Left-handed metamaterial enhancement for Casimir energy amplification.

Math: ρ_meta(d) = -1/√ε_eff × π²ℏc/(720 d⁴)

Next steps:
1. Design left-handed (μ<0, ε<0) unit cell with resonance at desired gap scale
2. Fabricate via nano-patterning (FIB, e-beam)
3. Characterize ε_eff(ω) with THz-TDS to plug into your models
"""

import numpy as np

class MetamaterialEnhancer:
    """
    Metamaterial Enhancer for Casimir energy amplification.
    
    Math: ρ_meta(d) = -1/√ε_eff × π²ℏc/(720 d⁴)
    """
    
    def __init__(self, epsilon_r: float, mu_r: float, loss_tangent: float = 0.01):
        """
        Initialize metamaterial enhancer.
        
        Args:
            epsilon_r: relative permittivity
            mu_r: relative permeability
            loss_tangent: material loss tangent
        """
        self.epsilon_r = epsilon_r
        self.mu_r = mu_r
        self.loss_tangent = loss_tangent
        self.ħ = 1.054571817e-34  # Planck constant [J⋅s]
        self.c = 2.99792458e8     # Speed of light [m/s]
    
    def get_effective_permittivity(self) -> complex:
        """Calculate effective permittivity including losses."""
        return self.epsilon_r * (1 + 1j * self.loss_tangent)
    
    def get_refractive_index(self) -> complex:
        """Calculate refractive index n = √(εᵣ μᵣ)."""
        eps_eff = self.get_effective_permittivity()
        return np.sqrt(eps_eff * self.mu_r)
    
    def is_left_handed(self) -> bool:
        """Check if material is left-handed (ε < 0 and μ < 0)."""
        return self.epsilon_r < 0 and self.mu_r < 0
    
    def calculate_enhancement(self) -> float:
        """
        Calculate Casimir energy enhancement factor.
        
        Returns:
            Enhancement factor (ratio to vacuum)
        """
        if self.epsilon_r > 0:
            # Right-handed material: enhancement = 1/√εᵣ
            return 1 / np.sqrt(abs(self.epsilon_r))
        else:
            # Left-handed material: can have significant enhancement
            eps_eff = abs(self.get_effective_permittivity())
            return 1 / np.sqrt(eps_eff)
    
    def enhanced_casimir_energy(self, gaps: np.ndarray) -> float:
        """
        Calculate enhanced Casimir energy per unit area.
        
        Args:
            gaps: array of gap sizes [m]
            
        Returns:
            Enhanced Casimir energy per unit area [J/m²]
        """
        enhancement = self.calculate_enhancement()
        
        # Enhanced energy density
        ρ = -enhancement * (np.pi**2 * self.ħ * self.c) / (720 * gaps**4)
        
        # Total energy per unit area
        return float(np.sum(ρ * gaps))
    
    def compare_to_vacuum(self, gaps: np.ndarray) -> dict:
        """Compare enhanced energy to vacuum Casimir energy."""
        # Vacuum Casimir energy
        vacuum_energy = np.sum(-np.pi**2 * self.ħ * self.c / (720 * gaps**4) * gaps)
        
        # Enhanced energy
        enhanced_energy = self.enhanced_casimir_energy(gaps)
        
        # Enhancement ratio
        enhancement_ratio = abs(enhanced_energy) / abs(vacuum_energy)
        
        return {
            'vacuum_energy': vacuum_energy,
            'enhanced_energy': enhanced_energy,
            'enhancement_ratio': enhancement_ratio
        }
    
    def get_material_properties(self) -> dict:
        """Get comprehensive material properties."""
        eps_eff = self.get_effective_permittivity()
        n = self.get_refractive_index()
        enhancement = self.calculate_enhancement()
        
        return {
            'epsilon_r': self.epsilon_r,
            'mu_r': self.mu_r,
            'loss_tangent': self.loss_tangent,
            'epsilon_effective': eps_eff,
            'refractive_index': n,
            'is_left_handed': self.is_left_handed(),
            'enhancement_factor': enhancement
        }

def metamaterial_casimir_energy(gaps: np.ndarray, ε_eff: float) -> float:
    """
    Standalone function for metamaterial-enhanced Casimir energy.
    
    Args:
        gaps: plate separations [m]
        ε_eff: effective permittivity of metamaterial
        
    Returns:
        enhanced Casimir energy per unit area [J/m²]
    """
    ħ = 1.054571817e-34
    c = 2.99792458e8
    ρ = -1/np.sqrt(abs(ε_eff)) * (np.pi**2 * ħ * c) / (720 * gaps**4)
    return float(np.sum(ρ * gaps))

# Legacy compatibility functions
def meta_density(d, eps_eff):
    """Legacy function for metamaterial density."""
    hbar, c = 1.054e-34, 3e8
    return -(np.pi**2 * hbar * c) / (720 * d**4) / np.sqrt(abs(eps_eff))

def optimize_meta(ds, eps_bounds):
    """Legacy optimization function."""
    eps_min, eps_max = eps_bounds
    epss = np.linspace(eps_min, eps_max, 50)
    vals = [np.sum(meta_density(ds, e)*ds) for e in epss]
    i = np.argmin(vals)
    return epss[i], vals[i]  # best eps_eff and energy/m²

def metamaterial_demonstration():
    """Demonstrate metamaterial enhancement for Casimir energy amplification."""
    
    print("🧪 METAMATERIAL ENHANCEMENT DEMONSTRATION")
    print("=" * 42)
    print()
    
    # Example configuration: left-handed metamaterial
    epsilon_r = -2.5    # Negative permittivity
    mu_r = -1.8        # Negative permeability  
    loss_tangent = 0.01  # 1% loss
    
    print(f"Relative permittivity εᵣ: {epsilon_r}")
    print(f"Relative permeability μᵣ: {mu_r}")
    print(f"Loss tangent: {loss_tangent}")
    print()
    
    # Create enhancer
    enhancer = MetamaterialEnhancer(epsilon_r, mu_r, loss_tangent)
    
    # Get material properties
    props = enhancer.get_material_properties()
    print("Material properties:")
    print(f"  Effective permittivity: {props['epsilon_effective']:.3f}")
    print(f"  Refractive index: {props['refractive_index']:.3f}")
    print(f"  Left-handed: {props['is_left_handed']}")
    print(f"  Enhancement factor: {props['enhancement_factor']:.2f}×")
    print()
    
    # Test with gap array
    gaps = np.array([6e-9, 7e-9, 8e-9, 7e-9, 6e-9])  # nm
    print(f"Test gap array: {gaps * 1e9} nm")
    
    # Calculate enhanced energy
    enhanced_energy = enhancer.enhanced_casimir_energy(gaps)
    comparison = enhancer.compare_to_vacuum(gaps)
    
    print()
    print(f"Vacuum Casimir energy: {comparison['vacuum_energy']:.3e} J/m²")
    print(f"Enhanced energy: {comparison['enhanced_energy']:.3e} J/m²")
    print(f"Enhancement ratio: {comparison['enhancement_ratio']:.2f}×")
    print()
    
    # Assessment
    enhancement_ratio = comparison['enhancement_ratio']
    if enhancement_ratio > 5:
        print("✅ Excellent enhancement achieved")
    elif enhancement_ratio > 2:
        print("✅ Good enhancement - substantial improvement")
    elif enhancement_ratio > 1.1:
        print("⚠️ Modest enhancement - consider optimization")
    else:
        print("❌ Poor enhancement - material design needs improvement")
    
    return {
        'enhancer': enhancer,
        'epsilon_r': epsilon_r,
        'mu_r': mu_r,
        'loss_tangent': loss_tangent,
        'enhanced_energy': enhanced_energy,
        'enhancement_ratio': enhancement_ratio,
        'properties': props
    }

if __name__ == "__main__":
    metamaterial_demonstration()
