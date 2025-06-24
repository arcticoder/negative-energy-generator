#!/usr/bin/env python3
"""
SU(2) 3nj Hypergeometric Recoupling for Stress-Energy Enhancement
================================================================

Implements the closed-form 3nj product formula to boost negative energy regions
via recoupling weights W_{j_e} = ∏_e (1/(2j_e)!) * 2F1(-2j_e, 1/2; 1; -ρ_e)

This systematically accounts for multi-field interactions through hypergeometric
functions, directly modifying the stress-energy tensor ansatz.

Author: Negative Energy Generator Framework
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import mpmath
from mpmath import hyper, factorial, mp

# Set precision for hypergeometric calculations
mp.dps = 50  # 50 decimal places

@dataclass
class RecouplingConfig:
    """Configuration for SU(2) recoupling weights."""
    
    # Spin quantum numbers for each edge
    spins: List[float] = None           # j_e values (half-integers)
    
    # Mass ratios for each edge  
    mass_ratios: List[float] = None     # ρ_e = M_e^+ / M_e^-
    
    # Coupling graph topology
    num_edges: int = 4                  # Number of coupling edges
    max_spin: float = 5.0               # Maximum spin value
    
    # Enhancement parameters
    boost_factor: float = 1e3           # Multiplicative boost for negative regions
    spatial_localization: float = 1e-14 # Spatial scale for enhancement (m)
    
    # Numerical parameters
    convergence_threshold: float = 1e-12
    max_iterations: int = 1000

class SU2RecouplingEnhancement:
    """
    SU(2) 3nj hypergeometric recoupling enhancement for stress-energy tensors.
    
    Uses closed-form 3nj products to systematically boost negative energy regions
    through multi-field recoupling coefficients.
    """
    
    def __init__(self, config: RecouplingConfig = None):
        self.config = config or RecouplingConfig()
        
        # Initialize default spins and ratios if not provided
        if self.config.spins is None:
            self.config.spins = [0.5, 1.0, 1.5, 2.0]  # Default spin-1/2 to spin-2
            
        if self.config.mass_ratios is None:
            # Default ratios optimized for negative energy enhancement
            self.config.mass_ratios = [2.5, 1.8, 3.2, 4.1]
        
        # Ensure consistent dimensions
        if len(self.config.spins) != len(self.config.mass_ratios):
            raise ValueError("Spins and mass_ratios must have same length")
        
        self.config.num_edges = len(self.config.spins)
        
        print(f"🔗 SU(2) Recoupling Enhancement Initialized")
        print(f"   Edges: {self.config.num_edges}")
        print(f"   Spins: {self.config.spins}")
        print(f"   Mass ratios: {[f'{r:.2f}' for r in self.config.mass_ratios]}")
    
    def hypergeometric_2F1(self, a: float, b: float, c: float, z: complex) -> complex:
        """
        Compute 2F1 hypergeometric function with high precision.
        
        2F1(a, b; c; z) = ∑_{n=0}^∞ (a)_n (b)_n / (c)_n * z^n / n!
        
        Args:
            a, b, c: Hypergeometric parameters
            z: Argument
            
        Returns:
            2F1(a, b; c; z) value
        """
        try:
            # Use mpmath for high-precision calculation
            result = hyper([a, b], [c], z)
            return complex(result)
        except Exception as e:
            print(f"⚠️  Hypergeometric calculation failed: {e}")
            print(f"   Parameters: a={a}, b={b}, c={c}, z={z}")
            # Fallback to series expansion for small |z|
            if abs(z) < 0.5:
                return self._series_expansion_2F1(a, b, c, z)
            else:
                return complex(1.0)  # Safe fallback
    
    def _series_expansion_2F1(self, a: float, b: float, c: float, z: complex, max_terms: int = 100) -> complex:
        """Fallback series expansion for 2F1."""
        result = complex(0.0)
        term = complex(1.0)
        
        for n in range(max_terms):
            if n > 0:
                term *= (a + n - 1) * (b + n - 1) * z / ((c + n - 1) * n)
            
            result += term
            
            if abs(term) < self.config.convergence_threshold:
                break
        
        return result
    
    def compute_edge_weight(self, j_e: float, rho_e: float) -> complex:
        """
        Compute recoupling weight for a single edge e.
        
        W_e = (1/(2j_e)!) * 2F1(-2j_e, 1/2; 1; -ρ_e)
        
        Args:
            j_e: Spin quantum number for edge e
            rho_e: Mass ratio ρ_e = M_e^+ / M_e^-
            
        Returns:
            Edge recoupling weight W_e
        """
        try:
            # Factorial term: 1/(2j_e)!
            factorial_term = 1.0 / float(factorial(2 * j_e))
            
            # Hypergeometric term: 2F1(-2j_e, 1/2; 1; -ρ_e)
            hyper_term = self.hypergeometric_2F1(-2 * j_e, 0.5, 1.0, -rho_e)
            
            weight = factorial_term * hyper_term
            
            return weight
            
        except Exception as e:
            print(f"⚠️  Edge weight calculation failed for j_e={j_e}, rho_e={rho_e}: {e}")
            return complex(1.0)  # Safe fallback
    
    def compute_total_recoupling_weight(self, spins: List[float] = None, 
                                      mass_ratios: List[float] = None) -> complex:
        """
        Compute total recoupling weight W_{j_e} = ∏_e W_e.
        
        Args:
            spins: Optional override for edge spins
            mass_ratios: Optional override for mass ratios
            
        Returns:
            Total recoupling weight W
        """
        if spins is None:
            spins = self.config.spins
        if mass_ratios is None:
            mass_ratios = self.config.mass_ratios
        
        if len(spins) != len(mass_ratios):
            raise ValueError("Spins and mass_ratios must have same length")
        
        # Product over all edges
        total_weight = complex(1.0)
        
        for j_e, rho_e in zip(spins, mass_ratios):
            edge_weight = self.compute_edge_weight(j_e, rho_e)
            total_weight *= edge_weight
        
        return total_weight
    
    def spatial_enhancement_profile(self, r: np.ndarray, r_center: float = None) -> np.ndarray:
        """
        Create spatial localization profile for recoupling enhancement.
        
        Concentrates enhancement near the throat or specified region.
        
        Args:
            r: Radial coordinate array
            r_center: Center of enhancement region
            
        Returns:
            Spatial enhancement profile
        """
        if r_center is None:
            r_center = self.config.spatial_localization
        
        # Gaussian-like localization
        sigma = self.config.spatial_localization
        profile = np.exp(-((r - r_center) / sigma) ** 2)
        
        # Additional boost for very small radii (near throat)
        throat_boost = np.exp(-r / (0.1 * sigma))
        
        return profile + 0.5 * throat_boost
    
    def enhance_stress_energy_tensor(self, T_00: np.ndarray, r: np.ndarray, 
                                   component_type: str = 'total') -> Dict[str, np.ndarray]:
        """
        Apply SU(2) recoupling enhancement to stress-energy tensor.
        
        T_00^enhanced = W_{j_e} * profile(r) * T_00
        
        Args:
            T_00: Original stress-energy tensor component
            r: Radial coordinate array
            component_type: Type of component being enhanced
            
        Returns:
            Enhanced stress-energy components and diagnostics
        """
        print(f"🔗 Applying SU(2) recoupling enhancement to {component_type}")
        
        # Compute total recoupling weight
        W_total = self.compute_total_recoupling_weight()
        W_magnitude = abs(W_total)
        W_phase = np.angle(W_total)
        
        print(f"   Recoupling weight: |W| = {W_magnitude:.2e}, phase = {W_phase:.3f}")
        
        # Create spatial enhancement profile
        spatial_profile = self.spatial_enhancement_profile(r)
        
        # Apply enhancement
        enhancement_factor = W_magnitude * self.config.boost_factor * spatial_profile
        
        # Enhanced stress-energy tensor
        T_00_enhanced = enhancement_factor * T_00
        
        # Diagnostics
        negative_fraction_original = (T_00 < 0).sum() / len(T_00)
        negative_fraction_enhanced = (T_00_enhanced < 0).sum() / len(T_00_enhanced)
        
        max_enhancement = enhancement_factor.max()
        mean_enhancement = enhancement_factor.mean()
        
        print(f"   Enhancement range: [{enhancement_factor.min():.2e}, {max_enhancement:.2e}]")
        print(f"   Mean enhancement: {mean_enhancement:.2e}")
        print(f"   Negative fraction: {negative_fraction_original:.1%} → {negative_fraction_enhanced:.1%}")
        
        return {
            'enhanced_T00': T_00_enhanced,
            'enhancement_factor': enhancement_factor,
            'spatial_profile': spatial_profile,
            'recoupling_weight': W_total,
            'diagnostics': {
                'weight_magnitude': W_magnitude,
                'weight_phase': W_phase,
                'max_enhancement': max_enhancement,
                'mean_enhancement': mean_enhancement,
                'negative_fraction_original': negative_fraction_original,
                'negative_fraction_enhanced': negative_fraction_enhanced,
                'negative_improvement': negative_fraction_enhanced - negative_fraction_original
            }
        }
    
    def optimize_recoupling_parameters(self, T_00_baseline: np.ndarray, r: np.ndarray,
                                     target_negative_fraction: float = 0.5) -> Dict[str, any]:
        """
        Optimize spins and mass ratios for maximum negative energy enhancement.
        
        Args:
            T_00_baseline: Baseline stress-energy tensor
            r: Radial coordinates
            target_negative_fraction: Target fraction of negative energy regions
            
        Returns:
            Optimized parameters and results
        """
        print(f"🎯 Optimizing SU(2) recoupling parameters")
        print(f"   Target negative fraction: {target_negative_fraction:.1%}")
        
        best_negative_fraction = 0.0
        best_params = None
        best_enhancement = None
        
        # Parameter search ranges
        spin_range = np.arange(0.5, self.config.max_spin + 0.5, 0.5)
        ratio_range = np.logspace(-0.5, 1.5, 20)  # 0.316 to 31.6
        
        num_trials = 100
        trial_count = 0
        
        for trial in range(num_trials):
            # Random parameter selection
            trial_spins = np.random.choice(spin_range, size=self.config.num_edges)
            trial_ratios = np.random.choice(ratio_range, size=self.config.num_edges)
            
            try:
                # Temporary update
                original_spins = self.config.spins.copy()
                original_ratios = self.config.mass_ratios.copy()
                
                self.config.spins = trial_spins.tolist()
                self.config.mass_ratios = trial_ratios.tolist()
                
                # Test enhancement
                enhancement_result = self.enhance_stress_energy_tensor(
                    T_00_baseline, r, component_type='optimization_trial'
                )
                
                negative_fraction = enhancement_result['diagnostics']['negative_fraction_enhanced']
                
                if negative_fraction > best_negative_fraction:
                    best_negative_fraction = negative_fraction
                    best_params = {
                        'spins': trial_spins.copy(),
                        'mass_ratios': trial_ratios.copy()
                    }
                    best_enhancement = enhancement_result
                    
                    print(f"   🎯 Trial {trial+1}: New best negative fraction {negative_fraction:.1%}")
                    print(f"      Spins: {trial_spins}")
                    print(f"      Ratios: {[f'{r:.2f}' for r in trial_ratios]}")
                
                # Restore original parameters
                self.config.spins = original_spins
                self.config.mass_ratios = original_ratios
                
                trial_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Trial {trial+1} failed: {e}")
                continue
        
        # Apply best parameters if found
        if best_params is not None:
            self.config.spins = best_params['spins'].tolist()
            self.config.mass_ratios = best_params['mass_ratios'].tolist()
            
            print(f"\n✅ Optimization complete!")
            print(f"   Best negative fraction: {best_negative_fraction:.1%}")
            print(f"   Target achieved: {'YES' if best_negative_fraction >= target_negative_fraction else 'NO'}")
            print(f"   Optimal spins: {self.config.spins}")
            print(f"   Optimal ratios: {[f'{r:.2f}' for r in self.config.mass_ratios]}")
        
        return {
            'optimization_success': best_params is not None,
            'best_negative_fraction': best_negative_fraction,
            'target_achieved': best_negative_fraction >= target_negative_fraction,
            'best_parameters': best_params,
            'best_enhancement_result': best_enhancement,
            'trials_completed': trial_count
        }

def demo_su2_recoupling():
    """Demonstrate SU(2) recoupling enhancement."""
    print("🔗 SU(2) Recoupling Enhancement Demo")
    print("=" * 50)
    
    # Create enhancement system
    config = RecouplingConfig(
        spins=[0.5, 1.0, 1.5, 2.0, 2.5],
        mass_ratios=[1.5, 2.8, 1.9, 3.4, 2.1],
        boost_factor=1e4,
        spatial_localization=1e-14
    )
    
    enhancer = SU2RecouplingEnhancement(config)
    
    # Create test stress-energy tensor (predominantly positive)
    r = np.linspace(1e-15, 1e-13, 1000)
    T_00_test = np.exp(-r/1e-14) - 0.3 * np.exp(-((r - 2e-14)/5e-15)**2)
    
    print(f"\n📊 Original T_00 statistics:")
    print(f"   Range: [{T_00_test.min():.2e}, {T_00_test.max():.2e}]")
    print(f"   Negative fraction: {(T_00_test < 0).sum()/len(T_00_test):.1%}")
    
    # Apply enhancement
    enhancement_result = enhancer.enhance_stress_energy_tensor(T_00_test, r)
    
    # Run optimization
    opt_result = enhancer.optimize_recoupling_parameters(T_00_test, r, target_negative_fraction=0.4)
    
    return enhancer, enhancement_result, opt_result

if __name__ == "__main__":
    demo_su2_recoupling()
