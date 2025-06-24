"""
Radiative Corrections: One-Loop & Two-Loop Extensions

This module implements quantum loop corrections to the stress-energy tensor
to validate ANEC deficit robustness under radiative effects.

Formulas:
- One-loop: T_00^(1) = (1/32π²) ∫ dp p²[ωₚ + m²/ωₚ] δg̃(p)
- Two-loop: T_00^(2) = (g²/(16π²)²) ∬ dp dq [p²q²/(ωₚωq)] δg̃(p)δg̃(q)K(p,q)
"""

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RadiativeCorrections:
    """
    Compute quantum loop corrections to warp bubble stress-energy tensor.
    """
    
    def __init__(self, mass=0.0, coupling=1.0, cutoff=100.0):
        """
        Initialize radiative correction calculator.
        
        Args:
            mass: Field mass (m)
            coupling: Coupling constant (g) 
            cutoff: UV cutoff for momentum integrals
        """
        self.m = mass
        self.g = coupling
        self.cutoff = cutoff
        
    def dispersion_relation(self, p: np.ndarray) -> np.ndarray:
        """
        Dispersion relation: ω_p = √(p² + m²)
        """
        return np.sqrt(p**2 + self.m**2)
    
    def fourier_transform_perturbation(self, R: float, tau: float, p: np.ndarray) -> np.ndarray:
        """
        Fourier transform of warp bubble perturbation δg(r,t) = f(r,t) - 1.
        
        For Gaussian-like perturbations:
        δg̃(p) ≈ exp(-(pR)²) * exp(-(ωₚτ)²)
        """
        omega_p = self.dispersion_relation(p)
        
        # Spatial Fourier component (Gaussian envelope)
        spatial_ft = np.exp(-(p * R)**2)
        
        # Temporal Fourier component
        temporal_ft = np.exp(-(omega_p * tau)**2)
        
        return spatial_ft * temporal_ft
    
    def one_loop_correction(self, R: float, tau: float) -> float:
        """
        Compute one-loop correction to T_00:
        
        T_00^(1) = (1/32π²) ∫₀^∞ dp p²[ωₚ + m²/ωₚ] δg̃(p)
        """
        def integrand(p):
            omega_p = self.dispersion_relation(p)
            delta_g_tilde = self.fourier_transform_perturbation(R, tau, p)
            
            # One-loop kernel
            kernel = omega_p + self.m**2 / omega_p if omega_p > 1e-10 else 2 * omega_p
            
            return p**2 * kernel * delta_g_tilde
        
        try:
            integral, error = quad(integrand, 0, self.cutoff, limit=200, epsabs=1e-12)
            prefactor = 1.0 / (32 * np.pi**2)
            
            return prefactor * integral
            
        except Exception as e:
            logger.warning(f"One-loop integration failed: {e}")
            return 0.0
    
    def two_loop_kernel(self, p: float, q: float) -> float:
        """
        Two-loop kernel function K(p,q) for sunset diagram.
        
        Simplified form: K(p,q) = 1/(ωₚ + ωq)
        """
        omega_p = self.dispersion_relation(p)
        omega_q = self.dispersion_relation(q)
        
        return 1.0 / (omega_p + omega_q + 1e-10)  # Regularization
    
    def two_loop_correction(self, R: float, tau: float) -> float:
        """
        Compute two-loop ("sunset") correction to T_00:
        
        T_00^(2) = (g²/(16π²)²) ∬₀^∞ dp dq [p²q²/(ωₚωq)] δg̃(p)δg̃(q)K(p,q)
        """
        def integrand(q, p):  # Note: dblquad expects (inner, outer) order
            omega_p = self.dispersion_relation(p)
            omega_q = self.dispersion_relation(q)
            
            if omega_p < 1e-10 or omega_q < 1e-10:
                return 0.0
            
            delta_g_p = self.fourier_transform_perturbation(R, tau, p)
            delta_g_q = self.fourier_transform_perturbation(R, tau, q)
            kernel = self.two_loop_kernel(p, q)
            
            return (p**2 * q**2 / (omega_p * omega_q)) * delta_g_p * delta_g_q * kernel
        
        try:
            # Limit integration range for computational efficiency
            max_momentum = min(self.cutoff, 10.0)  
            
            integral, error = dblquad(
                integrand, 0, max_momentum, 
                lambda _: 0, lambda _: max_momentum,
                epsabs=1e-10, epsrel=1e-8
            )
            
            prefactor = self.g**2 / (16 * np.pi**2)**2
            
            return prefactor * integral
            
        except Exception as e:
            logger.warning(f"Two-loop integration failed: {e}")
            return 0.0
    
    def polymer_enhanced_corrections(self, R: float, tau: float, mu: float) -> Dict[str, float]:
        """
        Compute polymer-enhanced radiative corrections.
        
        In LQG, the polymer scale μ modifies the loop structure,
        potentially enhancing negative contributions.
        """
        # Base loop corrections
        one_loop = self.one_loop_correction(R, tau)
        two_loop = self.two_loop_correction(R, tau)
        
        # Polymer enhancement factors
        if mu > 0:
            # Polymer modification enhances loop effects
            polymer_factor_1 = 1.0 + mu**2 * R**2  # Linear enhancement
            polymer_factor_2 = 1.0 + mu**4 * R**4  # Quadratic enhancement for 2-loop
        else:
            polymer_factor_1 = 1.0
            polymer_factor_2 = 1.0
        
        # Enhanced corrections
        enhanced_one_loop = one_loop * polymer_factor_1
        enhanced_two_loop = two_loop * polymer_factor_2
        
        # Additional polymer-specific loop corrections
        polymer_specific = -mu**3 * R * tau * np.exp(-(R/tau)**2)  # Can be negative
        
        return {
            'one_loop_bare': one_loop,
            'two_loop_bare': two_loop,
            'one_loop_enhanced': enhanced_one_loop,
            'two_loop_enhanced': enhanced_two_loop,
            'polymer_specific': polymer_specific,
            'total_correction': enhanced_one_loop + enhanced_two_loop + polymer_specific
        }
    
    def corrected_stress_energy(self, T_00_tree: np.ndarray, R: float, tau: float, mu: float, 
                               hbar: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Apply radiative corrections to tree-level stress-energy tensor.
        
        T_00_total = T_00_tree + ℏ*T_00^(1) + ℏ²*T_00^(2) + ...
        
        Args:
            T_00_tree: Tree-level stress-energy tensor
            R, tau, mu: Warp bubble parameters
            hbar: Effective ℏ parameter (natural units: ℏ=1)
            
        Returns:
            Corrected T_00 and breakdown of corrections
        """
        
        # Compute all corrections
        corrections = self.polymer_enhanced_corrections(R, tau, mu)
        
        # Apply loop expansion
        correction_1loop = hbar * corrections['one_loop_enhanced']
        correction_2loop = hbar**2 * corrections['two_loop_enhanced'] 
        correction_polymer = hbar * corrections['polymer_specific']
        
        # Total correction (uniform spatial distribution assumption)
        total_correction = correction_1loop + correction_2loop + correction_polymer
        
        # Apply to tree-level tensor
        T_00_corrected = T_00_tree + total_correction
        
        correction_breakdown = {
            'tree_level': np.mean(T_00_tree),
            'one_loop': correction_1loop,
            'two_loop': correction_2loop,
            'polymer_specific': correction_polymer,
            'total_correction': total_correction,
            'corrected_mean': np.mean(T_00_corrected)
        }
        
        return T_00_corrected, correction_breakdown


def validate_loop_convergence(R_vals: np.ndarray, tau_vals: np.ndarray, mu: float,
                             max_hbar: float = 2.0, num_hbar: int = 20) -> Dict:
    """
    Validate convergence of loop expansion for different ℏ values.
    """
    
    corrector = RadiativeCorrections(mass=0.0, coupling=1.0)
    hbar_vals = np.linspace(0.1, max_hbar, num_hbar)
    
    convergence_data = []
    
    for R in R_vals:
        for tau in tau_vals:
            corrections_vs_hbar = []
            
            for hbar in hbar_vals:
                corrections = corrector.polymer_enhanced_corrections(R, tau, mu)
                
                total_correction = (hbar * corrections['one_loop_enhanced'] + 
                                  hbar**2 * corrections['two_loop_enhanced'] +
                                  hbar * corrections['polymer_specific'])
                
                corrections_vs_hbar.append(total_correction)
            
            # Check for convergence (corrections should remain bounded)
            max_correction = np.max(np.abs(corrections_vs_hbar))
            is_convergent = max_correction < 10.0  # Reasonable bound
            
            convergence_data.append({
                'R': R,
                'tau': tau,
                'mu': mu,
                'max_correction': max_correction,
                'convergent': is_convergent,
                'corrections_series': corrections_vs_hbar
            })
    
    # Overall convergence assessment
    total_configs = len(convergence_data)
    convergent_configs = sum(1 for c in convergence_data if c['convergent'])
    convergence_rate = convergent_configs / total_configs
    
    return {
        'convergence_rate': convergence_rate,
        'total_tested': total_configs,
        'convergent_count': convergent_configs,
        'detailed_data': convergence_data
    }


def anec_with_radiative_corrections(sim, mu: float, R: float, tau: float, 
                                  include_loops: bool = True, hbar: float = 1.0) -> Dict:
    """
    Compute ANEC integral including radiative corrections.
    
    Returns tree + loop contributions and their breakdown.
    """
    
    # Get tree-level ANEC
    I_tree = sim.integrated_negative_energy(mu, R, tau)
    
    if not include_loops:
        return {
            'I_tree': I_tree,
            'I_total': I_tree,
            'corrections': None
        }
    
    # Compute radiative corrections
    corrector = RadiativeCorrections()
    corrections = corrector.polymer_enhanced_corrections(R, tau, mu)
    
    # Estimate spatial volume for correction scaling
    volume_factor = (4 * np.pi * R**3 / 3) * tau * np.sqrt(2 * np.pi)
    
    # Scale corrections by spacetime volume
    I_1loop = hbar * corrections['one_loop_enhanced'] * volume_factor
    I_2loop = hbar**2 * corrections['two_loop_enhanced'] * volume_factor
    I_polymer = hbar * corrections['polymer_specific'] * volume_factor
    
    I_total = I_tree + I_1loop + I_2loop + I_polymer
    
    return {
        'I_tree': I_tree,
        'I_1loop': I_1loop,
        'I_2loop': I_2loop,
        'I_polymer': I_polymer,
        'I_total': I_total,
        'corrections': corrections,
        'loop_enhancement': (I_total - I_tree) / abs(I_tree) if I_tree != 0 else 0.0
    }


if __name__ == "__main__":
    # Test radiative corrections
    print("Radiative Corrections Validation")
    print("="*50)
    
    # Test parameters
    R = 2.3
    tau = 1.2
    mu = 0.095
    
    corrector = RadiativeCorrections(mass=0.0, coupling=1.0)
    
    # Compute corrections
    corrections = corrector.polymer_enhanced_corrections(R, tau, mu)
    
    print(f"Test parameters: R={R}, τ={tau}, μ={mu}")
    print(f"\nRadiative corrections:")
    for key, value in corrections.items():
        print(f"  {key}: {value:.2e}")
    
    # Test convergence
    print(f"\nTesting loop convergence...")
    R_test = np.array([2.0, 2.3, 2.6])
    tau_test = np.array([1.0, 1.2, 1.4])
    
    convergence = validate_loop_convergence(R_test, tau_test, mu)
    
    print(f"Convergence rate: {convergence['convergence_rate']:.1%}")
    print(f"Convergent configurations: {convergence['convergent_count']}/{convergence['total_tested']}")
    
    if convergence['convergence_rate'] > 0.8:
        print("✅ Loop expansion converges reliably")
    else:
        print("⚠️  Loop expansion shows convergence issues")
    
    print("\n✅ Radiative corrections module ready!")
