"""
Radiative Corrections & Higher-Loop Analysis

Extends the LQG-ANEC formalism to include 1-loop and 2-loop contributions,
ensuring ANEC deficits remain robust under quantum corrections.
"""

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoopCorrection:
    """Results from loop correction calculation."""
    tree_level: float
    one_loop: float
    two_loop: float
    total: float
    convergent: bool


def compute_fourier_transform(f_vals: np.ndarray, r_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Fourier transform of perturbation δg = f - 1.
    Returns momentum values and FT amplitudes.
    """
    dr = r_vals[1] - r_vals[0]
    delta_g = f_vals - 1.0
    
    # Use FFT for numerical Fourier transform
    k_vals = fftfreq(len(r_vals), dr) * 2 * np.pi
    delta_g_ft = fft(delta_g) * dr
    
    # Take only positive frequencies and make real
    positive_k = k_vals >= 0
    k_positive = k_vals[positive_k]
    delta_g_positive = np.abs(delta_g_ft[positive_k])
    
    return k_positive, delta_g_positive


def omega_dispersion(p: np.ndarray, m: float = 0.0) -> np.ndarray:
    """Relativistic dispersion relation ω_p = √(p² + m²)."""
    return np.sqrt(p**2 + m**2)


def compute_one_loop_correction(mu: float, R: float, tau: float, 
                               m: float = 0.0, max_momentum: float = 100.0) -> float:
    """
    Compute 1-loop stress-energy correction:
    T₀₀⁽¹⁾(μ,R,τ) = (1/32π²) ∫₀^∞ dp p² [ω_p + m²/ω_p] δg̃(p;μ,R,τ)
    """
    # Create sample warp bubble profile
    r_vals = np.linspace(0.1, 5*R, 500)
    f_vals = 1 - ((r_vals - R) / (R/4))**4 * np.exp(-1/(2*tau**2))  # t=0 slice
    
    # Compute Fourier transform
    k_vals, delta_g_ft = compute_fourier_transform(f_vals, r_vals)
    
    # Interpolate for integration
    from scipy.interpolate import interp1d
    if len(k_vals) > 1:
        delta_g_interp = interp1d(k_vals, delta_g_ft, 
                                 bounds_error=False, fill_value=0.0)
    else:
        # Fallback for single point
        delta_g_interp = lambda p: np.exp(-(p*R)**2) * np.exp(-(omega_dispersion(p)*tau)**2)
    
    def integrand(p):
        if p <= 0:
            return 0.0
        omega_p = omega_dispersion(p, m)
        kinetic_term = omega_p + m**2/omega_p if omega_p > 0 else 0.0
        return p**2 * kinetic_term * delta_g_interp(p)
    
    prefactor = 1.0 / (32 * np.pi**2)
    
    try:
        integral, error = quad(integrand, 0, max_momentum, limit=200, epsabs=1e-10)
        return prefactor * integral
    except Exception as e:
        logger.warning(f"1-loop integration failed: {e}")
        return 0.0


def compute_two_loop_correction(mu: float, R: float, tau: float,
                               m: float = 0.0, max_momentum: float = 50.0) -> float:
    """
    Compute approximate 2-loop correction using "sunset" diagram contribution.
    This is a simplified implementation of the nested momentum integrals.
    """
    # Create delta_g interpolator as in 1-loop
    r_vals = np.linspace(0.1, 5*R, 200)  # Smaller grid for 2-loop
    f_vals = 1 - ((r_vals - R) / (R/4))**4 * np.exp(-1/(2*tau**2))
    k_vals, delta_g_ft = compute_fourier_transform(f_vals, r_vals)
    
    from scipy.interpolate import interp1d
    if len(k_vals) > 1:
        delta_g_interp = interp1d(k_vals, delta_g_ft, 
                                 bounds_error=False, fill_value=0.0)
    else:
        delta_g_interp = lambda p: np.exp(-(p*R)**2/2) * np.exp(-(omega_dispersion(p)*tau)**2/2)
    
    def integrand_2loop(p1, p2):
        if p1 <= 0 or p2 <= 0:
            return 0.0
        
        omega1 = omega_dispersion(p1, m)
        omega2 = omega_dispersion(p2, m)
        p_total = np.sqrt(p1**2 + p2**2 + 2*p1*p2*np.cos(np.pi/3))  # Approximate angle
        
        factor1 = p1**2 * (omega1 + m**2/omega1) * delta_g_interp(p1)
        factor2 = p2**2 * (omega2 + m**2/omega2) * delta_g_interp(p2)
        coupling = delta_g_interp(p_total) / (omega1 * omega2)
        
        return factor1 * factor2 * coupling
    
    prefactor = 1.0 / (1024 * np.pi**4)  # 2-loop factor
    
    try:
        # Use smaller integration range for 2-loop
        max_p = min(max_momentum, 20.0)
        integral, error = dblquad(integrand_2loop, 0, max_p, 0, max_p, 
                                 epsabs=1e-8, epsrel=1e-6)
        return prefactor * integral
    except Exception as e:
        logger.warning(f"2-loop integration failed: {e}")
        return 0.0


def compute_tree_level(mu: float, R: float, tau: float) -> float:
    """
    Compute tree-level (classical) stress-energy contribution.
    This provides the baseline for comparison with loop corrections.
    """
    # Simplified tree-level calculation
    # In full implementation, this would use the exact 6-term polynomial
    
    # Characteristic energy scale
    energy_scale = 1.0 / (R**2 * tau**2)
    
    # Include polymer corrections via μ dependence
    polymer_factor = np.sin(np.pi * mu) / (np.pi * mu) if mu > 0 else 1.0
    
    return -energy_scale * polymer_factor**2  # Negative for exotic matter


def analyze_loop_convergence(mu: float, R: float, tau: float, 
                           hbar: float = 1.0) -> LoopCorrection:
    """
    Analyze convergence of the loop expansion:
    ⟨T₀₀⟩ = T₀₀⁽⁰⁾ + ℏT₀₀⁽¹⁾ + ℏ²T₀₀⁽²⁾ + ...
    """
    logger.info(f"Computing loop expansion for μ={mu:.4f}, R={R:.3f}, τ={tau:.3f}")
    
    # Compute each order
    tree_level = compute_tree_level(mu, R, tau)
    one_loop = compute_one_loop_correction(mu, R, tau)
    two_loop = compute_two_loop_correction(mu, R, tau)
    
    # Include ℏ factors
    one_loop_contrib = hbar * one_loop
    two_loop_contrib = hbar**2 * two_loop
    
    total = tree_level + one_loop_contrib + two_loop_contrib
    
    # Check convergence
    convergent = True
    if abs(tree_level) > 0:
        one_loop_ratio = abs(one_loop_contrib / tree_level)
        two_loop_ratio = abs(two_loop_contrib / tree_level)
        
        # Series is convergent if each term is smaller than the previous
        convergent = (one_loop_ratio < 1.0 and two_loop_ratio < one_loop_ratio)
        
        logger.info(f"Tree level: {tree_level:.4e}")
        logger.info(f"1-loop ratio: {one_loop_ratio:.4e}")
        logger.info(f"2-loop ratio: {two_loop_ratio:.4e}")
        logger.info(f"Convergent: {convergent}")
    
    return LoopCorrection(
        tree_level=tree_level,
        one_loop=one_loop_contrib,
        two_loop=two_loop_contrib,
        total=total,
        convergent=convergent
    )


def loop_correction_parameter_sweep(mu_vals: np.ndarray, R_vals: np.ndarray, 
                                   tau_vals: np.ndarray) -> Dict:
    """
    Sweep over parameter space and analyze loop correction stability.
    """
    results = {
        'mu': [], 'R': [], 'tau': [],
        'tree_level': [], 'one_loop': [], 'two_loop': [], 'total': [],
        'convergent': [], 'correction_ratio': []
    }
    
    total_points = len(mu_vals) * len(R_vals) * len(tau_vals)
    count = 0
    
    for mu in mu_vals:
        for R in R_vals:
            for tau in tau_vals:
                count += 1
                if count % 10 == 0:
                    logger.info(f"Loop analysis progress: {count}/{total_points}")
                
                try:
                    correction = analyze_loop_convergence(mu, R, tau)
                    
                    # Store results
                    results['mu'].append(mu)
                    results['R'].append(R) 
                    results['tau'].append(tau)
                    results['tree_level'].append(correction.tree_level)
                    results['one_loop'].append(correction.one_loop)
                    results['two_loop'].append(correction.two_loop)
                    results['total'].append(correction.total)
                    results['convergent'].append(correction.convergent)
                    
                    # Correction ratio
                    if abs(correction.tree_level) > 0:
                        ratio = abs((correction.one_loop + correction.two_loop) / correction.tree_level)
                    else:
                        ratio = float('inf')
                    results['correction_ratio'].append(ratio)
                    
                except Exception as e:
                    logger.warning(f"Loop analysis failed for μ={mu}, R={R}, τ={tau}: {e}")
    
    return results


def validate_anec_robustness(tree_level_anec: float, loop_corrections: LoopCorrection,
                           tolerance: float = 0.1) -> Dict:
    """
    Check if ANEC violations remain robust under quantum corrections.
    """
    corrected_anec = loop_corrections.total
    
    # Check if sign is preserved (violation maintained)
    sign_preserved = (tree_level_anec < 0 and corrected_anec < 0)
    
    # Check if magnitude change is within tolerance
    if abs(tree_level_anec) > 0:
        relative_change = abs(corrected_anec - tree_level_anec) / abs(tree_level_anec)
        magnitude_stable = relative_change < tolerance
    else:
        magnitude_stable = False
    
    robust = sign_preserved and magnitude_stable and loop_corrections.convergent
    
    return {
        'robust': robust,
        'sign_preserved': sign_preserved,
        'magnitude_stable': magnitude_stable,
        'convergent': loop_corrections.convergent,
        'relative_change': relative_change if abs(tree_level_anec) > 0 else float('inf'),
        'corrected_anec': corrected_anec
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test single point
    mu, R, tau = 0.095, 2.3, 1.2
    print(f"Analyzing loop corrections for optimal parameters:")
    print(f"μ={mu}, R={R}, τ={tau}")
    
    correction = analyze_loop_convergence(mu, R, tau)
    print(f"Tree level: {correction.tree_level:.4e}")
    print(f"1-loop: {correction.one_loop:.4e}")
    print(f"2-loop: {correction.two_loop:.4e}")
    print(f"Total: {correction.total:.4e}")
    print(f"Convergent: {correction.convergent}")
    
    # Test ANEC robustness
    tree_anec = -3.58e5  # From barrier assessment
    robustness = validate_anec_robustness(tree_anec, correction)
    print(f"\\nANEC robustness analysis:")
    print(f"Robust: {robustness['robust']}")
    print(f"Relative change: {robustness['relative_change']:.2%}")
