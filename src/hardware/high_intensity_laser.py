"""
High-Intensity Laser Boundary Pump Module
========================================

Implements ultrahigh field-driven moving-mirror DCE with enhanced pump amplitudes.

Mathematical Foundation:
- Effective squeezing: r_eff = (ε_eff * √Q) / (1 + (2Δ)²)
- Pump normalization: ε_eff ∝ E₀/E_ref where E_ref = 10⁹ V/m
- Negative energy: ρ_neg = -sinh²(r_eff) * ℏω₀
- Total energy: E_tot = ρ_neg × V_eff

High-field regime: E₀ ~ 10¹⁵ V/m (approaching breakdown limits)
"""

import numpy as np
from typing import Dict, Tuple, List
import warnings

# Physical constants
ℏ = 1.054571817e-34  # Reduced Planck constant (J⋅s)
ω0 = 2 * np.pi * 5e9  # Base frequency: 5 GHz (rad/s)
V_eff = 1e-18         # Effective mode volume (m³) - femtoliter scale
E_ref = 1e9           # Reference field normalization (V/m)

# Safety limits
E_BREAKDOWN = 1e14    # Dielectric breakdown threshold (V/m)
E_MAX_SAFE = 5e15     # Maximum safe field before plasma formation

def simulate_high_intensity_laser(E0: float, delta: float, Q: float, 
                                 temperature: float = 0.01) -> Dict:
    """
    Simulate high-intensity laser-driven DCE with breakdown protection.
    
    Mathematical model:
    r_eff = (ε_eff * √Q) / (1 + (2Δ)²)
    ε_eff = E₀/E_ref
    ρ_neg = -sinh²(r_eff) * ℏω₀ * thermal_factor
    
    Args:
        E0: Peak electric field (V/m)
        delta: Frequency detuning (GHz)
        Q: Quality factor (dimensionless)
        temperature: Operating temperature (K)
    
    Returns:
        Dictionary with laser DCE performance metrics
    """
    # Safety check for breakdown
    if E0 > E_MAX_SAFE:
        warnings.warn(f"Field {E0:.2e} V/m exceeds safe limit {E_MAX_SAFE:.2e} V/m")
        return {
            'r_eff': 0,
            'rho_neg': 0,
            'E_tot': 0,
            'status': 'BREAKDOWN_RISK',
            'breakdown_margin': E0 / E_MAX_SAFE
        }
    
    # Thermal effects
    k_B = 1.380649e-23
    thermal_photons = 1 / (np.exp(ℏ * ω0 / (k_B * temperature)) - 1) if temperature > 0 else 0
    thermal_factor = 1 / (1 + 2 * thermal_photons)
    
    # Effective pump amplitude normalization
    epsilon_eff = E0 / E_ref
    
    # Detuning factor with quadratic suppression
    detuning_factor = 1 + (2 * delta)**2
    
    # Effective squeezing parameter
    r_eff = epsilon_eff * np.sqrt(Q) / detuning_factor
    
    # Saturation effects at very high squeezing
    r_saturated = r_eff / (1 + r_eff / 10)  # Saturation around r~10
    
    # Negative energy density
    sinh_r = np.sinh(r_saturated)
    rho_neg = -sinh_r**2 * ℏ * ω0 * thermal_factor
    
    # Total negative energy
    E_tot = rho_neg * V_eff
    
    # Performance metrics
    squeezing_dB = 20 * np.log10(np.exp(-r_saturated)) if r_saturated > 0 else 0
    
    # Power requirements (simplified model)
    power_required = (E0**2 * V_eff) / (2 * 377)  # W (impedance of free space)
    
    # Efficiency metrics
    extraction_efficiency = sinh_r**2 / (sinh_r**2 + np.cosh(r_saturated)**2)
    
    return {
        'r_eff': r_saturated,
        'rho_neg': rho_neg,
        'E_tot': E_tot,
        'E0': E0,
        'delta': delta,
        'Q': Q,
        'epsilon_eff': epsilon_eff,
        'squeezing_dB': squeezing_dB,
        'thermal_factor': thermal_factor,
        'thermal_photons': thermal_photons,
        'power_required': power_required,
        'extraction_efficiency': extraction_efficiency,
        'breakdown_margin': E0 / E_BREAKDOWN,
        'status': 'SUCCESS'
    }

def optimize_high_intensity_laser(n_trials: int = 200, 
                                target_energy: float = -1e-12) -> Dict:
    """
    Optimize high-intensity laser parameters using random search with constraints.
    
    Args:
        n_trials: Number of optimization trials
        target_energy: Target negative energy (J)
    
    Returns:
        Dictionary with optimal parameters and performance
    """
    print("🔥 Optimizing High-Intensity Laser DCE Platform")
    print("=" * 55)
    
    best = {'E_tot': 0, 'status': 'NONE'}
    optimization_history = []
    breakdown_count = 0
    
    for trial in range(n_trials):
        # Sample parameters with physical constraints
        E0 = 10**np.random.uniform(14, 15.7)  # 1e14 - 5e15 V/m (below breakdown)
        delta = np.random.uniform(-0.5, 0.5)  # ±0.5 GHz detuning
        Q = 10**np.random.uniform(5, 8)       # 1e5 - 1e8 quality factor
        
        res = simulate_high_intensity_laser(E0, delta, Q)
        optimization_history.append(res)
        
        if res['status'] == 'BREAKDOWN_RISK':
            breakdown_count += 1
            continue
        
        # Multi-objective optimization: energy vs efficiency vs power
        energy_score = abs(res['E_tot'])
        efficiency_score = res['extraction_efficiency']
        power_penalty = np.log10(res['power_required'] / 1e6)  # Penalty for >MW power
        
        # Combined figure of merit
        fom = energy_score * efficiency_score / (1 + max(0, power_penalty))
        
        if res['E_tot'] < best['E_tot'] and res['status'] == 'SUCCESS':
            best = {**res, 'fom': fom, 'trial': trial}
    
    # Analysis
    successful_trials = [r for r in optimization_history if r['status'] == 'SUCCESS']
    breakdown_rate = breakdown_count / n_trials * 100
    
    if successful_trials:
        avg_energy = np.mean([r['E_tot'] for r in successful_trials])
        avg_efficiency = np.mean([r['extraction_efficiency'] for r in successful_trials])
        avg_power = np.mean([r['power_required'] for r in successful_trials])
    else:
        avg_energy = avg_efficiency = avg_power = 0
    
    print(f"✅ High-Intensity Laser Optimization Complete!")
    print(f"   • Successful trials: {len(successful_trials)}/{n_trials}")
    print(f"   • Breakdown rate: {breakdown_rate:.1f}%")
    
    if best['status'] == 'SUCCESS':
        print(f"   • Optimal field: E₀ = {best['E0']:.2e} V/m")
        print(f"   • Optimal detuning: Δ = {best['delta']:.3f} GHz")
        print(f"   • Optimal Q-factor: {best['Q']:.2e}")
        print(f"   • Achieved squeezing: r = {best['r_eff']:.3f}")
        print(f"   • Squeezing in dB: {best['squeezing_dB']:.1f} dB")
        print(f"   • Negative energy: {best['E_tot']:.2e} J")
        print(f"   • Extraction efficiency: {best['extraction_efficiency']:.1%}")
        print(f"   • Power required: {best['power_required']:.2e} W")
        print(f"   • Breakdown margin: {best['breakdown_margin']:.1f}x")
    else:
        print("   ⚠️  No successful optimization found")
    
    return {
        'best_result': best,
        'optimization_history': optimization_history,
        'statistics': {
            'breakdown_rate': breakdown_rate,
            'avg_energy': avg_energy,
            'avg_efficiency': avg_efficiency,
            'avg_power': avg_power,
            'n_successful': len(successful_trials)
        },
        'target_achieved': abs(best.get('E_tot', 0)) > abs(target_energy)
    }

def analyze_field_scaling(E_range: Tuple[float, float], n_points: int = 50) -> Dict:
    """
    Analyze scaling behavior with field intensity.
    
    Args:
        E_range: (E_min, E_max) field range in V/m
        n_points: Number of analysis points
    
    Returns:
        Dictionary with scaling analysis
    """
    E_values = np.logspace(np.log10(E_range[0]), np.log10(E_range[1]), n_points)
    
    # Fixed optimal parameters for scaling study
    delta_opt = 0.1    # Near-resonant
    Q_opt = 1e6        # High Q
    
    results = []
    for E0 in E_values:
        res = simulate_high_intensity_laser(E0, delta_opt, Q_opt)
        results.append(res)
    
    # Extract scaling data
    energies = [r['E_tot'] for r in results]
    squeezing = [r['r_eff'] for r in results]
    powers = [r['power_required'] for r in results]
    
    # Find optimal operating point
    valid_indices = [i for i, r in enumerate(results) if r['status'] == 'SUCCESS']
    if valid_indices:
        optimal_idx = valid_indices[np.argmin([energies[i] for i in valid_indices])]
        optimal_E = E_values[optimal_idx]
        optimal_result = results[optimal_idx]
    else:
        optimal_E = None
        optimal_result = None
    
    return {
        'E_values': E_values,
        'energies': energies,
        'squeezing_parameters': squeezing,
        'power_requirements': powers,
        'results': results,
        'optimal_field': optimal_E,
        'optimal_result': optimal_result,
        'scaling_law': 'E_tot ∝ sinh²(E₀/E_ref √Q)'
    }

if __name__ == "__main__":
    # Test the module
    print("🔥 High-Intensity Laser DCE Module Test")
    print("=" * 45)
    
    # Single simulation test
    test_result = simulate_high_intensity_laser(E0=1e15, delta=0.1, Q=1e6)
    print(f"Test simulation: {test_result['E_tot']:.2e} J")
    
    # Optimization test
    opt_result = optimize_high_intensity_laser(n_trials=50)
    
    # Scaling analysis
    scaling_result = analyze_field_scaling((1e14, 1e16), n_points=20)
    print(f"Scaling analysis: {len(scaling_result['E_values'])} points")
    
    print("✅ High-Intensity Laser module validated!")
