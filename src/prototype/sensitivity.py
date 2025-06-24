# src/prototype/sensitivity.py
import numpy as np

def dimensionless_energy(tilde_ds, d_min):
    """Compute E factoring out d_min."""
    factor = - (np.pi**2 * 1.054e-34 * 3e8) / (720 * d_min**3)
    return factor * np.sum(tilde_ds**(-3))

def relative_error_bound(tilde_ds, sigma_rel):
    """First-order bound: ΔE/E ≤ 3·σ_rel."""
    return 3 * sigma_rel

def sensitivity_matrix(ds):
    """Compute sensitivity matrix ∂E/∂d_i for all gaps."""
    hbar, c = 1.054e-34, 3e8
    sensitivities = []
    
    for i, d in enumerate(ds):
        # Partial derivative of E with respect to d_i
        dE_dd_i = - (np.pi**2 * hbar * c) / (240 * d**4)
        sensitivities.append(dE_dd_i)
    
    return np.array(sensitivities)

def condition_number_analysis(ds):
    """Analyze condition number for numerical stability."""
    
    # Create Jacobian matrix (simplified for demonstration)
    J = np.diag(sensitivity_matrix(ds))
    
    # Condition number
    cond_num = np.linalg.cond(J)
    
    return cond_num

def tolerance_allocation(ds, total_error_budget):
    """Allocate tolerances optimally across gaps."""
    
    sensitivities = np.abs(sensitivity_matrix(ds))
    
    # Optimal allocation: σ_i ∝ 1/|∂E/∂d_i|
    inv_sensitivities = 1.0 / sensitivities
    norm_weights = inv_sensitivities / np.sum(inv_sensitivities)
    
    # Allocate error budget
    individual_tolerances = norm_weights * total_error_budget
    
    return individual_tolerances

def dimensional_analysis_report(ds, sigma_rel=0.01):
    """Generate comprehensive sensitivity and dimensional analysis."""
    
    print("📐 DIMENSIONAL SENSITIVITY ANALYSIS")
    print("=" * 36)
    print()
    
    # Basic parameters
    d_min = np.min(ds)
    d_max = np.max(ds)
    
    print(f"Gap range: {d_min*1e9:.1f} - {d_max*1e9:.1f} nm")
    print(f"Number of gaps: {len(ds)}")
    print(f"Relative uncertainty: {sigma_rel*100:.1f}%")
    print()
    
    # Dimensionless analysis
    print("🔢 DIMENSIONLESS ANALYSIS")
    print("-" * 25)
    
    tilde_ds = ds / d_min
    print(f"Dimensionless gaps: {tilde_ds}")
    print(f"Gap ratio (max/min): {np.max(tilde_ds):.2f}")
    
    # Energy scaling
    E_factor = - (np.pi**2 * 1.054e-34 * 3e8) / (720 * d_min**3)
    E_dimensionless = np.sum(tilde_ds**(-3))
    
    print(f"Energy factor: {E_factor:.3e} J/m²")
    print(f"Dimensionless sum: {E_dimensionless:.3f}")
    print()
    
    # First-order error bound
    print("📊 ERROR PROPAGATION")
    print("-" * 20)
    
    error_bound = relative_error_bound(tilde_ds, sigma_rel)
    print(f"First-order relative error bound: {error_bound*100:.2f}%")
    
    # Individual sensitivities
    sensitivities = sensitivity_matrix(ds)
    print(f"Max sensitivity: {np.max(np.abs(sensitivities)):.2e} J/m²/m")
    print(f"Min sensitivity: {np.min(np.abs(sensitivities)):.2e} J/m²/m")
    print(f"Sensitivity ratio: {np.max(np.abs(sensitivities))/np.min(np.abs(sensitivities)):.1f}")
    print()
    
    # Condition number
    print("🔍 NUMERICAL STABILITY")
    print("-" * 22)
    
    cond_num = condition_number_analysis(ds)
    print(f"Condition number: {cond_num:.2e}")
    
    if cond_num < 1e3:
        print("✅ Well-conditioned problem")
    elif cond_num < 1e6:
        print("⚠️ Moderately ill-conditioned")
    else:
        print("❌ Severely ill-conditioned")
    print()
    
    # Tolerance allocation
    print("🎯 OPTIMAL TOLERANCE ALLOCATION")
    print("-" * 31)
    
    total_budget = sigma_rel * abs(np.sum(- (np.pi**2 * 1.054e-34 * 3e8)/(720 * ds**3)))
    tolerances = tolerance_allocation(ds, total_budget)
    
    for i, (d, tol) in enumerate(zip(ds, tolerances)):
        rel_tol = tol / abs(sensitivities[i]) / abs(d) * 100
        print(f"Gap {i+1} ({d*1e9:.1f} nm): ±{abs(tol)/abs(sensitivities[i])*1e12:.1f} pm ({rel_tol:.2f}%)")
    
    print()
    
    # Design recommendations
    print("💡 DESIGN RECOMMENDATIONS")
    print("-" * 26)
    
    if np.max(tilde_ds) / np.min(tilde_ds) > 2:
        print("⚠️ Large gap ratio may cause numerical issues")
        print("   Recommendation: Use more uniform gaps")
    
    if error_bound > 0.05:
        print("⚠️ High error sensitivity")
        print(f"   Recommendation: Improve precision to ±{sigma_rel/error_bound*100:.2f}%")
    
    if cond_num > 1e6:
        print("⚠️ Poor numerical conditioning")
        print("   Recommendation: Redesign gap configuration")
    
    return {
        'dimensionless_gaps': tilde_ds,
        'error_bound': error_bound,
        'condition_number': cond_num,
        'sensitivities': sensitivities,
        'tolerances': tolerances
    }

def scaling_laws_analysis(d_range, N_gaps):
    """Analyze scaling laws for different configurations."""
    
    print("📈 SCALING LAWS ANALYSIS")
    print("-" * 24)
    print()
    
    configurations = [
        ('Uniform Small', np.full(N_gaps, d_range[0])),
        ('Uniform Large', np.full(N_gaps, d_range[1])),
        ('Linear Gradient', np.linspace(d_range[0], d_range[1], N_gaps)),
        ('Geometric Progression', np.geomspace(d_range[0], d_range[1], N_gaps))
    ]
    
    for name, gaps in configurations:
        energy = np.sum(- (np.pi**2 * 1.054e-34 * 3e8)/(720 * gaps**3))
        avg_gap = np.mean(gaps)
        error_bound = relative_error_bound(gaps/np.min(gaps), 0.01)
        
        print(f"{name}:")
        print(f"  Energy: {energy:.3e} J/m²")
        print(f"  Avg gap: {avg_gap*1e9:.1f} nm")
        print(f"  Error bound: {error_bound*100:.1f}%")
        print()

# Example usage
if __name__=='__main__':
    # Test case: 5 gaps with slight variation
    ds = np.array([6e-9, 7e-9, 8e-9, 7e-9, 6e-9])  # nm
    
    results = dimensional_analysis_report(ds, sigma_rel=0.02)
    
    print()
    print("=" * 36)
    print("📊 SCALING LAWS COMPARISON")
    print("=" * 36)
    
    scaling_laws_analysis([5e-9, 10e-9], 5)
    
    print("=" * 36)
    print("🎯 SUMMARY")
    print("=" * 36)
    print(f"Optimal error bound: {results['error_bound']*100:.1f}%")
    print(f"Numerical conditioning: {'Good' if results['condition_number'] < 1e3 else 'Needs attention'}")
    print("Next steps: Implement optimal tolerances in fabrication")
