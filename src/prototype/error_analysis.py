# src/prototype/error_analysis.py
import numpy as np

hbar, c = 1.054e-34, 3e8

def casimir_energy(d):
    return - (np.pi**2 * hbar * c) / (720 * d**3)

def casimir_energy_sensitivity(d):
    # dE/dd
    return - (np.pi**2 * hbar * c) / (240 * d**4)

def mc_uncertainty(ds, sigma_d, n_samples=10000):
    """Propagate gap‐size uncertainty via Monte Carlo."""
    d_samples = np.random.normal(ds, sigma_d, size=(n_samples, len(ds)))
    E_samples = np.sum(casimir_energy(d_samples), axis=1)
    return E_samples.mean(), E_samples.std()

def analytical_uncertainty(ds, sigma_d):
    """First-order analytical uncertainty propagation."""
    sensitivities = casimir_energy_sensitivity(ds)
    variance = np.sum((sensitivities * sigma_d)**2)
    return np.sqrt(variance)

def uncertainty_report(ds, sigma_d, n_samples=10000):
    """Generate comprehensive uncertainty analysis report."""
    
    print("🔬 CASIMIR ARRAY UNCERTAINTY ANALYSIS")
    print("=" * 38)
    print()
    
    # Nominal energy
    E_nominal = np.sum(casimir_energy(ds))
    print(f"Nominal energy: {E_nominal:.3e} J/m²")
    print(f"Number of gaps: {len(ds)}")
    print(f"Gap sizes: {[f'{d*1e9:.1f}' for d in ds]} nm")
    print(f"Gap uncertainty: ±{sigma_d * 1e9:.2f} nm")
    print()
    
    # Monte Carlo uncertainty
    print("📊 MONTE CARLO PROPAGATION")
    print("-" * 27)
    mc_mean, mc_std = mc_uncertainty(ds, sigma_d, n_samples)
    mc_rel_error = mc_std / abs(mc_mean) * 100
    
    print(f"MC mean energy: {mc_mean:.3e} J/m²")
    print(f"MC std deviation: {mc_std:.3e} J/m²")
    print(f"MC relative error: {mc_rel_error:.2f}%")
    print()
    
    # Analytical uncertainty
    print("📐 ANALYTICAL PROPAGATION")
    print("-" * 25)
    analytical_std = analytical_uncertainty(ds, sigma_d)
    analytical_rel_error = analytical_std / abs(E_nominal) * 100
    
    print(f"Analytical std: {analytical_std:.3e} J/m²")
    print(f"Analytical rel error: {analytical_rel_error:.2f}%")
    print()
    
    # Comparison
    print("🔍 METHOD COMPARISON")
    print("-" * 20)
    error_ratio = mc_std / analytical_std
    print(f"MC/Analytical ratio: {error_ratio:.3f}")
    
    if abs(error_ratio - 1.0) < 0.1:
        print("✅ Methods agree well (linear regime)")
    else:
        print("⚠️ Nonlinear effects present")
    
    print()
    
    # Sensitivity breakdown
    print("📊 SENSITIVITY BREAKDOWN")
    print("-" * 24)
    sensitivities = casimir_energy_sensitivity(ds)
    for i, (d, sens) in enumerate(zip(ds, sensitivities)):
        contribution = (sens * sigma_d)**2 / analytical_std**2 * 100
        print(f"Gap {i+1} ({d*1e9:.1f} nm): {contribution:.1f}% of variance")
    
    return {
        'nominal_energy': E_nominal,
        'mc_mean': mc_mean,
        'mc_std': mc_std,
        'analytical_std': analytical_std,
        'mc_rel_error': mc_rel_error,
        'analytical_rel_error': analytical_rel_error
    }

# Example usage
if __name__=='__main__':
    ds = np.full(5, 7e-9)  # 5 gaps at 7 nm
    sigma_d = 1e-10        # ±0.1 nm uncertainty
    
    results = uncertainty_report(ds, sigma_d)
    
    print("=" * 38)
    print("🎯 EXPERIMENTAL REQUIREMENTS")
    print("=" * 38)
    print(f"To achieve 1% energy uncertainty:")
    target_rel_error = 0.01
    required_precision = target_rel_error * abs(results['nominal_energy']) / abs(casimir_energy_sensitivity(ds[0]))
    print(f"Required gap precision: ±{required_precision*1e12:.1f} pm")
