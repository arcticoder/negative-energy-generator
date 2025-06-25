# File: src/hardware/laser_pump.py
"""
Laser-Based Boundary Pump Module

Physics Implementation:
- Mirror position: X(t) = X₀sin(Ωt)
- Effective squeezing: r_eff ∝ (dX/dt)/c √(Q/ω₀)
- Negative energy density: ρ_neg(t) ≈ -sinh²(r_eff(t))ℏω₀

This module simulates boundary modulation by a laser-driven mirror for
dynamical Casimir effect (DCE) negative energy generation.
"""

import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Physical constants
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
c = 299792458        # Speed of light (m/s)
k_B = 1.380649e-23   # Boltzmann constant (J/K)

def simulate_laser_pump(X0: float, Ω: float, ω0: float, Q: float, t: np.ndarray,
                       coupling_efficiency: float = 1.0, loss_factor: float = 0.95) -> dict:
    """
    Simulate boundary modulation by a laser-driven mirror for DCE.
    
    Mathematical Foundation:
    X(t) = X₀sin(Ωt)
    r_eff(t) = (dX/dt)/c × √(Q/ω₀) × coupling_efficiency
    ρ_neg(t) = -sinh²(r_eff(t)) × ℏω₀ × loss_factor
    
    Args:
        X0: Mirror displacement amplitude (m)
        Ω: Drive frequency (rad/s)
        ω0: Cavity resonance frequency (rad/s)
        Q: Cavity quality factor
        t: Time array (s)
        coupling_efficiency: Mirror-cavity coupling (0-1)
        loss_factor: Account for various losses (0-1)
    
    Returns:
        Dictionary with time-dependent DCE metrics
    """
    # Mirror motion and velocity
    X = X0 * np.sin(Ω * t)
    dX_dt = X0 * Ω * np.cos(Ω * t)
    
    # Effective squeezing parameter with coupling efficiency
    r_eff = (dX_dt / c) * np.sqrt(Q / ω0) * coupling_efficiency
    
    # Enhanced model for strong driving (nonlinear effects)
    r_max = 3.0  # Maximum achievable squeezing
    r_eff_clamped = np.tanh(np.abs(r_eff) / r_max) * r_max * np.sign(r_eff)
    
    # Instantaneous negative energy density
    sinh_r = np.sinh(r_eff_clamped)
    rho_neg = -sinh_r**2 * hbar * ω0 * loss_factor
    
    # Total energy in cavity mode volume (assume 1 femtoliter)
    mode_volume = 1e-18  # m³
    total_energy = rho_neg * mode_volume
    
    # Peak power metrics
    instantaneous_power = np.abs(np.gradient(total_energy, t))
    
    # Frequency domain analysis
    fft_rho = np.fft.fft(rho_neg)
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    spectral_density = np.abs(fft_rho)**2
    
    # Efficiency metrics
    drive_power = 0.5 * X0**2 * Ω**3  # Approximate mechanical drive power
    extraction_efficiency = np.abs(np.mean(total_energy)) / drive_power if drive_power > 0 else 0
    
    # Quality metrics
    coherence_time = Q / ω0
    bandwidth = ω0 / Q
    
    return {
        't': t,
        'X': X,
        'dX_dt': dX_dt,
        'r_eff': r_eff_clamped,
        'rho_neg': rho_neg,
        'total_energy': total_energy,
        'instantaneous_power': instantaneous_power,
        'peak_rho_neg': np.min(rho_neg),  # Most negative value
        'peak_energy': np.min(total_energy),
        'mean_energy': np.mean(total_energy),
        'extraction_efficiency': extraction_efficiency,
        'coherence_time': coherence_time,
        'bandwidth': bandwidth,
        'spectral_density': spectral_density,
        'frequencies': freqs,
        'optimization_score': -np.min(total_energy)  # For optimization
    }

def optimize_laser_pump_parameters(target_energy: float = -1e-15,
                                 X0_range: tuple = (1e-15, 1e-9),
                                 Ω_range: tuple = (1e8, 1e11),
                                 Q_range: tuple = (1e4, 1e7),
                                 n_samples: int = 1000) -> dict:
    """
    Optimize laser pump parameters to achieve target negative energy.
    
    Args:
        target_energy: Target negative energy (J)
        X0_range: Mirror amplitude range (m)
        Ω_range: Drive frequency range (rad/s)
        Q_range: Quality factor range
        n_samples: Number of optimization samples
    
    Returns:
        Dictionary with optimization results
    """
    print(f"🔧 Optimizing laser pump for target energy: {target_energy:.2e} J")
    
    best_params = None
    best_energy = 0
    results = []
    
    # Time array for simulation
    t_sim = np.linspace(0, 10e-9, 500)  # 10 ns simulation
    
    for i in range(n_samples):
        # Random parameter sampling
        X0 = np.random.uniform(*X0_range)
        Ω = np.random.uniform(*Ω_range)
        Q = np.random.uniform(*Q_range)
        ω0 = Ω * (0.8 + 0.4 * np.random.random())  # ω0 near Ω
        
        try:
            result = simulate_laser_pump(X0, Ω, ω0, Q, t_sim)
            peak_energy = result['peak_energy']
            
            results.append({
                'X0': X0,
                'Omega': Ω,
                'omega0': ω0,
                'Q': Q,
                'peak_energy': peak_energy,
                'efficiency': result['extraction_efficiency']
            })
            
            if peak_energy < best_energy:  # More negative is better
                best_energy = peak_energy
                best_params = {
                    'X0': X0,
                    'Omega': Ω,
                    'omega0': ω0,
                    'Q': Q
                }
        except:
            continue  # Skip failed evaluations
    
    # Find configurations meeting target
    successful_configs = [r for r in results if r['peak_energy'] <= target_energy]
    
    print(f"✅ Laser pump optimization complete!")
    print(f"   • Best energy achieved: {best_energy:.2e} J")
    print(f"   • Target met by: {len(successful_configs)}/{len(results)} configurations")
    if best_params:
        print(f"   • Best X₀: {best_params['X0']:.2e} m")
        print(f"   • Best Ω: {best_params['Omega']:.2e} rad/s")
        print(f"   • Best Q: {best_params['Q']:.1e}")
    
    return {
        'best_parameters': best_params,
        'best_energy': best_energy,
        'target_energy': target_energy,
        'all_results': results,
        'successful_configs': successful_configs,
        'success_rate': len(successful_configs) / len(results) if results else 0
    }

def laser_pump_sensitivity_analysis(X0_nominal: float, Ω_nominal: float, 
                                  ω0_nominal: float, Q_nominal: float,
                                  perturbation: float = 0.1) -> dict:
    """
    Analyze parameter sensitivity of laser pump performance.
    
    Args:
        X0_nominal, Ω_nominal, ω0_nominal, Q_nominal: Nominal parameters
        perturbation: Fractional perturbation for sensitivity analysis
    
    Returns:
        Dictionary with sensitivity metrics
    """
    t_sim = np.linspace(0, 5e-9, 250)
    
    # Baseline simulation
    baseline = simulate_laser_pump(X0_nominal, Ω_nominal, ω0_nominal, Q_nominal, t_sim)
    baseline_energy = baseline['peak_energy']
    
    sensitivities = {}
    
    # Test each parameter
    params = {
        'X0': X0_nominal,
        'Omega': Ω_nominal,
        'omega0': ω0_nominal,
        'Q': Q_nominal
    }
    
    for param_name, nominal_value in params.items():
        # Positive perturbation
        test_params = params.copy()
        test_params[param_name] = nominal_value * (1 + perturbation)
        
        if param_name == 'X0':
            result_pos = simulate_laser_pump(test_params['X0'], test_params['Omega'], 
                                           test_params['omega0'], test_params['Q'], t_sim)
        elif param_name == 'Omega':
            result_pos = simulate_laser_pump(test_params['X0'], test_params['Omega'], 
                                           test_params['omega0'], test_params['Q'], t_sim)
        elif param_name == 'omega0':
            result_pos = simulate_laser_pump(test_params['X0'], test_params['Omega'], 
                                           test_params['omega0'], test_params['Q'], t_sim)
        else:  # Q
            result_pos = simulate_laser_pump(test_params['X0'], test_params['Omega'], 
                                           test_params['omega0'], test_params['Q'], t_sim)
        
        # Calculate sensitivity
        energy_change = result_pos['peak_energy'] - baseline_energy
        sensitivity = (energy_change / baseline_energy) / perturbation
        
        sensitivities[param_name] = {
            'sensitivity': sensitivity,
            'energy_change': energy_change,
            'relative_change': energy_change / baseline_energy
        }
    
    return {
        'baseline_energy': baseline_energy,
        'sensitivities': sensitivities,
        'most_sensitive': max(sensitivities.keys(), 
                            key=lambda k: abs(sensitivities[k]['sensitivity']))
    }

# Example usage and testing
if __name__ == "__main__":
    print("🔬 LASER-BASED BOUNDARY PUMP SIMULATION")
    print("=" * 60)
    
    # Test basic simulation
    print("\n1️⃣  BASIC DCE SIMULATION")
    t = np.linspace(0, 2*np.pi/1e9, 1000)  # 1 GHz drive period
    result = simulate_laser_pump(
        X0=1e-12,        # 1 pm amplitude
        Ω=2*np.pi*1e9,   # 1 GHz drive frequency
        ω0=2*np.pi*5e9,  # 5 GHz cavity frequency
        Q=1e6,           # Quality factor
        t=t
    )
    
    print(f"   • Peak |ρ_neg| = {np.abs(result['peak_rho_neg']):.2e} J/m³")
    print(f"   • Peak energy = {result['peak_energy']:.2e} J")
    print(f"   • Extraction efficiency = {result['extraction_efficiency']:.2e}")
    print(f"   • Coherence time = {result['coherence_time']:.2e} s")
    
    # Test parameter optimization
    print("\n2️⃣  PARAMETER OPTIMIZATION")
    opt_result = optimize_laser_pump_parameters(
        target_energy=-1e-15,
        n_samples=500
    )
    
    # Test sensitivity analysis
    if opt_result['best_parameters']:
        print("\n3️⃣  SENSITIVITY ANALYSIS")
        bp = opt_result['best_parameters']
        sens_result = laser_pump_sensitivity_analysis(
            bp['X0'], bp['Omega'], bp['omega0'], bp['Q']
        )
        
        print(f"   • Most sensitive parameter: {sens_result['most_sensitive']}")
        for param, data in sens_result['sensitivities'].items():
            print(f"   • {param} sensitivity: {data['sensitivity']:.2f}")
    
    print(f"\n✅ Laser pump module validation complete!")
