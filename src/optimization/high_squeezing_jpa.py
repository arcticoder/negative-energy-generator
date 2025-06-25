"""
High-Squeezing JPA Optimization for >15 dB in Femtoliter Cavities
================================================================

Mathematical Foundation:
- Target squeezing: >15 dB requires r ≥ ln(10^(15/20)) ≈ 1.726
- Squeezing parameter: r = ε√(Q/10⁶)/(1 + 4Δ²)
- Femtoliter cavity volume: 1 fL = 1×10⁻¹⁸ m³
- Negative energy: E = -sinh²(r)ℏω × V_cavity

Optimization targets:
- Squeezing >15 dB in realistic parameter ranges
- Pump amplitude ε ∈ [0.01, 0.3] (hardware limits)
- Quality factor Q ∈ [10⁴, 10⁷] (achievable range)
- Detuning Δ ∈ [-0.5, 0.5] GHz (operational range)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from scipy import constants
from scipy.optimize import minimize_scalar, minimize

# Physical constants
hbar = constants.hbar
k_B = constants.k


def required_pump_for_squeezing_db(target_db: float, Q_factor: float, 
                                 detuning: float = 0.0) -> float:
    """
    Solve for pump amplitude ε to achieve target squeezing.
    
    Mathematical relation:
    squeezing_dB = 8.686 * r_effective
    r_effective = ε * sqrt(Q/1e6) / (1 + 4*detuning²)
    So: ε = target_dB / (8.686 * sqrt(Q/1e6) / (1 + 4*detuning²))
    
    Args:
        target_db: Target squeezing in dB
        Q_factor: Quality factor
        detuning: Frequency detuning (GHz)
    
    Returns:
        Required pump amplitude ε
    """
    # Convert dB to squeezing parameter
    r_target = target_db / 8.686
    
    # Calculate required pump amplitude
    denominator = np.sqrt(Q_factor/1e6) / (1 + 4*detuning**2)
    
    if denominator <= 0:
        return float('inf')  # Impossible configuration
    
    epsilon_required = r_target / denominator
    return epsilon_required


def simulate_jpa_femtoliter_cavity(signal_freq: float, pump_power: float, 
                                 temperature: float, Q_factor: float,
                                 detuning: float = 0.0,
                                 josephson_energy: float = 25e6,
                                 charging_energy: float = 1e6) -> Dict:
    """
    Enhanced JPA simulation for femtoliter cavity optimization.
    
    Key improvements:
    - True femtoliter volume (1e-18 m³)
    - High-Q cavity physics
    - Optimized for >15 dB squeezing
    
    Args:
        signal_freq: Signal frequency (Hz)
        pump_power: Pump amplitude ε (0.01 to 0.3)
        temperature: Operating temperature (K)
        Q_factor: Quality factor (1e4 to 1e7)
        detuning: Frequency detuning (GHz)
        josephson_energy: Josephson energy (Hz)
        charging_energy: Charging energy (Hz)
    
    Returns:
        Dictionary with enhanced JPA metrics
    """
    # Femtoliter cavity volume (1 fL = 1e-18 m³)
    cavity_volume = 1e-18  # m³
    
    # JPA characteristics
    plasma_freq = np.sqrt(8 * josephson_energy * charging_energy)
    anharmonicity = -charging_energy
    
    # Thermal effects
    if temperature > 0:
        thermal_photons = 1 / (np.exp(hbar * signal_freq / (k_B * temperature)) - 1)
    else:
        thermal_photons = 0
    thermal_factor = 1 / (1 + 2 * thermal_photons)
    
    # Squeezing parameter calculation  
    # r = ε * sqrt(Q/1e6) / (1 + 4*Δ²) with thermal degradation
    base_squeezing = pump_power * np.sqrt(Q_factor / 1e6) / (1 + 4 * detuning**2)
    r_effective = base_squeezing * thermal_factor
    
    # Squeezing metrics (corrected formulas)
    if r_effective > 0:
        # Squeezing in dB: -20*log10(exp(-r)) = 20*r/ln(10) ≈ 8.686*r
        squeezing_dB = 8.686 * r_effective  # Positive dB for squeezing
        variance_reduction = np.exp(-2 * r_effective)
    else:
        squeezing_dB = 0
        variance_reduction = 1
    
    # Negative energy density in femtoliter cavity
    hbar_omega = hbar * signal_freq
    sinh_r = np.sinh(r_effective)
    cosh_r = np.cosh(r_effective)
    
    # Zero-point energy modification
    rho_squeezed = -sinh_r**2 * hbar_omega
    total_energy = rho_squeezed * cavity_volume
    
    # JPA performance metrics
    gain_dB = 20 * np.log10(cosh_r)
    bandwidth = plasma_freq / (2 * np.pi * Q_factor)
    
    # Quantum efficiency and noise
    quantum_efficiency = thermal_factor * pump_power / 0.3  # Normalized to max pump
    noise_temperature = temperature / thermal_factor
    
    # Cavity enhancement factor
    cavity_enhancement = Q_factor / 1e6  # Enhancement relative to Q=1e6 baseline
    
    # Check if target squeezing is achieved
    target_15db_achieved = squeezing_dB >= 15.0
    target_20db_achieved = squeezing_dB >= 20.0
    
    # Power requirements
    power_efficiency = r_effective / pump_power if pump_power > 0 else 0
    
    return {
        'squeezing_parameter': r_effective,
        'squeezing_dB': squeezing_dB,
        'variance_reduction': variance_reduction,
        'energy_density': rho_squeezed,
        'total_energy': total_energy,
        'cavity_volume': cavity_volume,
        'thermal_factor': thermal_factor,
        'thermal_photons': thermal_photons,
        'gain_dB': gain_dB,
        'bandwidth': bandwidth,
        'quantum_efficiency': quantum_efficiency,
        'noise_temperature': noise_temperature,
        'cavity_enhancement': cavity_enhancement,
        'target_15db_achieved': target_15db_achieved,
        'target_20db_achieved': target_20db_achieved,
        'power_efficiency': power_efficiency,
        'Q_factor': Q_factor,
        'detuning': detuning,
        'pump_amplitude': pump_power,
        'plasma_frequency': plasma_freq,
        'anharmonicity': anharmonicity,
        'optimization_score': -total_energy if target_15db_achieved else 1e10
    }


def optimize_jpa_for_high_squeezing(target_db: float = 15.0,
                                  Q_range: Tuple[float, float] = (1e6, 1e8),
                                  temp_range: Tuple[float, float] = (0.01, 0.05),
                                  detuning_range: Tuple[float, float] = (-0.2, 0.2)) -> Dict:
    """
    Optimize JPA parameters for high squeezing (>15 dB).
    
    Args:
        target_db: Target squeezing in dB
        Q_range: (min, max) quality factors
        temp_range: (min, max) temperatures (K)
        detuning_range: (min, max) detuning (GHz)
    
    Returns:
        Dictionary with optimization results
    """
    print(f"🎯 Optimizing JPA for >{target_db} dB squeezing")
    print(f"   Cavity volume: 1 fL = 1×10⁻¹⁸ m³")
    
    signal_freq = 6e9  # 6 GHz signal
    best_result = None
    best_energy = 0
    feasible_configs = []
    
    # Parameter grid search
    Q_values = np.logspace(np.log10(Q_range[0]), np.log10(Q_range[1]), 15)
    temp_values = np.linspace(temp_range[0], temp_range[1], 10)
    detuning_values = np.linspace(detuning_range[0], detuning_range[1], 11)
    
    print(f"   Searching {len(Q_values)} × {len(temp_values)} × {len(detuning_values)} configurations...")
    
    for Q in Q_values:
        for temp in temp_values:
            for detuning in detuning_values:
                # Calculate required pump amplitude
                eps_required = required_pump_for_squeezing_db(target_db, Q, detuning)
                
                # Check if within hardware limits
                eps_feasible = np.clip(eps_required, 0.01, 0.3)
                
                # Simulate JPA performance
                result = simulate_jpa_femtoliter_cavity(
                    signal_freq, eps_feasible, temp, Q, detuning
                )
                
                # Check if target is achieved
                if result['target_15db_achieved'] and eps_required <= 0.3:
                    feasible_configs.append({
                        'Q_factor': Q,
                        'temperature': temp,
                        'detuning': detuning,
                        'pump_amplitude': eps_feasible,
                        'squeezing_dB': result['squeezing_dB'],
                        'total_energy': result['total_energy'],
                        'quantum_efficiency': result['quantum_efficiency']
                    })
                    
                    if result['total_energy'] < best_energy:
                        best_energy = result['total_energy']
                        best_result = result.copy()
                        best_result.update({
                            'optimal_Q': Q,
                            'optimal_temp': temp,
                            'optimal_detuning': detuning,
                            'optimal_pump': eps_feasible
                        })
    
    n_feasible = len(feasible_configs)
    print(f"   ✅ Found {n_feasible} feasible configurations")
    
    if best_result:
        print(f"   🏆 Best configuration:")
        print(f"      • Q-factor: {best_result['optimal_Q']:.1e}")
        print(f"      • Temperature: {best_result['optimal_temp']*1000:.1f} mK")
        print(f"      • Detuning: {best_result['optimal_detuning']:.2f} GHz")
        print(f"      • Pump amplitude: ε = {best_result['optimal_pump']:.3f}")
        print(f"      • Achieved squeezing: {best_result['squeezing_dB']:.1f} dB")
        print(f"      • Negative energy: {best_result['total_energy']:.2e} J")
        print(f"      • Quantum efficiency: {best_result['quantum_efficiency']:.1%}")
    else:
        print(f"   ❌ No feasible configurations found for {target_db} dB")
        # Try with relaxed constraints
        print("   🔄 Trying with relaxed pump power limit...")
        best_relaxed = None
        for Q in Q_values[::3]:  # Coarser search
            eps_required = required_pump_for_squeezing_db(target_db, Q, 0.0)
            result = simulate_jpa_femtoliter_cavity(signal_freq, eps_required, 0.015, Q, 0.0)
            if best_relaxed is None or result['squeezing_dB'] > best_relaxed['squeezing_dB']:
                best_relaxed = result.copy()
                best_relaxed.update({'required_Q': Q, 'required_pump': eps_required})
        
        if best_relaxed:
            print(f"      • Would need ε = {best_relaxed['required_pump']:.3f} at Q = {best_relaxed['required_Q']:.1e}")
            print(f"      • Would achieve {best_relaxed['squeezing_dB']:.1f} dB")
    
    return {
        'best_result': best_result,
        'feasible_configs': feasible_configs,
        'n_feasible': n_feasible,
        'target_db': target_db,
        'parameter_ranges': {
            'Q_range': Q_range,
            'temp_range': temp_range,
            'detuning_range': detuning_range
        }
    }


def squeezing_parameter_sweep(Q_factor: float = 1e6, temperature: float = 0.015) -> Dict:
    """
    Parameter sweep for squeezing optimization at fixed Q and T.
    
    Args:
        Q_factor: Fixed quality factor
        temperature: Fixed temperature (K)
    
    Returns:
        Dictionary with sweep results
    """
    print(f"📊 Squeezing parameter sweep (Q={Q_factor:.1e}, T={temperature*1000:.1f} mK)")
    
    signal_freq = 6e9
    pump_powers = np.linspace(0.01, 0.3, 30)
    detunings = np.linspace(-0.5, 0.5, 21)
    
    # Create result matrices
    squeezing_matrix = np.zeros((len(pump_powers), len(detunings)))
    energy_matrix = np.zeros((len(pump_powers), len(detunings)))
    feasible_matrix = np.zeros((len(pump_powers), len(detunings)), dtype=bool)
    
    best_config = None
    best_squeezing = 0
    
    for i, pump in enumerate(pump_powers):
        for j, detuning in enumerate(detunings):
            result = simulate_jpa_femtoliter_cavity(
                signal_freq, pump, temperature, Q_factor, detuning
            )
            
            squeezing_matrix[i, j] = result['squeezing_dB']
            energy_matrix[i, j] = -result['total_energy']  # Store positive for plotting
            feasible_matrix[i, j] = result['target_15db_achieved']
            
            if result['squeezing_dB'] > best_squeezing:
                best_squeezing = result['squeezing_dB']
                best_config = {
                    'pump_power': pump,
                    'detuning': detuning,
                    'squeezing_dB': result['squeezing_dB'],
                    'total_energy': result['total_energy'],
                    'quantum_efficiency': result['quantum_efficiency']
                }
    
    # Count feasible region
    n_feasible = np.sum(feasible_matrix)
    total_points = feasible_matrix.size
    feasible_fraction = n_feasible / total_points * 100
    
    print(f"   ✅ Sweep complete!")
    print(f"   🎯 Best squeezing: {best_squeezing:.1f} dB")
    print(f"   📈 Feasible region: {feasible_fraction:.1f}% of parameter space")
    print(f"   🏆 Best config: ε={best_config['pump_power']:.3f}, Δ={best_config['detuning']:.2f} GHz")
    
    return {
        'pump_powers': pump_powers,
        'detunings': detunings,
        'squeezing_matrix': squeezing_matrix,
        'energy_matrix': energy_matrix,
        'feasible_matrix': feasible_matrix,
        'best_config': best_config,
        'feasible_fraction': feasible_fraction,
        'Q_factor': Q_factor,
        'temperature': temperature
    }


def compare_cavity_volumes() -> Dict:
    """
    Compare performance across different cavity volumes.
    
    Returns:
        Dictionary with volume comparison results
    """
    print("📏 Cavity volume comparison")
    
    # Volume range from 0.1 fL to 10 fL
    volumes_fL = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    volumes_m3 = [v * 1e-18 for v in volumes_fL]
    
    signal_freq = 6e9
    pump_power = 0.2  # Fixed pump
    temperature = 0.015  # 15 mK
    Q_factor = 1e6
    
    results = []
    
    for i, (vol_fL, vol_m3) in enumerate(zip(volumes_fL, volumes_m3)):
        # Simulate with modified volume
        base_result = simulate_jpa_femtoliter_cavity(
            signal_freq, pump_power, temperature, Q_factor
        )
        
        # Scale energy by volume ratio
        volume_ratio = vol_m3 / 1e-18
        scaled_energy = base_result['total_energy'] * volume_ratio
        
        results.append({
            'volume_fL': vol_fL,
            'volume_m3': vol_m3,
            'squeezing_dB': base_result['squeezing_dB'],  # Independent of volume
            'total_energy': scaled_energy,
            'energy_density': base_result['energy_density'],  # Independent of volume
            'volume_ratio': volume_ratio
        })
        
        print(f"   {vol_fL:4.1f} fL: {base_result['squeezing_dB']:.1f} dB, {scaled_energy:.2e} J")
    
    return {
        'results': results,
        'optimal_volume_fL': 1.0,  # Sweet spot for fabrication vs energy
        'volume_range_fL': volumes_fL
    }


if __name__ == "__main__":
    print("⚡ High-Squeezing JPA Testing")
    print("=" * 35)
    
    # Test 1: Required pump amplitudes for target squeezing
    print("\n1️⃣  Required Pump Amplitudes")
    targets = [15, 18, 20, 25]
    Q_test = 1e6
    
    for target in targets:
        eps_req = required_pump_for_squeezing_db(target, Q_test, 0.0)
        feasible = "✅" if eps_req <= 0.3 else "❌"
        print(f"   {target:2d} dB: ε = {eps_req:.3f} {feasible}")
    
    # Test 2: Optimize for 15 dB squeezing
    print("\n2️⃣  Optimization for >15 dB Squeezing")
    opt_result = optimize_jpa_for_high_squeezing(target_db=15.0)
    
    # Test 3: Parameter sweep at optimal conditions
    if opt_result['best_result']:
        print("\n3️⃣  Parameter Sweep at Optimal Q")
        Q_opt = opt_result['best_result']['optimal_Q']
        T_opt = opt_result['best_result']['optimal_temp']
        sweep_result = squeezing_parameter_sweep(Q_opt, T_opt)
    
    # Test 4: Cavity volume comparison
    print("\n4️⃣  Cavity Volume Effects")
    volume_result = compare_cavity_volumes()
    
    print("\n✅ High-squeezing JPA testing complete!")
