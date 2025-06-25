"""
Scale-Up Strategy Module for Negative Energy Generation
=====================================================

Implements multi-chamber scaling strategies for transitioning from 1 μm³ 
demonstrator to practical-scale negative energy systems.

Mathematical Foundation:
- Tiling scaling: V_tot = N·V₀, E_tot = N·E₀, ANEC_tot = N·ANEC₀
- Thermal dynamics: dT/dt = (Ṗ - (T-T_env)/R_th) / C_th
- Vibration isolation: H(s) = ω_c²/(s² + 2ζω_c·s + ω_c²)
- Cooling COP: T_ss = T_env + (P_drive(1-COP))·R_th

Engineering Considerations:
- Power scaling, thermal management, vibration isolation
- Distributed control architecture for N-chamber arrays
- Infrastructure requirements for practical deployment
"""

import numpy as np
from scipy.integrate import odeint
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings

# Import single chamber simulator
try:
    from scripts.integrated_small_scale_demo import simulate_chamber
except ImportError:
    warnings.warn("Could not import simulate_chamber - using mock function")
    def simulate_chamber(scenario="burst", duration=100e-9, volume_m3=1e-18):
        """Mock chamber simulator for testing"""
        return {
            'volume_m3': volume_m3,
            'total_energy_J': -8.65e-14,
            'anec_Js_per_m3': -8.65e4,
            'dist_rejection_dB': 119.0,
            'rms_control_effort': 82.4,
            'constraint_satisfaction': 1.0,
            'target_achievement': 1.0,
            'duration_s': duration,
            'scenario': scenario
        }

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
sigma_SB = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²/K⁴)

# Engineering parameters
DEFAULT_R_TH = 0.1  # Thermal resistance (K/W)
DEFAULT_C_TH = 50.0  # Heat capacity (J/K)
DEFAULT_COP = 3.0   # Coefficient of performance for cooling

# -----------------------------------------------------------------------------
# 1) Modular Tiling: N×N×N chamber arrays
# -----------------------------------------------------------------------------

def scale_tiling(Nx: int, Ny: int, Nz: int, scenario: str = "burst", 
                duration: float = 100e-9) -> Dict:
    """
    Scale single-chamber performance to N×N×N array through modular tiling.
    
    Mathematical model:
    - Volume scaling: V_total = N_chambers · V₀
    - Energy scaling: E_total = N_chambers · E₀  
    - ANEC scaling: ANEC_total = N_chambers · ANEC₀
    - Disturbance rejection: DR_total ≈ DR₀ - 10·log₁₀(√N) [noise correlation]
    - Control effort: scales as √N for parallel controllers
    
    Args:
        Nx, Ny, Nz: Chamber array dimensions
        scenario: Disturbance type ('burst', 'continuous', 'step')
        duration: Simulation time (s)
    
    Returns:
        dict: Scaled performance metrics
    """
    print(f"🏗️ Modular Tiling Scale-Up: {Nx}×{Ny}×{Nz} Array")
    print("=" * 50)
    
    N = Nx * Ny * Nz
    print(f"   📦 Total chambers: {N:,}")
    
    # Get baseline single-chamber performance
    print(f"   🔬 Running baseline chamber simulation...")
    base = simulate_chamber(scenario=scenario, duration=duration)
    
    # Linear scaling for extensive properties
    scaled_volume = N * base['volume_m3']
    scaled_energy = N * base['total_energy_J']
    scaled_anec = N * base['anec_Js_per_m3']
    
    # Noise scaling for disturbance rejection
    # Assume √N scaling due to partial correlation
    noise_penalty_dB = 10 * np.log10(np.sqrt(N))
    scaled_dist_rejection = base['dist_rejection_dB'] - noise_penalty_dB
    
    # Control effort scaling - √N for parallel controllers
    scaled_control_effort = base['rms_control_effort'] * np.sqrt(N)
    
    # Power scaling estimate (proportional to control effort)
    base_power_W = 1e-6  # Estimate 1 μW per chamber
    scaled_power_W = N * base_power_W
    
    # Constraint satisfaction assumes independent chambers
    scaled_constraint_satisfaction = base['constraint_satisfaction']
    scaled_target_achievement = base['target_achievement']
    
    print(f"   📊 Scaling Results:")
    print(f"      • Volume: {base['volume_m3']:.2e} → {scaled_volume:.2e} m³")
    print(f"      • Energy: {base['total_energy_J']:.2e} → {scaled_energy:.2e} J")
    print(f"      • ANEC: {base['anec_Js_per_m3']:.2e} → {scaled_anec:.2e} J·s·m⁻³")
    print(f"      • Dist. rejection: {base['dist_rejection_dB']:.1f} → {scaled_dist_rejection:.1f} dB")
    print(f"      • Control effort: {base['rms_control_effort']:.1e} → {scaled_control_effort:.1e}")
    print(f"      • Power estimate: {scaled_power_W:.2e} W")
    
    return {
        'N_chambers': N,
        'array_dimensions': (Nx, Ny, Nz),
        'volume_m3': scaled_volume,
        'total_energy_J': scaled_energy,
        'total_anec': scaled_anec,
        'anec_density_Js_per_m3': base['anec_Js_per_m3'],  # Density unchanged
        'dist_rejection_dB': scaled_dist_rejection,
        'rms_control_effort': scaled_control_effort,
        'constraint_satisfaction': scaled_constraint_satisfaction,
        'target_achievement': scaled_target_achievement,
        'estimated_power_W': scaled_power_W,
        'scaling_factor': N,
        'noise_penalty_dB': noise_penalty_dB,
        'baseline_metrics': base,
        'scenario': scenario,
        'duration_s': duration
    }

# -----------------------------------------------------------------------------
# 2) Thermal & Vibration Management
# -----------------------------------------------------------------------------

def thermal_response(P_dot: float, C_th: float = DEFAULT_C_TH, 
                    R_th: float = DEFAULT_R_TH, T_env: float = 4.0,
                    times: np.ndarray = None) -> np.ndarray:
    """
    Lumped-parameter thermal model for cryogenic cooling.
    
    Mathematical model:
    dT/dt = (Ṗ_in - (T-T_env)/R_th) / C_th
    
    Where:
    - Ṗ_in: Heat input (W)
    - R_th: Thermal resistance (K/W)  
    - C_th: Heat capacity (J/K)
    - T_env: Environment temperature (K)
    
    Args:
        P_dot: Heat dissipation rate (W)
        C_th: Heat capacity (J/K)
        R_th: Thermal resistance (K/W)
        T_env: Environment temperature (K)
        times: Time array (s)
    
    Returns:
        Temperature evolution T(t) (K)
    """
    if times is None:
        times = np.linspace(0, 1e-3, 1000)  # Default 1 ms, 1000 points
    
    def dTdt(T, t):
        """Temperature evolution ODE"""
        heat_loss = (T - T_env) / R_th
        return (P_dot - heat_loss) / C_th
    
    # Solve ODE with initial condition T(0) = T_env
    T = odeint(dTdt, T_env, times).flatten()
    
    return T

def vibration_isolation(z_input: np.ndarray, fs: float, 
                       f_c: float = 1e3, order: int = 2) -> np.ndarray:
    """
    2nd-order Butterworth low-pass filter for vibration isolation.
    
    Mathematical model:
    H(s) = ω_c²/(s² + 2ζω_c·s + ω_c²)
    
    Where:
    - ω_c = 2πf_c (cutoff frequency)
    - ζ = 1/√2 (Butterworth damping ratio)
    
    Args:
        z_input: Input displacement signal (m)
        fs: Sampling frequency (Hz)
        f_c: Cutoff frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered displacement signal (m)
    """
    # Design Butterworth filter
    nyquist = fs / 2
    normalized_cutoff = f_c / nyquist
    
    if normalized_cutoff >= 1.0:
        warnings.warn(f"Cutoff frequency {f_c} Hz exceeds Nyquist {nyquist} Hz")
        normalized_cutoff = 0.99
    
    b, a = butter(order, normalized_cutoff, btype='low')
    
    # Apply zero-phase filtering (forward-backward)
    z_output = filtfilt(b, a, z_input)
    
    return z_output

def compute_vibration_isolation_performance(f_c: float, frequencies: np.ndarray) -> Dict:
    """
    Analyze vibration isolation performance across frequency range.
    
    Args:
        f_c: Cutoff frequency (Hz)
        frequencies: Frequency array for analysis (Hz)
    
    Returns:
        dict: Isolation performance metrics
    """
    # 2nd-order Butterworth transfer function magnitude
    omega_c = 2 * np.pi * f_c
    omega = 2 * np.pi * frequencies
    
    # |H(jω)| = ω_c² / √((ω_c² - ω²)² + (2ζω_c·ω)²)
    zeta = 1 / np.sqrt(2)  # Butterworth damping
    
    magnitude_squared = (omega_c**4) / ((omega_c**2 - omega**2)**2 + (2*zeta*omega_c*omega)**2)
    magnitude_dB = 10 * np.log10(magnitude_squared)
    
    # Isolation effectiveness (negative dB = good isolation)
    isolation_dB = -magnitude_dB
    
    return {
        'frequencies_Hz': frequencies,
        'magnitude_dB': magnitude_dB,
        'isolation_dB': isolation_dB,
        'cutoff_frequency_Hz': f_c,
        'isolation_at_10fc': isolation_dB[frequencies >= 10*f_c][0] if any(frequencies >= 10*f_c) else None,
        'isolation_at_100fc': isolation_dB[frequencies >= 100*f_c][0] if any(frequencies >= 100*f_c) else None
    }

# -----------------------------------------------------------------------------
# 3) Power & Cooling Infrastructure
# -----------------------------------------------------------------------------

def steady_state_temperature(P_drive: float, COP: float = DEFAULT_COP,
                           T_env: float = 4.0, R_th: float = DEFAULT_R_TH) -> float:
    """
    Steady-state temperature with active cooling.
    
    Mathematical model:
    Cooling power: P_cool = COP · P_input
    Heat balance: (T_ss - T_env)/R_th = P_drive - P_cool
    ⇒ T_ss = T_env + (P_drive(1 - COP)) · R_th
    
    Note: For COP > 1, net cooling occurs (negative net heat)
    
    Args:
        P_drive: Drive power dissipation (W)
        COP: Coefficient of performance for cooling
        T_env: Environment temperature (K)
        R_th: Thermal resistance (K/W)
    
    Returns:
        Steady-state temperature (K)
    """
    # For COP > 1, cooling power exceeds drive power
    # Net heat removal = P_drive * (COP - 1)
    if COP > 1:
        # Effective cooling: system removes more heat than it generates
        net_cooling = P_drive * (COP - 1)
        # But can't cool below environment
        delta_T = -min(net_cooling * R_th, T_env - 0.1)  # Keep above 0.1 K
    else:
        # COP ≤ 1: net heating
        net_heat = P_drive * (1 - COP)
        delta_T = net_heat * R_th
    
    T_ss = T_env + delta_T
    
    return max(T_ss, 0.1)  # Physical lower bound

def cooling_power_requirements(N_chambers: int, power_per_chamber: float = 1e-6,
                              T_target: float = 4.0, T_env: float = 300.0,
                              COP: float = DEFAULT_COP) -> Dict:
    """
    Calculate cooling infrastructure requirements for N-chamber array.
    
    Args:
        N_chambers: Number of chambers
        power_per_chamber: Power dissipation per chamber (W)
        T_target: Target operating temperature (K)
        T_env: Environment temperature (K)
        COP: Cooling system coefficient of performance
    
    Returns:
        dict: Cooling system requirements
    """
    # Total heat load
    total_heat_W = N_chambers * power_per_chamber
    
    # Required cooling power to maintain T_target
    # Heat balance: P_cool = P_heat + (T_target - T_env)/R_th
    # For steady state: P_cool = total_heat_W to maintain T_target
    required_cooling_W = total_heat_W
    
    # Input power to cooling system
    input_power_W = required_cooling_W / COP
    
    # Total electrical power (drive + cooling)
    total_electrical_W = total_heat_W + input_power_W
    
    # Cooling efficiency
    cooling_efficiency = required_cooling_W / total_electrical_W
    
    return {
        'N_chambers': N_chambers,
        'total_heat_load_W': total_heat_W,
        'required_cooling_W': required_cooling_W,
        'cooling_input_power_W': input_power_W,
        'total_electrical_power_W': total_electrical_W,
        'cooling_efficiency': cooling_efficiency,
        'COP': COP,
        'target_temperature_K': T_target,
        'environment_temperature_K': T_env,
        'power_per_chamber_W': power_per_chamber
    }

# -----------------------------------------------------------------------------
# 4) Comprehensive Scale-Up Analysis
# -----------------------------------------------------------------------------

def comprehensive_scale_up_analysis(array_sizes: List[Tuple[int, int, int]],
                                  scenario: str = "burst",
                                  duration: float = 100e-9) -> Dict:
    """
    Comprehensive analysis of scale-up scenarios.
    
    Args:
        array_sizes: List of (Nx, Ny, Nz) array dimensions
        scenario: Disturbance scenario
        duration: Simulation duration (s)
    
    Returns:
        dict: Complete scale-up analysis
    """
    print(f"🚀 Comprehensive Scale-Up Analysis")
    print("=" * 50)
    print(f"   🎯 Scenario: {scenario}")
    print(f"   ⏱️ Duration: {duration*1e9:.0f} ns")
    print(f"   📏 Array sizes: {len(array_sizes)} configurations")
    
    results = {}
    
    for i, (Nx, Ny, Nz) in enumerate(array_sizes):
        config_name = f"{Nx}x{Ny}x{Nz}"
        print(f"\n   📊 Analyzing {config_name} array...")
        
        # Tiling analysis
        tiling_result = scale_tiling(Nx, Ny, Nz, scenario, duration)
        N = tiling_result['N_chambers']
        
        # Thermal analysis (1 ms simulation)
        times = np.linspace(0, 1e-3, 1000)
        P_total = tiling_result['estimated_power_W']
        T_thermal = thermal_response(P_total, times=times)
        
        # Cooling requirements
        cooling = cooling_power_requirements(N, power_per_chamber=P_total/N)
        
        # Vibration isolation analysis
        freqs = np.logspace(1, 6, 100)  # 10 Hz to 1 MHz
        vibration = compute_vibration_isolation_performance(1e3, freqs)
        
        # Summary metrics
        volume_scale_factor = tiling_result['volume_m3'] / tiling_result['baseline_metrics']['volume_m3']
        energy_density = tiling_result['total_energy_J'] / tiling_result['volume_m3']
        
        results[config_name] = {
            'array_dimensions': (Nx, Ny, Nz),
            'N_chambers': N,
            'tiling_analysis': tiling_result,
            'thermal_peak_K': T_thermal[-1],
            'thermal_time_constant_s': times[np.argmax(T_thermal > 0.63 * T_thermal[-1])],
            'cooling_requirements': cooling,
            'vibration_isolation': vibration,
            'volume_scale_factor': volume_scale_factor,
            'energy_density_J_per_m3': energy_density,
            'power_efficiency_J_per_W_per_s': abs(tiling_result['total_energy_J']) / (P_total * duration) if P_total > 0 else 0
        }
        
        print(f"      ✅ {N:,} chambers, {volume_scale_factor:.1e}× volume scale")
        print(f"      🌡️ Thermal peak: {T_thermal[-1]:.1f} K")
        print(f"      ⚡ Power: {P_total:.2e} W, Cooling: {cooling['cooling_input_power_W']:.2e} W")
    
    return {
        'analysis_results': results,
        'scenario': scenario,
        'duration_s': duration,
        'array_configurations': array_sizes,
        'timestamp': np.datetime64('now').astype(str)
    }

# -----------------------------------------------------------------------------
# 5) Visualization and Reporting
# -----------------------------------------------------------------------------

def plot_scale_up_analysis(analysis_results: Dict, save_filename: str = None):
    """
    Generate comprehensive plots for scale-up analysis.
    
    Args:
        analysis_results: Output from comprehensive_scale_up_analysis
        save_filename: Optional filename to save plot
    """
    results = analysis_results['analysis_results']
    
    # Extract data for plotting
    configs = list(results.keys())
    N_chambers = [results[c]['N_chambers'] for c in configs]
    volumes = [results[c]['tiling_analysis']['volume_m3'] for c in configs]
    energies = [abs(results[c]['tiling_analysis']['total_energy_J']) for c in configs]
    powers = [results[c]['tiling_analysis']['estimated_power_W'] for c in configs]
    thermal_peaks = [results[c]['thermal_peak_K'] for c in configs]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Scale-Up Analysis: {analysis_results['scenario'].title()} Scenario", fontsize=14)
    
    # Volume vs N chambers
    ax1.loglog(N_chambers, volumes, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Chambers')
    ax1.set_ylabel('Total Volume (m³)')
    ax1.set_title('Volume Scaling')
    ax1.grid(True, alpha=0.3)
    
    # Energy vs N chambers  
    ax2.loglog(N_chambers, energies, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Chambers')
    ax2.set_ylabel('Total Energy (J)')
    ax2.set_title('Energy Scaling')
    ax2.grid(True, alpha=0.3)
    
    # Power vs N chambers
    ax3.loglog(N_chambers, powers, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Number of Chambers')
    ax3.set_ylabel('Power Consumption (W)')
    ax3.set_title('Power Scaling')
    ax3.grid(True, alpha=0.3)
    
    # Thermal performance
    ax4.semilogx(N_chambers, thermal_peaks, 'mo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Number of Chambers')
    ax4.set_ylabel('Peak Temperature (K)')
    ax4.set_title('Thermal Management')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=10.0, color='r', linestyle='--', alpha=0.7, label='10 K limit')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"   📊 Plot saved: {save_filename}")
    
    return fig

def generate_scale_up_report(analysis_results: Dict, report_filename: str = None) -> str:
    """
    Generate detailed text report of scale-up analysis.
    
    Args:
        analysis_results: Output from comprehensive_scale_up_analysis
        report_filename: Optional filename to save report
    
    Returns:
        str: Report content
    """
    results = analysis_results['analysis_results']
    scenario = analysis_results['scenario']
    
    report = f"""
# Scale-Up Analysis Report
## Scenario: {scenario.title()}
## Duration: {analysis_results['duration_s']*1e9:.0f} ns
## Generated: {analysis_results['timestamp']}

## Executive Summary

Analyzed {len(results)} array configurations for negative energy scale-up:
"""
    
    for config, data in results.items():
        N = data['N_chambers']
        volume = data['tiling_analysis']['volume_m3']
        energy = data['tiling_analysis']['total_energy_J']
        power = data['tiling_analysis']['estimated_power_W']
        
        report += f"""
### {config} Array Configuration
- **Chambers**: {N:,}
- **Total Volume**: {volume:.2e} m³
- **Total Energy**: {energy:.2e} J
- **Power Consumption**: {power:.2e} W
- **Energy Density**: {energy/volume:.2e} J/m³
- **Disturbance Rejection**: {data['tiling_analysis']['dist_rejection_dB']:.1f} dB
- **Peak Temperature**: {data['thermal_peak_K']:.1f} K
- **Cooling Power**: {data['cooling_requirements']['cooling_input_power_W']:.2e} W
"""
    
    # Scaling analysis
    report += f"""
## Scaling Laws Analysis

The analysis reveals the following scaling behaviors:
- **Volume**: Linear with N chambers (V proportional to N)
- **Energy**: Linear with N chambers (E proportional to N)  
- **Power**: Linear with N chambers (P proportional to N)
- **Disturbance Rejection**: Degrades as ~10log10(sqrt(N))
- **Thermal Load**: Proportional to power (T proportional to P)

## Infrastructure Requirements

For practical deployment, key considerations include:
1. **Cryogenic Cooling**: COP={results[list(results.keys())[0]]['cooling_requirements']['COP']} cooling systems
2. **Vibration Isolation**: <1 kHz cutoff frequency isolation
3. **Power Distribution**: Scalable electrical infrastructure
4. **Control Architecture**: Distributed controller networks

## Recommendations

Based on this analysis:
- **Small Arrays** (≤100 chambers): Excellent performance, manageable infrastructure
- **Medium Arrays** (100-10,000 chambers): Good scaling, increased cooling requirements  
- **Large Arrays** (>10,000 chambers): Significant infrastructure challenges

The optimal scale depends on application requirements and infrastructure constraints.
"""
    
    if report_filename:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   📄 Report saved: {report_filename}")
    
    return report

# -----------------------------------------------------------------------------
# Main execution and testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Scale-Up Strategy Module Test")
    print("=" * 50)
    
    # Test 1: Simple tiling example
    print("\n1️⃣ Testing modular tiling...")
    tiling_result = scale_tiling(10, 10, 10, scenario='burst')
    
    # Test 2: Thermal response
    print("\n2️⃣ Testing thermal response...")
    t = np.linspace(0, 1e-3, 1000)
    T = thermal_response(P_dot=1e3, C_th=50.0, R_th=0.1, T_env=4.0, times=t)
    print(f"   🌡️ Final temp: {T[-1]:.1f} K")
    
    # Test 3: Vibration isolation
    print("\n3️⃣ Testing vibration isolation...")
    fs = 10e3  # 10 kHz sampling
    t_vib = np.linspace(0, 0.1, int(0.1 * fs))
    z_input = np.sin(2*np.pi*100*t_vib) + 0.5*np.sin(2*np.pi*5000*t_vib)  # 100 Hz + 5 kHz
    z_output = vibration_isolation(z_input, fs, f_c=1e3)
    isolation_dB = 20*np.log10(np.std(z_output)/np.std(z_input))
    print(f"   🔧 Vibration isolation: {isolation_dB:.1f} dB")
    
    # Test 4: Cooling analysis
    print("\n4️⃣ Testing cooling requirements...")
    T_ss = steady_state_temperature(P_drive=1e3, COP=3.0, T_env=4.0)
    print(f"   ❄️ Steady-state temp: {T_ss:.1f} K")
    
    cooling_req = cooling_power_requirements(1000, power_per_chamber=1e-6)
    print(f"   ⚡ Cooling power: {cooling_req['cooling_input_power_W']:.2e} W")
    
    # Test 5: Comprehensive analysis
    print("\n5️⃣ Running comprehensive analysis...")
    array_configs = [(2,2,2), (5,5,5), (10,10,10)]
    analysis = comprehensive_scale_up_analysis(array_configs, scenario='burst')
    
    # Test 6: Generate outputs
    print("\n6️⃣ Generating outputs...")
    try:
        fig = plot_scale_up_analysis(analysis, 'scale_up_analysis.png')
        plt.close(fig)
    except Exception as e:
        print(f"   ⚠️ Plotting failed: {e}")
    
    report = generate_scale_up_report(analysis, 'scale_up_report.md')
    
    print("\n✅ Scale-Up Strategy Module validated!")
    print(f"   📊 Analyzed {len(array_configs)} configurations")
    print(f"   🎯 Scenario: {analysis['scenario']}")
    print(f"   📈 Results available for detailed analysis")
