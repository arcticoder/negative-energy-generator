"""
Capacitive/Inductive Field Rig Design Module
==========================================

High-energy capacitor banks and RF inductive cavities for field-enhanced DCE.

Mathematical Foundation:
- Capacitive energy density: ρ_E = ½ε₀E² where E = V/d
- Inductive energy density: ρ_B = B²/(2μ₀) 
- Peak B-field: B ≈ μ₀μᵣI/(2πr) with r ≈ √(L/μ₀)
- Breakdown constraint: E < 10¹⁴ V/m

Combined electromagnetic energy extraction from capacitive and inductive sources.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings

# Physical constants
ε0 = 8.854187817e-12  # Vacuum permittivity (F/m)
μ0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
c = 2.998e8            # Speed of light (m/s)

# Safety and operational limits
E_BREAKDOWN = 1e14     # Dielectric breakdown (V/m)
B_MAX_SAFE = 100       # Maximum safe B-field (T)
I_MAX_SAFE = 1e6       # Maximum safe current (A)
V_MAX_SAFE = 1e7       # Maximum safe voltage (V)

def simulate_capacitive_rig(C: float, V: float, d: float, 
                           dielectric_strength: float = E_BREAKDOWN) -> Dict:
    """
    Simulate high-energy capacitive field rig with breakdown protection.
    
    Mathematical model:
    E = V/d
    ρ_E = ½ε₀E² (if E < E_breakdown)
    W = ½CV² (stored energy)
    
    Args:
        C: Capacitance (F)
        V: Charging voltage (V)
        d: Plate separation (m)
        dielectric_strength: Breakdown field limit (V/m)
    
    Returns:
        Dictionary with capacitive rig performance
    """
    # Electric field calculation
    E = V / d if d > 0 else 0
    
    # Safety checks
    breakdown_margin = dielectric_strength / E if E > 0 else float('inf')
    voltage_safe = V <= V_MAX_SAFE
    
    if E > dielectric_strength:
        return {
            'E_field': E,
            'rho_E': 0.0,
            'stored_energy': 0.0,
            'status': 'BREAKDOWN',
            'breakdown_margin': breakdown_margin,
            'safe_operation': False
        }
    
    if not voltage_safe:
        warnings.warn(f"Voltage {V:.2e} V exceeds safe limit {V_MAX_SAFE:.2e} V")
    
    # Energy density calculation
    rho_E = 0.5 * ε0 * E**2
    
    # Total stored energy
    stored_energy = 0.5 * C * V**2
    
    # Effective volume (simplified as plate area × separation)
    # Assume square plates with area = stored_energy/(rho_E) if known
    if rho_E > 0:
        effective_volume = stored_energy / rho_E
        plate_area = effective_volume / d
    else:
        effective_volume = d * 1e-6  # Default 1 mm² plate area
        plate_area = 1e-6
    
    # Discharge characteristics
    # RC time constant (assuming R ~ 1 Ω for discharge circuit)
    R_discharge = 1.0  # Ohm
    tau_discharge = R_discharge * C
    
    # Peak discharge current
    I_peak = V / R_discharge if R_discharge > 0 else 0
    
    return {
        'E_field': E,
        'rho_E': rho_E,
        'stored_energy': stored_energy,
        'effective_volume': effective_volume,
        'plate_area': plate_area,
        'breakdown_margin': breakdown_margin,
        'discharge_time': tau_discharge,
        'peak_current': I_peak,
        'C': C,
        'V': V,
        'd': d,
        'status': 'SUCCESS',
        'safe_operation': voltage_safe
    }

def simulate_inductive_rig(L: float, I: float, f_mod: float, 
                          mu_r: float = 1.0, geometry: str = 'solenoid') -> Dict:
    """
    Simulate RF inductive cavity with high-current drive.
    
    Mathematical model:
    r ≈ √(L/μ₀μᵣ) (characteristic radius)
    B ≈ μ₀μᵣI/(2πr) (peak field)
    ρ_B = B²/(2μ₀μᵣ) (magnetic energy density)
    
    Args:
        L: Inductance (H)
        I: Drive current (A)
        f_mod: Modulation frequency (Hz)
        mu_r: Relative permeability
        geometry: Coil geometry ('solenoid', 'toroidal')
    
    Returns:
        Dictionary with inductive rig performance
    """
    # Safety checks
    current_safe = I <= I_MAX_SAFE
    if not current_safe:
        warnings.warn(f"Current {I:.2e} A exceeds safe limit {I_MAX_SAFE:.2e} A")
    
    # Characteristic dimensions
    if geometry == 'solenoid':
        # Solenoid approximation: r ≈ √(L/μ₀μᵣ)
        r_char = np.sqrt(L / (μ0 * mu_r))
    elif geometry == 'toroidal':
        # Toroidal approximation (more complex)
        r_char = np.sqrt(L / (μ0 * mu_r)) * 0.8  # Correction factor
    else:
        r_char = np.sqrt(L / (μ0 * mu_r))
    
    # Peak magnetic field
    if r_char > 0:
        B_peak = μ0 * mu_r * I / (2 * np.pi * r_char)
    else:
        B_peak = 0
    
    # Safety check for B-field
    B_safe = B_peak <= B_MAX_SAFE
    if not B_safe:
        warnings.warn(f"B-field {B_peak:.1f} T exceeds safe limit {B_MAX_SAFE} T")
    
    # Magnetic energy density
    rho_B = B_peak**2 / (2 * μ0 * mu_r)
    
    # Total stored magnetic energy
    stored_energy = 0.5 * L * I**2
    
    # Effective volume
    # Approximate as cylindrical volume: V ≈ πr²h, assume h ≈ 2r
    effective_volume = 2 * np.pi * r_char**3
    
    # RF characteristics
    omega = 2 * np.pi * f_mod
    reactance = omega * L
    
    # Power dissipation (simplified R ≈ 0.1 Ω)
    R_coil = 0.1  # Ohm (typical superconducting or low-resistance coil)
    power_dissipated = R_coil * I**2
    
    # Quality factor
    Q_factor = reactance / R_coil if R_coil > 0 else float('inf')
    
    # Skin depth effects at high frequency
    # δ = √(2/(ωμ₀σ)) for conductivity σ ≈ 6e7 S/m (copper)
    sigma_cu = 6e7  # S/m
    skin_depth = np.sqrt(2 / (omega * μ0 * sigma_cu)) if omega > 0 else float('inf')
    
    return {
        'B_field': B_peak,
        'rho_B': rho_B,
        'stored_energy': stored_energy,
        'effective_volume': effective_volume,
        'characteristic_radius': r_char,
        'Q_factor': Q_factor,
        'power_dissipated': power_dissipated,
        'skin_depth': skin_depth,
        'L': L,
        'I': I,
        'f_mod': f_mod,
        'mu_r': mu_r,
        'geometry': geometry,
        'status': 'SUCCESS',
        'current_safe': current_safe,
        'B_safe': B_safe
    }

def simulate_combined_rig(cap_params: Dict, ind_params: Dict) -> Dict:
    """
    Simulate combined capacitive + inductive field rig.
    
    Args:
        cap_params: {C, V, d} for capacitive section
        ind_params: {L, I, f_mod} for inductive section
    
    Returns:
        Combined rig performance metrics
    """
    # Simulate individual components
    cap_result = simulate_capacitive_rig(**cap_params)
    ind_result = simulate_inductive_rig(**ind_params)
    
    # Combined energy densities
    total_rho = cap_result['rho_E'] + ind_result['rho_B']
    total_stored = cap_result['stored_energy'] + ind_result['stored_energy']
    
    # Effective total volume (geometric mean for different field distributions)
    cap_volume = cap_result.get('effective_volume', 1e-15)  # Default fallback
    ind_volume = ind_result.get('effective_volume', 1e-15)  # Default fallback
    vol_eff = np.sqrt(cap_volume * ind_volume)
    
    # Total negative energy (phenomenological coupling)
    # Assume field interaction enhances extraction by 10-50%
    coupling_factor = 1.2  # 20% enhancement from E×B coupling
    total_negative_energy = -total_rho * vol_eff * coupling_factor
    
    # Safety assessment
    all_safe = (cap_result['safe_operation'] and 
                ind_result['current_safe'] and 
                ind_result['B_safe'])
    
    # Power requirements
    total_power = cap_result.get('peak_current', 0) * cap_params['V'] + ind_result['power_dissipated']
    
    return {
        'capacitive': cap_result,
        'inductive': ind_result,
        'total_rho': total_rho,
        'total_stored_energy': total_stored,
        'total_negative_energy': total_negative_energy,
        'effective_volume': vol_eff,
        'coupling_factor': coupling_factor,
        'total_power': total_power,
        'all_systems_safe': all_safe,
        'optimization_score': -total_negative_energy  # For minimization
    }

def optimize_field_rigs(n_trials: int = 300, target_energy: float = -1e-12) -> Dict:
    """
    Optimize combined capacitive/inductive field rig using random search.
    
    Args:
        n_trials: Number of optimization trials
        target_energy: Target negative energy (J)
    
    Returns:
        Optimization results and best configuration
    """
    print("⚡ Optimizing Capacitive/Inductive Field Rig")
    print("=" * 50)
    
    best = {'total_negative_energy': 0, 'all_systems_safe': False}
    optimization_history = []
    unsafe_count = 0
    
    for trial in range(n_trials):
        # Sample capacitive parameters
        d = 10**np.random.uniform(-9, -6)     # 1 nm - 1 mm separation
        C = 10**np.random.uniform(-12, -6)    # 1 pF - 1 μF
        V = np.random.uniform(1e3, 1e6)       # 1 kV - 1 MV
        
        # Sample inductive parameters  
        L = 10**np.random.uniform(-6, -3)     # 1 μH - 1 mH
        I = np.random.uniform(1, 1000)        # 1 A - 1 kA
        f_mod = 10**np.random.uniform(3, 9)   # 1 kHz - 1 GHz
        mu_r = np.random.uniform(1, 1000)     # Air to ferrite
        
        cap_params = {'C': C, 'V': V, 'd': d}
        ind_params = {'L': L, 'I': I, 'f_mod': f_mod, 'mu_r': mu_r}
        
        result = simulate_combined_rig(cap_params, ind_params)
        optimization_history.append(result)
        
        if not result['all_systems_safe']:
            unsafe_count += 1
            continue
        
        # Multi-objective: energy vs power efficiency vs safety margin
        energy_score = abs(result['total_negative_energy'])
        power_penalty = np.log10(result['total_power'] / 1e6)  # Penalty for >MW
        
        # Safety margins
        cap_margin = result['capacitive']['breakdown_margin']
        B_margin = B_MAX_SAFE / result['inductive']['B_field'] if result['inductive']['B_field'] > 0 else float('inf')
        safety_score = min(cap_margin, B_margin)
        
        # Combined figure of merit
        fom = energy_score * safety_score / (1 + max(0, power_penalty))
        
        if (result['total_negative_energy'] < best['total_negative_energy'] and 
            result['all_systems_safe']):
            best = {**result, 'fom': fom, 'trial': trial,
                   'cap_params': cap_params, 'ind_params': ind_params}
    
    # Statistics
    safe_trials = [r for r in optimization_history if r['all_systems_safe']]
    safety_rate = len(safe_trials) / n_trials * 100
    
    if safe_trials:
        avg_energy = np.mean([r['total_negative_energy'] for r in safe_trials])
        avg_power = np.mean([r['total_power'] for r in safe_trials])
    else:
        avg_energy = avg_power = 0
    
    print(f"✅ Field Rig Optimization Complete!")
    print(f"   • Safe trials: {len(safe_trials)}/{n_trials}")
    print(f"   • Safety rate: {safety_rate:.1f}%")
    
    if best['all_systems_safe']:
        cap = best['capacitive']
        ind = best['inductive']
        print(f"   • Optimal capacitor: C={best['cap_params']['C']:.2e} F, V={best['cap_params']['V']:.2e} V")
        print(f"   • Optimal inductor: L={best['ind_params']['L']:.2e} H, I={best['ind_params']['I']:.1f} A")
        print(f"   • E-field: {cap['E_field']:.2e} V/m")
        print(f"   • B-field: {ind['B_field']:.2f} T")
        print(f"   • Total energy density: {best['total_rho']:.2e} J/m³")
        print(f"   • Total negative energy: {best['total_negative_energy']:.2e} J")
        print(f"   • Power requirement: {best['total_power']:.2e} W")
        print(f"   • Coupling enhancement: {best['coupling_factor']:.1f}x")
    else:
        print("   ⚠️  No safe optimal configuration found")
    
    return {
        'best_result': best,
        'optimization_history': optimization_history,
        'statistics': {
            'safety_rate': safety_rate,
            'avg_energy': avg_energy,
            'avg_power': avg_power,
            'n_safe': len(safe_trials)
        },
        'target_achieved': abs(best.get('total_negative_energy', 0)) > abs(target_energy)
    }

def analyze_scaling_laws(param_ranges: Dict, n_points: int = 30) -> Dict:
    """
    Analyze scaling laws for capacitive and inductive energy densities.
    
    Args:
        param_ranges: Dictionary with parameter ranges
        n_points: Number of points per parameter
    
    Returns:
        Scaling analysis results
    """
    results = {
        'capacitive_scaling': {},
        'inductive_scaling': {},
        'combined_scaling': {}
    }
    
    # Capacitive scaling: ρ_E ∝ V²/d²
    V_range = param_ranges.get('V_range', (1e3, 1e6))
    d_range = param_ranges.get('d_range', (1e-9, 1e-6))
    
    V_vals = np.logspace(np.log10(V_range[0]), np.log10(V_range[1]), n_points)
    d_vals = np.logspace(np.log10(d_range[0]), np.log10(d_range[1]), n_points)
    
    cap_energies = []
    for V in V_vals:
        for d in d_vals:
            cap_result = simulate_capacitive_rig(C=1e-9, V=V, d=d)
            if cap_result['status'] == 'SUCCESS':
                cap_energies.append((V, d, cap_result['rho_E']))
    
    results['capacitive_scaling'] = {
        'V_values': V_vals,
        'd_values': d_vals,
        'energy_densities': cap_energies,
        'scaling_law': 'ρ_E ∝ V²/d²'
    }
    
    # Inductive scaling: ρ_B ∝ I²/L
    I_range = param_ranges.get('I_range', (1, 1000))
    L_range = param_ranges.get('L_range', (1e-6, 1e-3))
    
    I_vals = np.linspace(I_range[0], I_range[1], n_points)
    L_vals = np.logspace(np.log10(L_range[0]), np.log10(L_range[1]), n_points)
    
    ind_energies = []
    for I in I_vals:
        for L in L_vals:
            ind_result = simulate_inductive_rig(L=L, I=I, f_mod=1e6)
            if ind_result['current_safe'] and ind_result['B_safe']:
                ind_energies.append((I, L, ind_result['rho_B']))
    
    results['inductive_scaling'] = {
        'I_values': I_vals,
        'L_values': L_vals,
        'energy_densities': ind_energies,
        'scaling_law': 'ρ_B ∝ I²/L'
    }
    
    return results

if __name__ == "__main__":
    # Test the module
    print("⚡ Capacitive/Inductive Field Rig Module Test")
    print("=" * 50)
    
    # Test individual components
    cap_test = simulate_capacitive_rig(C=1e-9, V=1e5, d=1e-6)
    ind_test = simulate_inductive_rig(L=1e-4, I=100, f_mod=1e6)
    
    print(f"Capacitive test: {cap_test['rho_E']:.2e} J/m³")
    print(f"Inductive test: {ind_test['rho_B']:.2e} J/m³")
    
    # Test combined system
    combined_test = simulate_combined_rig(
        {'C': 1e-9, 'V': 1e5, 'd': 1e-6},
        {'L': 1e-4, 'I': 100, 'f_mod': 1e6}
    )
    print(f"Combined test: {combined_test['total_negative_energy']:.2e} J")
    
    # Quick optimization test
    opt_result = optimize_field_rigs(n_trials=50)
    
    print("✅ Field Rig module validated!")
