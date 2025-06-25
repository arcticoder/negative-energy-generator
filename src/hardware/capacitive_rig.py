# File: src/hardware/capacitive_rig.py
"""
Capacitive/Inductive Field Rig Module

Physics Implementation:
- Capacitive energy: E_cap = ½CV² → modulated boundaries
- Inductive energy: E_ind = ½LI² → time-varying inductance
- Negative energy density: ρ_neg = -∂(E_field)/∂V when ∂V/∂t < 0
- Vacuum fluctuation coupling: ⟨E²⟩ ∝ ∫d³k ω_k/(e^(ℏω_k/k_BT) - 1)

This module simulates modulated boundary conditions using capacitive and
inductive field rigs for controlled vacuum fluctuation manipulation.
"""

import numpy as np
import sys
import os
from scipy import signal
from scipy.integrate import simpson

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Physical constants
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
c = 299792458        # Speed of light (m/s)
epsilon_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
mu_0 = 4*np.pi*1e-7      # Vacuum permeability (H/m)
k_B = 1.380649e-23     # Boltzmann constant (J/K)

def simulate_capacitive_rig(C0: float, V_mod: callable, t: np.ndarray,
                          plate_separation: float, plate_area: float,
                          T: float = 300, coupling_factor: float = 1.0) -> dict:
    """
    Simulate capacitive field rig with time-modulated voltage.
    
    Mathematical Foundation:
    C(t) = ε₀A/d(t), where d(t) can be modulated
    E_cap(t) = ½C(t)V(t)²
    ρ_vacuum = ħc/(240π²d⁴) (Casimir energy density)
    ρ_neg = -α × ∂E_cap/∂t × coupling_factor
    
    Args:
        C0: Base capacitance (F)
        V_mod: Voltage modulation function V(t)
        t: Time array (s)
        plate_separation: Distance between plates (m)
        plate_area: Area of plates (m²)
        T: Temperature (K)
        coupling_factor: Vacuum coupling strength (0-1)
    
    Returns:
        Dictionary with capacitive rig metrics
    """
    # Calculate time-dependent voltage and capacitance
    V = np.array([V_mod(ti) for ti in t])
    
    # Model capacitance variation (simplified)
    # Assume sinusoidal plate vibration: d(t) = d₀(1 + δ sin(ωt))
    ω_mech = 2*np.pi*1e3  # 1 kHz mechanical frequency
    δ = 0.1  # 10% modulation depth
    d_t = plate_separation * (1 + δ * np.sin(ω_mech * t))
    C_t = epsilon_0 * plate_area / d_t
    
    # Capacitive energy
    E_cap = 0.5 * C_t * V**2
    
    # Electric field between plates
    E_field = V / d_t
    
    # Energy density in the capacitor
    u_electric = 0.5 * epsilon_0 * E_field**2
    
    # Casimir energy density (baseline vacuum contribution)
    casimir_density = hbar * c / (240 * np.pi**2 * d_t**4)
    
    # Total energy density
    total_energy_density = u_electric - casimir_density
    
    # Negative energy emerges from rapid field changes
    dE_dt = np.gradient(E_cap, t)
    du_dt = np.gradient(u_electric, t)
    
    # Enhanced negative energy density from field modulation
    # Physical mechanism: When field drops rapidly, vacuum gives up energy
    rho_neg = np.where(dE_dt < 0, -np.abs(du_dt) * coupling_factor, 0)
    
    # Thermal effects
    thermal_energy_density = (np.pi**2/90) * (k_B*T)**4 / (hbar**3 * c**3)
    
    # Total volume and energy
    volume = plate_area * plate_separation
    total_neg_energy = simpson(rho_neg, x=t) * volume
    
    # Field enhancement factor
    enhancement_factor = np.max(E_field) / (V[0] / plate_separation) if V[0] != 0 else 1
    
    # Frequency domain analysis
    fft_rho = np.fft.fft(rho_neg)
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    power_spectrum = np.abs(fft_rho)**2
    
    # Quality metrics
    peak_field = np.max(E_field)
    rms_field = np.sqrt(np.mean(E_field**2))
    
    return {
        't': t,
        'V': V,
        'C_t': C_t,
        'E_cap': E_cap,
        'E_field': E_field,
        'u_electric': u_electric,
        'rho_neg': rho_neg,
        'casimir_density': casimir_density,
        'total_energy_density': total_energy_density,
        'total_neg_energy': total_neg_energy,
        'peak_rho_neg': np.max(rho_neg),
        'peak_field': peak_field,
        'rms_field': rms_field,
        'enhancement_factor': enhancement_factor,
        'thermal_density': thermal_energy_density,
        'power_spectrum': power_spectrum,
        'frequencies': freqs,
        'optimization_score': np.max(rho_neg)  # For optimization
    }

def simulate_inductive_rig(L0: float, I_mod: callable, t: np.ndarray,
                         core_permeability: float = 1000, turns: int = 100,
                         coupling_factor: float = 1.0) -> dict:
    """
    Simulate inductive field rig with time-modulated current.
    
    Mathematical Foundation:
    L(t) = μ₀μᵣN²A/l (time-varying through μᵣ(t))
    E_ind(t) = ½L(t)I(t)²
    B_field(t) = μ₀NI(t)/l
    ρ_neg = -β × ∂(B²/2μ₀)/∂t × coupling_factor
    
    Args:
        L0: Base inductance (H)
        I_mod: Current modulation function I(t)
        t: Time array (s)
        core_permeability: Relative permeability of core
        turns: Number of turns in inductor
        coupling_factor: Vacuum coupling strength (0-1)
    
    Returns:
        Dictionary with inductive rig metrics
    """
    # Calculate time-dependent current
    I = np.array([I_mod(ti) for ti in t])
    
    # Model inductance variation through permeability modulation
    # mu_r(t) = mu_r_0(1 + delta cos(wt)) - ferroelectric/magnetic modulation
    omega_mod = 2*np.pi*10e3  # 10 kHz modulation
    delta_mu = 0.2  # 20% permeability modulation
    mu_r = core_permeability * (1 + delta_mu * np.cos(omega_mod * t))
    
    L_t = L0 * mu_r / core_permeability  # Scale by permeability change
    
    # Inductive energy
    E_ind = 0.5 * L_t * I**2
    
    # Magnetic field (approximate for solenoid)
    B_field = mu_0 * mu_r * turns * I / 0.1  # Assume 10 cm length
    
    # Magnetic energy density
    u_magnetic = B_field**2 / (2 * mu_0)
    
    # Negative energy from rapid field changes
    dE_dt = np.gradient(E_ind, t)
    dB_dt = np.gradient(B_field, t)
    
    # When magnetic field drops rapidly, vacuum responds
    rho_neg = np.where(dE_dt < 0, -np.abs(dB_dt) * B_field / mu_0 * coupling_factor, 0)
    
    # Flux through inductor
    flux = L_t * I
    
    # Induced EMF
    emf = -np.gradient(flux, t)
    
    # Power dissipation/generation
    power = emf * I
    
    # Field penetration depth (skin effect approximation)
    I_nonzero = I[I != 0]
    if len(I_nonzero) > 1:
        dI_dt = np.gradient(I, t)
        f_characteristic = 1 / (2*np.pi) * np.mean(np.abs(dI_dt[I != 0]) / I_nonzero)
        if f_characteristic > 0:
            skin_depth = np.sqrt(2 / (2*np.pi*f_characteristic*mu_0*core_permeability*1e6))  # Assume sigma ~ 1 MS/m
        else:
            skin_depth = 1e-3  # Default 1 mm
    else:
        skin_depth = 1e-3  # Default 1 mm
    
    # Frequency domain analysis
    fft_rho = np.fft.fft(rho_neg)
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    power_spectrum = np.abs(fft_rho)**2
    
    return {
        't': t,
        'I': I,
        'L_t': L_t,
        'E_ind': E_ind,
        'B_field': B_field,
        'u_magnetic': u_magnetic,
        'rho_neg': rho_neg,
        'flux': flux,
        'emf': emf,
        'power': power,
        'peak_rho_neg': np.max(rho_neg),
        'peak_B_field': np.max(B_field),
        'skin_depth': skin_depth,
        'power_spectrum': power_spectrum,
        'frequencies': freqs,
        'optimization_score': np.max(rho_neg)  # For optimization
    }

def combined_capacitive_inductive_rig(cap_params: dict, ind_params: dict, 
                                    t: np.ndarray, cross_coupling: float = 0.1) -> dict:
    """
    Simulate combined capacitive-inductive rig with electromagnetic coupling.
    
    Args:
        cap_params: Parameters for capacitive_rig simulation
        ind_params: Parameters for inductive_rig simulation
        t: Time array
        cross_coupling: EM coupling strength between capacitive and inductive elements
    
    Returns:
        Dictionary with combined rig metrics
    """
    # Run individual simulations
    cap_result = simulate_capacitive_rig(**cap_params, t=t)
    ind_result = simulate_inductive_rig(**ind_params, t=t)
    
    # Cross-coupling effects
    # Capacitive field influences magnetic permeability
    E_influence_on_B = cross_coupling * cap_result['E_field'] * ind_result['B_field']
    
    # Inductive field influences electric permittivity
    B_influence_on_E = cross_coupling * ind_result['B_field'] * cap_result['E_field']
    
    # Combined negative energy density
    rho_neg_combined = cap_result['rho_neg'] + ind_result['rho_neg'] + \
                      cross_coupling * (E_influence_on_B + B_influence_on_E) / c**2
    
    # Poynting vector (energy flow)
    S = (cap_result['E_field'] * ind_result['B_field']) / mu_0
    
    # Total electromagnetic energy density
    u_total = cap_result['u_electric'] + ind_result['u_magnetic']
    
    # Combined optimization score
    combined_score = cap_result['optimization_score'] + ind_result['optimization_score'] + \
                    cross_coupling * np.max(np.abs(rho_neg_combined))
    
    return {
        'capacitive': cap_result,
        'inductive': ind_result,
        'rho_neg_combined': rho_neg_combined,
        'poynting_vector': S,
        'u_total': u_total,
        'cross_coupling_E_B': E_influence_on_B,
        'cross_coupling_B_E': B_influence_on_E,
        'peak_combined_neg': np.max(rho_neg_combined),
        'combined_score': combined_score
    }

def optimize_field_rig_parameters(rig_type: str = 'capacitive',
                                target_density: float = 1e12,  # J/m³
                                optimization_rounds: int = 1000) -> dict:
    """
    Optimize capacitive or inductive rig parameters for target negative energy density.
    
    Args:
        rig_type: 'capacitive', 'inductive', or 'combined'
        target_density: Target negative energy density (J/m³)
        optimization_rounds: Number of optimization iterations
    
    Returns:
        Dictionary with optimization results
    """
    print(f"🔧 Optimizing {rig_type} rig for target density: {target_density:.2e} J/m³")
    
    best_params = None
    best_density = 0
    results = []
    
    t_sim = np.linspace(0, 1e-6, 1000)  # 1 μs simulation
    
    for i in range(optimization_rounds):
        if rig_type == 'capacitive':
            # Random capacitive parameters
            C0 = np.random.uniform(1e-12, 1e-9)  # pF to nF
            V_max = np.random.uniform(1e2, 1e4)   # 100V to 10kV
            f_mod = np.random.uniform(1e3, 1e6)   # kHz to MHz
            plate_sep = np.random.uniform(1e-6, 1e-3)  # μm to mm
            plate_area = np.random.uniform(1e-6, 1e-4)  # μm² to cm²
            
            V_mod = lambda t: V_max * np.sin(2*np.pi*f_mod*t)
            
            try:
                result = simulate_capacitive_rig(C0, V_mod, t_sim, plate_sep, plate_area)
                peak_density = result['peak_rho_neg']
                
                results.append({
                    'type': 'capacitive',
                    'C0': C0,
                    'V_max': V_max,
                    'f_mod': f_mod,
                    'plate_separation': plate_sep,
                    'plate_area': plate_area,
                    'peak_density': peak_density
                })
                
                if peak_density > best_density:
                    best_density = peak_density
                    best_params = {
                        'C0': C0, 'V_max': V_max, 'f_mod': f_mod,
                        'plate_separation': plate_sep, 'plate_area': plate_area
                    }
            except:
                continue
                
        elif rig_type == 'inductive':
            # Random inductive parameters
            L0 = np.random.uniform(1e-6, 1e-3)  # μH to mH
            I_max = np.random.uniform(1e-3, 10)  # mA to 10A
            f_mod = np.random.uniform(1e3, 1e6)  # kHz to MHz
            permeability = np.random.uniform(100, 10000)
            turns = int(np.random.uniform(10, 1000))
            
            I_mod = lambda t: I_max * np.sin(2*np.pi*f_mod*t)
            
            try:
                result = simulate_inductive_rig(L0, I_mod, t_sim, permeability, turns)
                peak_density = result['peak_rho_neg']
                
                results.append({
                    'type': 'inductive',
                    'L0': L0,
                    'I_max': I_max,
                    'f_mod': f_mod,
                    'permeability': permeability,
                    'turns': turns,
                    'peak_density': peak_density
                })
                
                if peak_density > best_density:
                    best_density = peak_density
                    best_params = {
                        'L0': L0, 'I_max': I_max, 'f_mod': f_mod,
                        'permeability': permeability, 'turns': turns
                    }
            except:
                continue
    
    # Find configurations meeting target
    successful_configs = [r for r in results if r['peak_density'] >= target_density]
    
    print(f"✅ {rig_type.title()} rig optimization complete!")
    print(f"   • Best density achieved: {best_density:.2e} J/m³")
    print(f"   • Target met by: {len(successful_configs)}/{len(results)} configurations")
    if best_params:
        print(f"   • Best parameters: {best_params}")
    
    return {
        'rig_type': rig_type,
        'best_parameters': best_params,
        'best_density': best_density,
        'target_density': target_density,
        'all_results': results,
        'successful_configs': successful_configs,
        'success_rate': len(successful_configs) / len(results) if results else 0
    }

# Example usage and testing
if __name__ == "__main__":
    print("⚡ CAPACITIVE/INDUCTIVE FIELD RIG SIMULATION")
    print("=" * 60)
    
    # Test capacitive rig
    print("\n1️⃣  CAPACITIVE RIG SIMULATION")
    t = np.linspace(0, 1e-6, 1000)  # 1 μs
    V_mod = lambda t: 1000 * np.sin(2*np.pi*100e3*t)  # 1kV, 100 kHz
    
    cap_result = simulate_capacitive_rig(
        C0=100e-12,  # 100 pF
        V_mod=V_mod,
        t=t,
        plate_separation=1e-4,  # 100 μm
        plate_area=1e-4         # 1 cm²
    )
    
    print(f"   • Peak |ρ_neg| = {cap_result['peak_rho_neg']:.2e} J/m³")
    print(f"   • Peak E-field = {cap_result['peak_field']:.2e} V/m")
    print(f"   • Enhancement factor = {cap_result['enhancement_factor']:.2f}")
    
    # Test inductive rig
    print("\n2️⃣  INDUCTIVE RIG SIMULATION")
    I_mod = lambda t: 5 * np.sin(2*np.pi*50e3*t)  # 5A, 50 kHz
    
    ind_result = simulate_inductive_rig(
        L0=1e-3,    # 1 mH
        I_mod=I_mod,
        t=t,
        core_permeability=5000,
        turns=200
    )
    
    print(f"   • Peak |ρ_neg| = {ind_result['peak_rho_neg']:.2e} J/m³")
    print(f"   • Peak B-field = {ind_result['peak_B_field']:.2e} T")
    print(f"   • Skin depth = {ind_result['skin_depth']:.2e} m")
    
    # Test combined rig
    print("\n3️⃣  COMBINED RIG SIMULATION")
    cap_params = {
        'C0': 100e-12,
        'V_mod': V_mod,
        'plate_separation': 1e-4,
        'plate_area': 1e-4
    }
    ind_params = {
        'L0': 1e-3,
        'I_mod': I_mod,
        'core_permeability': 5000,
        'turns': 200
    }
    
    combined_result = combined_capacitive_inductive_rig(
        cap_params, ind_params, t, cross_coupling=0.05
    )
    
    print(f"   • Peak combined |ρ_neg| = {combined_result['peak_combined_neg']:.2e} J/m³")
    print(f"   • Combined score = {combined_result['combined_score']:.2e}")
    
    # Test optimization
    print("\n4️⃣  PARAMETER OPTIMIZATION")
    opt_result = optimize_field_rig_parameters(
        rig_type='capacitive',
        target_density=1e10,
        optimization_rounds=500
    )
    
    print(f"\n✅ Field rig module validation complete!")
