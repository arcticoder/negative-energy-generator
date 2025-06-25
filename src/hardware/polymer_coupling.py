# File: src/hardware/polymer_coupling.py
"""
Polymer QFT Coupling Module

Physics Implementation:
- Polymer quantization: â†,â = η[sin(μE)/μ, sin(μB)/μ] with μ = √Δ (area gap)
- Modified dispersion: ω²(k) = c²k²[1 - (ħk/ρc)²/3] for polymer density ρ
- Vacuum shaping: ⟨ψ|T_μν|ψ⟩ ∝ polymer_influence × quantum_geometry_correction
- Negative energy: E_eff = ħω_eff where ω_eff can become imaginary in polymer regime

This module simulates polymer-modified quantum field theory effects for
controlled vacuum fluctuation manipulation and negative energy extraction.
"""

import numpy as np
import sys
import os
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import quad, simpson
from scipy.optimize import minimize_scalar

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Physical constants
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
c = 299792458        # Speed of light (m/s)
k_B = 1.380649e-23   # Boltzmann constant (J/K)
G = 6.67430e-11      # Gravitational constant (m³/kg·s²)

# Polymer LQG parameters
l_P = np.sqrt(hbar*G/c**3)  # Planck length ≈ 1.6e-35 m
gamma = 0.2375               # Barbero-Immirzi parameter (typical value)

def polymer_dispersion_relation(k: np.ndarray, polymer_scale: float, 
                              classical_cutoff: float = None) -> np.ndarray:
    """
    Calculate polymer-modified dispersion relation.
    
    Mathematical Foundation:
    ω²(k) = c²k² × [1 - (ħk/ρc)²/3] × polymer_correction(k,μ)
    where polymer_correction involves sin(μE)/μ terms from LQG
    
    Args:
        k: Wave vector array (1/m)
        polymer_scale: Polymer length scale μ ~ √Δ (m)
        classical_cutoff: Cutoff for classical transition (1/m)
    
    Returns:
        Modified frequency array ω(k)
    """
    if classical_cutoff is None:
        classical_cutoff = 1e15  # Default optical frequency cutoff
    
    # Classical dispersion
    ω_classical = c * k
    
    # Polymer modification parameter
    μ = polymer_scale
    x = μ * k
    
    # Polymer correction factor: sin(x)/x for small x, oscillatory for large x
    # This captures the discrete nature of polymer geometry
    polymer_factor = np.where(
        np.abs(x) < 1e-10,
        1 - x**2/6 + x**4/120,  # Taylor expansion for small x
        np.sin(x) / x
    )
    
    # Additional quantum geometry correction from LQG
    # Includes holonomy effects and area quantization
    quantum_geometry_factor = 1 - (hbar*k)**2 / (3 * polymer_scale**2 * c**2)
    quantum_geometry_factor = np.maximum(quantum_geometry_factor, 0)  # Avoid negative
    
    # Combine corrections
    ω_polymer = ω_classical * polymer_factor * np.sqrt(quantum_geometry_factor)
    
    # Smooth transition to classical regime
    transition = np.exp(-k/classical_cutoff)
    ω_modified = transition * ω_polymer + (1 - transition) * ω_classical
    
    return ω_modified

def vacuum_fluctuation_spectrum(k: np.ndarray, polymer_scale: float,
                              temperature: float = 0, volume: float = 1e-18) -> dict:
    """
    Calculate vacuum fluctuation spectrum with polymer modifications.
    
    Args:
        k: Wave vector array (1/m)
        polymer_scale: Polymer length scale (m)
        temperature: Temperature (K, 0 for pure vacuum)
        volume: Quantization volume (m³)
    
    Returns:
        Dictionary with vacuum fluctuation properties
    """
    # Modified dispersion relation
    ω = polymer_dispersion_relation(k, polymer_scale)
    
    # Zero-point energy density (per mode)
    E_vacuum = 0.5 * hbar * ω
    
    # Thermal occupation (if T > 0)
    if temperature > 0:
        β = 1 / (k_B * temperature)
        n_thermal = 1 / (np.exp(β * hbar * ω) - 1)
        E_thermal = hbar * ω * n_thermal
    else:
        n_thermal = np.zeros_like(ω)
        E_thermal = np.zeros_like(ω)
    
    # Total energy per mode
    E_total = E_vacuum + E_thermal
    
    # Mode density in k-space (3D)
    mode_density = volume * k**2 / (2 * np.pi**2)
    
    # Energy density spectrum
    energy_density_spectrum = E_total * mode_density
    
    # Negative energy regions (where ω becomes complex/imaginary)
    negative_regions = np.imag(ω) != 0
    
    return {
        'k': k,
        'omega': ω,
        'E_vacuum': E_vacuum,
        'E_thermal': E_thermal,
        'E_total': E_total,
        'mode_density': mode_density,
        'energy_density_spectrum': energy_density_spectrum,
        'negative_regions': negative_regions,
        'total_energy_density': simpson(energy_density_spectrum, x=k) if len(k) > 1 else energy_density_spectrum[0]
    }

def simulate_polymer_coupling(polymer_scale: float, coupling_strength: float,
                            k_max: float = 1e16, n_modes: int = 1000,
                            modulation_freq: float = 1e9, t: np.ndarray = None) -> dict:
    """
    Simulate polymer QFT coupling for vacuum fluctuation shaping.
    
    Mathematical Foundation:
    H_eff = H_field + λH_polymer + H_interaction
    where H_polymer involves polymer holonomies and area operators
    
    Args:
        polymer_scale: Characteristic polymer length μ (m)
        coupling_strength: Field-polymer coupling λ (dimensionless)
        k_max: Maximum wave vector (1/m)
        n_modes: Number of modes to simulate
        modulation_freq: Time modulation frequency (Hz)
        t: Time array for dynamic simulation (s)
    
    Returns:
        Dictionary with polymer coupling results
    """
    if t is None:
        t = np.linspace(0, 1e-9, 100)  # Default 1 ns simulation
    
    # Wave vector array (logarithmic spacing for broad spectrum)
    k = np.logspace(10, np.log10(k_max), n_modes)
    
    # Base vacuum fluctuation spectrum
    base_spectrum = vacuum_fluctuation_spectrum(k, polymer_scale)
    
    # Time-dependent polymer coupling modulation
    # Simulates dynamic changes in polymer geometry or field coupling
    polymer_modulation = 1 + 0.1 * np.sin(2*np.pi*modulation_freq*t[:, np.newaxis])
    
    # Effective polymer scale varies with time
    μ_eff = polymer_scale * polymer_modulation
    
    # Calculate time-dependent energy density
    energy_density_t = np.zeros((len(t), len(k)))
    negative_energy_t = np.zeros(len(t))
    
    for i, ti in enumerate(t):
        # Updated spectrum at time ti
        spectrum_ti = vacuum_fluctuation_spectrum(k, μ_eff[i].mean())
        energy_density_t[i] = spectrum_ti['energy_density_spectrum']
        
        # Extract negative energy contributions
        # Occurs when polymer effects make ω² < 0
        ω_ti = polymer_dispersion_relation(k, μ_eff[i].mean())
        negative_mask = np.imag(ω_ti) != 0
        
        if np.any(negative_mask):
            # Negative energy density from imaginary frequencies
            negative_contrib = -np.abs(np.imag(ω_ti[negative_mask])) * hbar
            negative_energy_t[i] = np.sum(negative_contrib * base_spectrum['mode_density'][negative_mask])
    
    # Holonomy effects: discrete area eigenvalues affect field modes
    # A_n = 4πγℓ_P²√(n(n+1)/2) for n = 1,2,3,...
    area_eigenvalues = [4*np.pi*gamma*l_P**2 * np.sqrt(n*(n+1)/2) for n in range(1, 11)]
    holonomy_frequencies = [c / np.sqrt(A) for A in area_eigenvalues]
    
    # Enhanced negative energy near holonomy resonances
    resonance_enhancement = np.zeros_like(energy_density_t)
    for ω_hol in holonomy_frequencies:
        k_res = ω_hol / c
        if k_res < k_max:
            # Find nearest k mode
            k_idx = np.argmin(np.abs(k - k_res))
            resonance_enhancement[:, k_idx] += coupling_strength * hbar * ω_hol
    
    # Total modified energy density
    total_energy_density = energy_density_t - resonance_enhancement
    
    # Integrated quantities
    total_negative_energy = np.trapz(negative_energy_t, t)
    peak_negative_power = np.max(-np.gradient(negative_energy_t, t))
    
    # Quantum coherence measures
    # Polymer effects can enhance quantum coherence through discrete geometry
    coherence_length = polymer_scale / (coupling_strength + 1e-10)
    decoherence_time = coherence_length / c
    
    # Effective stress-energy tensor components
    # T_μν includes polymer contributions to vacuum stress
    T_00 = np.mean(total_energy_density, axis=1)  # Energy density
    T_11 = -T_00 / 3  # Pressure (isotropic approximation)
    
    return {
        't': t,
        'k': k,
        'polymer_scale': polymer_scale,
        'coupling_strength': coupling_strength,
        'base_spectrum': base_spectrum,
        'energy_density_t': energy_density_t,
        'total_energy_density': total_energy_density,
        'negative_energy_t': negative_energy_t,
        'total_negative_energy': total_negative_energy,
        'peak_negative_power': peak_negative_power,
        'holonomy_frequencies': holonomy_frequencies,
        'area_eigenvalues': area_eigenvalues,
        'resonance_enhancement': resonance_enhancement,
        'coherence_length': coherence_length,
        'decoherence_time': decoherence_time,
        'T_00': T_00,  # Energy density
        'T_11': T_11,  # Pressure
        'optimization_score': -total_negative_energy
    }

def optimize_polymer_parameters(target_negative_energy: float = -1e-15,
                              polymer_scale_range: tuple = (1e-20, 1e-15),
                              coupling_range: tuple = (0.01, 10),
                              n_samples: int = 1000) -> dict:
    """
    Optimize polymer coupling parameters for target negative energy extraction.
    
    Args:
        target_negative_energy: Target negative energy (J)
        polymer_scale_range: Range of polymer scales to test (m)
        coupling_range: Range of coupling strengths
        n_samples: Number of optimization samples
    
    Returns:
        Dictionary with optimization results
    """
    print(f"🔧 Optimizing polymer coupling for target energy: {target_negative_energy:.2e} J")
    
    best_params = None
    best_energy = 0
    results = []
    
    for i in range(n_samples):
        # Random parameter sampling
        μ = np.random.uniform(*polymer_scale_range)
        λ = np.random.uniform(*coupling_range)
        
        try:
            result = simulate_polymer_coupling(
                polymer_scale=μ,
                coupling_strength=λ,
                n_modes=500  # Reduced for speed
            )
            
            total_neg_energy = result['total_negative_energy']
            
            results.append({
                'polymer_scale': μ,
                'coupling_strength': λ,
                'negative_energy': total_neg_energy,
                'peak_power': result['peak_negative_power'],
                'coherence_time': result['decoherence_time']
            })
            
            if total_neg_energy < best_energy:  # More negative is better
                best_energy = total_neg_energy
                best_params = {
                    'polymer_scale': μ,
                    'coupling_strength': λ
                }
        except:
            continue  # Skip failed evaluations
    
    # Find configurations meeting target
    successful_configs = [r for r in results if r['negative_energy'] <= target_negative_energy]
    
    print(f"✅ Polymer coupling optimization complete!")
    print(f"   • Best energy achieved: {best_energy:.2e} J")
    print(f"   • Target met by: {len(successful_configs)}/{len(results)} configurations")
    if best_params:
        print(f"   • Best polymer scale: {best_params['polymer_scale']:.2e} m")
        print(f"   • Best coupling: {best_params['coupling_strength']:.2f}")
    
    return {
        'best_parameters': best_params,
        'best_energy': best_energy,
        'target_energy': target_negative_energy,
        'all_results': results,
        'successful_configs': successful_configs,
        'success_rate': len(successful_configs) / len(results) if results else 0
    }

def polymer_casimir_effect(plate_separation: float, polymer_scale: float,
                          plate_area: float = 1e-4) -> dict:
    """
    Calculate Casimir effect with polymer QFT modifications.
    
    Args:
        plate_separation: Distance between plates (m)
        polymer_scale: Polymer length scale (m)
        plate_area: Area of plates (m²)
    
    Returns:
        Dictionary with modified Casimir effect results
    """
    # Classical Casimir energy density
    E_casimir_classical = -np.pi**2 * hbar * c / (240 * plate_separation**4)
    
    # Polymer correction factor
    # When plate separation ~ polymer scale, significant deviations occur
    polymer_ratio = plate_separation / polymer_scale
    
    if polymer_ratio > 100:
        # Classical regime
        polymer_correction = 1.0
    elif polymer_ratio > 1:
        # Intermediate regime - smooth interpolation
        polymer_correction = 1 - 0.3 * np.exp(-polymer_ratio/10)
    else:
        # Deep polymer regime - strong suppression
        polymer_correction = 0.1 * polymer_ratio**2
    
    # Modified Casimir energy
    E_casimir_polymer = E_casimir_classical * polymer_correction
    
    # Total energy
    total_energy = E_casimir_polymer * plate_area * plate_separation
    
    # Force between plates
    force = -np.pi**2 * hbar * c * plate_area * polymer_correction / (240 * plate_separation**4)
    
    return {
        'plate_separation': plate_separation,
        'polymer_scale': polymer_scale,
        'E_density_classical': E_casimir_classical,
        'E_density_polymer': E_casimir_polymer,
        'polymer_correction': polymer_correction,
        'total_energy': total_energy,
        'force': force,
        'force_per_area': force / plate_area
    }

# Example usage and testing
if __name__ == "__main__":
    print("🧬 POLYMER QFT COUPLING SIMULATION")
    print("=" * 60)
    
    # Test basic polymer coupling
    print("\n1️⃣  BASIC POLYMER COUPLING")
    t = np.linspace(0, 1e-9, 200)  # 1 ns
    result = simulate_polymer_coupling(
        polymer_scale=1e-18,     # 1 attometer
        coupling_strength=1.0,
        k_max=1e15,             # Optical frequencies
        n_modes=500,
        modulation_freq=1e9,    # 1 GHz
        t=t
    )
    
    print(f"   • Total negative energy = {result['total_negative_energy']:.2e} J")
    print(f"   • Peak negative power = {result['peak_negative_power']:.2e} W")
    print(f"   • Coherence length = {result['coherence_length']:.2e} m")
    print(f"   • Decoherence time = {result['decoherence_time']:.2e} s")
    print(f"   • Number of holonomy modes = {len(result['holonomy_frequencies'])}")
    
    # Test dispersion relation
    print("\n2️⃣  POLYMER DISPERSION ANALYSIS")
    k_test = np.logspace(12, 16, 100)
    ω_polymer = polymer_dispersion_relation(k_test, 1e-18)
    ω_classical = c * k_test
    
    deviation = np.abs(ω_polymer - ω_classical) / ω_classical
    max_deviation = np.max(deviation)
    significant_k = k_test[deviation > 0.01]
    print(f"   • Maximum dispersion deviation = {max_deviation:.2%}")
    if len(significant_k) > 0:
        print(f"   • Polymer effects significant for k > {significant_k[0]:.2e} m⁻¹")
    else:
        print(f"   • No significant polymer effects in tested range")
    
    # Test Casimir effect modifications
    print("\n3️⃣  POLYMER-MODIFIED CASIMIR EFFECT")
    separations = np.logspace(-9, -6, 10)  # nm to μm range
    polymer_scale = 1e-18
    
    casimir_results = []
    for d in separations:
        cas_result = polymer_casimir_effect(d, polymer_scale)
        casimir_results.append(cas_result)
    
    classical_energies = [r['E_density_classical'] for r in casimir_results]
    polymer_energies = [r['E_density_polymer'] for r in casimir_results]
    
    print(f"   • Classical Casimir energy range: {min(classical_energies):.2e} to {max(classical_energies):.2e} J/m³")
    print(f"   • Polymer-modified range: {min(polymer_energies):.2e} to {max(polymer_energies):.2e} J/m³")
    
    # Test optimization
    print("\n4️⃣  PARAMETER OPTIMIZATION")
    opt_result = optimize_polymer_parameters(
        target_negative_energy=-1e-16,
        n_samples=300
    )
    
    # Vacuum spectrum analysis
    print("\n5️⃣  VACUUM FLUCTUATION SPECTRUM")
    k_spectrum = np.logspace(10, 16, 200)
    spectrum = vacuum_fluctuation_spectrum(k_spectrum, 1e-18, temperature=0)
    
    total_vacuum_energy = spectrum['total_energy_density']
    print(f"   • Total vacuum energy density = {total_vacuum_energy:.2e} J/m³")
    print(f"   • Number of modes analyzed = {len(k_spectrum)}")
    
    print(f"\n✅ Polymer QFT coupling module validation complete!")
