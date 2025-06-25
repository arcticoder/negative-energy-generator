#!/usr/bin/env python3
"""
Ultra-High Energy Cosmic Ray (UHECR) Lorentz Invariance Violation Simulator
==========================================================================

Simulates LIV-shifted photopion production threshold for UHECR protons
interacting with CMB photons via the GZK cutoff mechanism.

Mathematical Foundation:
- Modified dispersion: E² = p²c² + m²c⁴ ± ξ(p^n c^n)/(E_LIV^(n-2))
- Threshold condition: p + γ_CMB → p + π⁰
- Shifted threshold: E_th = E_th0 × [1 + η(E_th/E_LIV)^(n-2)]
- CMB photon energy: ε_CMB ≈ 6×10⁻⁴ eV

Physical Parameters:
- Proton mass: m_p = 938.3 MeV
- Pion mass: m_π = 135.0 MeV  
- CMB temperature: T_CMB = 2.725 K
- Planck scale: E_Planck ≈ 1.22×10¹⁹ GeV

LIV Phenomenology:
- Superluminal (ξ > 0): Higher threshold, extended GZK cutoff
- Subluminal (ξ < 0): Lower threshold, enhanced UHECR suppression
- Order n: Power law dependence on energy scale
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings

# Physical constants
c = 2.99792458e8           # Speed of light (m/s)
h_bar = 1.054571817e-34    # Reduced Planck constant (J⋅s)
k_B = 1.380649e-23         # Boltzmann constant (J/K)
eV = 1.602176634e-19       # Electron volt (J)

# Particle masses (eV)
m_p = 0.938272e9           # Proton mass
m_pi = 0.13498e9           # Neutral pion mass
m_e = 0.5109989e6          # Electron mass

# CMB parameters
T_CMB = 2.72548            # CMB temperature (K)
epsilon_cmb = 2.7 * k_B * T_CMB / eV  # Peak CMB photon energy ≈ 6×10⁻⁴ eV

# LIV scales and parameters
E_PLANCK = 1.22e19 * 1e9   # Planck energy (eV)
E_LIV_DEFAULT = 1e28       # Default LIV scale (eV) - just below Planck
QG_SCALE = 1e16 * 1e9      # Quantum gravity scale (eV)

# GZK cutoff parameters
GZK_THRESHOLD_STANDARD = 5e19 * eV / eV  # Standard GZK threshold (eV)

class UHECRLIVSimulator:
    """
    Simulator for UHECR propagation with Lorentz Invariance Violation.
    
    Implements modified dispersion relations and threshold calculations
    for photopion production in cosmic ray interactions.
    """
    
    def __init__(self, E_LIV: float = E_LIV_DEFAULT, order: int = 3, 
                 xi: float = 1.0, cmb_energy: float = epsilon_cmb):
        """
        Initialize UHECR LIV simulator.
        
        Args:
            E_LIV: LIV energy scale (eV)
            order: LIV correction order (n)
            xi: Dimensionless LIV coefficient
            cmb_energy: CMB photon energy (eV)
        """
        self.E_LIV = E_LIV
        self.n = order
        self.xi = xi
        self.epsilon_cmb = cmb_energy
        
        # Calculate standard threshold
        self.E_th_standard = self._standard_threshold()
        
        print(f"🚀 UHECR LIV Simulator Initialized")
        print(f"   • LIV scale: E_LIV = {E_LIV:.2e} eV ({E_LIV/E_PLANCK:.1e} × E_Planck)")
        print(f"   • LIV order: n = {order}")
        print(f"   • LIV strength: ξ = {xi}")
        print(f"   • CMB energy: ε_CMB = {cmb_energy*1e3:.1f} meV")
        print(f"   • Standard threshold: {self.E_th_standard:.2e} eV")
    
    def _standard_threshold(self) -> float:
        """
        Calculate standard photopion production threshold.
        
        Threshold condition: s = (p_p + p_γ)² = (m_p + m_π)²
        In lab frame: E_th = [(m_π + m_p)² - m_p²] / (4ε_CMB)
        
        Returns:
            Standard threshold energy (eV)
        """
        invariant_mass_squared = (m_pi + m_p)**2 - m_p**2
        threshold = invariant_mass_squared / (4 * self.epsilon_cmb)
        return threshold
    
    def modified_threshold(self, xi_sign: int = 1) -> float:
        """
        Calculate LIV-modified photopion threshold.
        
        Mathematical model:
        E_th = E_th0 × [1 + η(E_th/E_LIV)^(n-2)]
        where η = ±ξ(n-1) and sign determines superluminal/subluminal
        
        Args:
            xi_sign: +1 for superluminal, -1 for subluminal LIV
            
        Returns:
            Modified threshold energy (eV)
        """
        # LIV correction parameter
        eta = xi_sign * self.xi * (self.n - 1)
        
        # Iterative solution for self-consistent threshold
        E_th = self.E_th_standard  # Initial guess
        
        for iteration in range(10):  # Converges quickly
            correction = eta * (E_th / self.E_LIV)**(self.n - 2)
            E_th_new = self.E_th_standard * (1 + correction)
            
            if abs(E_th_new - E_th) / E_th < 1e-10:
                break
            E_th = E_th_new
        
        return E_th
    
    def threshold_shift_ratio(self, xi_sign: int = 1) -> float:
        """
        Calculate relative threshold shift: ΔE_th / E_th0
        
        Args:
            xi_sign: +1 for superluminal, -1 for subluminal
            
        Returns:
            Relative threshold shift
        """
        E_th_modified = self.modified_threshold(xi_sign)
        return (E_th_modified - self.E_th_standard) / self.E_th_standard
    
    def gzk_cutoff_position(self, xi_sign: int = 1) -> float:
        """
        Estimate GZK cutoff position with LIV corrections.
        
        Args:
            xi_sign: LIV sign
            
        Returns:
            Modified GZK cutoff energy (eV)
        """
        # Simplified model: cutoff scales with threshold
        threshold_ratio = self.modified_threshold(xi_sign) / self.E_th_standard
        return GZK_THRESHOLD_STANDARD * threshold_ratio
    
    def propagation_length(self, energy: float, xi_sign: int = 1) -> float:
        """
        Calculate UHECR propagation length with LIV.
        
        Simplified model based on threshold modification.
        
        Args:
            energy: Proton energy (eV)
            xi_sign: LIV sign
            
        Returns:
            Propagation length (Mpc)
        """
        E_th = self.modified_threshold(xi_sign)
        
        if energy < E_th:
            # Below threshold: limited by Hubble distance
            return 4000  # Mpc (Hubble distance ≈ 4 Gpc)
        else:
            # Above threshold: energy-dependent interaction length
            # Simplified scaling: λ ∝ (E_th/E)²
            baseline_length = 50  # Mpc at standard threshold
            scaling_factor = (E_th / energy)**2
            return baseline_length * max(scaling_factor, 0.01)  # Minimum 0.5 Mpc

def scan_liv_parameter_space(energy_scales: List[float], orders: List[int], 
                           strengths: List[float]) -> Dict:
    """
    Comprehensive scan of LIV parameter space.
    
    Args:
        energy_scales: List of E_LIV values (eV)
        orders: List of LIV orders n
        strengths: List of LIV strengths ξ
        
    Returns:
        Dictionary with scan results
    """
    print("🔍 Scanning LIV Parameter Space")
    print("=" * 35)
    
    results = {
        'energy_scales': energy_scales,
        'orders': orders,
        'strengths': strengths,
        'threshold_shifts': {},
        'gzk_modifications': {},
        'propagation_changes': {}
    }
    
    scan_count = len(energy_scales) * len(orders) * len(strengths)
    current_scan = 0
    
    for E_LIV in energy_scales:
        for n in orders:
            for xi in strengths:
                current_scan += 1
                
                # Create simulator for this parameter point
                sim = UHECRLIVSimulator(E_LIV=E_LIV, order=n, xi=xi)
                
                # Calculate observables for both LIV signs
                for xi_sign, label in [(+1, 'superluminal'), (-1, 'subluminal')]:
                    key = f"E_LIV_{E_LIV:.0e}_n_{n}_xi_{xi}_{label}"
                    
                    # Threshold modifications
                    threshold_shift = sim.threshold_shift_ratio(xi_sign)
                    results['threshold_shifts'][key] = threshold_shift
                    
                    # GZK cutoff modifications
                    gzk_cutoff = sim.gzk_cutoff_position(xi_sign)
                    results['gzk_modifications'][key] = gzk_cutoff
                    
                    # Propagation length at 10^20 eV
                    test_energy = 1e20 * eV / eV
                    prop_length = sim.propagation_length(test_energy, xi_sign)
                    results['propagation_changes'][key] = prop_length
                
                if current_scan % max(1, scan_count // 10) == 0:
                    progress = current_scan / scan_count * 100
                    print(f"   Progress: {progress:.0f}% ({current_scan}/{scan_count})")
    
    return results

def benchmark_liv_scenarios() -> Dict:
    """
    Benchmark specific LIV scenarios of experimental interest.
    
    Returns:
        Dictionary with benchmark results
    """
    print("🎯 Benchmarking LIV Scenarios")
    print("=" * 30)
    
    scenarios = {
        'quantum_gravity': {
            'E_LIV': QG_SCALE,
            'order': 1,
            'xi': 1.0,
            'description': 'Linear quantum gravity corrections'
        },
        'string_theory': {
            'E_LIV': E_PLANCK,
            'order': 2,
            'xi': 0.1,
            'description': 'Quadratic string theory modifications'
        },
        'rainbow_gravity': {
            'E_LIV': 1e24,  # eV
            'order': 3,
            'xi': 1.0,
            'description': 'Cubic rainbow gravity effects'
        },
        'phenomenological': {
            'E_LIV': 1e26,  # eV
            'order': 4,
            'xi': 0.5,
            'description': 'Generic high-order phenomenology'
        }
    }
    
    benchmark_results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\n   📊 Scenario: {scenario_name}")
        print(f"      {params['description']}")
        
        sim = UHECRLIVSimulator(
            E_LIV=params['E_LIV'],
            order=params['order'],
            xi=params['xi']
        )
        
        # Calculate key observables
        results = {}
        
        for xi_sign, label in [(+1, 'superluminal'), (-1, 'subluminal')]:
            threshold_shift = sim.threshold_shift_ratio(xi_sign)
            gzk_cutoff = sim.gzk_cutoff_position(xi_sign)
            
            results[label] = {
                'threshold_shift_percent': threshold_shift * 100,
                'gzk_cutoff_eV': gzk_cutoff,
                'gzk_shift_percent': (gzk_cutoff / GZK_THRESHOLD_STANDARD - 1) * 100
            }
            
            print(f"      {label.capitalize()}: "
                  f"Δ_th = {threshold_shift*100:+.2f}%, "
                  f"GZK = {gzk_cutoff:.2e} eV")
        
        benchmark_results[scenario_name] = {
            'parameters': params,
            'results': results
        }
    
    return benchmark_results

def visualize_liv_effects(parameter_scan: Dict, benchmark: Dict, 
                         save_filename: str = 'uhecr_liv_analysis.png'):
    """
    Create comprehensive visualization of LIV effects on UHECR propagation.
    
    Args:
        parameter_scan: Results from scan_liv_parameter_space
        benchmark: Results from benchmark_liv_scenarios
        save_filename: Output filename for plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('UHECR Lorentz Invariance Violation Analysis', fontsize=16)
    
    # Plot 1: Threshold shift vs LIV energy scale
    ax1.set_title('Photopion Threshold Shifts')
    ax1.set_xlabel('LIV Energy Scale (eV)')
    ax1.set_ylabel('Relative Threshold Shift')
    ax1.set_xscale('log')
    ax1.set_yscale('symlog', linthresh=1e-10)
    
    # Extract data for fixed order and strength
    energy_scales = parameter_scan['energy_scales']
    threshold_data_super = []
    threshold_data_sub = []
    
    for E_LIV in energy_scales:
        key_super = f"E_LIV_{E_LIV:.0e}_n_3_xi_1.0_superluminal"
        key_sub = f"E_LIV_{E_LIV:.0e}_n_3_xi_1.0_subluminal"
        
        if key_super in parameter_scan['threshold_shifts']:
            threshold_data_super.append(parameter_scan['threshold_shifts'][key_super])
            threshold_data_sub.append(parameter_scan['threshold_shifts'][key_sub])
    
    if threshold_data_super:
        ax1.loglog(energy_scales[:len(threshold_data_super)], 
                   np.abs(threshold_data_super), 'b-o', 
                   label='Superluminal (ξ > 0)', markersize=4)
        ax1.loglog(energy_scales[:len(threshold_data_sub)], 
                   np.abs(threshold_data_sub), 'r-s', 
                   label='Subluminal (ξ < 0)', markersize=4)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GZK cutoff modification
    ax2.set_title('GZK Cutoff Energy Shifts')
    ax2.set_xlabel('LIV Energy Scale (eV)')
    ax2.set_ylabel('GZK Cutoff Energy (eV)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    gzk_data_super = []
    gzk_data_sub = []
    
    for E_LIV in energy_scales:
        key_super = f"E_LIV_{E_LIV:.0e}_n_3_xi_1.0_superluminal"
        key_sub = f"E_LIV_{E_LIV:.0e}_n_3_xi_1.0_subluminal"
        
        if key_super in parameter_scan['gzk_modifications']:
            gzk_data_super.append(parameter_scan['gzk_modifications'][key_super])
            gzk_data_sub.append(parameter_scan['gzk_modifications'][key_sub])
    
    if gzk_data_super:
        ax2.loglog(energy_scales[:len(gzk_data_super)], gzk_data_super, 
                   'b-o', label='Superluminal', markersize=4)
        ax2.loglog(energy_scales[:len(gzk_data_sub)], gzk_data_sub, 
                   'r-s', label='Subluminal', markersize=4)
    
    # Standard GZK line
    ax2.axhline(y=GZK_THRESHOLD_STANDARD, color='k', linestyle='--', 
                alpha=0.7, label='Standard GZK')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Benchmark scenario comparison
    ax3.set_title('Benchmark LIV Scenarios')
    ax3.set_ylabel('Threshold Shift (%)')
    
    scenario_names = list(benchmark.keys())
    x_pos = np.arange(len(scenario_names))
    width = 0.35
    
    super_shifts = []
    sub_shifts = []
    
    for scenario in scenario_names:
        super_shifts.append(benchmark[scenario]['results']['superluminal']['threshold_shift_percent'])
        sub_shifts.append(benchmark[scenario]['results']['subluminal']['threshold_shift_percent'])
    
    ax3.bar(x_pos - width/2, super_shifts, width, label='Superluminal', alpha=0.8)
    ax3.bar(x_pos + width/2, sub_shifts, width, label='Subluminal', alpha=0.8)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([name.replace('_', ' ').title() for name in scenario_names], 
                        rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy dependence of propagation length
    ax4.set_title('UHECR Propagation Length vs Energy')
    ax4.set_xlabel('Proton Energy (eV)')
    ax4.set_ylabel('Propagation Length (Mpc)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    # Test energies
    test_energies = np.logspace(19, 21, 20)
    
    # Calculate for benchmark quantum gravity scenario
    qg_params = benchmark['quantum_gravity']['parameters']
    sim_qg = UHECRLIVSimulator(
        E_LIV=qg_params['E_LIV'],
        order=qg_params['order'],
        xi=qg_params['xi']
    )
    
    standard_lengths = [sim_qg.propagation_length(E, 0) for E in test_energies]
    super_lengths = [sim_qg.propagation_length(E, +1) for E in test_energies]
    sub_lengths = [sim_qg.propagation_length(E, -1) for E in test_energies]
    
    ax4.loglog(test_energies, standard_lengths, 'k-', 
               label='Standard', linewidth=2)
    ax4.loglog(test_energies, super_lengths, 'b--', 
               label='Superluminal LIV', linewidth=2)
    ax4.loglog(test_energies, sub_lengths, 'r:', 
               label='Subluminal LIV', linewidth=2)
    
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"   📊 Visualization saved: {save_filename}")
    
    return fig

def main():
    """Main execution function for UHECR LIV analysis."""
    print("🌌 UHECR Lorentz Invariance Violation Simulator")
    print("=" * 50)
    
    # Basic demonstration
    print("\n1️⃣ Basic LIV threshold calculation...")
    sim = UHECRLIVSimulator(E_LIV=1e28, order=3, xi=1.0)
    
    for sign, label in [(+1, 'Superluminal'), (-1, 'Subluminal')]:
        E_th = sim.modified_threshold(sign)
        shift = sim.threshold_shift_ratio(sign)
        print(f"   {label} LIV: E_th = {E_th:.2e} eV "
              f"(shift: {shift*100:+.3f}%)")
    
    # Parameter space scan
    print("\n2️⃣ Parameter space scan...")
    energy_scales = np.logspace(24, 30, 7)  # 10^24 to 10^30 eV
    orders = [1, 2, 3, 4]
    strengths = [0.1, 1.0]
    
    scan_results = scan_liv_parameter_space(energy_scales, orders, strengths)
    
    # Benchmark scenarios
    print("\n3️⃣ Benchmark scenarios...")
    benchmark_results = benchmark_liv_scenarios()
    
    # Generate visualization
    print("\n4️⃣ Creating visualization...")
    fig = visualize_liv_effects(scan_results, benchmark_results)
    plt.close(fig)
    
    print("\n✅ UHECR LIV Analysis Complete!")
    print(f"   • Scanned {len(energy_scales)*len(orders)*len(strengths)} parameter combinations")
    print(f"   • Benchmarked {len(benchmark_results)} scenarios")
    print(f"   • Threshold shifts: {scan_results['threshold_shifts']}")
    
    return {
        'simulator': sim,
        'scan_results': scan_results,
        'benchmark_results': benchmark_results
    }

if __name__ == "__main__":
    results = main()
