#!/usr/bin/env python3
"""
Prototype Vacuum-Engineering Device Models
=========================================

When theory targets are met, this script provides models for:
1. Casimir-array demonstrator
2. Dynamic Casimir effect simulation
3. Squeezed-vacuum cavity design

These are the next steps when readiness assessment indicates
theory-to-prototype transition is ready.

Usage:
    python prototype_vacuum_devices.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson
import os

# Physical constants
HBAR = 1.054571817e-34  # J⋅s
C = 2.99792458e8        # m/s
PI = np.pi

class CasimirArrayDemonstrator:
    """Casimir array for negative energy demonstration."""
    
    def __init__(self):
        self.epsilon_0 = 8.854187817e-12  # F/m
        
    def casimir_energy(self, d):
        """
        Casimir energy between two plates:
        E(d) = -π²ℏc/(720d⁴)
        """
        return -PI**2 * HBAR * C / (720 * d**4)
    
    def casimir_force(self, d):
        """
        Casimir force: F = -dE/dd = -π²ℏc/(240d⁵)
        """
        return -PI**2 * HBAR * C / (240 * d**5)
    
    def casimir_pressure(self, d):
        """
        Casimir pressure (energy density):
        P = π²ℏc/(240d⁴)
        """
        return PI**2 * HBAR * C / (240 * d**4)
    
    def optimize_gap_spacing(self, d_min=1e-9, d_max=1e-6, n_points=100):
        """Find optimal gap spacing for maximum negative energy density."""
        
        d_values = np.logspace(np.log10(d_min), np.log10(d_max), n_points)
        energies = [self.casimir_energy(d) for d in d_values]
        pressures = [self.casimir_pressure(d) for d in d_values]
        
        # Find optimal spacing (most negative energy)
        min_idx = np.argmin(energies)
        optimal_d = d_values[min_idx]
        optimal_energy = energies[min_idx]
        optimal_pressure = pressures[min_idx]
        
        results = {
            'd_values': d_values,
            'energies': energies,
            'pressures': pressures,
            'optimal_spacing': optimal_d,
            'optimal_energy': optimal_energy,
            'optimal_pressure': optimal_pressure
        }
        
        return results
    
    def design_plate_array(self, n_plates=10, base_spacing=5e-9):
        """Design multi-plate Casimir array."""
        
        print(f"🔧 CASIMIR ARRAY DESIGN")
        print("-" * 25)
        print(f"   Number of plates: {n_plates}")
        print(f"   Base spacing: {base_spacing:.1e} m")
        
        # Calculate spacing for each gap
        spacings = [base_spacing * (1 + 0.1*i) for i in range(n_plates-1)]
        
        total_energy = 0
        total_force = 0
        
        print(f"   Gap analysis:")
        for i, d in enumerate(spacings):
            energy = self.casimir_energy(d)
            force = self.casimir_force(d)
            
            total_energy += energy
            total_force += force
            
            print(f"     Gap {i+1}: d={d:.1e} m, E={energy:.2e} J/m², F={force:.2e} N/m²")
        
        print(f"   Total array energy: {total_energy:.2e} J/m²")
        print(f"   Total array force: {total_force:.2e} N/m²")
        
        return {
            'n_plates': n_plates,
            'spacings': spacings,
            'total_energy': total_energy,
            'total_force': total_force,
            'energy_density': total_energy / sum(spacings)
        }

class DynamicCasimirSimulator:
    """Dynamic Casimir effect with time-varying boundaries."""
    
    def __init__(self):
        pass
    
    def time_varying_spacing(self, t, d0, A, omega):
        """
        Time-varying plate separation:
        d(t) = d₀ + A sin(ωt)
        """
        return d0 + A * np.sin(omega * t)
    
    def instantaneous_energy(self, d):
        """Instantaneous Casimir energy."""
        return -PI**2 * HBAR * C / (720 * d**4)
    
    def dynamic_energy_injection(self, d0, A, omega, T_period):
        """
        Calculate net energy injection over one period.
        """
        
        # Time grid over one period
        t_values = np.linspace(0, T_period, 1000)
        
        # Calculate instantaneous energies
        d_values = [self.time_varying_spacing(t, d0, A, omega) for t in t_values]
        energies = [self.instantaneous_energy(d) for d in d_values]
        
        # Net energy injection (integral over period)
        net_energy = simpson(energies, t_values) / T_period
        
        # Energy modulation amplitude
        energy_amplitude = (max(energies) - min(energies)) / 2
        
        return {
            't_values': t_values,
            'd_values': d_values,
            'energies': energies,
            'net_energy': net_energy,
            'energy_amplitude': energy_amplitude,
            'modulation_depth': energy_amplitude / abs(np.mean(energies))
        }
    
    def optimize_dynamic_parameters(self):
        """Optimize dynamic Casimir parameters for maximum energy injection."""
        
        print(f"⚡ DYNAMIC CASIMIR OPTIMIZATION")
        print("-" * 32)
        
        # Parameter ranges
        d0_values = np.logspace(-8, -6, 10)  # Base spacing
        A_ratios = np.linspace(0.1, 0.5, 5)  # Amplitude as fraction of d0
        frequencies = np.logspace(6, 12, 10)  # Modulation frequency
        
        best_result = {'net_energy': 0, 'params': None}
        
        for d0 in d0_values:
            for A_ratio in A_ratios:
                A = A_ratio * d0
                for omega in frequencies:
                    T_period = 2*PI / omega
                    
                    try:
                        result = self.dynamic_energy_injection(d0, A, omega, T_period)
                        
                        if abs(result['net_energy']) > abs(best_result['net_energy']):
                            best_result = {
                                'net_energy': result['net_energy'],
                                'params': {'d0': d0, 'A': A, 'omega': omega},
                                'modulation_depth': result['modulation_depth'],
                                'energy_amplitude': result['energy_amplitude']
                            }
                    except:
                        continue
        
        if best_result['params']:
            params = best_result['params']
            print(f"   Optimal parameters:")
            print(f"     Base spacing d₀: {params['d0']:.1e} m")
            print(f"     Amplitude A: {params['A']:.1e} m")
            print(f"     Frequency ω: {params['omega']:.1e} rad/s")
            print(f"   Results:")
            print(f"     Net energy injection: {best_result['net_energy']:.2e} J/m²")
            print(f"     Modulation depth: {best_result['modulation_depth']:.1%}")
            print(f"     Energy amplitude: {best_result['energy_amplitude']:.2e} J/m²")
        
        return best_result

class SqueezedVacuumCavity:
    """Squeezed vacuum state generation for negative energy."""
    
    def __init__(self):
        pass
    
    def squeezed_energy_density(self, r, xi, omega, sigma):
        """
        Energy density for squeezed vacuum:
        ρ(r) = -ℏω/2 sinh(2ξ) exp(-r²/σ²)
        """
        return -HBAR * omega / 2 * np.sinh(2*xi) * np.exp(-r**2/sigma**2)
    
    def cavity_finesse_factor(self, R1, R2, L, wavelength):
        """
        Cavity finesse for optical cavity:
        F = π√(R1*R2) / (1 - R1*R2)
        """
        if R1 * R2 >= 1:
            return float('inf')  # Perfect cavity
        
        return PI * np.sqrt(R1 * R2) / (1 - R1 * R2)
    
    def optimize_squeezed_parameters(self):
        """Optimize squeezed vacuum parameters."""
        
        print(f"🌀 SQUEEZED VACUUM OPTIMIZATION")
        print("-" * 31)
        
        # Parameter ranges
        xi_values = np.linspace(0.5, 3.0, 10)    # Squeezing parameter
        omega_values = np.logspace(14, 16, 10)   # Optical frequency
        sigma_values = np.logspace(-6, -4, 10)   # Beam waist
        
        # Target energy density
        target_density = -1e-10  # J/m³
        
        best_result = {'density': 0, 'params': None}
        
        for xi in xi_values:
            for omega in omega_values:
                for sigma in sigma_values:
                    # Calculate peak energy density (at r=0)
                    peak_density = self.squeezed_energy_density(0, xi, omega, sigma)
                    
                    if peak_density < best_result['density']:
                        best_result = {
                            'density': peak_density,
                            'params': {'xi': xi, 'omega': omega, 'sigma': sigma}
                        }
        
        if best_result['params']:
            params = best_result['params']
            print(f"   Optimal parameters:")
            print(f"     Squeezing parameter ξ: {params['xi']:.2f}")
            print(f"     Frequency ω: {params['omega']:.2e} rad/s")
            print(f"     Beam waist σ: {params['sigma']:.1e} m")
            print(f"   Results:")
            print(f"     Peak energy density: {best_result['density']:.2e} J/m³")
            print(f"     Target achieved: {'✅' if best_result['density'] <= target_density else '❌'}")
        
        return best_result
    
    def design_opo_cavity(self, target_xi=2.0):
        """Design optical parametric oscillator cavity for squeezing."""
        
        print(f"🔬 OPO CAVITY DESIGN")
        print("-" * 20)
        
        # Cavity parameters
        wavelength = 1064e-9  # Nd:YAG wavelength
        cavity_length = 0.1   # 10 cm cavity
        
        # Mirror reflectivities for target squeezing
        R1_values = np.linspace(0.95, 0.999, 20)
        R2_values = np.linspace(0.95, 0.999, 20)
        
        best_cavity = {'finesse': 0, 'params': None}
        
        for R1 in R1_values:
            for R2 in R2_values:
                finesse = self.cavity_finesse_factor(R1, R2, cavity_length, wavelength)
                
                # Estimate achievable squeezing (simplified model)
                max_squeezing = np.log(finesse / 100)  # Rough estimate
                
                if max_squeezing >= target_xi and finesse > best_cavity['finesse']:
                    best_cavity = {
                        'finesse': finesse,
                        'params': {'R1': R1, 'R2': R2, 'max_squeezing': max_squeezing},
                        'cavity_length': cavity_length,
                        'wavelength': wavelength
                    }
        
        if best_cavity['params']:
            params = best_cavity['params']
            print(f"   Cavity design:")
            print(f"     Length: {cavity_length:.2f} m")
            print(f"     Wavelength: {wavelength*1e9:.0f} nm")
            print(f"     Mirror R1: {params['R1']:.3f}")
            print(f"     Mirror R2: {params['R2']:.3f}")
            print(f"   Performance:")
            print(f"     Finesse: {best_cavity['finesse']:.1f}")
            print(f"     Max squeezing: {params['max_squeezing']:.2f}")
            print(f"     Target achieved: {'✅' if params['max_squeezing'] >= target_xi else '❌'}")
        
        return best_cavity

def main():
    """Main prototype device evaluation."""
    
    print("🔧 PROTOTYPE VACUUM-ENGINEERING DEVICES")
    print("=" * 45)
    print("Models for next-phase hardware implementation")
    print("when theory validation targets are met.\n")
    
    # 1. Casimir Array Demonstrator
    print("="*60)
    casimir_demo = CasimirArrayDemonstrator()
    
    # Optimize gap spacing
    opt_result = casimir_demo.optimize_gap_spacing()
    print(f"   Optimal gap spacing: {opt_result['optimal_spacing']:.1e} m")
    print(f"   Maximum negative energy: {opt_result['optimal_energy']:.2e} J/m²")
    print(f"   Pressure magnitude: {opt_result['optimal_pressure']:.2e} Pa")
    
    # Design multi-plate array
    array_design = casimir_demo.design_plate_array(n_plates=5, base_spacing=5e-9)
    print(f"   Array energy density: {array_design['energy_density']:.2e} J/m³")
    
    print()
    
    # 2. Dynamic Casimir Effect
    print("="*60)
    dynamic_casimir = DynamicCasimirSimulator()
    
    # Optimize dynamic parameters
    dynamic_result = dynamic_casimir.optimize_dynamic_parameters()
    
    print()
    
    # 3. Squeezed Vacuum Cavity
    print("="*60)
    squeezed_cavity = SqueezedVacuumCavity()
    
    # Optimize squeezed parameters
    squeezed_result = squeezed_cavity.optimize_squeezed_parameters()
    
    # Design OPO cavity
    opo_design = squeezed_cavity.design_opo_cavity(target_xi=2.0)
    
    print()
    
    # Summary and recommendations
    print("="*60)
    print("🚀 PROTOTYPE RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    # Casimir recommendation
    if abs(opt_result['optimal_energy']) > 1e-6:
        recommendations.append("✅ Casimir array: Promising for demonstration")
    else:
        recommendations.append("⚠️ Casimir array: May need larger arrays")
    
    # Dynamic Casimir recommendation  
    if dynamic_result['params'] and abs(dynamic_result['net_energy']) > 1e-8:
        recommendations.append("✅ Dynamic Casimir: Viable for energy injection")
    else:
        recommendations.append("⚠️ Dynamic Casimir: Requires higher frequencies")
    
    # Squeezed vacuum recommendation
    if squeezed_result['params'] and abs(squeezed_result['density']) > 1e-12:
        recommendations.append("✅ Squeezed vacuum: Achievable with OPO")
    else:
        recommendations.append("⚠️ Squeezed vacuum: Needs stronger squeezing")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Build Casimir demonstrator with {opt_result['optimal_spacing']:.0e} m gaps")
    print(f"   2. Test dynamic modulation at {dynamic_result['params']['omega']:.1e} rad/s")
    print(f"   3. Implement OPO with {opo_design['params']['R1']:.3f} reflectivity mirrors")
    print(f"   4. Measure negative energy densities > 10⁻¹⁰ J/m³")
    
    print("\n" + "="*60)
    print("Ready for experimental implementation when theory targets are met!")
    print("="*60)

if __name__ == "__main__":
    main()
