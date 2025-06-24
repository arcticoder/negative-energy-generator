#!/usr/bin/env python3
"""
Combined Prototype Integration
=============================

Unified vacuum generator integrating all four negative energy sources.

Components:
1. Casimir Array Demonstrator
2. Dynamic Casimir Cavity
3. Squeezed Vacuum Source  
4. Metamaterial Enhancement

Usage: Combined system for maximum negative energy generation.
"""

import numpy as np
from casimir_array import casimir_array_energy, CasimirArrayDemonstrator
from dynamic_casimir import dynamic_casimir_energy, DynamicCasimirCavity
from squeezed_vacuum import squeezed_vacuum_energy, SqueezedVacuumSource
from metamaterial import metamaterial_casimir_energy, MetamaterialEnhancer

class UnifiedVacuumGenerator:
    """
    Unified Vacuum Generator integrating all four negative energy sources.
    
    Combines:
    - Casimir arrays
    - Dynamic Casimir cavities
    - Squeezed vacuum states
    - Metamaterial enhancement
    """
    
    def __init__(self):
        """Initialize unified vacuum generator."""
        self.casimir_array = None
        self.dynamic_cavity = None
        self.squeezed_source = None
        self.metamaterial = None
        
        # Default test configuration
        self._setup_default_configuration()
    
    def _setup_default_configuration(self):
        """Set up default test configuration."""
        # Casimir array with optimized gaps
        gaps = np.array([6e-9, 7e-9, 8e-9, 7e-9, 6e-9])
        self.casimir_array = CasimirArrayDemonstrator(gaps)
        
        # Dynamic Casimir cavity
        d0 = 1e-6         # 1 μm mean gap
        omega = 1e12      # 1 THz modulation
        amplitude = 0.1 * d0  # 10% modulation
        self.dynamic_cavity = DynamicCasimirCavity(d0, omega, amplitude)
        
        # Squeezed vacuum source  
        pump_power = 1e-3      # 1 mW
        nonlinearity = 1e-12   # χ(2)
        finesse = 1000         # High-Q cavity
        self.squeezed_source = SqueezedVacuumSource(pump_power, nonlinearity, finesse)
        
        # Metamaterial enhancement
        epsilon_r = -2.5   # Negative permittivity
        mu_r = -1.8       # Negative permeability
        loss = 0.01       # 1% loss
        self.metamaterial = MetamaterialEnhancer(epsilon_r, mu_r, loss)
    
    def calculate_total_energy(self) -> float:
        """
        Calculate total combined energy density.
        
        Returns:
            Total energy density [J/m²]
        """
        total_energy = 0
        
        # Casimir array contribution
        if self.casimir_array:
            E_casimir = self.casimir_array.calculate_energy_density()
            total_energy += E_casimir
        
        # Dynamic Casimir contribution
        if self.dynamic_cavity:
            E_dynamic = self.dynamic_cavity.calculate_time_averaged_energy()
            total_energy += E_dynamic
        
        # Squeezed vacuum contribution (convert J/m³ to J/m² assuming 1 μm thickness)
        if self.squeezed_source:
            E_squeezed = self.squeezed_source.calculate_energy_density()
            thickness = 1e-6  # 1 μm thickness assumption
            total_energy += E_squeezed * thickness
        
        # Metamaterial enhancement (multiply Casimir by enhancement factor)
        if self.metamaterial and self.casimir_array:
            enhancement = self.metamaterial.calculate_enhancement()
            E_enhanced = self.casimir_array.calculate_energy_density() * (enhancement - 1)
            total_energy += E_enhanced
        
        return total_energy
    
    def estimate_power_output(self) -> float:
        """
        Estimate total power output.
        
        Returns:
            Power output [W]
        """
        total_power = 0
        
        # Dynamic Casimir photon production
        if self.dynamic_cavity:
            photon_rate = self.dynamic_cavity.calculate_photon_production()
            energy_per_photon = 1.055e-34 * self.dynamic_cavity.omega
            total_power += photon_rate * energy_per_photon
        
        # Squeezed vacuum power (rough estimate)
        if self.squeezed_source:
            total_power += self.squeezed_source.pump_power * 0.01  # 1% conversion efficiency
        
        return total_power
    
    def get_component_contributions(self) -> dict:
        """Get individual component energy contributions."""
        contributions = {}
        
        if self.casimir_array:
            contributions['casimir'] = self.casimir_array.calculate_energy_density()
        
        if self.dynamic_cavity:
            contributions['dynamic'] = self.dynamic_cavity.calculate_time_averaged_energy()
        
        if self.squeezed_source:
            E_squeezed = self.squeezed_source.calculate_energy_density()
            contributions['squeezed'] = E_squeezed * 1e-6  # Convert to J/m²
        
        if self.metamaterial and self.casimir_array:
            enhancement = self.metamaterial.calculate_enhancement()
            base_energy = self.casimir_array.calculate_energy_density()
            contributions['metamaterial'] = base_energy * (enhancement - 1)
        
        return contributions
    
    def optimize_combined_system(self) -> dict:
        """Optimize the combined system for maximum energy density."""
        # This is a simplified optimization - in practice would use
        # sophisticated multi-objective optimization
        
        current_energy = self.calculate_total_energy()
        contributions = self.get_component_contributions()
        
        recommendations = {}
        
        # Identify largest contributor
        largest_component = max(contributions, key=lambda k: abs(contributions[k]))
        
        recommendations['primary_focus'] = largest_component
        recommendations['current_energy'] = current_energy
        recommendations['contributions'] = contributions
        
        # Component-specific recommendations
        if abs(contributions.get('casimir', 0)) > abs(current_energy) * 0.3:
            recommendations['casimir'] = "Optimize gap configuration for smaller gaps"
        
        if abs(contributions.get('dynamic', 0)) > abs(current_energy) * 0.3:
            recommendations['dynamic'] = "Increase modulation frequency and depth"
        
        if abs(contributions.get('squeezed', 0)) > abs(current_energy) * 0.1:
            recommendations['squeezed'] = "Increase squeeze parameter and mode count"
        
        if abs(contributions.get('metamaterial', 0)) > abs(current_energy) * 0.2:
            recommendations['metamaterial'] = "Optimize for stronger negative index"
        
        return recommendations
    
    def parallel_prototyping_roadmap(self) -> dict:
        """
        Detailed parallel prototyping roadmap for each vacuum-engineering module.
        
        Returns optimization strategies while theory targets are being refined.
        """
        roadmap = {}
        
        # 2.1 Casimir Array Demonstrator
        roadmap['casimir_array'] = {
            'math': 'ρ_C(d_i) = -π²ℏc/(720 d_i⁴), E_C = Σ_i ρ_C(d_i) d_i',
            'current_gaps': list(self.casimir_array.gaps * 1e9),  # nm
            'optimization_strategy': 'Local gap pattern scanning',
            'next_steps': [
                'Generate gap patterns around sweet spot',
                'Explore ±0.5 nm variations systematically', 
                'Fabricate 1 cm² arrays with 5-10 nm precision',
                'Set up precision force measurement',
                'Target: |E| > 1 μJ/m² with <5% uncertainty'
            ],
            'scan_parameters': {
                'gap_range': '5-10 nm',
                'delta_scan': '±0.5 nm steps',
                'array_size': '1 cm²',
                'precision': '±0.1 nm'
            }
        }
        
        # 2.2 Dynamic Casimir Cavity  
        roadmap['dynamic_casimir'] = {
            'math': 'd(t) = d₀ + A sin(ωt), Ē_C = (ω/2π) ∫ -π²ℏc/[720d(t)⁴] dt',
            'current_params': {
                'd0': f"{self.dynamic_cavity.d0 * 1e6:.1f} μm",
                'amplitude': f"{self.dynamic_cavity.amplitude/self.dynamic_cavity.d0*100:.0f}%",
                'frequency': f"{self.dynamic_cavity.omega/1e12:.1f} THz"
            },
            'optimization_strategy': 'Amplitude and frequency sweeping',
            'next_steps': [
                'Sweep modulation 5-20% amplitude',
                'Explore 0.5-2 THz frequency range',
                'Build precision actuator systems',
                'Implement GHz-THz modulation capability',
                'Target: Measurable photon production'
            ],
            'scan_parameters': {
                'amplitude_range': '5-20%',
                'frequency_range': '0.5-2 THz',
                'time_resolution': 'sub-ps',
                'stability': '<1% RMS'
            }
        }
        
        # 2.3 Squeezed Vacuum Source
        roadmap['squeezed_vacuum'] = {
            'math': 'ρ_sq = -Σⱼ (ℏωⱼ)/(2Vⱼ) sinh(2rⱼ)',
            'current_params': {
                'pump_power': f"{self.squeezed_source.pump_power * 1e3:.0f} mW",
                'finesse': self.squeezed_source.cavity_finesse,
                'nonlinearity': f"{self.squeezed_source.nonlinearity:.1e}"
            },
            'optimization_strategy': 'Squeezing parameter optimization',
            'next_steps': [
                'Optimize squeezing parameter r = 0.5-2.0 rad',
                'Build optical parametric oscillator (OPO)',
                'Implement high-Q cavity (F > 1000)',
                'Multi-mode squeezing implementation',
                'Target: r > 1.5 with |ρ| maximized'
            ],
            'scan_parameters': {
                'squeeze_range': '0.5-2.0 rad',
                'mode_count': '1-10 modes',
                'cavity_finesse': '>1000',
                'pump_efficiency': '>80%'
            }
        }
        
        # 2.4 Metamaterial Enhancement
        roadmap['metamaterial'] = {
            'math': 'ρ_meta(d) = -1/√ε_eff × π²ℏc/(720 d⁴)',
            'current_params': {
                'epsilon_eff': self.metamaterial.epsilon_r,
                'mu_eff': self.metamaterial.mu_r,
                'loss_factor': self.metamaterial.loss_tangent
            },
            'optimization_strategy': 'Effective permittivity scanning',
            'next_steps': [
                'Scan effective permittivity ε = -1.5 to -3.0',
                'Synthesize left-handed materials (ε<0, μ<0)',
                'Minimize loss mechanisms (<1%)',
                'Scale to array-compatible geometries',
                'Target: >2× enhancement factor'
            ],
            'scan_parameters': {
                'epsilon_range': '-3.0 to -1.5',
                'mu_range': '-2.0 to -1.5', 
                'loss_target': '<1%',
                'frequency_range': '100 THz'
            }
        }
        
        return roadmap
    
    def run_parallel_optimization_scans(self) -> dict:
        """
        Run parallel optimization scans for all components.
        
        Implements the detailed scanning strategies from the roadmap.
        """
        results = {}
        
        print("🔬 PARALLEL PROTOTYPING SCANS")
        print("=" * 31)
        print()
        
        # 2.1 Casimir Array Local Scan
        print("1️⃣ CASIMIR ARRAY OPTIMIZATION")
        print("-" * 30)
        
        best_casimir = {'energy': 0, 'gaps': None}
        base_gaps = np.array([6, 7, 8, 7, 6]) * 1e-9
        
        for delta in np.linspace(-0.5, 0.5, 11) * 1e-9:  # ±0.5 nm
            gaps = base_gaps + delta
            demo = CasimirArrayDemonstrator(gaps)
            E = demo.calculate_energy_density()
            
            if abs(E) > abs(best_casimir['energy']):
                best_casimir = {'energy': E, 'gaps': gaps}
            
            print(f"Gaps={gaps*1e9} nm → E_C={E:.2e} J/m²")
        
        results['casimir'] = best_casimir
        print(f"Best: E={best_casimir['energy']:.2e} J/m², gaps={best_casimir['gaps']*1e9} nm")
        print()
        
        # 2.2 Dynamic Casimir Amplitude/Frequency Sweep
        print("2️⃣ DYNAMIC CASIMIR OPTIMIZATION")
        print("-" * 32)
        
        best_dynamic = {'energy': 0, 'params': None}
        
        for A_frac in [0.05, 0.1, 0.2]:  # 5-20%
            for f in [0.5e12, 1e12, 2e12]:  # 0.5-2 THz
                cavity = DynamicCasimirCavity(d0=1e-6, omega=f, amplitude=A_frac*1e-6)
                E_dyn = cavity.calculate_time_averaged_energy()
                
                if abs(E_dyn) > abs(best_dynamic['energy']):
                    best_dynamic = {'energy': E_dyn, 'params': (A_frac, f)}
                
                print(f"A={A_frac*100:.0f}%, f={f/1e12:.1f} THz → Ē_C={E_dyn:.2e} J/m²")
        
        results['dynamic'] = best_dynamic
        A_best, f_best = best_dynamic['params']
        print(f"Best: E={best_dynamic['energy']:.2e} J/m², A={A_best*100:.0f}%, f={f_best/1e12:.1f} THz")
        print()
        
        # 2.3 Squeezed Vacuum Parameter Optimization
        print("3️⃣ SQUEEZED VACUUM OPTIMIZATION")
        print("-" * 32)
        
        best_squeezed = {'energy': 0, 'r': None}
        
        for r in np.linspace(0.5, 2.0, 16):  # 0.5-2.0 rad
            src = SqueezedVacuumSource(
                pump_power=1e-3, 
                nonlinearity=1e-12, 
                cavity_finesse=1000
            )
            # Manually set the squeeze parameter for this test
            src.squeeze_parameters = [r]
            src.omegas = [1e14]
            src.volumes = [1e-15]
            ρ = src.calculate_energy_density()
            
            if abs(ρ) > abs(best_squeezed['energy']):
                best_squeezed = {'energy': ρ, 'r': r}
            
            print(f"r={r:.2f} → ρ_sq={ρ:.2e} J/m³")
        
        results['squeezed'] = best_squeezed
        print(f"Best: ρ={best_squeezed['energy']:.2e} J/m³, r={best_squeezed['r']:.2f}")
        print()
        
        # 2.4 Metamaterial Permittivity Scan
        print("4️⃣ METAMATERIAL OPTIMIZATION")
        print("-" * 28)
        
        best_meta = {'enhancement': 1, 'eps': None}
        
        for eps in [-1.5, -2.0, -2.5, -3.0]:
            enh = MetamaterialEnhancer(epsilon_r=eps, mu_r=-1.8, loss_tangent=0.01)
            factor = enh.calculate_enhancement()
            
            if factor > best_meta['enhancement']:
                best_meta = {'enhancement': factor, 'eps': eps}
            
            print(f"ε_eff={eps} → enhancement={factor:.2f}×")
        
        results['metamaterial'] = best_meta
        print(f"Best: {best_meta['enhancement']:.2f}× enhancement, ε={best_meta['eps']}")
        print()
        
        return results
    
    def assess_prototype_readiness(self) -> dict:
        """
        Assess readiness of each prototype module for experimental construction.
        
        Checks if each module meets the minimum viability criteria.
        """
        assessment = {}
        
        # Energy targets for each component
        targets = {
            'casimir': 1e-6,      # 1 μJ/m²
            'dynamic': 1e-7,      # 100 nJ/m²  
            'squeezed': 1e-4,     # 100 μJ/m³
            'metamaterial': 2.0   # 2× enhancement
        }
        
        contributions = self.get_component_contributions()
        
        # Casimir array assessment
        casimir_energy = abs(contributions.get('casimir', 0))
        assessment['casimir'] = {
            'energy': casimir_energy,
            'target': targets['casimir'],
            'ready': casimir_energy >= targets['casimir'],
            'readiness_pct': min(100, casimir_energy / targets['casimir'] * 100)
        }
        
        # Dynamic Casimir assessment
        dynamic_energy = abs(contributions.get('dynamic', 0))
        assessment['dynamic'] = {
            'energy': dynamic_energy,
            'target': targets['dynamic'],
            'ready': dynamic_energy >= targets['dynamic'],
            'readiness_pct': min(100, dynamic_energy / targets['dynamic'] * 100)
        }
        
        # Squeezed vacuum assessment (convert to volume density)
        squeezed_energy = abs(contributions.get('squeezed', 0)) / 1e-6  # Back to J/m³
        assessment['squeezed'] = {
            'energy': squeezed_energy,
            'target': targets['squeezed'],
            'ready': squeezed_energy >= targets['squeezed'],
            'readiness_pct': min(100, squeezed_energy / targets['squeezed'] * 100)
        }
        
        # Metamaterial assessment
        if self.metamaterial:
            meta_enhancement = self.metamaterial.calculate_enhancement()
        else:
            meta_enhancement = 1.0
            
        assessment['metamaterial'] = {
            'enhancement': meta_enhancement,
            'target': targets['metamaterial'],
            'ready': meta_enhancement >= targets['metamaterial'],
            'readiness_pct': min(100, meta_enhancement / targets['metamaterial'] * 100)
        }
        
        # Overall assessment
        ready_count = sum(1 for comp in assessment.values() if comp['ready'])
        assessment['overall'] = {
            'ready_components': ready_count,
            'total_components': len(assessment) - 1,  # Exclude 'overall'
            'overall_ready': ready_count >= 3,  # At least 3 of 4 ready
            'readiness_pct': ready_count / 4 * 100
        }
        
        return assessment

# Legacy compatibility functions
def optimize_casimir(N, d_min, d_max):
    """Legacy function for Casimir optimization."""
    from casimir_array import optimize_casimir as legacy_optimize
    return legacy_optimize(N, d_min, d_max)

def sweep_dynamic(d0_range, A_range, ω_range):
    """Legacy function for dynamic sweep."""
    from dynamic_casimir import sweep_dynamic as legacy_sweep
    return legacy_sweep(d0_range, A_range, ω_range)

def optimize_single_mode(omega, V, r_max=3.0):
    """Legacy function for squeezed vacuum optimization."""
    from squeezed_vacuum import optimize_single_mode as legacy_optimize
    return legacy_optimize(omega, V, r_max)

def optimize_meta(ds, eps_bounds):
    """Legacy function for metamaterial optimization."""
    from metamaterial import optimize_meta as legacy_optimize
    return legacy_optimize(ds, eps_bounds)

def combined_prototype_demonstration():
    """Demonstrate unified vacuum generator with parallel development strategy."""
    
    print("🔗 UNIFIED VACUUM GENERATOR + PARALLEL DEVELOPMENT")
    print("=" * 52)
    print()
    
    # Create unified generator
    generator = UnifiedVacuumGenerator()
    
    print("Components initialized:")
    print("✅ Casimir Array Demonstrator")
    print("✅ Dynamic Casimir Cavity")
    print("✅ Squeezed Vacuum Source")
    print("✅ Metamaterial Enhancement")
    print()
    
    # Calculate individual contributions
    contributions = generator.get_component_contributions()
    print("📊 INDIVIDUAL CONTRIBUTIONS:")
    for component, energy in contributions.items():
        print(f"  {component.capitalize()}: {energy:.3e} J/m²")
    print()
    
    # Calculate total energy and power
    total_energy = generator.calculate_total_energy()
    total_power = generator.estimate_power_output()
    
    print(f"Total energy density: {total_energy:.3e} J/m²")
    print(f"Total power output: {total_power:.3e} W")
    print()
    
    # Energy per unit area (1 cm²)
    area = 1e-4  # 1 cm²
    energy_per_cm2 = total_energy * area
    print(f"Energy per cm²: {energy_per_cm2:.3e} J")
    print()
    
    # 🚀 NEW: Parallel prototyping roadmap
    print("🛣️ PARALLEL PROTOTYPING ROADMAP")
    print("=" * 33)
    roadmap = generator.parallel_prototyping_roadmap()
    
    for component, details in roadmap.items():
        print(f"\n🔧 {component.replace('_', ' ').title()}:")
        print(f"   Math: {details['math']}")
        print(f"   Strategy: {details['optimization_strategy']}")
        print("   Next steps:")
        for step in details['next_steps']:
            print(f"     • {step}")
    print()
    
    # 🚀 NEW: Run optimization scans
    print("🔬 RUNNING PARALLEL OPTIMIZATION SCANS")
    print("=" * 37)
    scan_results = generator.run_parallel_optimization_scans()
    
    # 🚀 NEW: Prototype readiness assessment
    print("🏗️ PROTOTYPE READINESS ASSESSMENT")
    print("=" * 33)
    readiness = generator.assess_prototype_readiness()
    
    for component, assessment in readiness.items():
        if component == 'overall':
            continue
        
        status = "✅ READY" if assessment['ready'] else "⚠️ DEVELOPING"
        print(f"{component.capitalize()}: {status} ({assessment['readiness_pct']:.0f}%)")
        
        if 'energy' in assessment:
            print(f"  Current: {assessment['energy']:.2e}, Target: {assessment['target']:.2e}")
        elif 'enhancement' in assessment:
            print(f"  Current: {assessment['enhancement']:.2f}×, Target: {assessment['target']:.2f}×")
    
    overall = readiness['overall']
    overall_status = "✅ READY" if overall['overall_ready'] else "🔄 IN PROGRESS"
    print(f"\nOverall: {overall_status} ({overall['ready_components']}/{overall['total_components']} ready)")
    print()
    
    # Theory-experiment coordination message
    print("🎯 THEORY-EXPERIMENT COORDINATION")
    print("=" * 33)
    print("Current status: PARALLEL_DEVELOPMENT")
    print("• Theory: Continue LQG-ANEC refinement until both targets met")
    print("• Experiments: Build and optimize vacuum-engineering testbeds") 
    print("• Validation: Deploy de-risking framework for all modules")
    print()
    print("🚦 Proceed to full demonstrator only when:")
    print("   best_anec_2d ≤ -1e5  AND  best_rate_2d ≥ 0.5")
    print()
    
    # Performance assessment with parallel development context
    if abs(total_energy) > 1e-5:
        print("🚀 EXCELLENT: Strong negative energy generation")
        print("   → Ready for large-scale integration once theory targets met")
    elif abs(total_energy) > 1e-6:
        print("✅ GOOD: Substantial negative energy achieved") 
        print("   → Continue optimization while advancing theory")
    elif abs(total_energy) > 1e-7:
        print("⚠️ MODERATE: Measurable but limited generation")
        print("   → Focus on enhancement while parallel theory work")
    else:
        print("❌ LOW: Minimal negative energy - needs enhancement")
        print("   → Prioritize optimization alongside theory refinement")
    
    return {
        'generator': generator,
        'total_energy': total_energy,
        'total_power': total_power,
        'contributions': contributions,
        'energy_per_cm2': energy_per_cm2,
        'roadmap': roadmap,
        'scan_results': scan_results,
        'readiness': readiness,
        'development_mode': 'PARALLEL_DEVELOPMENT'
    }

# Legacy demo for compatibility
N = 5

if __name__ == "__main__":
    # Run new demonstration
    combined_prototype_demonstration()
    
    print("\n" + "="*42)
    print("LEGACY COMPATIBILITY TEST")
    print("="*42)
    
    # Run legacy calculations for compatibility
    try:
        ds_opt, E_array = optimize_casimir(N, 5e-9, 1e-8)
        E_dyn, (d0,A,ω) = sweep_dynamic((5e-9,1e-8),(0,5e-9),(1e9,1e12))
        r_opt, E_sq = optimize_single_mode(2*np.pi*1e14, 1e-12)
        eps_opt, E_meta = optimize_meta(ds_opt, (1e-4,1))
        
        print("Casimir array:", E_array)
        print("Dynamic Casimir:", E_dyn)
        print("Squeezed vac:", E_sq)
        print("Metamaterial:", E_meta)
    except Exception as e:
        print(f"Legacy compatibility test failed: {e}")
