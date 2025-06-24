#!/usr/bin/env python3
"""
Unified ANEC Pipeline: Comprehensive Negative Energy Validation
==============================================================

This module integrates all theoretical components for comprehensive ANEC
integral calculation and negative energy validation:
- Traversable wormhole (Morris-Thorne/Krasnikov) ansatz
- Casimir effect enhancement shells  
- Squeezed vacuum state contributions
- Complete radiative corrections (1-loop, 2-loop, 3-loop Monte Carlo)
- Parameter optimization for maximum ANEC violation

The goal is to achieve robust negative ANEC integrals (< -10⁵ J·s·m⁻³)
with high violation rates (≥30%) across the extended parameter space.

Author: Negative Energy Generator Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import scipy.optimize as optimize
import sys
import os

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'theoretical'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'corrections'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'quantum'))

# Import all components
try:
    from wormhole_ansatz import TraversableWormholeAnsatz, WormholeConfig
    from casimir_enhancement import CasimirEnhancement, CasimirConfig
    from squeezed_vacuum import SqueezedVacuumStates, SqueezedVacuumConfig
    from radiative import RadiativeCorrections
    from field_algebra import PolymerFieldAlgebra
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    print("Some functionality may be limited.")

@dataclass
class UnifiedConfig:
    """Unified configuration for all negative energy components."""
    
    # Wormhole parameters
    throat_radius: float = 1e-15        # r₀ (m)
    shell_thickness: float = 1e-14      # R (m)
    redshift_param: float = 0.1         # τ
    shape_param: float = 2.0            # s
    exotic_strength: float = 1e-3       # Δ
    
    # Casimir parameters
    casimir_plate_separation: float = 5e-15     # d (m)
    casimir_modulation_freq: float = 2e15       # f_mod (Hz)
    casimir_vacuum_coupling: float = 1e-2       # g_vac
    
    # Squeezed vacuum parameters
    squeezing_parameter: float = 2.0            # r
    squeezing_phase: float = 0.0                # φ
    coherent_amplitude: float = 1.5             # α
    vacuum_coupling: float = 1e-2               # g_sq
    
    # Radiative correction parameters
    coupling_constant: float = 1.0              # g
    field_mass: float = 0.0                     # m
    uv_cutoff: float = 100.0                    # Λ
    
    # LQG polymer parameters
    polymer_scale: float = 1e-35                # μ (Planck scale)
    holonomy_eigenvalue: float = 0.5            # λ
    
    # Computational parameters
    grid_points: int = 1000                     # Spatial resolution
    time_steps: int = 500                       # Temporal resolution
    mc_samples: int = 5000                      # Monte Carlo samples
    
    # Target criteria
    target_anec: float = -1e5                   # Target ANEC (J·s·m⁻³)
    target_violation_rate: float = 0.30         # Target violation rate
    ford_roman_factor: float = 100.0            # Ford-Roman enhancement

class UnifiedANECPipeline:
    """
    Unified pipeline for comprehensive negative energy generation validation.
    
    Integrates wormhole geometries, Casimir effects, squeezed vacuum states,
    and complete radiative corrections for robust ANEC violation.
    """
    
    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()
        
        # Initialize all components
        self._initialize_components()
        
        # Results cache
        self.results_cache = {}
        self.optimization_history = []
        
        print("🚀 Unified ANEC Pipeline Initialized")
        print("=" * 50)
        print(f"Target ANEC: {self.config.target_anec:.2e} J·s·m⁻³")
        print(f"Target violation rate: {self.config.target_violation_rate:.1%}")
        print(f"Components: Wormhole + Casimir + Squeezed + Radiative")
    
    def _initialize_components(self):
        """Initialize all theoretical components."""
        
        # Wormhole ansatz
        wormhole_config = WormholeConfig(
            throat_radius=self.config.throat_radius,
            shell_thickness=self.config.shell_thickness,
            redshift_param=self.config.redshift_param,
            shape_param=self.config.shape_param,
            exotic_strength=self.config.exotic_strength
        )
        self.wormhole = TraversableWormholeAnsatz(wormhole_config)
        
        # Casimir enhancement
        casimir_config = CasimirConfig(
            plate_separation=self.config.casimir_plate_separation,
            modulation_frequency=self.config.casimir_modulation_freq,
            vacuum_coupling=self.config.casimir_vacuum_coupling
        )
        self.casimir = CasimirEnhancement(casimir_config)
        
        # Squeezed vacuum states
        squeezed_config = SqueezedVacuumConfig(
            squeezing_parameter=self.config.squeezing_parameter,
            squeezing_phase=self.config.squeezing_phase,
            coherent_amplitude=self.config.coherent_amplitude,
            vacuum_coupling=self.config.vacuum_coupling
        )
        self.squeezed = SqueezedVacuumStates(squeezed_config)
        
        # Radiative corrections
        self.radiative = RadiativeCorrections(
            mass=self.config.field_mass,
            coupling=self.config.coupling_constant,
            cutoff=self.config.uv_cutoff
        )
        
        print("✅ All components initialized successfully")
    
    def create_radial_grid(self, r_min: float = None, r_max: float = None) -> np.ndarray:
        """Create optimized radial coordinate grid."""
        if r_min is None:
            r_min = self.config.throat_radius * 1.01
        if r_max is None:
            r_max = self.config.throat_radius + 10 * self.config.shell_thickness
            
        # Non-uniform grid with higher resolution near throat
        throat_region = np.linspace(r_min, r_min + 2*self.config.shell_thickness, 
                                  self.config.grid_points // 2)
        outer_region = np.linspace(r_min + 2*self.config.shell_thickness, r_max,
                                 self.config.grid_points // 2)
        
        return np.concatenate([throat_region, outer_region[1:]])
    
    def compute_total_energy_density(self, r: np.ndarray, t: float = 0.0, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute total energy density from all sources.
        
        ρ_total = ρ_wormhole + ρ_casimir + ρ_squeezed + corrections
        
        Args:
            r: Radial coordinate array
            t: Time coordinate
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary of energy density components
        """
        if verbose:
            print(f"⚡ Computing total energy density at t={t:.2e}s...")
        
        # Wormhole contribution
        wormhole_stress = self.wormhole.stress_energy_tensor(r)
        rho_wormhole = wormhole_stress['energy_density']
        
        # Casimir shell contribution
        rho_casimir = self.casimir.casimir_shell_around_throat(
            r, self.config.throat_radius, self.config.shell_thickness
        )
        
        # Squeezed vacuum contribution
        rho_squeezed = self.squeezed.total_squeezed_energy_density(r, t)
        
        # Combined tree-level density
        rho_tree = rho_wormhole + rho_casimir + rho_squeezed
        
        # Apply radiative corrections
        rho_corrected, correction_breakdown = self.radiative.corrected_stress_energy(
            rho_tree, 
            self.config.shell_thickness,
            self.config.redshift_param,
            self.config.polymer_scale
        )
        
        components = {
            'wormhole': rho_wormhole,
            'casimir': rho_casimir,
            'squeezed': rho_squeezed,
            'tree_level': rho_tree,
            'corrected_total': rho_corrected,
            'radiative_breakdown': correction_breakdown
        }
        
        if verbose:
            print(f"✓ Wormhole density range: [{rho_wormhole.min():.2e}, {rho_wormhole.max():.2e}]")
            print(f"✓ Casimir density range: [{rho_casimir.min():.2e}, {rho_casimir.max():.2e}]")
            print(f"✓ Squeezed density range: [{rho_squeezed.min():.2e}, {rho_squeezed.max():.2e}]")
            print(f"✓ Total corrected range: [{rho_corrected.min():.2e}, {rho_corrected.max():.2e}]")
        
        return components
    
    def compute_unified_anec_integral(self, r_grid: np.ndarray = None) -> Dict[str, float]:
        """
        Compute ANEC integral including all contributions.
        
        ANEC = ∫ ρ_total(r) dr along null geodesic
        
        Args:
            r_grid: Radial grid (auto-generated if None)
            
        Returns:
            ANEC computation results
        """
        print("🔍 Computing unified ANEC integral...")
        
        if r_grid is None:
            r_grid = self.create_radial_grid()
        
        # Compute total energy density
        energy_components = self.compute_total_energy_density(r_grid)
        
        # ANEC integrals for each component
        dr = r_grid[1] - r_grid[0]  # Assume uniform spacing for integration
        
        anec_wormhole = np.trapz(energy_components['wormhole'], r_grid)
        anec_casimir = np.trapz(energy_components['casimir'], r_grid)
        anec_squeezed = np.trapz(energy_components['squeezed'], r_grid)
        anec_tree = np.trapz(energy_components['tree_level'], r_grid)
        anec_total = np.trapz(energy_components['corrected_total'], r_grid)
        
        # Radiative correction contributions
        rad_breakdown = energy_components['radiative_breakdown']
        
        results = {
            'anec_wormhole': anec_wormhole,
            'anec_casimir': anec_casimir,
            'anec_squeezed': anec_squeezed,
            'anec_tree_level': anec_tree,
            'anec_total': anec_total,
            'radiative_corrections': {
                'one_loop': rad_breakdown['one_loop'],
                'two_loop': rad_breakdown['two_loop'],
                'three_loop': rad_breakdown.get('three_loop', 0.0),
                'polymer_specific': rad_breakdown['polymer_specific']
            },
            'negative_anec_achieved': anec_total < 0,
            'target_met': anec_total < self.config.target_anec,
            'violation_magnitude': abs(anec_total) if anec_total < 0 else 0
        }
        
        print(f"📊 ANEC Results:")
        print(f"   Wormhole: {anec_wormhole:.2e} J·s·m⁻³")
        print(f"   Casimir: {anec_casimir:.2e} J·s·m⁻³")
        print(f"   Squeezed: {anec_squeezed:.2e} J·s·m⁻³")
        print(f"   Total: {anec_total:.2e} J·s·m⁻³")
        print(f"   Negative ANEC: {'YES' if anec_total < 0 else 'NO'}")
        print(f"   Target achieved: {'YES' if anec_total < self.config.target_anec else 'NO'}")
        
        return results
    
    def optimize_unified_parameters(self, n_iterations: int = 50, max_evaluations: int = 1000) -> Dict[str, float]:
        """
        Optimize all parameters simultaneously for maximum ANEC violation.
        
        Uses differential evolution to explore the high-dimensional parameter space.
        """
        print("🔧 Starting unified parameter optimization...")
        print(f"   Optimizing {12} parameters across {n_iterations} iterations")
        print(f"   Maximum evaluations: {max_evaluations}")
        print(f"   Target: ANEC < {self.config.target_anec:.2e} J·s·m⁻³")
        
        # Progress tracking
        self.best_anec = float('inf')
        self.evaluation_count = 0
        self.progress_history = []
        self.stagnation_count = 0  # Track how long we've been stagnant
        
        def objective(params):
            """Objective function: minimize ANEC (maximize violation)."""
            try:
                self.evaluation_count += 1
                
                # Early termination if too many evaluations
                if self.evaluation_count > max_evaluations:
                    print(f"   ⏰ Maximum evaluations ({max_evaluations}) reached!")
                    return 1e15  # High penalty to stop optimization
                
                # Unpack parameters
                (throat_radius, shell_thickness, redshift_param, shape_param, exotic_strength,
                 casimir_plate_sep, casimir_mod_freq, casimir_vac_coupling,
                 squeeze_param, squeeze_phase, coherent_amp, vacuum_coup) = params
                
                # Update configurations (suppress output for optimization)
                self._update_all_configs(params)
                
                # Compute ANEC (suppress detailed output)
                r_grid = self.create_radial_grid()
                energy_components = self.compute_total_energy_density(r_grid, verbose=False)
                anec_total = np.trapz(energy_components['corrected_total'], r_grid)
                
                # Track progress
                improvement_threshold = 0.01  # 1% improvement required
                if anec_total < self.best_anec * (1 - improvement_threshold):
                    self.best_anec = anec_total
                    self.stagnation_count = 0  # Reset stagnation
                    improvement = "🎯 NEW BEST!" if anec_total < 0 else "⬇️ Improving"
                    negative_status = "✅ NEGATIVE!" if anec_total < 0 else "❌ Positive"
                    target_status = "🚀 TARGET MET!" if anec_total < self.config.target_anec else "📈 Toward target"
                    
                    print(f"   Eval {self.evaluation_count:4d}: {improvement}")
                    print(f"      ANEC: {anec_total:.2e} J·s·m⁻³ {negative_status}")
                    print(f"      Progress: {target_status}")
                    print(f"      Best params: r₀={throat_radius:.2e}, R={shell_thickness:.2e}, τ={redshift_param:.3f}")
                else:
                    self.stagnation_count += 1
                
                # Show periodic progress even without improvement
                if self.evaluation_count % 50 == 0:
                    progress_pct = (self.evaluation_count / max_evaluations) * 100
                    stagnation_info = f"Stagnant: {self.stagnation_count}" if self.stagnation_count > 100 else ""
                    print(f"   Progress: {progress_pct:.1f}% | Best ANEC: {self.best_anec:.2e} | Evaluations: {self.evaluation_count} {stagnation_info}")
                
                # Early termination if stagnant for too long
                if self.stagnation_count > 200:  # No improvement for 200 evaluations
                    print(f"   ⏸️ Optimization stagnant for {self.stagnation_count} evaluations - terminating early")
                    return 1e15  # High penalty to stop optimization
                
                self.progress_history.append({
                    'evaluation': self.evaluation_count,
                    'anec': anec_total,
                    'negative': anec_total < 0,
                    'target_met': anec_total < self.config.target_anec
                })
                
                # Objective: minimize ANEC (want maximum negative value)
                return anec_total
                
            except Exception as e:
                print(f"   ❌ Optimization error at eval {self.evaluation_count}: {e}")
                return 1e10  # Penalty for failed evaluation
        
        # Parameter bounds
        bounds = [
            (1e-16, 1e-13),     # throat_radius
            (1e-15, 1e-12),     # shell_thickness
            (0.01, 1.0),        # redshift_param
            (1.0, 5.0),         # shape_param
            (1e-6, 1e-1),       # exotic_strength
            (1e-16, 1e-13),     # casimir_plate_separation
            (1e14, 1e16),       # casimir_modulation_frequency
            (1e-4, 1e-1),       # casimir_vacuum_coupling
            (0.1, 5.0),         # squeezing_parameter
            (-np.pi, np.pi),    # squeezing_phase
            (0.1, 3.0),         # coherent_amplitude
            (1e-4, 1e-1)        # vacuum_coupling
        ]
        
        # Initial guess (current configuration)
        x0 = [
            self.config.throat_radius,
            self.config.shell_thickness,
            self.config.redshift_param,
            self.config.shape_param,
            self.config.exotic_strength,
            self.config.casimir_plate_separation,
            self.config.casimir_modulation_freq,
            self.config.casimir_vacuum_coupling,
            self.config.squeezing_parameter,
            self.config.squeezing_phase,
            self.config.coherent_amplitude,
            self.config.vacuum_coupling
        ]
        
        # Differential evolution optimization with early termination
        result = optimize.differential_evolution(
            objective, bounds, seed=42, maxiter=n_iterations,
            popsize=15, tol=1e-12, disp=False,  # Suppress scipy output
            callback=self._optimization_callback
        )
        
        # Apply best parameters
        if result.success:
            self._update_all_configs(result.x)
        
        # Final ANEC calculation with full output
        print(f"\n📊 Final ANEC Calculation (after {self.evaluation_count} evaluations):")
        final_anec_results = self.compute_unified_anec_integral()
        
        # Optimization summary
        negative_count = sum(1 for p in self.progress_history if p['negative'])
        target_count = sum(1 for p in self.progress_history if p['target_met'])
        
        print(f"\n🔍 Optimization Summary:")
        print(f"   Total evaluations: {self.evaluation_count}")
        print(f"   Negative ANEC found: {negative_count} times ({negative_count/self.evaluation_count*100:.1f}%)")
        print(f"   Target achieved: {target_count} times ({target_count/self.evaluation_count*100:.1f}%)")
        print(f"   Best ANEC: {self.best_anec:.2e} J·s·m⁻³")
        
        success_criteria = {
            'negative_anec': final_anec_results['anec_total'] < 0,
            'target_met': final_anec_results['target_met'],
            'optimization_converged': result.success,
            'violation_rate': negative_count / self.evaluation_count if self.evaluation_count > 0 else 0
        }
        
        print("✅ Unified optimization complete!")
        print(f"   Final ANEC: {final_anec_results['anec_total']:.2e} J·s·m⁻³")
        print(f"   Negative ANEC: {'YES' if success_criteria['negative_anec'] else 'NO'}")
        print(f"   Target achieved: {'YES' if success_criteria['target_met'] else 'NO'}")
        
        return {
            'optimization_result': result,
            'final_anec_results': final_anec_results,
            'success_criteria': success_criteria,
            'best_parameters': dict(zip([
                'throat_radius', 'shell_thickness', 'redshift_param', 'shape_param', 'exotic_strength',
                'casimir_plate_separation', 'casimir_modulation_frequency', 'casimir_vacuum_coupling',
                'squeezing_parameter', 'squeezing_phase', 'coherent_amplitude', 'vacuum_coupling'
            ], result.x if result.success else x0))
        }
    
    def _update_all_configs(self, params: List[float]):
        """Update all component configurations with new parameters."""
        (throat_radius, shell_thickness, redshift_param, shape_param, exotic_strength,
         casimir_plate_sep, casimir_mod_freq, casimir_vac_coupling,
         squeeze_param, squeeze_phase, coherent_amp, vacuum_coup) = params
        
        # Update main config
        self.config.throat_radius = throat_radius
        self.config.shell_thickness = shell_thickness
        self.config.redshift_param = redshift_param
        self.config.shape_param = shape_param
        self.config.exotic_strength = exotic_strength
        self.config.casimir_plate_separation = casimir_plate_sep
        self.config.casimir_modulation_freq = casimir_mod_freq
        self.config.casimir_vacuum_coupling = casimir_vac_coupling
        self.config.squeezing_parameter = squeeze_param
        self.config.squeezing_phase = squeeze_phase
        self.config.coherent_amplitude = coherent_amp
        self.config.vacuum_coupling = vacuum_coup
        
        # Update component configs
        self.wormhole.config.throat_radius = throat_radius
        self.wormhole.config.shell_thickness = shell_thickness
        self.wormhole.config.redshift_param = redshift_param
        self.wormhole.config.shape_param = shape_param
        self.wormhole.config.exotic_strength = exotic_strength
        
        self.casimir.config.plate_separation = casimir_plate_sep
        self.casimir.config.modulation_frequency = casimir_mod_freq
        self.casimir.config.vacuum_coupling = casimir_vac_coupling
        
        self.squeezed.config.squeezing_parameter = squeeze_param
        self.squeezed.config.squeezing_phase = squeeze_phase
        self.squeezed.config.coherent_amplitude = coherent_amp
        self.squeezed.config.vacuum_coupling = vacuum_coup
    
    def run_comprehensive_validation(self) -> Dict[str, any]:
        """
        Run comprehensive validation of the unified negative energy framework.
        """
        print("🧪 Running Comprehensive Validation")
        print("=" * 50)
        
        validation_results = {}
        
        # 1. Initial ANEC calculation
        print("\n=== Initial ANEC Calculation ===")
        initial_anec = self.compute_unified_anec_integral()
        validation_results['initial_anec'] = initial_anec
        
        # 2. Parameter optimization
        print("\n=== Parameter Optimization ===")
        optimization_results = self.optimize_unified_parameters(n_iterations=20, max_evaluations=500)
        validation_results['optimization'] = optimization_results
        
        # 3. Final validation
        print("\n=== Final Validation ===")
        final_anec = optimization_results['final_anec_results']
        
        # Success criteria assessment
        success_metrics = {
            'negative_anec_achieved': final_anec['anec_total'] < 0,
            'target_anec_met': final_anec['anec_total'] < self.config.target_anec,
            'violation_magnitude': abs(final_anec['anec_total']) if final_anec['anec_total'] < 0 else 0,
            'ford_roman_factor': abs(final_anec['anec_total']) / 1e10 if final_anec['anec_total'] < 0 else 0,  # Simplified
            'radiative_stability': all(
                abs(v) < 1e-5 for v in final_anec['radiative_corrections'].values()
            )
        }
        
        validation_results['success_metrics'] = success_metrics
        
        # Summary
        print(f"\n📊 FINAL VALIDATION SUMMARY")
        print(f"   Negative ANEC achieved: {'✅ YES' if success_metrics['negative_anec_achieved'] else '❌ NO'}")
        print(f"   Target ANEC met: {'✅ YES' if success_metrics['target_anec_met'] else '❌ NO'}")
        print(f"   ANEC magnitude: {success_metrics['violation_magnitude']:.2e} J·s·m⁻³")
        print(f"   Radiative stability: {'✅ STABLE' if success_metrics['radiative_stability'] else '⚠️ UNSTABLE'}")
        
        overall_success = (
            success_metrics['negative_anec_achieved'] and
            success_metrics['target_anec_met'] and
            success_metrics['radiative_stability']
        )
        
        print(f"\n🎯 OVERALL SUCCESS: {'✅ ACHIEVED' if overall_success else '❌ PENDING'}")
        
        validation_results['overall_success'] = overall_success
        return validation_results

    def _optimization_callback(self, xk, convergence):
        """Callback function for optimization progress and early termination."""
        # Early termination if target achieved consistently
        if len(self.progress_history) >= 10:
            recent_targets = [p['target_met'] for p in self.progress_history[-10:]]
            if sum(recent_targets) >= 5:  # 50% of recent evaluations hit target
                print(f"   🎯 Early termination: Target achieved consistently!")
                return True  # Stop optimization
        
        # Early termination if we find very negative ANEC
        if self.best_anec < self.config.target_anec * 10:  # 10x better than target
            print(f"   🚀 Early termination: Exceptional ANEC found!")
            return True
        
        # Early termination if stagnant
        if hasattr(self, 'stagnation_count') and self.stagnation_count > 150:
            print(f"   ⏸️ Early termination: Optimization stagnant for {self.stagnation_count} evaluations")
            return True
        
        return False  # Continue optimization
    
    def diagnose_anec_problem(self) -> Dict[str, any]:
        """
        Diagnose why ANEC integral is not achieving negative values.
        """
        print("🔍 ANEC Problem Diagnosis")
        print("=" * 40)
        
        r_grid = self.create_radial_grid()
        energy_components = self.compute_total_energy_density(r_grid)
        
        # Analyze each component separately
        print(f"\n📊 Component Analysis:")
        print(f"   Grid points: {len(r_grid)}")
        print(f"   Radial range: [{r_grid.min():.2e}, {r_grid.max():.2e}] m")
        
        # Check where energy is negative vs positive
        wormhole_negative = (energy_components['wormhole'] < 0).sum()
        casimir_negative = (energy_components['casimir'] < 0).sum()
        squeezed_negative = (energy_components['squeezed'] < 0).sum()
        total_negative = (energy_components['corrected_total'] < 0).sum()
        
        print(f"\n🎯 Negative Energy Regions:")
        print(f"   Wormhole: {wormhole_negative}/{len(r_grid)} points ({wormhole_negative/len(r_grid)*100:.1f}%)")
        print(f"   Casimir: {casimir_negative}/{len(r_grid)} points ({casimir_negative/len(r_grid)*100:.1f}%)")
        print(f"   Squeezed: {squeezed_negative}/{len(r_grid)} points ({squeezed_negative/len(r_grid)*100:.1f}%)")
        print(f"   Total: {total_negative}/{len(r_grid)} points ({total_negative/len(r_grid)*100:.1f}%)")
        
        # Integration analysis
        anec_wormhole = np.trapz(energy_components['wormhole'], r_grid)
        anec_casimir = np.trapz(energy_components['casimir'], r_grid)
        anec_squeezed = np.trapz(energy_components['squeezed'], r_grid)
        anec_total = np.trapz(energy_components['corrected_total'], r_grid)
        
        print(f"\n⚖️ Integration Results:")
        print(f"   Wormhole ANEC: {anec_wormhole:.2e} J·s·m⁻³")
        print(f"   Casimir ANEC: {anec_casimir:.2e} J·s·m⁻³") 
        print(f"   Squeezed ANEC: {anec_squeezed:.2e} J·s·m⁻³")
        print(f"   Total ANEC: {anec_total:.2e} J·s·m⁻³")
        
        # Identify dominant contribution
        contributions = {
            'wormhole': abs(anec_wormhole),
            'casimir': abs(anec_casimir),
            'squeezed': abs(anec_squeezed)
        }
        dominant = max(contributions.keys(), key=lambda k: contributions[k])
        
        print(f"\n🏆 Dominant contribution: {dominant} ({contributions[dominant]:.2e})")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if anec_wormhole > 0 and abs(anec_wormhole) > abs(anec_casimir) + abs(anec_squeezed):
            print(f"   • Wormhole geometry produces too much positive energy")
            print(f"   • Try: smaller throat radius, larger exotic strength")
        
        if anec_casimir >= 0:
            print(f"   • Casimir effect not producing enough negative energy")
            print(f"   • Try: smaller plate separation, higher modulation frequency")
            
        if anec_squeezed >= 0:
            print(f"   • Squeezed vacuum not effective")
            print(f"   • Try: higher squeezing parameter, better phase tuning")
        
        if total_negative < len(r_grid) * 0.3:
            print(f"   • Too few negative energy regions ({total_negative/len(r_grid)*100:.1f}%)")
            print(f"   • Need better spatial localization of negative contributions")
        
        return {
            'component_analysis': {
                'wormhole_negative_fraction': wormhole_negative / len(r_grid),
                'casimir_negative_fraction': casimir_negative / len(r_grid),
                'squeezed_negative_fraction': squeezed_negative / len(r_grid),
                'total_negative_fraction': total_negative / len(r_grid)
            },
            'anec_values': {
                'wormhole': anec_wormhole,
                'casimir': anec_casimir,
                'squeezed': anec_squeezed,
                'total': anec_total
            },
            'dominant_contribution': dominant,
            'recommendations': {
                'wormhole_dominant': anec_wormhole > abs(anec_casimir) + abs(anec_squeezed),
                'insufficient_negative_regions': total_negative < len(r_grid) * 0.3,
                'casimir_ineffective': anec_casimir >= 0,
                'squeezed_ineffective': anec_squeezed >= 0
            }
        }
    
def demo_unified_pipeline():
    """Demonstrate the unified ANEC pipeline."""
    print("🚀 UNIFIED ANEC PIPELINE DEMO")
    print("=" * 60)
    
    # Create pipeline with enhanced configuration
    config = UnifiedConfig(
        throat_radius=5e-15,
        shell_thickness=2e-14,
        exotic_strength=5e-3,
        casimir_plate_separation=3e-15,
        squeezing_parameter=3.0,
        target_anec=-1e5
    )
    
    pipeline = UnifiedANECPipeline(config)
    
    # Run comprehensive validation
    results = pipeline.run_comprehensive_validation()
    
    return pipeline, results

if __name__ == "__main__":
    demo_unified_pipeline()
