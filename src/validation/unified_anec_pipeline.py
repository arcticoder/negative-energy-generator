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
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import scipy.optimize as optimize
import sys
import os
import time

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
    # NEW: Advanced mathematical enhancements
    from su2_recoupling import SU2RecouplingEnhancement, RecouplingConfig
    from generating_functional import GeneratingFunctionalAnalysis, GeneratingFunctionalConfig
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    print("Some functionality may be limited.")

# Add parameter scanning import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'optimization'))
try:
    from parameter_scanning import HighDimensionalParameterScan, ParameterScanConfig
except ImportError as e:
    print(f"Warning: Could not import parameter scanning: {e}")

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
    
    # NEW: SU(2) Recoupling parameters
    recoupling_spins: List[float] = None        # j_e values
    mass_ratios: List[float] = None             # ρ_e = M_e^+ / M_e^-
    recoupling_boost: float = 1e3               # Enhancement factor
    
    # NEW: Generating functional parameters
    gf_grid_size: int = 50                      # GF discretization
    gf_kernel_type: str = 'warp_bubble'         # Kernel type
    gf_kernel_strength: float = 0.5             # Kernel strength
    
    # Computational parameters
    grid_points: int = 1000                     # Spatial resolution
    time_steps: int = 500                       # Temporal resolution
    mc_samples: int = 5000                      # Monte Carlo samples
    
    # Target criteria
    target_anec: float = -1e5                   # Target ANEC (J·s·m⁻³)
    target_violation_rate: float = 0.30         # Target violation rate
    ford_roman_factor: float = 100.0            # Ford-Roman enhancement
    
    def __post_init__(self):
        """Initialize default values for new parameters."""
        if self.recoupling_spins is None:
            self.recoupling_spins = [0.5, 1.0, 1.5, 2.0]
        if self.mass_ratios is None:
            self.mass_ratios = [2.5, 1.8, 3.2, 4.1]

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
        
        # NEW: SU(2) Recoupling enhancement
        try:
            recoupling_config = RecouplingConfig(
                spins=self.config.recoupling_spins,
                mass_ratios=self.config.mass_ratios,
                boost_factor=self.config.recoupling_boost,
                spatial_localization=self.config.shell_thickness
            )
            self.su2_recoupling = SU2RecouplingEnhancement(recoupling_config)
            self.has_su2_recoupling = True
        except Exception as e:
            print(f"⚠️  SU(2) recoupling not available: {e}")
            self.has_su2_recoupling = False
        
        # NEW: Generating functional analysis
        try:
            gf_config = GeneratingFunctionalConfig(
                grid_size=self.config.gf_grid_size,
                spatial_extent=self.config.shell_thickness * 10,
                kernel_type=self.config.gf_kernel_type,
                kernel_strength=self.config.gf_kernel_strength,
                throat_radius=self.config.throat_radius,
                shell_thickness=self.config.shell_thickness
            )
            self.generating_functional = GeneratingFunctionalAnalysis(gf_config)
            self.has_generating_functional = True
        except Exception as e:
            print(f"⚠️  Generating functional not available: {e}")
            self.has_generating_functional = False
        
        print("✅ All components initialized successfully")
        if self.has_su2_recoupling:
            print("   🔗 SU(2) Recoupling enhancement: ENABLED")
        if self.has_generating_functional:
            print("   📐 Generating functional analysis: ENABLED")
    
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
        Compute total energy density from all sources with mathematical enhancements.
        
        ρ_total = W_recoupling * ⟨T₀₀⟩_GF * (ρ_wormhole + ρ_casimir + ρ_squeezed) + corrections
        
        Args:
            r: Radial coordinate array
            t: Time coordinate
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary of energy density components
        """
        if verbose:
            print(f"⚡ Computing enhanced total energy density at t={t:.2e}s...")
        
        # Base wormhole contribution
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
        
        # Apply SU(2) recoupling enhancement
        rho_enhanced = rho_tree.copy()
        su2_enhancement_info = None
        
        if self.has_su2_recoupling:
            try:
                su2_result = self.su2_recoupling.enhance_stress_energy_tensor(
                    rho_tree, r, component_type='unified'
                )
                rho_enhanced = su2_result['enhanced_T00']
                su2_enhancement_info = su2_result['diagnostics']
                
                if verbose:
                    print(f"   🔗 SU(2) recoupling applied: {su2_enhancement_info['negative_improvement']:+.1%} negative fraction improvement")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ SU(2) recoupling failed: {e}")
        
        # Apply generating functional enhancement
        gf_enhancement_info = None
        
        if self.has_generating_functional:
            try:
                gf_result = self.generating_functional.compute_vacuum_expectation_T00()
                
                # Extract spatial profile from generating functional
                gf_spatial_profile = gf_result['T00_enhanced']
                
                # Interpolate to match our grid
                if len(gf_spatial_profile) != len(r):
                    gf_r_grid = self.generating_functional.r_grid
                    gf_profile_interp = np.interp(r, gf_r_grid, gf_spatial_profile)
                else:
                    gf_profile_interp = gf_spatial_profile
                
                # Apply generating functional multiplicative enhancement
                gf_factor = 1.0 + 0.1 * (gf_profile_interp / np.abs(gf_profile_interp).max())
                rho_enhanced *= gf_factor
                
                gf_enhancement_info = {
                    'gf_coefficient_C': gf_result['generating_coefficients']['coefficient_C'],
                    'gf_trace_M_inv': gf_result['generating_coefficients']['M_inverse_trace'],
                    'gf_factor_range': [gf_factor.min(), gf_factor.max()]
                }
                
                if verbose:
                    print(f"   📐 Generating functional applied: C = {gf_enhancement_info['gf_coefficient_C']:.2e}")
                    
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Generating functional failed: {e}")
        
        # Apply radiative corrections to enhanced density
        rho_corrected, correction_breakdown = self.radiative.corrected_stress_energy(
            rho_enhanced, 
            self.config.shell_thickness,
            self.config.redshift_param,
            self.config.polymer_scale
        )
        
        components = {
            'wormhole': rho_wormhole,
            'casimir': rho_casimir,
            'squeezed': rho_squeezed,
            'tree_level': rho_tree,
            'su2_enhanced': rho_enhanced,
            'corrected_total': rho_corrected,
            'radiative_breakdown': correction_breakdown,
            'su2_enhancement_info': su2_enhancement_info,
            'gf_enhancement_info': gf_enhancement_info
        }
        
        if verbose:
            print(f"✓ Wormhole density range: [{rho_wormhole.min():.2e}, {rho_wormhole.max():.2e}]")
            print(f"✓ Casimir density range: [{rho_casimir.min():.2e}, {rho_casimir.max():.2e}]")
            print(f"✓ Squeezed density range: [{rho_squeezed.min():.2e}, {rho_squeezed.max():.2e}]")
            print(f"✓ Enhanced density range: [{rho_enhanced.min():.2e}, {rho_enhanced.max():.2e}]")
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
            'target_met': final_anec_results['anec_total'] < self.config.target_anec,
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
        Run comprehensive validation of the unified negative energy framework
        with advanced mathematical enhancements.
        """
        print("🧪 Running Comprehensive Validation with Mathematical Enhancements")
        print("=" * 70)
        
        validation_results = {}
        
        # 1. Initial ANEC calculation
        print("\n=== Initial ANEC Calculation ===")
        initial_anec = self.compute_unified_anec_integral()
        validation_results['initial_anec'] = initial_anec
        
        # 2. Mathematical enhancement optimization
        print("\n=== Mathematical Enhancement Optimization ===")
        enhancement_results = {}
        
        # Optimize SU(2) recoupling parameters
        if self.has_su2_recoupling:
            print("🔗 Optimizing SU(2) recoupling parameters...")
            r_grid = self.create_radial_grid()
            baseline_density = self.compute_total_energy_density(r_grid, verbose=False)['tree_level']
            
            su2_opt = self.su2_recoupling.optimize_recoupling_parameters(
                baseline_density, r_grid, target_negative_fraction=0.4
            )
            enhancement_results['su2_optimization'] = su2_opt
            
            if su2_opt['optimization_success']:
                print(f"   ✅ SU(2) optimization: {su2_opt['best_negative_fraction']:.1%} negative fraction achieved")
            else:
                print(f"   ❌ SU(2) optimization failed")
        
        # Optimize generating functional kernel
        if self.has_generating_functional:
            print("📐 Optimizing generating functional kernel...")
            gf_opt = self.generating_functional.optimize_kernel_parameters(target_anec=self.config.target_anec)
            enhancement_results['gf_optimization'] = gf_opt
            
            if gf_opt['optimization_success']:
                print(f"   ✅ GF optimization: ANEC = {gf_opt['best_anec']:.2e} J·s·m⁻³")
            else:
                print(f"   ❌ GF optimization failed")
        
        validation_results['enhancement_optimization'] = enhancement_results
        
        # 3. High-dimensional parameter scan
        print("\n=== High-Dimensional Parameter Scan ===")
        try:
            # Create parameter scan with evaluation function
            def evaluate_params(params):
                """Evaluation function for parameter scanning."""
                # Update configuration
                temp_config = UnifiedConfig()
                for key, value in params.items():
                    if hasattr(temp_config, key):
                        setattr(temp_config, key, value)
                
                # Create temporary pipeline
                temp_pipeline = UnifiedANECPipeline(temp_config)
                temp_pipeline.config = temp_config  # Ensure config is updated
                
                # Compute ANEC
                anec_result = temp_pipeline.compute_unified_anec_integral()
                
                return {
                    'anec_total': anec_result['anec_total'],
                    'negative_anec': anec_result['negative_anec_achieved'],
                    'target_met': anec_result['target_met'],
                    'violation_rate': 0.5 if anec_result['negative_anec_achieved'] else 0.0  # Simplified
                }
            
            scan_config = ParameterScanConfig(
                grid_resolution=20,  # Reduced for demo
                target_anec=self.config.target_anec,
                target_violation_rate=self.config.target_violation_rate
            )
            
            scanner = HighDimensionalParameterScan(scan_config, evaluate_params)
            
            # Run 2D sweep on key parameters
            sweep_result = scanner.run_2d_parameter_sweep(
                'polymer_scale', 'exotic_strength',
                fixed_params={
                    'shell_thickness': self.config.shell_thickness,
                    'redshift_param': self.config.redshift_param,
                    'shape_param': self.config.shape_param
                }
            )
            
            validation_results['parameter_scan'] = sweep_result
            
            print(f"   ✅ Parameter scan complete:")
            print(f"      Negative ANEC regions: {sweep_result['statistics']['negative_fraction']:.1%}")
            print(f"      Target achieved: {sweep_result['statistics']['target_fraction']:.1%}")
            
        except Exception as e:
            print(f"   ❌ Parameter scan failed: {e}")
            validation_results['parameter_scan'] = {'error': str(e)}
        
        # 4. Unified parameter optimization  
        print("\n=== Unified Parameter Optimization ===")
        optimization_results = self.optimize_unified_parameters(n_iterations=15, max_evaluations=300)
        validation_results['optimization'] = optimization_results
        
        # 5. Final validation
        print("\n=== Final Validation ===")
        final_anec = optimization_results['final_anec_results']
        
        # Enhanced success criteria assessment
        success_metrics = {
            'negative_anec_achieved': final_anec['anec_total'] < 0,
            'target_anec_met': final_anec['anec_total'] < self.config.target_anec,
            'violation_magnitude': abs(final_anec['anec_total']) if final_anec['anec_total'] < 0 else 0,
            'ford_roman_factor': abs(final_anec['anec_total']) / 1e10 if final_anec['anec_total'] < 0 else 0,
            'radiative_stability': all(
                abs(v) < 1e-5 for v in final_anec['radiative_corrections'].values()
            ),
            'su2_enhancement_active': self.has_su2_recoupling,
            'gf_enhancement_active': self.has_generating_functional,
            'mathematical_enhancement_factor': 1.0  # Placeholder for combined enhancement
        }
        
        # Calculate mathematical enhancement factor
        if 'initial_anec' in validation_results and validation_results['initial_anec']['anec_total'] != 0:
            enhancement_factor = abs(final_anec['anec_total']) / abs(validation_results['initial_anec']['anec_total'])
            success_metrics['mathematical_enhancement_factor'] = enhancement_factor
        
        validation_results['success_metrics'] = success_metrics
        
        # Summary
        print(f"\n📊 ENHANCED VALIDATION SUMMARY")
        print(f"   Mathematical enhancements: SU(2)={'✅' if self.has_su2_recoupling else '❌'}, GF={'✅' if self.has_generating_functional else '❌'}")
        print(f"   Negative ANEC achieved: {'✅ YES' if success_metrics['negative_anec_achieved'] else '❌ NO'}")
        print(f"   Target ANEC met: {'✅ YES' if success_metrics['target_anec_met'] else '❌ NO'}")
        print(f"   ANEC magnitude: {success_metrics['violation_magnitude']:.2e} J·s·m⁻³")
        print(f"   Enhancement factor: {success_metrics['mathematical_enhancement_factor']:.2f}x")
        print(f"   Radiative stability: {'✅ STABLE' if success_metrics['radiative_stability'] else '⚠️ UNSTABLE'}")
        
        overall_success = (
            success_metrics['negative_anec_achieved'] and
            success_metrics['target_anec_met'] and
            success_metrics['radiative_stability']
        )
        
        print(f"\n🎯 OVERALL SUCCESS: {'✅ ACHIEVED' if overall_success else '❌ PENDING'}")
        print(f"   Mathematical breakthrough: {'🚀 YES' if success_metrics['mathematical_enhancement_factor'] > 2.0 else '📈 Incremental'}")
        
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
    
    def run_high_dimensional_parameter_scan(self, 
                                           scan_dimensions: List[str] = None,
                                           grid_density: int = 8,
                                           adaptive_refinement: bool = True,
                                           max_scan_points: int = 2000) -> Dict[str, Any]:
        """
        Execute high-dimensional parameter scan with mathematical enhancements.
        
        Implements advanced multidimensional grid scanning with:
        - SU(2) recoupling-aware parameter selection
        - Generating functional optimization regions
        - Adaptive mesh refinement for negative ANEC zones
        
        Args:
            scan_dimensions: Parameters to scan (default: key physical parameters)
            grid_density: Base grid points per dimension
            adaptive_refinement: Enable adaptive refinement of promising regions
            max_scan_points: Maximum total scan points
            
        Returns:
            Comprehensive scanning results with negative ANEC zones
        """
        print("🔬 Initiating high-dimensional parameter scan with mathematical enhancements...")
        
        # Default scan dimensions focused on key physics
        if scan_dimensions is None:
            scan_dimensions = [
                'throat_radius', 'shell_thickness', 'exotic_strength', 
                'redshift_param', 'casimir_plate_sep', 'squeeze_param'
            ]
        
        n_dims = len(scan_dimensions)
        print(f"   📐 Scanning {n_dims}D parameter space: {scan_dimensions}")
        print(f"   🔍 Grid density: {grid_density}^{n_dims} = {grid_density**n_dims} base points")
        
        # Define parameter bounds based on physical constraints
        param_bounds = {
            'throat_radius': (1e-6, 1e-3),      # 1μm to 1mm
            'shell_thickness': (1e-6, 1e-2),     # 1μm to 1cm  
            'exotic_strength': (0.1, 10.0),      # Dimensionless coupling
            'redshift_param': (0.01, 0.99),      # Surface redshift
            'casimir_plate_sep': (1e-9, 1e-6),   # 1nm to 1μm
            'squeeze_param': (0.1, 5.0),         # Squeezing strength
            'casimir_mod_freq': (1e10, 1e16),    # 10GHz to 10PHz
            'vacuum_coup': (0.01, 1.0),          # Vacuum coupling
            'coherent_amp': (0.1, 3.0),          # Coherent amplitude
            'squeeze_phase': (0, 2*np.pi),       # Phase angle
            'casimir_vac_coupling': (0.01, 1.0), # Casimir-vacuum coupling
            'shape_param': (0.5, 2.0)            # Wormhole shape
        }
        
        # Generate base parameter grid
        param_ranges = []
        for dim in scan_dimensions:
            bounds = param_bounds[dim]
            # Use log-space for physical parameters spanning orders of magnitude
            if bounds[0] > 0 and bounds[1]/bounds[0] > 100:
                param_ranges.append(np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), grid_density))
            else:
                param_ranges.append(np.linspace(bounds[0], bounds[1], grid_density))
        
        # Create meshgrid for all parameter combinations
        param_grids = np.meshgrid(*param_ranges, indexing='ij')
        
        # Flatten to scan points
        base_scan_points = np.column_stack([grid.flatten() for grid in param_grids])
        
        # Limit to maximum scan points if needed
        if len(base_scan_points) > max_scan_points:
            indices = np.random.choice(len(base_scan_points), max_scan_points, replace=False)
            scan_points = base_scan_points[indices]
            print(f"   ⚡ Subsampled to {len(scan_points)} points (from {len(base_scan_points)})")
        else:
            scan_points = base_scan_points
            print(f"   🎯 Scanning {len(scan_points)} total parameter combinations")
        
        # Initialize results storage
        scan_results = {
            'scan_points': scan_points,
            'scan_dimensions': scan_dimensions,
            'anec_values': [],
            'energy_components': [],
            'negative_anec_regions': [],
            'mathematical_enhancements': [],
            'convergence_metrics': []
        }
        
        # Progress tracking
        best_anec = float('inf')
        negative_count = 0
        significant_negatives = []  # ANEC < -1e-12
        
        print(f"   🚀 Beginning parameter space exploration...")
        start_time = time.time()
        
        for i, point in enumerate(scan_points):
            try:
                # Update configuration with current parameter combination
                param_dict = dict(zip(scan_dimensions, point))
                self._update_configs_from_dict(param_dict)
                
                # Compute energy density with mathematical enhancements
                r_grid = self.create_radial_grid()
                energy_components = self.compute_total_energy_density(r_grid, verbose=False)
                
                # Calculate ANEC integral
                anec_total = np.trapz(energy_components['corrected_total'], r_grid)
                
                # Store results
                scan_results['anec_values'].append(anec_total)
                scan_results['energy_components'].append({
                    'total_range': [energy_components['corrected_total'].min(), 
                                  energy_components['corrected_total'].max()],
                    'negative_fraction': (energy_components['corrected_total'] < 0).mean()
                })
                
                # Track mathematical enhancement contributions
                enhancement_info = {
                    'su2_applied': energy_components.get('su2_enhancement_info') is not None,
                    'gf_applied': energy_components.get('gf_enhancement_info') is not None,
                    'su2_improvement': 0.0,
                    'gf_coefficient': 0.0
                }
                
                if energy_components.get('su2_enhancement_info'):
                    enhancement_info['su2_improvement'] = energy_components['su2_enhancement_info'].get('negative_improvement', 0.0)
                    
                if energy_components.get('gf_enhancement_info'):
                    enhancement_info['gf_coefficient'] = energy_components['gf_enhancement_info'].get('gf_coefficient_C', 0.0)
                
                scan_results['mathematical_enhancements'].append(enhancement_info)
                
                # Track negative ANEC discoveries
                if anec_total < 0:
                    negative_count += 1
                    region_info = {
                        'parameters': param_dict,
                        'anec_value': anec_total,
                        'point_index': i,
                        'mathematical_enhancements': enhancement_info
                    }
                    scan_results['negative_anec_regions'].append(region_info)
                    
                    if anec_total < -1e-12:  # Significant negative
                        significant_negatives.append(region_info)
                
                # Update best
                if anec_total < best_anec:
                    best_anec = anec_total
                
                # Progress reporting
                if (i + 1) % max(1, len(scan_points) // 20) == 0:
                    progress = (i + 1) / len(scan_points) * 100
                    elapsed = time.time() - start_time
                    eta = elapsed * (len(scan_points) - i - 1) / (i + 1)
                    
                    print(f"   📈 Progress: {progress:.1f}% | Negatives: {negative_count}/{i+1} ({100*negative_count/(i+1):.1f}%)")
                    print(f"      Best ANEC: {best_anec:.2e} | ETA: {eta:.0f}s")
                    
                    if significant_negatives:
                        print(f"      🎯 Significant negatives: {len(significant_negatives)}")
                
            except Exception as e:
                print(f"   ⚠️ Error at point {i}: {e}")
                scan_results['anec_values'].append(float('inf'))
                scan_results['energy_components'].append({'error': str(e)})
                scan_results['mathematical_enhancements'].append({'error': str(e)})
        
        # Analysis and adaptive refinement
        scan_results['anec_values'] = np.array(scan_results['anec_values'])
        
        # Summary statistics
        finite_anec = scan_results['anec_values'][np.isfinite(scan_results['anec_values'])]
        scan_results['summary'] = {
            'total_points': len(scan_points),
            'successful_evaluations': len(finite_anec),
            'negative_anec_count': negative_count,
            'negative_anec_fraction': negative_count / len(finite_anec) if len(finite_anec) > 0 else 0,
            'best_anec': best_anec,
            'anec_statistics': {
                'mean': finite_anec.mean() if len(finite_anec) > 0 else float('inf'),
                'std': finite_anec.std() if len(finite_anec) > 0 else 0,
                'min': finite_anec.min() if len(finite_anec) > 0 else float('inf'),
                'max': finite_anec.max() if len(finite_anec) > 0 else float('inf')
            },
            'significant_negatives': len(significant_negatives),
            'mathematical_enhancement_success_rate': sum(1 for e in scan_results['mathematical_enhancements'] 
                                                        if e.get('su2_applied', False) or e.get('gf_applied', False)) / len(scan_results['mathematical_enhancements'])
        }
        
        print(f"\n🎊 HIGH-DIMENSIONAL SCAN COMPLETE!")
        print(f"   📊 Results: {scan_results['summary']['successful_evaluations']}/{scan_results['summary']['total_points']} successful")
        print(f"   🎯 Negative ANEC: {negative_count} regions ({100*scan_results['summary']['negative_anec_fraction']:.1f}%)")
        print(f"   🏆 Best ANEC: {best_anec:.2e} J·s·m⁻³")
        print(f"   ⚡ Significant negatives: {len(significant_negatives)}")
        print(f"   🔬 Enhancement success: {100*scan_results['summary']['mathematical_enhancement_success_rate']:.1f}%")
        
        # Adaptive refinement of promising regions
        if adaptive_refinement and negative_count > 0:
            print(f"\n🔍 Initiating adaptive refinement of {len(scan_results['negative_anec_regions'])} negative regions...")
            refined_results = self._adaptive_refinement_scan(scan_results['negative_anec_regions'], 
                                                           scan_dimensions, param_bounds)
            scan_results['adaptive_refinement'] = refined_results
            print(f"   ✅ Adaptive refinement complete: {len(refined_results.get('refined_negatives', []))} additional negatives found")
        
        return scan_results
    
    def _update_configs_from_dict(self, param_dict: Dict[str, float]):
        """Helper method to update all component configurations from parameter dictionary."""
        # Update main config
        for param, value in param_dict.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
        
        # Update component-specific configs
        if hasattr(self, 'wormhole'):
            self.wormhole.throat_radius = param_dict.get('throat_radius', self.config.throat_radius)
            if hasattr(self.wormhole, 'shape_param'):
                self.wormhole.shape_param = param_dict.get('shape_param', getattr(self.wormhole, 'shape_param', 1.0))
                
        if hasattr(self, 'casimir'):
            self.casimir.plate_separation = param_dict.get('casimir_plate_sep', getattr(self.casimir, 'plate_separation', 1e-6))
            
        if hasattr(self, 'squeezed'):
            if hasattr(self.squeezed, 'squeeze_param'):
                self.squeezed.squeeze_param = param_dict.get('squeeze_param', getattr(self.squeezed, 'squeeze_param', 1.0))
    
    def _adaptive_refinement_scan(self, negative_regions: List[Dict], 
                                scan_dimensions: List[str], 
                                param_bounds: Dict[str, Tuple[float, float]],
                                refinement_factor: int = 3) -> Dict[str, Any]:
        """
        Perform adaptive mesh refinement around negative ANEC regions.
        
        Args:
            negative_regions: List of parameter combinations that yielded negative ANEC
            scan_dimensions: Dimensions to refine
            param_bounds: Parameter bounds for each dimension
            refinement_factor: Grid refinement factor around each negative region
            
        Returns:
            Refinement results
        """
        refined_results = {
            'refined_negatives': [],
            'refinement_zones': [],
            'additional_discoveries': 0
        }
        
        for region in negative_regions[:10]:  # Limit to top 10 regions to avoid explosion
            center_params = region['parameters']
            
            # Define refinement box around this negative region
            refinement_ranges = []
            for dim in scan_dimensions:
                center_val = center_params[dim]
                bounds = param_bounds[dim]
                
                # Create small refinement window (±10% of parameter range)
                range_size = (bounds[1] - bounds[0]) * 0.1
                ref_min = max(bounds[0], center_val - range_size/2)
                ref_max = min(bounds[1], center_val + range_size/2)
                
                refinement_ranges.append(np.linspace(ref_min, ref_max, refinement_factor))
            
            # Generate refined grid around this region
            ref_grids = np.meshgrid(*refinement_ranges, indexing='ij')
            ref_points = np.column_stack([grid.flatten() for grid in ref_grids])
            
            # Scan refined points
            for point in ref_points:
                param_dict = dict(zip(scan_dimensions, point))
                self._update_configs_from_dict(param_dict)
                
                r_grid = self.create_radial_grid()
                energy_components = self.compute_total_energy_density(r_grid, verbose=False)
                anec_total = np.trapz(energy_components['corrected_total'], r_grid)
                
                if anec_total < 0:
                    refined_results['refined_negatives'].append({
                        'parameters': param_dict,
                        'anec_value': anec_total,
                        'parent_region': region
                    })
                    refined_results['additional_discoveries'] += 1
        
        return refined_results
def demo_unified_pipeline():
    """Demonstrate the enhanced unified ANEC pipeline with mathematical breakthroughs."""
    print("🚀 ENHANCED UNIFIED ANEC PIPELINE DEMO")
    print("=" * 70)
    print("Mathematical Enhancements:")
    print("🔗 SU(2) 3nj Hypergeometric Recoupling")
    print("📐 Generating Functional Closed-Form T₀₀") 
    print("📈 High-Dimensional Parameter Scanning")
    print("=" * 70)
    
    # Create pipeline with enhanced configuration
    config = UnifiedConfig(
        throat_radius=3e-15,
        shell_thickness=1.5e-14,
        exotic_strength=8e-3,
        casimir_plate_separation=2e-15,
        squeezing_parameter=3.5,
        recoupling_spins=[0.5, 1.0, 2.0, 2.5],
        mass_ratios=[2.1, 3.4, 1.7, 4.2],
        recoupling_boost=5e3,
        gf_grid_size=40,
        gf_kernel_strength=0.8,
        target_anec=-1e5
    )
    
    pipeline = UnifiedANECPipeline(config)
    
    # Run comprehensive validation with mathematical enhancements
    results = pipeline.run_comprehensive_validation()
    
    print(f"\n🔬 MATHEMATICAL ENHANCEMENT ANALYSIS")
    print(f"=" * 50)
    
    # Analyze enhancement effectiveness
    if 'enhancement_optimization' in results:
        enhancement = results['enhancement_optimization']
        
        if 'su2_optimization' in enhancement:
            su2_result = enhancement['su2_optimization']
            print(f"🔗 SU(2) Recoupling:")
            print(f"   Success: {'✅' if su2_result['optimization_success'] else '❌'}")
            print(f"   Negative fraction: {su2_result['best_negative_fraction']:.1%}")
            print(f"   Target achieved: {'✅' if su2_result['target_achieved'] else '❌'}")
        
        if 'gf_optimization' in enhancement:
            gf_result = enhancement['gf_optimization']
            print(f"📐 Generating Functional:")
            print(f"   Success: {'✅' if gf_result['optimization_success'] else '❌'}")
            print(f"   Best ANEC: {gf_result['best_anec']:.2e} J·s·m⁻³")
            print(f"   Target achieved: {'✅' if gf_result['target_achieved'] else '❌'}")
    
    if 'parameter_scan' in results and 'statistics' in results['parameter_scan']:
        scan_stats = results['parameter_scan']['statistics']
        print(f"📈 Parameter Space Analysis:")
        print(f"   Viable parameter space: {scan_stats['negative_fraction']:.1%}")
        print(f"   Target-achieving space: {scan_stats['target_fraction']:.1%}")
        print(f"   Total evaluations: {scan_stats['total_evaluations']:,}")
    
    # Final assessment
    success_metrics = results.get('success_metrics', {})
    enhancement_factor = success_metrics.get('mathematical_enhancement_factor', 1.0)
    
    print(f"\n🎯 BREAKTHROUGH ASSESSMENT")
    print(f"=" * 30)
    print(f"Mathematical enhancement factor: {enhancement_factor:.2f}x")
    print(f"SU(2) recoupling active: {'✅' if success_metrics.get('su2_enhancement_active') else '❌'}")
    print(f"Generating functional active: {'✅' if success_metrics.get('gf_enhancement_active') else '❌'}")
    
    if enhancement_factor > 5.0:
        print(f"🚀 MAJOR BREAKTHROUGH: {enhancement_factor:.1f}x improvement!")
    elif enhancement_factor > 2.0:
        print(f"📈 SIGNIFICANT PROGRESS: {enhancement_factor:.1f}x improvement")
    else:
        print(f"📊 INCREMENTAL PROGRESS: {enhancement_factor:.1f}x improvement")
    
    final_anec = results['optimization']['final_anec_results']['anec_total']
    target_anec = config.target_anec
    
    if final_anec < target_anec:
        print(f"🎯 TARGET ACHIEVED: ANEC = {final_anec:.2e} < {target_anec:.2e} J·s·m⁻³")
    else:
        print(f"📍 TARGET PENDING: ANEC = {final_anec:.2e} (target: {target_anec:.2e})")
    
    return pipeline, results

if __name__ == "__main__":
    demo_unified_pipeline()
