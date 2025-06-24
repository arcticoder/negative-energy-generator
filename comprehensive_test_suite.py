#!/usr/bin/env python3
"""
Comprehensive Test Suite for New Theoretical Components
=====================================================

Tests all newly implemented theoretical components:
- Traversable wormhole ansatz
- Casimir effect enhancement
- Squeezed vacuum states
- Extended radiative corrections (3-loop)
- Unified ANEC pipeline

This script validates functionality and integration before
proceeding with the comprehensive negative energy validation.
"""

import sys
import os
import numpy as np
from typing import Dict, Any
import traceback

# Add module paths
sys.path.append('src')
sys.path.append('src/theoretical')
sys.path.append('src/corrections')
sys.path.append('src/validation')

def test_wormhole_ansatz() -> Dict[str, Any]:
    """Test the traversable wormhole ansatz implementation."""
    print("🌀 Testing Traversable Wormhole Ansatz...")
    
    try:
        from wormhole_ansatz import TraversableWormholeAnsatz, WormholeConfig
        
        # Create test configuration
        config = WormholeConfig(
            throat_radius=1e-14,
            shell_thickness=5e-14,
            exotic_strength=1e-2,
            grid_points=100  # Reduced for testing
        )
        
        wormhole = TraversableWormholeAnsatz(config)
        
        # Test basic geometry calculations
        r_test = np.linspace(config.throat_radius * 1.1, 
                           config.throat_radius + 3 * config.shell_thickness, 
                           50)
        
        # Test shape function
        shape_func = wormhole.morris_thorne_shape_function(r_test)
        print(f"   ✓ Shape function range: [{shape_func.min():.2e}, {shape_func.max():.2e}]")
        
        # Test metric components
        metric = wormhole.metric_components(r_test)
        print(f"   ✓ Metric g_tt range: [{metric['g_tt'].min():.2e}, {metric['g_tt'].max():.2e}]")
        
        # Test stress-energy tensor
        stress_energy = wormhole.stress_energy_tensor(r_test)
        energy_density = stress_energy['energy_density']
        print(f"   ✓ Energy density range: [{energy_density.min():.2e}, {energy_density.max():.2e}]")
        
        # Test ANEC calculation
        anec_value = wormhole.compute_anec_integral()
        print(f"   ✓ Initial ANEC: {anec_value:.2e} J·s·m⁻³")
        
        # Test optimization
        opt_results = wormhole.optimize_for_negative_anec()
        print(f"   ✓ Optimized ANEC: {opt_results['anec_value']:.2e} J·s·m⁻³")
        print(f"   ✓ Negative ANEC achieved: {opt_results['negative_anec_achieved']}")
        
        return {
            'status': 'SUCCESS',
            'anec_initial': anec_value,
            'anec_optimized': opt_results['anec_value'],
            'negative_achieved': opt_results['negative_anec_achieved'],
            'optimization_success': opt_results['optimization_success']
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}

def test_casimir_enhancement() -> Dict[str, Any]:
    """Test the Casimir effect enhancement implementation."""
    print("\n⚡ Testing Casimir Effect Enhancement...")
    
    try:
        from casimir_enhancement import CasimirEnhancement, CasimirConfig
        
        # Create test configuration
        config = CasimirConfig(
            plate_separation=5e-15,
            modulation_frequency=2e15,
            modulation_amplitude=1e-16,
            vacuum_coupling=1e-2
        )
        
        casimir = CasimirEnhancement(config)
        
        # Test energy density calculations
        r_test = np.linspace(1e-14, 1e-13, 50)
        
        # Static Casimir energy
        static_density = casimir.static_casimir_energy_density(r_test)
        print(f"   ✓ Static density range: [{static_density.min():.2e}, {static_density.max():.2e}] J/m³")
        
        # Dynamic enhancement
        dynamic_density = casimir.dynamic_casimir_enhancement(r_test, 0.0)
        print(f"   ✓ Dynamic enhancement range: [{dynamic_density.min():.2e}, {dynamic_density.max():.2e}] J/m³")
        
        # Total enhanced density
        total_density = casimir.total_casimir_energy_density(r_test)
        print(f"   ✓ Total density range: [{total_density.min():.2e}, {total_density.max():.2e}] J/m³")
        
        # Test shell configuration
        throat_radius = 1e-14
        shell_thickness = 5e-14
        shell_density = casimir.casimir_shell_around_throat(r_test, throat_radius, shell_thickness)
        shell_integral = np.trapz(shell_density, r_test)
        print(f"   ✓ Shell integrated energy: {shell_integral:.2e} J/m²")
        
        # Test optimization
        opt_results = casimir.optimize_casimir_parameters()
        print(f"   ✓ Optimized density: {opt_results['energy_density']:.2e} J/m³")
        print(f"   ✓ Target achieved: {opt_results['target_achieved']}")
        
        return {
            'status': 'SUCCESS',
            'static_density_min': static_density.min(),
            'total_density_min': total_density.min(),
            'shell_integral': shell_integral,
            'optimized_density': opt_results['energy_density'],
            'target_achieved': opt_results['target_achieved']
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}

def test_squeezed_vacuum() -> Dict[str, Any]:
    """Test the squeezed vacuum states implementation."""
    print("\n🌌 Testing Squeezed Vacuum States...")
    
    try:
        from squeezed_vacuum import SqueezedVacuumStates, SqueezedVacuumConfig
        
        # Create test configuration
        config = SqueezedVacuumConfig(
            squeezing_parameter=2.5,
            coherent_amplitude=2.0,
            localization_length=2e-14,
            vacuum_coupling=1e-2
        )
        
        squeezed = SqueezedVacuumStates(config)
        
        # Test energy calculations
        r_test = np.linspace(1e-14, 5e-14, 50)
        
        # Energy expectation values
        energy_exp = squeezed.squeezed_energy_expectation(
            config.squeezing_parameter, config.squeezing_phase, config.mode_frequency
        )
        print(f"   ✓ Squeezed energy expectation: {energy_exp:.2e} J")
        
        # Two-mode energy
        two_mode_energy = squeezed.two_mode_squeezed_energy(
            config.squeezing_parameter, config.mode_frequency, config.coupling_frequency
        )
        print(f"   ✓ Two-mode energy: {two_mode_energy:.2e} J")
        
        # Total energy density
        energy_density = squeezed.total_squeezed_energy_density(r_test)
        print(f"   ✓ Energy density range: [{energy_density.min():.2e}, {energy_density.max():.2e}] J/m³")
        
        # Localized energy bump
        bump_energy = squeezed.squeezed_vacuum_bump(r_test, 2e-14, 5e-15, 1e-12)
        print(f"   ✓ Bump energy minimum: {bump_energy.min():.2e} J/m³")
        
        # Test optimization
        opt_results = squeezed.optimize_squeezing_parameters()
        print(f"   ✓ Optimized energy: {opt_results['energy_density']:.2e} J/m³")
        print(f"   ✓ Target achieved: {opt_results['target_achieved']}")
        
        return {
            'status': 'SUCCESS',
            'energy_expectation': energy_exp,
            'two_mode_energy': two_mode_energy,
            'density_min': energy_density.min(),
            'bump_min': bump_energy.min(),
            'optimized_energy': opt_results['energy_density'],
            'target_achieved': opt_results['target_achieved']
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}

def test_extended_radiative() -> Dict[str, Any]:
    """Test the extended radiative corrections with 3-loop Monte Carlo."""
    print("\n🔄 Testing Extended Radiative Corrections...")
    
    try:
        from radiative import RadiativeCorrections
        
        # Create radiative corrections calculator
        corrector = RadiativeCorrections(mass=0.0, coupling=1.0, cutoff=50.0)
        
        # Test parameters
        R = 1e-14
        tau = 1e-15
        mu = 1e-35
        
        # Test 1-loop and 2-loop (existing functionality)
        one_loop = corrector.one_loop_correction(R, tau)
        two_loop = corrector.two_loop_correction(R, tau)
        print(f"   ✓ One-loop correction: {one_loop:.2e}")
        print(f"   ✓ Two-loop correction: {two_loop:.2e}")
        
        # Test new 3-loop Monte Carlo
        three_loop = corrector.three_loop_monte_carlo(R, tau, n_samples=1000)  # Reduced samples for testing
        print(f"   ✓ Three-loop correction: {three_loop:.2e}")
        
        # Test enhanced corrections with polymer effects
        enhanced_corrections = corrector.polymer_enhanced_corrections(R, tau, mu)
        print(f"   ✓ Enhanced one-loop: {enhanced_corrections['one_loop_enhanced']:.2e}")
        print(f"   ✓ Enhanced two-loop: {enhanced_corrections['two_loop_enhanced']:.2e}")
        print(f"   ✓ Enhanced three-loop: {enhanced_corrections['three_loop_enhanced']:.2e}")
        print(f"   ✓ Polymer-specific: {enhanced_corrections['polymer_specific']:.2e}")
        
        # Test corrected stress-energy
        T_00_tree = np.array([1e10, 5e9, -2e9, 1e10])  # Sample tree-level values
        T_00_corrected, breakdown = corrector.corrected_stress_energy(T_00_tree, R, tau, mu)
        
        print(f"   ✓ Tree-level mean: {breakdown['tree_level']:.2e}")
        print(f"   ✓ Corrected mean: {breakdown['corrected_mean']:.2e}")
        print(f"   ✓ Total correction: {breakdown['total_correction']:.2e}")
        
        return {
            'status': 'SUCCESS',
            'one_loop': one_loop,
            'two_loop': two_loop,
            'three_loop': three_loop,
            'enhanced_corrections': enhanced_corrections,
            'correction_breakdown': breakdown
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}

def test_unified_pipeline() -> Dict[str, Any]:
    """Test the unified ANEC pipeline integration."""
    print("\n🚀 Testing Unified ANEC Pipeline...")
    
    try:
        from unified_anec_pipeline import UnifiedANECPipeline, UnifiedConfig
        
        # Create test configuration
        config = UnifiedConfig(
            throat_radius=5e-15,
            shell_thickness=2e-14,
            exotic_strength=5e-3,
            casimir_plate_separation=3e-15,
            squeezing_parameter=3.0,
            target_anec=-1e5,
            grid_points=100,  # Reduced for testing
            mc_samples=1000   # Reduced for testing
        )
        
        pipeline = UnifiedANECPipeline(config)
        
        # Test radial grid creation
        r_grid = pipeline.create_radial_grid()
        print(f"   ✓ Radial grid: {len(r_grid)} points, range [{r_grid.min():.2e}, {r_grid.max():.2e}] m")
        
        # Test total energy density computation
        energy_components = pipeline.compute_total_energy_density(r_grid)
        
        print(f"   ✓ Wormhole density range: [{energy_components['wormhole'].min():.2e}, {energy_components['wormhole'].max():.2e}]")
        print(f"   ✓ Casimir density range: [{energy_components['casimir'].min():.2e}, {energy_components['casimir'].max():.2e}]")
        print(f"   ✓ Squeezed density range: [{energy_components['squeezed'].min():.2e}, {energy_components['squeezed'].max():.2e}]")
        print(f"   ✓ Total corrected range: [{energy_components['corrected_total'].min():.2e}, {energy_components['corrected_total'].max():.2e}]")
        
        # Test ANEC integral computation
        anec_results = pipeline.compute_unified_anec_integral(r_grid)
        
        print(f"   ✓ ANEC wormhole: {anec_results['anec_wormhole']:.2e} J·s·m⁻³")
        print(f"   ✓ ANEC Casimir: {anec_results['anec_casimir']:.2e} J·s·m⁻³")
        print(f"   ✓ ANEC squeezed: {anec_results['anec_squeezed']:.2e} J·s·m⁻³")
        print(f"   ✓ ANEC total: {anec_results['anec_total']:.2e} J·s·m⁻³")
        print(f"   ✓ Negative ANEC achieved: {anec_results['negative_anec_achieved']}")
        print(f"   ✓ Target met: {anec_results['target_met']}")
        
        return {
            'status': 'SUCCESS',
            'anec_results': anec_results,
            'energy_components': {
                'wormhole_min': energy_components['wormhole'].min(),
                'casimir_min': energy_components['casimir'].min(),
                'squeezed_min': energy_components['squeezed'].min(),
                'total_min': energy_components['corrected_total'].min()
            },
            'pipeline_functional': True
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}

def run_comprehensive_test_suite():
    """Run the complete test suite for all new components."""
    print("🧪 COMPREHENSIVE TEST SUITE FOR NEW THEORETICAL COMPONENTS")
    print("=" * 70)
    
    test_results = {}
    
    # Run all tests
    test_results['wormhole'] = test_wormhole_ansatz()
    test_results['casimir'] = test_casimir_enhancement()
    test_results['squeezed'] = test_squeezed_vacuum()
    test_results['radiative'] = test_extended_radiative()
    test_results['unified'] = test_unified_pipeline()
    
    # Summarize results
    print(f"\n📊 TEST SUITE SUMMARY")
    print("=" * 30)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() if result['status'] == 'SUCCESS')
    
    for test_name, result in test_results.items():
        status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_emoji} {test_name.title()} Test: {result['status']}")
    
    print(f"\nOverall Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    
    # Key achievements
    if test_results['unified']['status'] == 'SUCCESS':
        anec_results = test_results['unified']['anec_results']
        print(f"\n🎯 KEY ACHIEVEMENTS:")
        print(f"   • Unified pipeline functional: ✅")
        print(f"   • Total ANEC: {anec_results['anec_total']:.2e} J·s·m⁻³")
        print(f"   • Negative ANEC achieved: {'✅' if anec_results['negative_anec_achieved'] else '❌'}")
        print(f"   • Target met: {'✅' if anec_results['target_met'] else '❌'}")
    
    # Next steps
    if successful_tests == total_tests:
        print(f"\n🚀 READY FOR COMPREHENSIVE VALIDATION")
        print("   All components tested successfully!")
        print("   Proceeding to full parameter optimization...")
    else:
        print(f"\n⚠️  ISSUES DETECTED")
        print("   Some components failed testing.")
        print("   Review errors before proceeding.")
    
    return test_results

if __name__ == "__main__":
    results = run_comprehensive_test_suite()
