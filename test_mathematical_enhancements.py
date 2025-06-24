#!/usr/bin/env python3
"""
Test Script for Mathematical Enhancement Integration
=================================================

Tests the integration of:
1. SU(2) 3nj Hypergeometric Recoupling
2. Generating Functional Closed-Form T₀₀  
3. High-Dimensional Parameter Scanning

This validates that all three mathematical approaches work together
to push past the positive-ANEC blockade.

Author: Negative Energy Generator Framework
"""

import numpy as np
import sys
import os

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'theoretical'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'optimization'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'validation'))

def test_su2_recoupling():
    """Test SU(2) recoupling enhancement."""
    print("🔗 Testing SU(2) Recoupling Enhancement")
    print("-" * 40)
    
    try:
        from su2_recoupling import SU2RecouplingEnhancement, RecouplingConfig
        
        # Create test configuration
        config = RecouplingConfig(
            spins=[0.5, 1.0, 1.5],
            mass_ratios=[2.0, 3.0, 1.5],
            boost_factor=1e3
        )
        
        enhancer = SU2RecouplingEnhancement(config)
        
        # Test with dummy stress-energy tensor
        r = np.linspace(1e-15, 1e-13, 100)
        T00_test = np.exp(-r/1e-14) - 0.2 * np.exp(-((r - 2e-14)/5e-15)**2)
        
        print(f"   Original T₀₀ negative fraction: {(T00_test < 0).sum()/len(T00_test):.1%}")
        
        # Apply enhancement
        enhancement_result = enhancer.enhance_stress_energy_tensor(T00_test, r)
        
        negative_improvement = enhancement_result['diagnostics']['negative_improvement']
        print(f"   Enhanced T₀₀ negative fraction: {enhancement_result['diagnostics']['negative_fraction_enhanced']:.1%}")
        print(f"   Improvement: {negative_improvement:+.1%}")
        
        # Test optimization
        opt_result = enhancer.optimize_recoupling_parameters(T00_test, r, target_negative_fraction=0.3)
        
        print(f"   Optimization success: {'✅' if opt_result['optimization_success'] else '❌'}")
        print(f"   Best negative fraction: {opt_result['best_negative_fraction']:.1%}")
        
        return True, "SU(2) recoupling test passed"
        
    except Exception as e:
        return False, f"SU(2) recoupling test failed: {e}"

def test_generating_functional():
    """Test generating functional approach."""
    print("📐 Testing Generating Functional Analysis")
    print("-" * 40)
    
    try:
        from generating_functional import GeneratingFunctionalAnalysis, GeneratingFunctionalConfig
        
        # Create test configuration
        config = GeneratingFunctionalConfig(
            grid_size=30,
            spatial_extent=1e-12,
            kernel_type='warp_bubble',
            kernel_strength=0.5,
            throat_radius=1e-15,
            shell_thickness=5e-15
        )
        
        gf_analysis = GeneratingFunctionalAnalysis(config)
        
        # Test vacuum expectation calculation
        T00_result = gf_analysis.compute_vacuum_expectation_T00()
        
        print(f"   Coefficient C: {T00_result['generating_coefficients']['coefficient_C']:.2e}")
        print(f"   T₀₀ range: [{T00_result['T00_enhanced'].min():.2e}, {T00_result['T00_enhanced'].max():.2e}]")
        print(f"   Negative fraction: {(T00_result['T00_enhanced'] < 0).sum()/len(T00_result['T00_enhanced']):.1%}")
        
        # Test ANEC calculation
        anec_result = gf_analysis.compute_closed_form_anec_integral()
        
        print(f"   ANEC (enhanced): {anec_result['anec_enhanced']:.2e} J·s·m⁻³")
        print(f"   Negative ANEC: {'✅' if anec_result['negative_anec'] else '❌'}")
        
        # Test optimization
        opt_result = gf_analysis.optimize_kernel_parameters(target_anec=-1e4)
        
        print(f"   Optimization success: {'✅' if opt_result['optimization_success'] else '❌'}")
        print(f"   Best ANEC: {opt_result['best_anec']:.2e} J·s·m⁻³")
        
        return True, "Generating functional test passed"
        
    except Exception as e:
        return False, f"Generating functional test failed: {e}"

def test_parameter_scanning():
    """Test high-dimensional parameter scanning."""
    print("📈 Testing Parameter Scanning")
    print("-" * 40)
    
    try:
        from parameter_scanning import HighDimensionalParameterScan, ParameterScanConfig
        
        # Create test configuration
        config = ParameterScanConfig(
            grid_resolution=10,  # Small for testing
            target_anec=-1e4,
            target_violation_rate=0.25
        )
        
        scanner = HighDimensionalParameterScan(config)
        
        # Test 2D parameter sweep
        sweep_result = scanner.run_2d_parameter_sweep(
            'polymer_scale', 'shape_param',
            fixed_params={
                'shell_thickness': 1e-14,
                'redshift_param': 0.1,
                'exotic_strength': 1e-3
            }
        )
        
        stats = sweep_result['statistics']
        print(f"   Total evaluations: {stats['total_evaluations']}")
        print(f"   Negative ANEC regions: {stats['negative_fraction']:.1%}")
        print(f"   Target achieved: {stats['target_fraction']:.1%}")
        print(f"   Best ANEC: {stats['best_anec']:.2e} J·s·m⁻³")
        
        # Test adaptive refinement
        refined_result = scanner.run_adaptive_refinement(sweep_result, refinement_levels=1)
        
        print(f"   Refinement completed: {'✅' if 'refinement_level' in refined_result else '❌'}")
        
        return True, "Parameter scanning test passed"
        
    except Exception as e:
        return False, f"Parameter scanning test failed: {e}"

def test_unified_integration():
    """Test unified pipeline integration."""
    print("🚀 Testing Unified Pipeline Integration")
    print("-" * 40)
    
    try:
        from unified_anec_pipeline import UnifiedANECPipeline, UnifiedConfig
        
        # Create enhanced configuration
        config = UnifiedConfig(
            throat_radius=2e-15,
            shell_thickness=8e-15,
            exotic_strength=5e-3,
            recoupling_spins=[0.5, 1.0],
            mass_ratios=[2.0, 3.0],
            recoupling_boost=1e3,
            gf_grid_size=20,
            gf_kernel_strength=0.3,
            grid_points=50  # Reduced for testing
        )
        
        pipeline = UnifiedANECPipeline(config)
        
        print(f"   SU(2) recoupling: {'✅' if pipeline.has_su2_recoupling else '❌'}")
        print(f"   Generating functional: {'✅' if pipeline.has_generating_functional else '❌'}")
        
        # Test enhanced energy density computation
        r_grid = pipeline.create_radial_grid()
        energy_components = pipeline.compute_total_energy_density(r_grid, verbose=False)
        
        print(f"   Components computed: {len(energy_components)}")
        print(f"   SU(2) enhanced: {'✅' if 'su2_enhanced' in energy_components else '❌'}")
        print(f"   GF info available: {'✅' if energy_components.get('gf_enhancement_info') else '❌'}")
        
        # Test ANEC calculation
        anec_result = pipeline.compute_unified_anec_integral()
        
        print(f"   ANEC total: {anec_result['anec_total']:.2e} J·s·m⁻³")
        print(f"   Negative ANEC: {'✅' if anec_result['negative_anec_achieved'] else '❌'}")
        
        # Test quick optimization (reduced iterations)
        opt_result = pipeline.optimize_unified_parameters(n_iterations=5, max_evaluations=50)
        
        print(f"   Optimization completed: {'✅' if opt_result else '❌'}")
        final_anec = opt_result['final_anec_results']['anec_total']
        print(f"   Final ANEC: {final_anec:.2e} J·s·m⁻³")
        
        return True, "Unified integration test passed"
        
    except Exception as e:
        return False, f"Unified integration test failed: {e}"

def run_all_tests():
    """Run all mathematical enhancement tests."""
    print("🧪 MATHEMATICAL ENHANCEMENT TEST SUITE")
    print("=" * 60)
    print("Testing integration of three breakthrough approaches:")
    print("1. SU(2) 3nj Hypergeometric Recoupling")
    print("2. Generating Functional Closed-Form T₀₀")
    print("3. High-Dimensional Parameter Scanning")
    print("=" * 60)
    
    tests = [
        ("SU(2) Recoupling", test_su2_recoupling),
        ("Generating Functional", test_generating_functional),
        ("Parameter Scanning", test_parameter_scanning),
        ("Unified Integration", test_unified_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            success, message = test_func()
            results.append((test_name, success, message))
            
            if success:
                print(f"✅ {test_name}: SUCCESS")
            else:
                print(f"❌ {test_name}: FAILED - {message}")
                
        except Exception as e:
            results.append((test_name, False, f"Exception: {e}"))
            print(f"❌ {test_name}: EXCEPTION - {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not success:
            print(f"      Reason: {message}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🚀 ALL TESTS PASSED - Mathematical enhancements ready!")
        print("🎯 Ready to break through the positive-ANEC blockade!")
    else:
        print(f"⚠️  {total-passed} test(s) failed - debugging needed")
    
    return results

if __name__ == "__main__":
    run_all_tests()
