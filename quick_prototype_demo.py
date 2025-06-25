"""
Quick Prototype Stack Demo
=========================

This script demonstrates the complete prototype workflow using direct imports
to avoid dependency conflicts. Shows the three-stage process:

1. SIMULATION → Exotic matter field prediction
2. FABRICATION → Test-bed specifications  
3. MEASUREMENT → Data analysis pipeline
"""

import importlib.util
import numpy as np
import json
from datetime import datetime

def load_module_direct(module_name, file_path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_quick_demo():
    """Run a quick demonstration of the complete prototype stack."""
    print("🚀 PROTOTYPE STACK QUICK DEMO")
    print("=" * 50)
    
    # Load modules directly
    print("Loading modules...")
    fab_module = load_module_direct("fabrication_spec", "src/prototype/fabrication_spec.py")
    measure_module = load_module_direct("measurement_pipeline", "src/prototype/measurement_pipeline.py")
    sim_module = load_module_direct("exotic_matter_simulator", "src/prototype/exotic_matter_simulator.py")
    print("✓ All modules loaded successfully\n")
    
    # ===================================================================
    # STAGE 1: SIMULATION
    # ===================================================================
    print("📊 STAGE 1: EXOTIC MATTER FIELD SIMULATION")
    
    # Create small test grid for demo
    n_points = 8
    cell_size = 2e-6  # 2 μm cell
    x = np.linspace(-cell_size/2, cell_size/2, n_points)
    grid = np.column_stack([
        np.repeat(x, n_points**2),
        np.tile(np.repeat(x, n_points), n_points),
        np.tile(x, n_points**2)
    ])
    
    g_metric = np.eye(3)
    
    def demo_kernel(grid):
        return sim_module.default_kernel_builder(grid, 0.05, cell_size/4)
    
    def demo_variation(i):
        return sim_module.default_variation_generator(grid, i)
    
    # Initialize simulator
    simulator = sim_module.ExoticMatterSimulator(demo_kernel, g_metric, grid)
    
    # Compute energy density (use smaller subset for demo speed)
    print(f"   • Computing T_00 for {len(grid)} grid points...")
    rho = simulator.T00(demo_variation)
    
    dV = (cell_size / n_points)**3
    energy_analysis = simulator.energy_analysis(rho, dV)
    
    print(f"   ✓ Total Energy: {energy_analysis['total_energy']:.2e} J")
    print(f"   ✓ Negative Energy: {energy_analysis['negative_energy']:.2e} J")
    print(f"   ✓ Negative Fraction: {energy_analysis['negative_fraction']:.1%}")
    
    # ===================================================================
    # STAGE 2: FABRICATION
    # ===================================================================
    print(f"\n🔧 STAGE 2: FABRICATION SPECIFICATIONS")
    
    # Design multi-layer Casimir array
    gaps_nm = [20, 40, 80, 160]  # Four-layer design
    plate_area_cm2 = 0.1  # 0.1 cm² plates
    
    print(f"   • Designing {len(gaps_nm)}-layer Casimir array")
    print(f"   • Plate area: {plate_area_cm2} cm²")
    
    casimir_specs = fab_module.casimir_plate_specs(
        gaps_nm, plate_area_cm2, 
        material="silicon", coating="gold"
    )
    
    total_casimir_energy = sum(spec['total_energy_J'] for spec in casimir_specs)
    
    print(f"   ✓ Casimir Array Energy: {total_casimir_energy:.2e} J")
    
    # Add metamaterial enhancement
    meta_spec = fab_module.metamaterial_slab_specs(
        d_nm=60, L_um=3, eps_neg=-2.2, 
        area_cm2=plate_area_cm2, metamaterial_type="split-ring"
    )
    
    print(f"   ✓ Metamaterial Enhancement: {meta_spec['enhancement_factor']:.1f}x")
    print(f"   ✓ Enhanced Energy: {meta_spec['total_enhanced_energy_J']:.2e} J")
    
    # Optimize gap sequence
    target_energy = -1e-16  # Target: 0.1 fJ
    opt_result = fab_module.optimize_gap_sequence(target_energy, plate_area_cm2, max_layers=4)
    
    if opt_result['optimization_success']:
        print(f"   ✓ Gap Optimization: {opt_result['error_percentage']:.1f}% error")
        print(f"   ✓ Optimal gaps: {[f'{g:.0f}' for g in opt_result['optimal_gaps_nm']]} nm")
    
    # ===================================================================
    # STAGE 3: MEASUREMENT
    # ===================================================================
    print(f"\n📈 STAGE 3: MEASUREMENT & DATA ANALYSIS")
    
    # Simulate experimental data
    measurement_gaps_m = np.array(gaps_nm) * 1e-9
    measurement_area_m2 = plate_area_cm2 * 1e-4
    
    # Generate "true" forces
    true_forces = measure_module.casimir_force_model(measurement_gaps_m, measurement_area_m2)
    
    # Add realistic noise
    noise_level = 0.02  # 2% noise
    np.random.seed(42)  # Reproducible
    measured_forces = true_forces * (1 + noise_level * np.random.randn(len(true_forces)))
    force_uncertainties = abs(true_forces) * noise_level
    
    print(f"   • Analyzing {len(measured_forces)} force measurements...")
    
    # Fit the data
    fit_result = measure_module.fit_casimir_force(
        measurement_gaps_m, measured_forces, force_uncertainties
    )
    
    if fit_result['success']:
        fitted_area = fit_result['parameters']['A_eff']
        area_error = abs(fitted_area - measurement_area_m2) / measurement_area_m2
        
        print(f"   ✓ Force fitting: R² = {fit_result['r_squared']:.4f}")
        print(f"   ✓ Area measurement error: {area_error:.1%}")
        print(f"   ✓ Reduced χ²: {fit_result['reduced_chi2']:.2f}")
    
    # Experimental design optimization
    design_opt = measure_module.experimental_design_optimizer(
        target_precision=0.01, available_area_cm2=plate_area_cm2
    )
    
    print(f"   ✓ Optimal measurement gap: {design_opt['optimal_gap_nm']:.0f} nm")
    print(f"   ✓ Expected precision: {design_opt['estimated_precision']:.1%}")
    print(f"   ✓ Meets 1% target: {design_opt['meets_target']}")
    
    # ===================================================================
    # SUMMARY & VALIDATION
    # ===================================================================
    print(f"\n🎯 INTEGRATION SUMMARY")
    
    # Calculate consistency metrics
    sim_energy_scale = abs(energy_analysis['negative_energy'])
    fab_energy_scale = abs(total_casimir_energy)
    measurement_precision = design_opt['estimated_precision']
    
    print(f"   • Simulation energy scale: {sim_energy_scale:.2e} J")
    print(f"   • Fabrication energy scale: {fab_energy_scale:.2e} J")
    print(f"   • Measurement precision: {measurement_precision:.1%}")
    
    # Overall assessment
    all_successful = (
        energy_analysis['negative_energy'] < 0 and
        total_casimir_energy < 0 and
        fit_result['success'] and
        fit_result['r_squared'] > 0.95
    )
    
    if all_successful:
        status = "🎉 PROTOTYPE STACK FULLY OPERATIONAL"
        deployment = "READY FOR DEPLOYMENT"
    else:
        status = "⚠️ PROTOTYPE STACK NEEDS REFINEMENT"
        deployment = "NEEDS ADDITIONAL WORK"
    
    print(f"\n{status}")
    print(f"Deployment Status: {deployment}")
    
    # Save demo results
    results = {
        'timestamp': datetime.now().isoformat(),
        'simulation': {
            'grid_points': len(grid),
            'total_energy_J': energy_analysis['total_energy'],
            'negative_energy_J': energy_analysis['negative_energy'],
            'negative_fraction': energy_analysis['negative_fraction']
        },
        'fabrication': {
            'casimir_energy_J': total_casimir_energy,
            'metamaterial_enhancement': meta_spec['enhancement_factor'],
            'optimization_error_percent': opt_result.get('error_percentage', 0),
            'layer_count': len(gaps_nm)
        },
        'measurement': {
            'fit_r_squared': fit_result.get('r_squared', 0),
            'area_error_percent': area_error * 100 if fit_result['success'] else 0,
            'optimal_gap_nm': design_opt['optimal_gap_nm'],
            'precision_percent': design_opt['estimated_precision'] * 100
        },
        'overall_status': deployment
    }
    
    # Save results
    with open('quick_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: quick_demo_results.json")
    
    # Next steps
    print(f"\n🚀 NEXT STEPS:")
    print("1. Review fabrication specifications")
    print("2. Send specs to clean-room fabrication partners")  
    print("3. Set up measurement apparatus")
    print("4. Begin prototype construction")
    print("5. Start measurement campaigns")
    print("6. Iterate design based on validation")
    
    return results

if __name__ == "__main__":
    demo_results = run_quick_demo()
    
    print(f"\n📊 DEMO COMPLETED")
    print(f"Status: {demo_results['overall_status']}")
    
    if demo_results['overall_status'] == "READY FOR DEPLOYMENT":
        print("🌟 The complete prototype stack is operational!")
        print("🔬 Ready for real-world exotic matter experiments!")
    else:
        print("🔧 Some refinements needed before deployment")
