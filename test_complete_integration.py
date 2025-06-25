"""
Complete Integration Test for ML-Powered Negative Energy Prototype Stack
========================================================================

This script tests all components of the modernized prototype stack including:
- Core modules (simulator, fabrication, measurement)
- Hardware modules (resonator, JPA, photonic crystal) 
- ML optimization modules (Bayesian, genetic, PINN)
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_core_modules():
    """Test core prototype modules."""
    print("🔧 Testing Core Prototype Modules...")
    
    try:
        from src.prototype import ExoticMatterSimulator, default_kernel_builder, default_variation_generator
        import numpy as np
        
        # Create proper grid and parameters
        n_grid = 10
        x = np.linspace(-0.5, 0.5, n_grid)
        grid = np.array([
            np.tile(np.repeat(x, n_grid), n_grid), 
            np.tile(np.repeat(x, n_grid), n_grid), 
            np.tile(x, n_grid**2)
        ])
        g_metric = np.eye(3)
        
        def test_kernel(grid):
            return default_kernel_builder(grid, coupling_strength=0.05, decay_length=0.3)
        
        def test_variation(i):
            return default_variation_generator(grid, i)
        
        # Create simulator with proper interface
        simulator = ExoticMatterSimulator(test_kernel, g_metric, grid)
        rho = simulator.T00(test_variation)
        
        dV = (1.0 / n_grid)**3
        analysis = simulator.energy_analysis(rho, dV)
        total_energy = analysis['total_energy']
        
        print(f"   ✅ ExoticMatterSimulator: Energy = {total_energy:.3e}")
    except Exception as e:
        print(f"   ❌ ExoticMatterSimulator: {e}")
        return False
    
    try:
        from src.prototype import casimir_plate_specs
        specs = casimir_plate_specs([100], 1.0)  # 100nm gap, 1 cm² area
        print(f"   ✅ FabricationSpec: Force = {specs[0]['casimir_force_N']:.2e} N")
    except Exception as e:
        print(f"   ❌ FabricationSpec: {e}")
        return False
    
    try:
        from src.prototype import fit_casimir_force
        import numpy as np
        gaps = np.linspace(50e-9, 200e-9, 10)
        forces = [-2.4e-16 / gap**4 for gap in gaps]
        fit_result = fit_casimir_force(gaps, forces)
        print(f"   ✅ MeasurementPipeline: R² = {fit_result['r_squared']:.4f}")
    except Exception as e:
        print(f"   ❌ MeasurementPipeline: {e}")
        return False
    
    return True

def test_hardware_modules():
    """Test modern hardware modules."""
    print("\n⚡ Testing Hardware Modules...")
    
    try:
        from src.prototype import SuperconductingResonator
        resonator = SuperconductingResonator(base_frequency=5e9, quality_factor=1e6)
        result = resonator.set_parametric_pump(amplitude=0.1, frequency=10e9)
        print(f"   ✅ SuperconductingResonator: DCE rate = {result['dce_rate_theoretical']:.2e} s⁻¹")
    except Exception as e:
        print(f"   ❌ SuperconductingResonator: {e}")
        return False
    
    try:
        from src.prototype import JosephsonParametricAmplifier
        jpa = JosephsonParametricAmplifier(signal_frequency=6e9, pump_frequency=12e9)
        result = jpa.configure_squeezing(target_squeezing_db=10.0)
        print(f"   ✅ JosephsonParametricAmplifier: Squeezing = {result['achieved_squeezing_db']:.1f} dB")
    except Exception as e:
        print(f"   ❌ JosephsonParametricAmplifier: {e}")
        return False
    
    try:
        from src.prototype import PhotonicCrystalEngine
        crystal = PhotonicCrystalEngine(lattice_constant=500e-9, filling_fraction=0.3)
        band_structure = crystal.calculate_band_structure(n_k_points=20)
        print(f"   ✅ PhotonicCrystalEngine: Band gaps = {len(band_structure['band_gaps'])}")
    except Exception as e:
        print(f"   ❌ PhotonicCrystalEngine: {e}")
        return False
    
    return True

def test_ml_modules():
    """Test ML optimization modules."""
    print("\n🧠 Testing ML Optimization Modules...")
    
    # Test Bayesian Optimization
    try:
        from src.ml.bo_ansatz_opt import ExoticMatterBayesianOptimizer
        
        def mock_objective(params):
            return -(params[0]**2 + params[1]**2)  # Simple quadratic
        
        optimizer = ExoticMatterBayesianOptimizer(mock_objective, [(-2, 2), (-2, 2)])
        result = optimizer.optimize(n_calls=10)
        print(f"   ✅ BayesianOptimization: Best = {result.fun:.3f} at {result.x}")
    except Exception as e:
        print(f"   ❌ BayesianOptimization: {e}")
    
    # Test Genetic Algorithm
    try:
        from src.ml.genetic_ansatz import GeneticAnsatzOptimizer
        
        def fitness_function(individual):
            return (-sum(x**2 for x in individual),)
        
        optimizer = GeneticAnsatzOptimizer(lambda x: fitness_function(x)[0], genome_length=3)
        result = optimizer.optimize(n_generations=10)
        print(f"   ✅ GeneticAlgorithm: Best fitness = {result['best_fitness']:.3f}")
    except Exception as e:
        print(f"   ❌ GeneticAlgorithm: {e}")
    
    # Test PINN (if PyTorch available)
    try:
        from src.ml import ProfileNet, ExoticMatterPINN, mock_warp_bubble_energy_computer
        
        network = ProfileNet(input_dim=3, hidden_dim=16, n_layers=2)
        pinn = ExoticMatterPINN(network, mock_warp_bubble_energy_computer)
        result = pinn.train(n_epochs=10, batch_size=100, verbose=False)
        print(f"   ✅ PINN: Final loss = {result['final_loss']:.3e}")
    except Exception as e:
        print(f"   ❌ PINN: {e}")

def test_integration():
    """Test full system integration."""
    print("\n🚀 Testing Full System Integration...")
    
    try:
        from src.prototype import create_combined_system
        system = create_combined_system()
        
        core_count = len([k for k, v in system['core'].items() if v is not None])
        hardware_count = len(system['hardware'])
        
        print(f"   ✅ Combined System: {core_count} core + {hardware_count} hardware modules")
        print(f"   ✅ Integration ready: {system['capabilities']['integration_ready']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Integration: {e}")
        return False

def run_complete_test():
    """Run complete test suite."""
    print("=" * 70)
    print("🧪 COMPLETE ML-POWERED NEGATIVE ENERGY PROTOTYPE STACK TEST")
    print("=" * 70)
    
    results = {
        'core': test_core_modules(),
        'hardware': test_hardware_modules(),
        'integration': test_integration()
    }
    
    # ML tests (optional dependencies)
    print("\n🧠 Testing ML Modules (Optional Dependencies)...")
    test_ml_modules()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY:")
    print("=" * 70)
    
    for category, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {category.upper()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nCore Tests: {total_passed}/{total_tests} passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL CORE TESTS PASSED! Prototype stack ready for deployment.")
        print("\n🎯 NEXT STEPS:")
        print("1. Install ML dependencies: pip install scikit-optimize deap torch")
        print("2. Run ML optimization benchmarks")
        print("3. Connect to experimental hardware")
        print("4. Begin negative energy extraction experiments")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Check module imports and dependencies.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = run_complete_test()
    sys.exit(0 if success else 1)
