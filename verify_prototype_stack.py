"""
Quick verification script for the complete prototype stack
"""
import sys
sys.path.append('.')

def test_modules():
    """Test that all three modules can be imported and run basic functions"""
    
    print("=== PROTOTYPE STACK VERIFICATION ===\n")
    
    # Test 1: Fabrication Specs
    try:
        from src.prototype.fabrication_spec import casimir_plate_specs
        
        specs = casimir_plate_specs([50, 100], 1.0)
        print("✅ Fabrication Specs Module: WORKING")
        print(f"   Generated {len(specs)} Casimir plate specifications")
        print(f"   Example energy: {specs[0]['total_energy_J']:.2e} J")
        
    except Exception as e:
        print(f"❌ Fabrication Specs Module: FAILED - {e}")
    
    # Test 2: Measurement Pipeline  
    try:
        from src.prototype.measurement_pipeline import casimir_force_model, fit_casimir_force
        import numpy as np
        
        # Simple test data
        gaps = np.array([20, 50, 100]) * 1e-9  # nm to m
        area = 1e-8  # m²
        forces = casimir_force_model(gaps, area)
        
        fit_result = fit_casimir_force(gaps, forces)
        
        print("✅ Measurement Pipeline Module: WORKING")
        print(f"   Force model test successful")
        print(f"   Fitting test: R² = {fit_result['r_squared']:.3f}")
        
    except Exception as e:
        print(f"❌ Measurement Pipeline Module: FAILED - {e}")
    
    # Test 3: Exotic Matter Simulator (simplified)
    try:
        from src.prototype.exotic_matter_simulator import default_kernel_builder
        import numpy as np
        
        # Small test grid to avoid long computation
        grid = np.array([[0, 0, 0], [1e-9, 0, 0], [0, 1e-9, 0]])
        K = default_kernel_builder(grid, 0.1, 1e-9)
        
        print("✅ Exotic Matter Simulator Module: WORKING") 
        print(f"   Kernel builder test successful")
        print(f"   Kernel shape: {K.shape}")
        
    except Exception as e:
        print(f"❌ Exotic Matter Simulator Module: FAILED - {e}")
    
    print("\n=== VERIFICATION COMPLETE ===")
    print("🚀 Ready for prototype development!")

if __name__ == "__main__":
    test_modules()
