#!/usr/bin/env python3
"""
Quick diagnostic test for ANEC pipeline issues.
"""

import sys
sys.path.append('src/validation')
sys.path.append('src/theoretical')
sys.path.append('src/corrections')

# Test direct import
try:
    from unified_anec_pipeline import UnifiedANECPipeline, UnifiedConfig
    print('✅ Import successful')
    
    # Create pipeline with diagnostic-friendly config
    config = UnifiedConfig(
        target_anec=-1e4, 
        grid_points=200,  # Manageable size
        throat_radius=1e-15,
        shell_thickness=5e-15,
        exotic_strength=1e-2,  # Higher for more negative energy
        casimir_plate_separation=1e-15,  # Smaller for stronger Casimir
        squeezing_parameter=3.0,  # Higher squeezing
    )
    
    pipeline = UnifiedANECPipeline(config)
    print('✅ Pipeline initialization successful')
    
    # Run diagnostic first
    print('\n=== ANEC Problem Diagnosis ===')
    diagnostic = pipeline.diagnose_anec_problem()
    
    # Test limited optimization with timeout
    print('\n=== Limited Optimization Test ===')
    results = pipeline.optimize_unified_parameters(n_iterations=5, max_evaluations=100)
    print(f'✅ Optimization complete: {pipeline.evaluation_count} evaluations')
    print(f'Best ANEC: {pipeline.best_anec:.2e} J·s·m⁻³')
    print(f'Stagnation count: {pipeline.stagnation_count}')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
