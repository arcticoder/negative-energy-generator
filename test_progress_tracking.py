#!/usr/bin/env python3
"""
Test script for unified ANEC pipeline with progress tracking.
"""

import sys
import os
sys.path.append('src')

from src.validation.unified_anec_pipeline import UnifiedANECPipeline, UnifiedConfig

def test_progress_tracking():
    """Test the progress tracking functionality."""
    print("🧪 Testing Progress Tracking for Unified ANEC Pipeline")
    print("=" * 60)
    
    # Create configuration with reasonable parameters for quick testing
    config = UnifiedConfig(
        throat_radius=2e-15,
        shell_thickness=1e-14,
        exotic_strength=1e-2,
        casimir_plate_separation=1e-15,
        squeezing_parameter=2.5,
        target_anec=-1e4,  # More achievable target for testing
        grid_points=200,   # Smaller grid for faster computation
    )
    
    print(f"Configuration:")
    print(f"  Target ANEC: {config.target_anec:.2e} J·s·m⁻³")
    print(f"  Grid points: {config.grid_points}")
    print(f"  Throat radius: {config.throat_radius:.2e} m")
    
    # Initialize pipeline
    pipeline = UnifiedANECPipeline(config)
    
    # Test initial ANEC calculation
    print(f"\n=== Initial ANEC Test ===")
    initial_results = pipeline.compute_unified_anec_integral()
    
    # Quick optimization test (few iterations)
    print(f"\n=== Quick Optimization Test (10 iterations) ===")
    optimization_results = pipeline.optimize_unified_parameters(n_iterations=10)
    
    print(f"\n✅ Progress tracking test complete!")
    print(f"   Evaluations performed: {pipeline.evaluation_count}")
    print(f"   Progress history length: {len(pipeline.progress_history)}")
    print(f"   Best ANEC found: {pipeline.best_anec:.2e} J·s·m⁻³")
    
    return pipeline, optimization_results

if __name__ == "__main__":
    test_progress_tracking()
