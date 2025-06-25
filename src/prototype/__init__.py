"""
Unified Negative Energy Generation Prototype Stack
================================================

This package provides a complete prototype stack for exotic matter research
and negative energy generation, transitioning from theoretical frameworks
to practical ML-powered implementations.

The stack includes:

**Core Modules:**
1. ExoticMatterSimulator - Field simulation and T₀₀ computation
2. FabricationSpec - Manufacturing specifications and optimization
3. MeasurementPipeline - Real-time data analysis and design optimization

**Modern Hardware Modules:**
4. SuperconductingResonator - DCE-based negative energy via superconducting cavities
5. JosephsonParametricAmplifier - Squeezed vacuum states for negative energy extraction
6. PhotonicCrystalEngine - Engineered vacuum fluctuations via photonic band gaps

All modules are designed to be independently testable, well-documented,
and ready for deployment in experimental settings.
"""

import numpy as np

# Core prototype modules
try:
    from .exotic_matter_simulator import ExoticMatterSimulator, default_kernel_builder, default_variation_generator
    from .fabrication_spec import (
        casimir_plate_specs, 
        metamaterial_slab_specs, 
        multi_layer_casimir_specs,
        optimize_gap_sequence
    )
    from .measurement_pipeline import (
        casimir_force_model,
        fit_casimir_force,
        analyze_time_series,
        frequency_shift_analysis,
        real_time_data_processor,
        experimental_design_optimizer,
        enhanced_casimir_model
    )
    
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import core prototype modules: {e}")
    CORE_MODULES_AVAILABLE = False

# Modern hardware modules
try:
    from .superconducting_resonator import SuperconductingResonator
    from .jpa_squeezer_vacuum import JosephsonParametricAmplifier
    from .photonic_crystal_engine import PhotonicCrystalEngine
    
    HARDWARE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import hardware modules: {e}")
    HARDWARE_MODULES_AVAILABLE = False

__version__ = "2.0.0"
__author__ = "Negative Energy Research Team"

# Build exports based on what's available
__all__ = []

if CORE_MODULES_AVAILABLE:
    __all__.extend([
        # Exotic matter simulation
        'ExoticMatterSimulator',
        'default_kernel_builder', 
        'default_variation_generator',
        
        # Fabrication specifications
        'casimir_plate_specs',
        'metamaterial_slab_specs',
        'multi_layer_casimir_specs', 
        'optimize_gap_sequence',
        'enhanced_casimir_model',
        
        # Measurement pipeline
        'casimir_force_model',
        'fit_casimir_force',
        'analyze_time_series',
        'frequency_shift_analysis',
        'real_time_data_processor',
        'experimental_design_optimizer'
    ])

if HARDWARE_MODULES_AVAILABLE:
    __all__.extend([
        # Hardware modules
        'SuperconductingResonator',
        'JosephsonParametricAmplifier',
        'PhotonicCrystalEngine'
    ])

def get_prototype_info():
    """Get information about the prototype stack."""
    return {
        "version": __version__,
        "modules_total": len(__all__),
        "core_modules_available": CORE_MODULES_AVAILABLE,
        "hardware_modules_available": HARDWARE_MODULES_AVAILABLE,
        "core_capabilities": [
            "Exotic matter field simulation",
            "Casimir effect modeling", 
            "Fabrication specification generation",
            "Real-time measurement analysis",
            "Design optimization workflows"
        ],
        "hardware_capabilities": [
            "Superconducting DCE resonators",
            "JPA squeezed vacuum generation", 
            "Photonic crystal vacuum modification",
            "Multi-frequency operation",
            "Real-time parameter tuning"
        ] if HARDWARE_MODULES_AVAILABLE else [],
        "integration_ready": CORE_MODULES_AVAILABLE and HARDWARE_MODULES_AVAILABLE,
        "ml_compatible": True
    }

def create_combined_system():
    """
    Create a combined system integrating all prototype modules.
    
    Returns:
        Dictionary with initialized system components
    """
    if not CORE_MODULES_AVAILABLE:
        raise ImportError("Core modules not available - cannot create combined system")
    
    print("🚀 Initializing Complete Negative Energy Prototype System")
    
    # Create proper grid and kernel for ExoticMatterSimulator
    n_grid = 15
    x = np.linspace(-1.0, 1.0, n_grid)
    grid = np.array([
        np.tile(np.repeat(x, n_grid), n_grid), 
        np.tile(np.repeat(x, n_grid), n_grid), 
        np.tile(x, n_grid**2)
    ])
    g_metric = np.eye(3)
    
    def system_kernel(grid):
        return default_kernel_builder(grid, coupling_strength=0.1, decay_length=0.5)
    
    # Core modules
    system = {
        'core': {
            'simulator': ExoticMatterSimulator(system_kernel, g_metric, grid),
            'fabrication': None,  # FabricationSpec is function-based
            'measurement': None   # MeasurementPipeline is function-based
        },
        'hardware': {},
        'capabilities': get_prototype_info()
    }
    
    # Add hardware modules if available
    if HARDWARE_MODULES_AVAILABLE:
        system['hardware'] = {
            'resonator': SuperconductingResonator(
                base_frequency=5.0e9,
                quality_factor=1e6,
                temperature=0.01
            ),
            'jpa': JosephsonParametricAmplifier(
                signal_frequency=6.0e9,
                pump_frequency=12.0e9,
                temperature=0.01
            ),
            'crystal': PhotonicCrystalEngine(
                lattice_constant=500e-9,
                filling_fraction=0.3,
                operating_frequency=600e12
            )
        }
    
    print("✅ Complete prototype system initialized")
    print(f"   • Core modules: {len([k for k, v in system['core'].items() if v is not None])}")
    print(f"   • Hardware modules: {len(system['hardware'])}")
    print(f"   • Integration ready: {system['capabilities']['integration_ready']}")
    
    return system
