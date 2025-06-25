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

# Core prototype modules
try:
    from .exotic_matter_simulator import ExoticMatterSimulator, default_kernel_builder, default_variation_generator
    from .fabrication_spec import (
        casimir_plate_specs, 
        metamaterial_slab_specs, 
        multi_layer_casimir_specs,
        optimize_gap_sequence,
        enhanced_casimir_model
    )
    from .measurement_pipeline import (
        casimir_force_model,
        fit_casimir_force,
        analyze_time_series,
        frequency_shift_analysis,
        real_time_data_processor,
        experimental_design_optimizer
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
    
    # Core modules
    system = {
        'core': {
            'simulator': ExoticMatterSimulator(grid_size=100, domain_size=2.0),
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
            np.tile(np.repeat(x, n_grid), n_grid), 
            np.tile(x, n_grid**2)
        ])
        
        g_metric = np.eye(3)
        
        def test_kernel(grid):
            return default_kernel_builder(grid, coupling_strength=0.05, decay_length=0.3)
        
        def test_variation(i):
            return default_variation_generator(grid, i)
        
        simulator = ExoticMatterSimulator(test_kernel, g_metric, grid)
        rho = simulator.T00(test_variation)
        
        dV = (1.0 / n_grid)**3
        analysis = simulator.energy_analysis(rho, dV)
        
        print(f"   ✓ Total Energy: {analysis['total_energy']:.2e} J")
        print(f"   ✓ Negative Energy: {analysis['negative_energy']:.2e} J")
        print(f"   ✓ Negative Fraction: {analysis['negative_fraction']:.1%}")
        
        # 2. Fabrication Specifications
        print("\n2. FABRICATION: Test-bed Specifications")
        print("   - Generating Casimir array specs...")
        
        gaps = [20, 50, 100]  # nm
        area = 0.1  # cm²
        specs = casimir_plate_specs(gaps, area, material="silicon", coating="gold")
        
        total_casimir_energy = sum(spec['total_energy_J'] for spec in specs)
        print(f"   ✓ {len(specs)} Casimir layers designed")
        print(f"   ✓ Total Casimir Energy: {total_casimir_energy:.2e} J")
        
        # Add metamaterial enhancement
        meta_spec = metamaterial_slab_specs(50, 5, -3.0, area_cm2=area)
        print(f"   ✓ Metamaterial Enhancement: {meta_spec['enhancement_factor']:.1f}x")
        print(f"   ✓ Enhanced Energy: {meta_spec['total_enhanced_energy_J']:.2e} J")
        
        # 3. Measurement Pipeline  
        print("\n3. MEASUREMENT: Data Analysis Pipeline")
        print("   - Setting up real-time processing...")
        
        # Simulate measurement data
        test_gaps = np.array([20, 30, 50, 80, 120]) * 1e-9  # nm to m
        test_area = area * 1e-4  # cm² to m²
        true_forces = casimir_force_model(test_gaps, test_area)
        noisy_forces = true_forces * (1 + 0.03 * np.random.randn(len(true_forces)))
        
        fit_result = fit_casimir_force(test_gaps, noisy_forces)
        
        if fit_result['success']:
            fitted_area = fit_result['parameters']['A_eff']
            relative_error = abs(fitted_area - test_area) / test_area
            print(f"   ✓ Force fitting successful")
            print(f"   ✓ Area measurement error: {relative_error:.1%}")
            print(f"   ✓ Fit R²: {fit_result['r_squared']:.3f}")
        
        # Experimental design optimization
        design = experimental_design_optimizer(
            target_precision=0.02,  # 2% precision target
            available_area_cm2=area
        )
        
        print(f"   ✓ Optimal gap: {design['optimal_gap_nm']:.1f} nm")
        print(f"   ✓ Expected precision: {design['estimated_precision']:.1%}")
        
        print("\n🚀 COMPLETE PROTOTYPE STACK OPERATIONAL")
        print("   → Ready for build → measure → validate → iterate")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False

# Auto-run demo if module is imported
if __name__ == "__main__":
    status = get_prototype_status()
    print("Prototype Stack Status:")
    for module, info in status['modules'].items():
        status_symbol = "✓" if info['available'] else "✗"
        print(f"  {status_symbol} {module}: {info['description']}")
    
    if all(info['available'] for info in status['modules'].values()):
        print("\n" + "="*50)
        demo_complete_stack()
    else:
        print("\nSome modules not available - check imports")
