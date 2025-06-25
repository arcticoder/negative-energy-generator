"""
Hardware Integration Module
==========================

High-intensity field drivers for enhanced negative energy extraction.

This module provides three advanced hardware platforms:
1. High-Intensity Laser Boundary Pump (E₀ ~ 10¹⁵ V/m)
2. Capacitive/Inductive Field Rigs (combined E×B fields)
3. Polymer QFT Coupling Inserts (4D ansatz mapping)

Each module implements real physics models with safety constraints
and optimization algorithms for parameter discovery.
"""

from .high_intensity_laser import (
    simulate_high_intensity_laser,
    optimize_high_intensity_laser,
    analyze_field_scaling
)

from .field_rig_design import (
    simulate_capacitive_rig,
    simulate_inductive_rig,
    simulate_combined_rig,
    optimize_field_rigs,
    analyze_scaling_laws
)

from .polymer_insert import (
    generate_polymer_mesh,
    compute_polymer_negative_energy,
    optimize_polymer_insert,
    gaussian_ansatz_4d,
    vortex_ansatz_4d,
    standing_wave_ansatz_4d,
    benchmark_ansatz_functions
)

__all__ = [
    # High-intensity laser module
    'simulate_high_intensity_laser',
    'optimize_high_intensity_laser',
    'analyze_field_scaling',
    
    # Field rig design module
    'simulate_capacitive_rig',
    'simulate_inductive_rig', 
    'simulate_combined_rig',
    'optimize_field_rigs',
    'analyze_scaling_laws',
    
    # Polymer insert module
    'generate_polymer_mesh',
    'compute_polymer_negative_energy',
    'optimize_polymer_insert',
    'gaussian_ansatz_4d',
    'vortex_ansatz_4d',
    'standing_wave_ansatz_4d',
    'benchmark_ansatz_functions'
]

__version__ = "1.0.0"
__author__ = "Physics-Driven Prototype Team"
