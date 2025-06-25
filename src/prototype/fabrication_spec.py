"""
Fabrication Specifications for Test-bed Geometries
==================================================

This module provides fabrication specifications for experimental test-bed geometries
including Casimir plate arrays and metamaterial slabs. It calculates the theoretical
energy densities and total energies for various configurations.

Mathematical Foundation:
    Casimir Plates: ρ_C(d) = -π²ℏc/(720d⁴), E_C = ρ_C(d) * A
    Metamaterial: E_meta = -ρ_C(d) * A * L / √(|ε_eff|)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json


# Physical constants (SI units)
HBAR = 1.054571817e-34  # Reduced Planck constant [J⋅s]
C = 2.99792458e8        # Speed of light [m/s]
PI = np.pi


def casimir_plate_specs(gaps_nm: List[float], plate_area_cm2: float, 
                       material: str = "silicon", coating: Optional[str] = None) -> List[Dict]:
    """
    Generate fabrication specifications for Casimir plate arrays.
    
    For parallel plates separated by gap d with area A:
    ρ_C(d) = -π²ℏc/(720d⁴)
    E_C = ρ_C(d) × A
    
    Args:
        gaps_nm: List of gap widths in nanometers
        plate_area_cm2: Single-plate area in cm²
        material: Plate material (affects fabrication specs)
        coating: Optional coating material for enhanced effect
        
    Returns:
        List of dictionaries with fabrication specifications
    """
    A = plate_area_cm2 * 1e-4  # Convert cm² to m²
    specs = []
    
    for d_nm in gaps_nm:
        d = d_nm * 1e-9  # Convert nm to m
        
        # Casimir energy density
        rho = -PI**2 * HBAR * C / (720 * d**4)
        
        # Total Casimir energy
        E = rho * A
        
        # Force per unit area (pressure)
        pressure = -PI**2 * HBAR * C / (240 * d**4)
        
        # Total force
        force = pressure * A
        
        # Fabrication tolerances
        tolerance_nm = max(0.1, d_nm * 0.001)  # 0.1% of gap or 0.1 nm minimum
        
        spec = {
            'gap_nm': d_nm,
            'gap_m': d,
            'plate_area_cm2': plate_area_cm2,
            'plate_area_m2': A,
            'material': material,
            'coating': coating,
            'energy_density_J_per_m2': rho,
            'total_energy_J': E,
            'casimir_pressure_Pa': pressure,
            'casimir_force_N': force,
            'tolerance_nm': tolerance_nm,
            'fabrication_notes': _get_fabrication_notes(d_nm, material, coating)
        }
        
        specs.append(spec)
    
    return specs


def metamaterial_slab_specs(d_nm: float, L_um: float, eps_neg: float, 
                           area_cm2: float = 1.0, metamaterial_type: str = "split-ring") -> Dict:
    """
    Generate fabrication specifications for metamaterial slabs.
    
    For a metamaterial slab with effective permittivity ε < 0:
    E_meta = -ρ_C(d) × A × L / √(|ε_eff|)
    
    Args:
        d_nm: Gap width in nanometers
        L_um: Slab thickness in micrometers
        eps_neg: Negative effective permittivity
        area_cm2: Slab area in cm²
        metamaterial_type: Type of metamaterial structure
        
    Returns:
        Dictionary with fabrication specifications
    """
    # Unit conversions
    d = d_nm * 1e-9      # nm to m
    L = L_um * 1e-6      # μm to m
    A = area_cm2 * 1e-4  # cm² to m²
    
    # Base Casimir energy density
    rho_c = -PI**2 * HBAR * C / (720 * d**4)
    
    # Enhanced energy in metamaterial
    E_meta = rho_c * A * L / np.sqrt(abs(eps_neg))
    
    # Enhanced energy density
    rho_meta = E_meta / (A * L)
    
    # Fabrication requirements
    feature_size_nm = d_nm / 10  # Metamaterial features should be ~10x smaller than gap
    
    spec = {
        'gap_nm': d_nm,
        'thickness_um': L_um,
        'area_cm2': area_cm2,
        'eps_negative': eps_neg,
        'metamaterial_type': metamaterial_type,
        'base_casimir_density_J_per_m3': rho_c / d,  # Volume density
        'enhanced_energy_density_J_per_m3': rho_meta,
        'total_enhanced_energy_J': E_meta,
        'enhancement_factor': 1 / np.sqrt(abs(eps_neg)),
        'min_feature_size_nm': feature_size_nm,
        'fabrication_method': _get_metamaterial_method(metamaterial_type),
        'fabrication_notes': _get_metamaterial_notes(metamaterial_type, feature_size_nm)
    }
    
    return spec


def multi_layer_casimir_specs(layer_gaps_nm: List[float], plate_area_cm2: float, 
                             n_layers: int) -> Dict:
    """
    Generate specifications for multi-layer Casimir arrays.
    
    Args:
        layer_gaps_nm: List of gap widths for each layer
        plate_area_cm2: Area of each plate
        n_layers: Number of layers in the array
        
    Returns:
        Comprehensive specifications for multi-layer system
    """
    total_energy = 0
    layer_specs = []
    
    for i, gap_nm in enumerate(layer_gaps_nm):
        layer_spec = casimir_plate_specs([gap_nm], plate_area_cm2)[0]
        layer_spec['layer_index'] = i
        layer_specs.append(layer_spec)
        total_energy += layer_spec['total_energy_J']
    
    # Overall system specs
    total_thickness_um = sum(layer_gaps_nm) / 1000  # nm to μm
    
    spec = {
        'n_layers': n_layers,
        'total_thickness_um': total_thickness_um,
        'total_area_cm2': plate_area_cm2,
        'total_energy_J': total_energy,
        'average_gap_nm': np.mean(layer_gaps_nm),
        'gap_variation_nm': np.std(layer_gaps_nm),
        'layer_specifications': layer_specs,
        'fabrication_complexity': _assess_fabrication_complexity(layer_gaps_nm),
        'assembly_notes': _get_assembly_notes(n_layers, layer_gaps_nm)
    }
    
    return spec


def optimize_gap_sequence(target_energy_J: float, plate_area_cm2: float, 
                         max_layers: int = 10) -> Dict:
    """
    Optimize gap sequence to achieve target negative energy.
    
    Args:
        target_energy_J: Target total energy (negative)
        plate_area_cm2: Available plate area
        max_layers: Maximum number of layers allowed
        
    Returns:
        Optimized gap sequence and specifications
    """
    from scipy.optimize import minimize
    
    def energy_objective(gaps_nm):
        """Objective: minimize |total_energy - target_energy|"""
        specs = [casimir_plate_specs([gap], plate_area_cm2)[0] for gap in gaps_nm]
        total_E = sum(spec['total_energy_J'] for spec in specs)
        return abs(total_E - target_energy_J)
    
    # Initial guess: uniform gaps around optimal single-gap value
    A = plate_area_cm2 * 1e-4
    d_optimal = (PI**2 * HBAR * C * A / (720 * abs(target_energy_J)))**(1/4)
    initial_gaps = [d_optimal * 1e9] * min(max_layers, 5)  # Convert to nm
    
    # Bounds: gaps between 1 nm and 1000 nm
    bounds = [(1, 1000) for _ in initial_gaps]
    
    # Optimize
    result = minimize(energy_objective, initial_gaps, bounds=bounds, method='L-BFGS-B')
    
    optimal_gaps = result.x
    achieved_specs = multi_layer_casimir_specs(optimal_gaps, plate_area_cm2, len(optimal_gaps))
    
    return {
        'target_energy_J': target_energy_J,
        'achieved_energy_J': achieved_specs['total_energy_J'],
        'error_percentage': 100 * abs(achieved_specs['total_energy_J'] - target_energy_J) / abs(target_energy_J),
        'optimal_gaps_nm': optimal_gaps.tolist(),
        'n_layers_used': len(optimal_gaps),
        'specifications': achieved_specs,
        'optimization_success': result.success,
        'optimization_message': result.message
    }


def _get_fabrication_notes(gap_nm: float, material: str, coating: Optional[str]) -> List[str]:
    """Generate fabrication notes based on gap size and materials."""
    notes = []
    
    if gap_nm < 10:
        notes.append("Requires electron-beam lithography for precision")
        notes.append("Ultra-clean environment essential")
        notes.append("Atomic force microscopy for gap verification")
    elif gap_nm < 100:
        notes.append("Photolithography suitable")
        notes.append("Standard cleanroom protocols")
        notes.append("Optical interferometry for gap measurement")
    else:
        notes.append("Standard microfabrication techniques")
        notes.append("Mechanical spacers acceptable")
    
    if material == "silicon":
        notes.append("Standard silicon wafer processing")
        notes.append("DRIE etching for vertical walls")
    elif material == "gold":
        notes.append("Sputter deposition on substrate")
        notes.append("Consider adhesion layer (Ti/Cr)")
    
    if coating:
        notes.append(f"Apply {coating} coating via {_get_coating_method(coating)}")
    
    return notes


def _get_metamaterial_method(metamaterial_type: str) -> str:
    """Get fabrication method for metamaterial type."""
    methods = {
        "split-ring": "Electron-beam lithography + metal deposition",
        "wire-array": "Focused ion beam milling",
        "fishnet": "Multi-layer lithography",
        "hyperbolic": "Alternating thin film deposition"
    }
    return methods.get(metamaterial_type, "Custom fabrication required")


def _get_metamaterial_notes(metamaterial_type: str, feature_size_nm: float) -> List[str]:
    """Generate metamaterial fabrication notes."""
    notes = []
    
    if feature_size_nm < 50:
        notes.append("Extreme UV lithography or electron-beam required")
        notes.append("Sub-wavelength feature control critical")
    
    if metamaterial_type == "split-ring":
        notes.append("Ensure gap continuity in ring structures")
        notes.append("Control ring coupling distance precisely")
    elif metamaterial_type == "hyperbolic":
        notes.append("Layer thickness uniformity critical")
        notes.append("Interface roughness must be minimized")
    
    return notes


def _get_coating_method(coating: str) -> str:
    """Get deposition method for coating material."""
    methods = {
        "gold": "sputtering or evaporation",
        "silver": "thermal evaporation",
        "aluminum": "sputtering",
        "graphene": "CVD transfer"
    }
    return methods.get(coating, "specialized deposition")


def _assess_fabrication_complexity(gaps_nm: List[float]) -> str:
    """Assess overall fabrication complexity."""
    min_gap = min(gaps_nm)
    gap_ratio = max(gaps_nm) / min_gap
    
    if min_gap < 10 or gap_ratio > 100:
        return "High - requires specialized techniques"
    elif min_gap < 50 or gap_ratio > 10:
        return "Medium - advanced lithography needed"
    else:
        return "Low - standard microfabrication"


def _get_assembly_notes(n_layers: int, gaps_nm: List[float]) -> List[str]:
    """Generate assembly notes for multi-layer systems."""
    notes = []
    
    if n_layers > 5:
        notes.append("Consider modular assembly approach")
        notes.append("Precision alignment fixtures required")
    
    if any(gap < 20 for gap in gaps_nm):
        notes.append("Assembly in controlled atmosphere")
        notes.append("Anti-stiction coatings recommended")
    
    notes.append(f"Total stack height: {sum(gaps_nm)/1000:.2f} μm")
    notes.append("Verify gap uniformity across full area")
    
    return notes


# Example usage and testing
if __name__ == "__main__":
    print("=== Casimir Plate Array Specifications ===")
    
    # Test single gaps
    gaps = [10, 20, 50, 100, 200]  # nm
    area = 1.0  # cm²
    
    specs = casimir_plate_specs(gaps, area, material="silicon", coating="gold")
    
    for spec in specs:
        print(f"\nGap: {spec['gap_nm']} nm")
        print(f"  Energy: {spec['total_energy_J']:.2e} J")
        print(f"  Force: {spec['casimir_force_N']:.2e} N")
        print(f"  Tolerance: ±{spec['tolerance_nm']:.2f} nm")
    
    print("\n=== Metamaterial Slab Specifications ===")
    
    # Test metamaterial
    meta_spec = metamaterial_slab_specs(50, 10, -2.5, area_cm2=1.0)
    print(f"Enhanced Energy: {meta_spec['total_enhanced_energy_J']:.2e} J")
    print(f"Enhancement Factor: {meta_spec['enhancement_factor']:.2f}")
    print(f"Min Feature Size: {meta_spec['min_feature_size_nm']:.1f} nm")
    
    print("\n=== Multi-Layer Optimization ===")
    
    # Test optimization
    target_energy = -1e-15  # 1 fJ negative energy
    opt_result = optimize_gap_sequence(target_energy, area, max_layers=5)
    print(f"Target: {target_energy:.2e} J")
    print(f"Achieved: {opt_result['achieved_energy_J']:.2e} J")
    print(f"Error: {opt_result['error_percentage']:.1f}%")
    print(f"Optimal gaps: {[f'{g:.1f}' for g in opt_result['optimal_gaps_nm']]} nm")
