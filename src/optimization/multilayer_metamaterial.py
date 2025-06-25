"""
Multi-layer Metamaterial Stacking with Inter-layer Coupling
===========================================================

Mathematical Foundation:
Extends single-layer metamaterial analysis to N≥10 layers with:
- Per-layer attenuation factor η ≲ 1
- Saturation exponent β to model coupling degradation
- Layer amplification: Σ(k=1 to N) η·k^(-β)

Total energy: E_total = E₀ × layer_amplification

Physical basis:
- Each additional layer contributes diminishing returns
- Inter-layer electromagnetic coupling causes saturation
- Optimal N exists due to competing enhancement vs. losses
"""

import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


def simulate_multilayer_metamaterial(lattice_const: float, filling_fraction: float,
                                   n_layers: int, eta: float = 0.95, beta: float = 0.5,
                                   n_rod: float = 3.5, n_matrix: float = 1.0) -> Dict:
    """
    Model N-layer metamaterial stacking with per-layer attenuation and saturation.
    
    Mathematical model:
    layer_amplification = Σ(k=1 to N) η·k^(-β)
    total_energy = E₀ × layer_amplification
    
    Args:
        lattice_const: Lattice constant (m)
        filling_fraction: Rod area / unit cell area
        n_layers: Number of metamaterial layers (recommend N≥10)
        eta: Per-layer efficiency factor (≲1, accounts for losses)
        beta: Saturation exponent (>0, models coupling degradation)
        n_rod: Rod refractive index
        n_matrix: Matrix refractive index
    
    Returns:
        Dictionary with multi-layer metamaterial results
    """
    # Import the base single-layer simulation
    try:
        # Try to import from the main physics validation script
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from physics_driven_prototype_validation import simulate_photonic_metamaterial_energy
        
        # Get base result for single layer
        base_result = simulate_photonic_metamaterial_energy(
            lattice_const, filling_fraction, n_layers=1, n_rod=n_rod, n_matrix=n_matrix
        )
        E0 = base_result['total_energy']
        base_enhancement = base_result['enhancement_factor']
        
    except ImportError:
        # Fallback implementation
        print("   ⚠️  Using fallback single-layer model")
        # Simplified baseline Casimir energy with proper scaling
        base_casimir = -1e-15  # J (baseline Casimir energy between plates)
        
        # Geometric optimization factors
        optimal_lattice = 250e-9
        optimal_filling = 0.35
        lattice_deviation = abs(lattice_const - optimal_lattice) / optimal_lattice
        filling_deviation = abs(filling_fraction - optimal_filling) / optimal_filling
        
        # Enhancement factors (ensure positive values)
        lattice_factor = 1 / (1 + 5 * lattice_deviation)
        filling_factor = 1 / (1 + 10 * filling_deviation)
        
        # Index contrast enhancement (realistic for Si/air)
        index_contrast = abs(n_rod**2 - n_matrix**2) / (n_rod**2 + n_matrix**2)
        contrast_factor = 1 + 3 * index_contrast  # Enhancement due to high contrast
        
        # Photonic density of states modification
        freq_factor = 2.0  # Typical DOS enhancement in photonic crystals
        
        # Total single-layer enhancement
        base_enhancement = lattice_factor * filling_factor * contrast_factor * freq_factor
        E0 = base_casimir * base_enhancement
        
        print(f"   📊 Fallback: E0 = {E0:.2e} J, enhancement = {base_enhancement:.2f}")
        
        # Ensure we have a valid base result structure
        base_result = {
            'total_energy': E0,
            'enhancement_factor': base_enhancement
        }
    
    # Extract base values for multi-layer calculation
    E0 = base_result['total_energy']
    base_enhancement = base_result['enhancement_factor']
    
    # Multi-layer amplification with saturation
    # Σ(k=1 to N) η·k^(-β)
    k_values = np.arange(1, n_layers + 1)
    layer_contributions = eta * k_values**(-beta)
    layer_amplification = np.sum(layer_contributions)
    
    # Total energy with multi-layer enhancement
    total_energy = E0 * layer_amplification
    
    # Energy density (per unit cell)
    unit_cell_volume = lattice_const**3
    energy_density = total_energy / unit_cell_volume
    
    # Layer efficiency metrics
    linear_prediction = n_layers * eta  # What we'd get without saturation
    saturation_factor = layer_amplification / linear_prediction
    effective_layers = layer_amplification / eta  # Equivalent number of ideal layers
    
    # Fabrication complexity metrics
    total_thickness = n_layers * lattice_const
    aspect_ratio = total_thickness / lattice_const
    fabrication_difficulty = 1 + 0.1 * np.log(n_layers)  # Logarithmic complexity growth
    
    # Inter-layer coupling strength
    coupling_strength = 1 - saturation_factor  # Higher coupling → more saturation
    
    return {
        'total_energy': total_energy,
        'energy_density': energy_density,
        'layer_amplification': layer_amplification,
        'base_energy': E0,
        'base_enhancement': base_enhancement,
        'n_layers': n_layers,
        'eta': eta,
        'beta': beta,
        'layer_contributions': layer_contributions,
        'saturation_factor': saturation_factor,
        'effective_layers': effective_layers,
        'linear_prediction': linear_prediction,
        'total_thickness': total_thickness,
        'aspect_ratio': aspect_ratio,
        'fabrication_difficulty': fabrication_difficulty,
        'coupling_strength': coupling_strength,
        'lattice_constant': lattice_const,
        'filling_fraction': filling_fraction,
        'optimization_score': -total_energy  # For minimization
    }


def optimize_layer_count(lattice_const: float, filling_fraction: float,
                        eta: float = 0.95, beta: float = 0.5,
                        max_layers: int = 50) -> Dict:
    """
    Find optimal number of layers for maximum negative energy.
    
    Args:
        lattice_const: Lattice constant (m)
        filling_fraction: Filling fraction
        eta: Layer efficiency
        beta: Saturation exponent
        max_layers: Maximum layers to consider
    
    Returns:
        Dictionary with optimization results
    """
    print(f"🔍 Optimizing layer count (max N={max_layers})")
    
    layers_range = range(1, max_layers + 1)
    energies = []
    amplifications = []
    
    best_result = None
    best_energy = 0  # Looking for most negative
    
    for N in layers_range:
        result = simulate_multilayer_metamaterial(
            lattice_const, filling_fraction, N, eta, beta
        )
        
        energies.append(result['total_energy'])
        amplifications.append(result['layer_amplification'])
        
        if result['total_energy'] < best_energy:
            best_energy = result['total_energy']
            best_result = result.copy()
    
    # Find optimal point and saturation characteristics
    energies = np.array(energies)
    optimal_idx = np.argmin(energies)
    optimal_layers = layers_range[optimal_idx]
    
    # Saturation analysis
    energy_gradient = np.gradient(energies)
    saturation_point = None
    for i, grad in enumerate(energy_gradient[5:], 5):  # Skip first few points
        if abs(grad) < 0.01 * abs(energies[0]):  # 1% of initial energy
            saturation_point = layers_range[i]
            break
    
    print(f"   ✅ Optimal layers: N = {optimal_layers}")
    print(f"   🎯 Best energy: {best_energy:.2e} J")
    print(f"   📈 Amplification: {best_result['layer_amplification']:.2f}")
    print(f"   🔄 Saturation at: N ≈ {saturation_point or 'Not reached'}")
    
    return {
        'optimal_layers': optimal_layers,
        'optimal_energy': best_energy,
        'optimal_result': best_result,
        'layers_range': list(layers_range),
        'energies': energies.tolist(),
        'amplifications': amplifications,
        'saturation_point': saturation_point,
        'energy_gradient': energy_gradient.tolist()
    }


def parameter_sweep_multilayer(lattice_range: Tuple[float, float] = (200e-9, 400e-9),
                             filling_range: Tuple[float, float] = (0.2, 0.5),
                             layer_range: Tuple[int, int] = (10, 30),
                             n_points: int = 15) -> Dict:
    """
    Parameter sweep for multi-layer metamaterial optimization.
    
    Args:
        lattice_range: (min, max) lattice constants (m)
        filling_range: (min, max) filling fractions
        layer_range: (min, max) layer counts
        n_points: Number of points per dimension
    
    Returns:
        Dictionary with sweep results
    """
    print(f"🔄 Multi-layer parameter sweep ({n_points}³ points)")
    
    # Parameter grids
    lattice_values = np.linspace(lattice_range[0], lattice_range[1], n_points)
    filling_values = np.linspace(filling_range[0], filling_range[1], n_points)
    layer_values = np.linspace(layer_range[0], layer_range[1], n_points, dtype=int)
    
    best_result = None
    best_energy = 0
    all_results = []
    
    total_evaluations = n_points**3
    evaluation_count = 0
    
    for i, lattice in enumerate(lattice_values):
        for j, filling in enumerate(filling_values):
            for k, layers in enumerate(layer_values):
                evaluation_count += 1
                
                result = simulate_multilayer_metamaterial(lattice, filling, layers)
                all_results.append({
                    'lattice_const': lattice,
                    'filling_fraction': filling,
                    'n_layers': layers,
                    'total_energy': result['total_energy'],
                    'layer_amplification': result['layer_amplification'],
                    'fabrication_difficulty': result['fabrication_difficulty']
                })
                
                if result['total_energy'] < best_energy:
                    best_energy = result['total_energy']
                    best_result = result.copy()
                    best_result['lattice_const'] = lattice
                    best_result['filling_fraction'] = filling
                
                # Progress update
                if evaluation_count % (total_evaluations // 10) == 0:
                    progress = evaluation_count / total_evaluations * 100
                    print(f"   Progress: {progress:.0f}% complete")
    
    print(f"✅ Parameter sweep complete!")
    print(f"   🎯 Best configuration:")
    print(f"      • Lattice: {best_result['lattice_const']*1e9:.1f} nm")
    print(f"      • Filling: {best_result['filling_fraction']:.2f}")
    print(f"      • Layers: {best_result['n_layers']}")
    print(f"      • Energy: {best_result['total_energy']:.2e} J")
    print(f"      • Amplification: {best_result['layer_amplification']:.2f}")
    
    return {
        'best_result': best_result,
        'all_results': all_results,
        'parameter_ranges': {
            'lattice': lattice_range,
            'filling': filling_range,
            'layers': layer_range
        },
        'n_evaluations': evaluation_count
    }


if __name__ == "__main__":
    print("🧪 Multi-layer Metamaterial Testing")
    print("=" * 40)
    
    # Test 1: Quick layer count optimization for N≥10
    print("\n1️⃣  Layer Count Optimization (N≥10)")
    best_layers = None
    best_energy = 0
    
    for N in range(10, 21):
        result = simulate_multilayer_metamaterial(250e-9, 0.3, N)
        print(f"   N={N:2d}: {result['total_energy']:.2e} J (amp: {result['layer_amplification']:.2f})")
        
        if result['total_energy'] < best_energy:
            best_energy = result['total_energy']
            best_layers = {'N': N, **result}
    
    print(f"\n   🏆 Best N≥10: N={best_layers['N']} with {best_layers['total_energy']:.2e} J")
    
    # Test 2: Saturation analysis
    print("\n2️⃣  Saturation Analysis")
    saturation_result = optimize_layer_count(250e-9, 0.3, max_layers=30)
    
    # Test 3: Parameter sensitivity
    print("\n3️⃣  Parameter Sensitivity")
    eta_values = [0.9, 0.95, 0.99]
    beta_values = [0.3, 0.5, 0.7]
    
    print("   η (efficiency) vs β (saturation):")
    for eta in eta_values:
        for beta in beta_values:
            result = simulate_multilayer_metamaterial(250e-9, 0.3, 15, eta, beta)
            print(f"      η={eta:.2f}, β={beta:.1f}: {result['total_energy']:.2e} J")
    
    print("\n✅ Multi-layer metamaterial testing complete!")
