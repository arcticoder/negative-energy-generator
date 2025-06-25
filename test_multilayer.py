"""
Standalone test for multilayer metamaterial optimization
"""

import numpy as np

def test_multilayer_metamaterial():
    """
    Test the multilayer metamaterial simulation with direct implementation.
    """
    # Physical parameters
    lattice_const = 250e-9  # 250 nm
    filling_fraction = 0.3
    n_layers = 10
    eta = 0.95  # Layer efficiency
    beta = 0.5  # Saturation exponent
    
    # Baseline Casimir energy
    base_casimir = -1e-15  # J
    
    # Geometric optimization factors
    optimal_lattice = 250e-9
    optimal_filling = 0.35
    lattice_deviation = abs(lattice_const - optimal_lattice) / optimal_lattice
    filling_deviation = abs(filling_fraction - optimal_filling) / optimal_filling
    
    # Enhancement factors
    lattice_factor = 1 / (1 + 5 * lattice_deviation)
    filling_factor = 1 / (1 + 10 * filling_deviation)
    
    # Index contrast enhancement (Si/air)
    n_rod = 3.5
    n_matrix = 1.0
    index_contrast = abs(n_rod**2 - n_matrix**2) / (n_rod**2 + n_matrix**2)
    contrast_factor = 1 + 3 * index_contrast
    
    # Photonic density of states
    freq_factor = 2.0
    
    # Base enhancement
    base_enhancement = lattice_factor * filling_factor * contrast_factor * freq_factor
    E0 = base_casimir * base_enhancement
    
    print(f"Base calculation:")
    print(f"  Lattice factor: {lattice_factor:.3f}")
    print(f"  Filling factor: {filling_factor:.3f}")
    print(f"  Contrast factor: {contrast_factor:.3f}")
    print(f"  Freq factor: {freq_factor:.3f}")
    print(f"  Base enhancement: {base_enhancement:.3f}")
    print(f"  E0: {E0:.2e} J")
    
    # Multi-layer amplification
    k_values = np.arange(1, n_layers + 1)
    layer_contributions = eta * k_values**(-beta)
    layer_amplification = np.sum(layer_contributions)
    
    # Total energy
    total_energy = E0 * layer_amplification
    
    print(f"\nMulti-layer calculation:")
    print(f"  Layer amplification: {layer_amplification:.3f}")
    print(f"  Total energy: {total_energy:.2e} J")
    
    return {
        'base_energy': E0,
        'layer_amplification': layer_amplification,
        'total_energy': total_energy,
        'base_enhancement': base_enhancement
    }

if __name__ == "__main__":
    print("🧪 Standalone Multilayer Metamaterial Test")
    print("=" * 45)
    
    result = test_multilayer_metamaterial()
    
    print(f"\n✅ Test complete!")
    print(f"   Energy enhancement: {abs(result['total_energy']) / 1e-15:.1f}x baseline Casimir")
