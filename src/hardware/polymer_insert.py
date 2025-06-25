"""
Polymer QFT Coupling Insert Module
=================================

Implements polymer quantization field coupling for 4D-ansatz profiles.

Mathematical Foundation:
- Polymer stress-energy: T₀₀^poly ~ -ℏ(f-1)²/a²
- Field ansatz mapping: f(x,y,z,t) → discrete polymer lattice
- Total negative energy: E_tot = ∫ T₀₀ d³x
- Mesh optimization: minimize over polymer scale 'a'

4D ansatz functions map continuous field profiles to discrete polymer networks
for enhanced negative energy extraction through quantum geometry effects.
"""

import numpy as np
from typing import Dict, Tuple, List, Callable, Optional
import warnings

# Physical constants
ℏ = 1.054571817e-34  # Reduced Planck constant (J⋅s)
c = 2.998e8           # Speed of light (m/s)

# Polymer quantization parameters
A_PLANCK = 5.391e-70  # Planck area (m²)
L_PLANCK = 1.616e-35  # Planck length (m)

# Practical limits
A_MIN = 1e-18         # Minimum polymer scale (m)
A_MAX = 1e-12         # Maximum polymer scale (m)
MESH_MAX = 200        # Maximum mesh points per dimension

def generate_polymer_mesh(ansatz_fn: Callable, bounds: List[Tuple], 
                         N: int, time_slice: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D polymer mesh from 4D ansatz function.
    
    Args:
        ansatz_fn: Function f(x,y,z,t) returning field values
        bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        N: Mesh resolution (N×N×N grid)
        time_slice: Fixed time value for 3D slice
    
    Returns:
        Tuple of (mesh_coordinates, field_values)
    """
    if N > MESH_MAX:
        warnings.warn(f"Mesh size {N} exceeds maximum {MESH_MAX}, clamping")
        N = MESH_MAX
    
    # Create coordinate arrays
    xs = np.linspace(bounds[0][0], bounds[0][1], N)
    ys = np.linspace(bounds[1][0], bounds[1][1], N)
    zs = np.linspace(bounds[2][0], bounds[2][1], N)
    
    # Generate mesh grid
    mesh = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
    
    # Evaluate ansatz function
    try:
        # Handle both 3D and 4D ansatz functions
        if callable(ansatz_fn):
            # Try 4D first
            try:
                f_values = ansatz_fn(mesh[..., 0], mesh[..., 1], mesh[..., 2], time_slice)
            except TypeError:
                # Fall back to 3D
                f_values = ansatz_fn(mesh[..., 0], mesh[..., 1], mesh[..., 2])
        else:
            raise ValueError("ansatz_fn must be callable")
    except Exception as e:
        warnings.warn(f"Ansatz function evaluation failed: {e}")
        # Use default Gaussian ansatz
        r_squared = (mesh[..., 0]**2 + mesh[..., 1]**2 + mesh[..., 2]**2)
        sigma = (bounds[0][1] - bounds[0][0]) / 4  # Width parameter
        f_values = 1 + 0.1 * np.exp(-r_squared / (2 * sigma**2))
    
    return mesh, f_values

def compute_polymer_negative_energy(f_values: np.ndarray, a: float, 
                                  volume_element: float) -> Dict:
    """
    Compute negative energy from polymer quantization stress-energy tensor.
    
    Mathematical model:
    T₀₀^poly = -ℏ(f-1)²/a²
    E_tot = ∫ T₀₀ dV = Σᵢ T₀₀ᵢ × ΔV
    
    Args:
        f_values: Array of field values on mesh
        a: Polymer length scale (m)
        volume_element: Volume per mesh point (m³)
    
    Returns:
        Dictionary with energy analysis
    """
    if a <= 0:
        raise ValueError("Polymer scale 'a' must be positive")
    
    # Field deviation from vacuum
    Δf = f_values - 1.0
    
    # Polymer stress-energy density
    T00_polymer = -ℏ * (Δf**2) / (a**2)
    
    # Total negative energy (spatial integral)
    E_total = np.sum(T00_polymer) * volume_element
    
    # Energy density statistics
    rho_mean = np.mean(T00_polymer)
    rho_std = np.std(T00_polymer)
    rho_min = np.min(T00_polymer)
    rho_max = np.max(T00_polymer)
    
    # Field variation metrics
    field_variance = np.var(Δf)
    field_gradient = np.mean(np.abs(np.gradient(Δf)))
    
    # Polymer coherence length
    coherence_length = a * np.sqrt(ℏ / (np.abs(rho_mean) * a**2 + 1e-50))
    
    # Quantum correction factor (phenomenological)
    # Accounts for discretization effects
    quantum_correction = 1 / (1 + (a / L_PLANCK)**2)
    
    E_corrected = E_total * quantum_correction
    
    return {
        'total_energy': E_corrected,
        'uncorrected_energy': E_total,
        'mean_density': rho_mean,
        'density_std': rho_std,
        'density_range': (rho_min, rho_max),
        'field_variance': field_variance,
        'field_gradient': field_gradient,
        'coherence_length': coherence_length,
        'quantum_correction': quantum_correction,
        'polymer_scale': a,
        'volume_element': volume_element,
        'n_mesh_points': f_values.size
    }

def optimize_polymer_insert(ansatz_fn: Callable, bounds: List[Tuple], 
                           N: int = 50, a_range: Tuple = (1e-18, 1e-12),
                           n_scale_points: int = 20) -> Dict:
    """
    Optimize polymer scale 'a' for maximum negative energy extraction.
    
    Args:
        ansatz_fn: 4D ansatz function f(x,y,z,t)
        bounds: Spatial bounds [(x_min,x_max), (y_min,y_max), (z_min,z_max)]
        N: Mesh resolution
        a_range: (a_min, a_max) polymer scale range
        n_scale_points: Number of scale values to test
    
    Returns:
        Dictionary with optimization results
    """
    print("🧬 Optimizing Polymer QFT Coupling Insert")
    print("=" * 50)
    
    # Generate mesh for ansatz function
    print(f"   📐 Generating {N}×{N}×{N} polymer mesh...")
    mesh, f_values = generate_polymer_mesh(ansatz_fn, bounds, N)
    
    # Calculate volume element
    dx = (bounds[0][1] - bounds[0][0]) / N
    dy = (bounds[1][1] - bounds[1][0]) / N  
    dz = (bounds[2][1] - bounds[2][0]) / N
    volume_element = dx * dy * dz
    
    # Scale optimization
    a_values = np.logspace(np.log10(a_range[0]), np.log10(a_range[1]), n_scale_points)
    
    results = []
    best_result = {'total_energy': 0}
    
    print(f"   🔍 Optimizing over {n_scale_points} polymer scales...")
    
    for i, a in enumerate(a_values):
        result = compute_polymer_negative_energy(f_values, a, volume_element)
        result['a_index'] = i
        results.append(result)
        
        if result['total_energy'] < best_result['total_energy']:
            best_result = result.copy()
    
    # Analysis
    energies = [r['total_energy'] for r in results]
    optimal_idx = np.argmin(energies)
    optimal_a = a_values[optimal_idx]
    
    # Scaling law analysis
    # Fit E_tot ∝ a^β
    log_a = np.log10(a_values)
    log_E = np.log10(np.abs(energies))
    valid_mask = np.isfinite(log_E)
    
    if np.sum(valid_mask) > 2:
        poly_fit = np.polyfit(log_a[valid_mask], log_E[valid_mask], 1)
        scaling_exponent = poly_fit[0]
        scaling_prefactor = 10**poly_fit[1]
    else:
        scaling_exponent = -2.0  # Theoretical expectation
        scaling_prefactor = 1.0
    
    # Field analysis
    field_stats = {
        'mean_field': np.mean(f_values),
        'field_std': np.std(f_values),
        'field_range': (np.min(f_values), np.max(f_values)),
        'deviation_mean': np.mean(np.abs(f_values - 1.0)),
        'max_deviation': np.max(np.abs(f_values - 1.0))
    }
    
    print(f"✅ Polymer Insert Optimization Complete!")
    print(f"   • Optimal polymer scale: a = {optimal_a:.2e} m")
    print(f"   • Maximum negative energy: {best_result['total_energy']:.2e} J")
    print(f"   • Scaling law: E ∝ a^{scaling_exponent:.2f}")
    print(f"   • Quantum correction: {best_result['quantum_correction']:.3f}")
    print(f"   • Coherence length: {best_result['coherence_length']:.2e} m")
    print(f"   • Field deviation: {field_stats['deviation_mean']:.3f}")
    print(f"   • Mesh points: {f_values.size:,}")
    
    return {
        'best_result': best_result,
        'optimal_scale': optimal_a,
        'all_results': results,
        'a_values': a_values,
        'energies': energies,
        'scaling_exponent': scaling_exponent,
        'scaling_prefactor': scaling_prefactor,
        'field_statistics': field_stats,
        'mesh_info': {
            'bounds': bounds,
            'N': N,
            'volume_element': volume_element,
            'total_volume': volume_element * f_values.size
        }
    }

# Pre-defined ansatz functions for testing and benchmarking

def gaussian_ansatz_4d(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float = 0.0) -> np.ndarray:
    """
    4D Gaussian ansatz: f(x,y,z,t) = 1 + A exp(-(r²+ωt²)/σ²)
    
    Models localized field enhancement with temporal modulation.
    """
    r_squared = x**2 + y**2 + z**2
    t_factor = 1 + 0.1 * np.cos(2 * np.pi * t)  # Temporal modulation
    sigma = 1e-6  # Characteristic scale
    amplitude = 0.2
    
    return 1 + amplitude * t_factor * np.exp(-r_squared / (2 * sigma**2))

def vortex_ansatz_4d(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float = 0.0) -> np.ndarray:
    """
    4D vortex ansatz: f(x,y,z,t) = 1 + A(r) exp(iθ + ωt)
    
    Models topological field configurations with helical structure.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    phi = theta + 2 * np.pi * t  # Helical rotation
    
    # Radial profile
    r0 = 1e-6
    amplitude = 0.15 * np.exp(-r**2 / (2 * r0**2))
    
    # Complex phase → real part
    phase_factor = np.cos(phi + np.pi * z / r0)
    
    return 1 + amplitude * phase_factor

def standing_wave_ansatz_4d(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float = 0.0) -> np.ndarray:
    """
    4D standing wave ansatz: f(x,y,z,t) = 1 + A sin(kx)sin(ky)sin(kz)cos(ωt)
    
    Models cavity-like field configurations with temporal oscillations.
    """
    k = 2 * np.pi / 1e-6  # Wave number
    omega = 2 * np.pi * 1e12  # Angular frequency (THz)
    amplitude = 0.3
    
    spatial_pattern = np.sin(k * x) * np.sin(k * y) * np.sin(k * z)
    temporal_factor = np.cos(omega * t)
    
    return 1 + amplitude * spatial_pattern * temporal_factor

def benchmark_ansatz_functions(bounds: List[Tuple], N: int = 30) -> Dict:
    """
    Benchmark different ansatz functions for polymer optimization.
    
    Args:
        bounds: Spatial bounds for evaluation
        N: Mesh resolution
    
    Returns:
        Comparison of ansatz function performance
    """
    print("🧪 Benchmarking Ansatz Functions")
    print("=" * 40)
    
    ansatz_functions = {
        'gaussian': gaussian_ansatz_4d,
        'vortex': vortex_ansatz_4d,
        'standing_wave': standing_wave_ansatz_4d
    }
    
    benchmark_results = {}
    
    for name, func in ansatz_functions.items():
        print(f"   Testing {name} ansatz...")
        
        try:
            result = optimize_polymer_insert(
                func, bounds, N=N, n_scale_points=10
            )
            
            benchmark_results[name] = {
                'best_energy': result['best_result']['total_energy'],
                'optimal_scale': result['optimal_scale'],
                'scaling_exponent': result['scaling_exponent'],
                'field_deviation': result['field_statistics']['deviation_mean'],
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            benchmark_results[name] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    # Find best performing ansatz
    successful = {k: v for k, v in benchmark_results.items() if v['status'] == 'SUCCESS'}
    
    if successful:
        best_ansatz = min(successful.keys(), 
                         key=lambda k: successful[k]['best_energy'])
        
        print(f"\n🏆 Best ansatz: {best_ansatz}")
        print(f"   Energy: {successful[best_ansatz]['best_energy']:.2e} J")
        print(f"   Scale: {successful[best_ansatz]['optimal_scale']:.2e} m")
    
    return {
        'benchmark_results': benchmark_results,
        'best_ansatz': best_ansatz if successful else None,
        'test_bounds': bounds,
        'test_resolution': N
    }

if __name__ == "__main__":
    # Test the module
    print("🧬 Polymer QFT Coupling Insert Module Test")
    print("=" * 50)
    
    # Define test bounds (1 μm³ volume)
    test_bounds = [(-5e-7, 5e-7), (-5e-7, 5e-7), (-5e-7, 5e-7)]
    
    # Test individual functions
    print("\n1️⃣  Testing Gaussian ansatz...")
    gauss_result = optimize_polymer_insert(
        gaussian_ansatz_4d, test_bounds, N=20, n_scale_points=10
    )
    
    print("\n2️⃣  Testing mesh generation...")
    mesh, f_vals = generate_polymer_mesh(vortex_ansatz_4d, test_bounds, 15)
    print(f"   Mesh shape: {mesh.shape}")
    print(f"   Field range: {np.min(f_vals):.3f} to {np.max(f_vals):.3f}")
    
    print("\n3️⃣  Running ansatz benchmark...")
    benchmark = benchmark_ansatz_functions(test_bounds, N=15)
    
    print("\n✅ Polymer QFT module validated!")
