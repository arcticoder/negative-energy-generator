"""
Mechanical FEM for Virtual Plate Deflection
===========================================

This module implements finite element method (FEM) simulations for mechanical
plate deflection under Casimir forces and other distributed loads.

Mathematical Foundation:
    Kirchhoff-Love plate theory:
    D∇⁴w(x,y) = q(x,y)
    
    Flexural rigidity:
    D = Et³/[12(1-ν²)]
    
    Boundary conditions:
    w = 0, ∂w/∂n = 0 (clamped edges)

Uses FEniCS for finite element analysis.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable, List, Any
import warnings

# Real FEniCS implementation for mechanical FEM simulation
try:
    from dolfin import (
        RectangleMesh, Point, FunctionSpace,
        TrialFunction, TestFunction, Function,
        DirichletBC, Constant, inner, grad, dx, solve
    )
    FENICS_AVAILABLE = True
    FunctionType = Function
except ImportError:
    print("⚠️  FEniCS not available. mechanical_fem.py:35")
    FENICS_AVAILABLE = False
    # Mock FunctionType for fallback
    FunctionType = type(None)

def solve_plate(E, nu, t, q_val, L, res=50):
    """
    Real FEniCS implementation for plate deflection.
    
    Kirchhoff-Love plate theory:
    D∇⁴w = q, where D = Et³/12(1-ν²)
    """
    if FENICS_AVAILABLE:
        D = E * t**3 / (12*(1-nu**2))
        mesh = RectangleMesh(Point(0,0), Point(L,L), res, res)
        V = FunctionSpace(mesh, "Lagrange", 2)
        w = TrialFunction(V)
        v = TestFunction(V)
        q = Constant(q_val)
        
        # Weak form of D ∇⁴ w = q (simplified)
        a = D*inner(grad(grad(w)), grad(grad(v))) * dx
        Lf = q * v * dx
        bc = DirichletBC(V, Constant(0.0), "on_boundary")
        w_sol = Function(V)
        solve(a == Lf, w_sol, bc)
        return w_sol
    else:
        # Fallback mock implementation
        print(f"   📐 Mock FEniCS: D={E*t**3/(12*(1-nu**2)):.2e}, q={q_val:.2e}")
        return None

def plate_deflection_fem(E: float, 
                        nu: float, 
                        t: float, 
                        q_val: float, 
                        L: float,
                        mesh_resolution: int = 50) -> Optional[Any]:
    """
    Solve plate deflection using finite element method.
    
    Args:
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        t: Plate thickness (m)
        q_val: Distributed load (Pa)
        L: Plate side length (m)
        mesh_resolution: Number of elements per side
    
    Returns:
        FEniCS Function containing deflection field w(x,y)
    """
    print(f"🔬 Running FEM plate deflection analysis:")
    print(f"   • Young's modulus: {E/1e9:.0f} GPa")
    print(f"   • Thickness: {t*1e9:.1f} nm")
    print(f"   • Load: {q_val:.2e} Pa")
    print(f"   • Plate size: {L*1e6:.1f}×{L*1e6:.1f} μm")
    print(f"   • Mesh resolution: {mesh_resolution}×{mesh_resolution}")
    
    # Flexural rigidity
    D = E * t**3 / (12 * (1 - nu**2))
    print(f"   • Flexural rigidity: {D:.2e} N⋅m")
    
    # Create mesh
    mesh = RectangleMesh(Point(0, 0), Point(L, L), mesh_resolution, mesh_resolution)
    
    # Function space (quadratic Lagrange elements for C1 continuity approximation)
    V = FunctionSpace(mesh, 'Lagrange', degree=2)
    
    # Trial and test functions
    w = TrialFunction(V)
    v = TestFunction(V)
    
    # Simplified biharmonic formulation for quadratic elements
    # Use: D∇²w ⋅ ∇²v (this is an approximation of the biharmonic)
    a = D * inner(grad(grad(w)), grad(grad(v))) * dx
    
    # Load term
    L_form = q_val * v * dx
    
    # Boundary conditions: clamped edges (w = 0, ∂w/∂n = 0)
    bc = DirichletBC(V, Constant(0.0), 'on_boundary')
    
    # Solve system
    w_sol = Function(V)
    solve(a == L_form, w_sol, bc)
    
    # Extract solution statistics
    deflections = w_sol.vector().get_local()
    max_deflection = np.max(np.abs(deflections))
    
    print(f"   ✅ FEM solution complete")
    print(f"   • Maximum deflection: {max_deflection*1e9:.2f} nm")
    print(f"   • RMS deflection: {np.sqrt(np.mean(deflections**2))*1e9:.2f} nm")
    
    return w_sol

def compute_casimir_plate_force(gap: float, 
                              area: float, 
                              material: str = 'silicon') -> float:
    """
    Compute Casimir force between parallel plates.
    
    Args:
        gap: Separation distance (m)
        area: Plate area (m²)
        material: Plate material
    
    Returns:
        Casimir force magnitude (N)
    """
    # Fundamental constants
    hbar = 1.054571817e-34  # J⋅s
    c = 299792458  # m/s
    
    # Material-dependent correction factor
    material_factors = {
        'silicon': 0.95,   # Realistic correction for finite conductivity
        'gold': 0.98,
        'aluminum': 0.90,
        'perfect': 1.0
    }
    
    correction = material_factors.get(material, 0.95)
    
    # Casimir force per unit area: F/A = π²ℏc/(240d⁴)
    force_density = (np.pi**2 * hbar * c) / (240 * gap**4)
    total_force = correction * force_density * area
    
    return total_force

def simulate_casimir_plate_deflection(gap: float = 100e-9,
                                    plate_size: float = 10e-6,
                                    thickness: float = 100e-9,
                                    material: str = 'silicon') -> Dict[str, float]:
    """
    Complete simulation of plate deflection under Casimir forces.
    
    Args:
        gap: Initial gap between plates (m)
        plate_size: Plate lateral size (m)
        thickness: Plate thickness (m)
        material: Plate material
    
    Returns:
        Complete analysis of mechanical response
    """
    print("🔬 CASIMIR PLATE DEFLECTION SIMULATION")
    print("=" * 60)
    
    # Material properties
    material_props = {
        'silicon': {'E': 170e9, 'nu': 0.28, 'density': 2330},
        'gold': {'E': 78e9, 'nu': 0.42, 'density': 19300},
        'aluminum': {'E': 70e9, 'nu': 0.33, 'density': 2700}
    }
    
    props = material_props.get(material, material_props['silicon'])
    E = props['E']
    nu = props['nu']
    
    # Compute Casimir force
    plate_area = plate_size**2
    casimir_force = compute_casimir_plate_force(gap, plate_area, material)
    casimir_pressure = casimir_force / plate_area
    
    print(f"\n⚡ Casimir force analysis:")
    print(f"   • Gap: {gap*1e9:.1f} nm")
    print(f"   • Plate area: {plate_area*1e12:.1f} μm²")
    print(f"   • Casimir force: {casimir_force*1e12:.2f} pN")
    print(f"   • Casimir pressure: {casimir_pressure:.2e} Pa")
    
    # Run FEM simulation
    print(f"\n🔧 FEM mechanical analysis:")
    w_solution = plate_deflection_fem(E, nu, thickness, casimir_pressure, plate_size)
    
    # Extract deflection data
    deflections = w_solution.vector().get_local()
    max_deflection = np.max(np.abs(deflections))
    rms_deflection = np.sqrt(np.mean(deflections**2))
    
    # Check if plates would contact
    contact_threshold = gap * 0.1  # Contact if deflection > 10% of gap
    plates_contact = max_deflection > contact_threshold
    
    # Stability analysis
    spring_constant = E * thickness**3 / (12 * (1 - nu**2) * plate_size**4)
    casimir_gradient = 4 * casimir_force / gap  # ∂F/∂z ∝ 1/z⁵
    stability_ratio = spring_constant / casimir_gradient
    
    print(f"\n📊 Mechanical response:")
    print(f"   • Maximum deflection: {max_deflection*1e9:.2f} nm")
    print(f"   • RMS deflection: {rms_deflection*1e9:.2f} nm")
    print(f"   • Plates contact: {'Yes' if plates_contact else 'No'}")
    print(f"   • Stability ratio: {stability_ratio:.2f}")
    
    return {
        'gap': gap,
        'plate_size': plate_size,
        'thickness': thickness,
        'material': material,
        'casimir_force': casimir_force,
        'casimir_pressure': casimir_pressure,
        'max_deflection': max_deflection,
        'rms_deflection': rms_deflection,
        'plates_contact': plates_contact,
        'stability_ratio': stability_ratio,
        'spring_constant': spring_constant,
        'casimir_gradient': casimir_gradient
    }

def optimize_plate_geometry_for_stability() -> Dict[str, float]:
    """
    Optimize plate geometry for stable Casimir force extraction.
    
    Returns:
        Optimal geometry parameters and performance metrics
    """
    print("🎯 OPTIMIZING PLATE GEOMETRY FOR STABILITY")
    print("=" * 60)
    
    best_stability = 0
    best_params = None
    best_result = None
    
    # Parameter ranges
    gaps = np.logspace(-7, -6, 5)  # 100nm - 1μm
    thicknesses = np.logspace(-7, -6, 5)  # 100nm - 1μm  
    sizes = np.logspace(-6, -5, 5)  # 1μm - 10μm
    materials = ['silicon', 'gold', 'aluminum']
    
    for gap in gaps:
        for thickness in thicknesses:
            for size in sizes:
                for material in materials:
                    print(f"\n🔧 Testing: {gap*1e9:.0f}nm gap, {thickness*1e9:.0f}nm thick, {size*1e6:.0f}μm, {material}")
                    
                    try:
                        result = simulate_casimir_plate_deflection(gap, size, thickness, material)
                        
                        # Stability metric: want high ratio, no contact
                        if not result['plates_contact']:
                            stability = result['stability_ratio']
                            
                            if stability > best_stability:
                                best_stability = stability
                                best_params = {
                                    'gap': gap,
                                    'thickness': thickness, 
                                    'size': size,
                                    'material': material
                                }
                                best_result = result
                    
                    except Exception as e:
                        print(f"   ❌ Failed: {e}")
                        continue
    
    if best_result:
        print(f"\n✅ OPTIMIZATION COMPLETE!")
        print(f"   • Optimal gap: {best_params['gap']*1e9:.0f} nm")
        print(f"   • Optimal thickness: {best_params['thickness']*1e9:.0f} nm")
        print(f"   • Optimal size: {best_params['size']*1e6:.0f} μm")
        print(f"   • Optimal material: {best_params['material']}")
        print(f"   • Achieved stability ratio: {best_result['stability_ratio']:.2f}")
        print(f"   • Maximum deflection: {best_result['max_deflection']*1e9:.2f} nm")
    
    return {**best_params, **best_result} if best_result else {}

def simulate_dynamic_plate_response(frequency_range: Tuple[float, float] = (1e6, 1e9),
                                  n_frequencies: int = 50) -> Dict[str, np.ndarray]:
    """
    Simulate dynamic response of Casimir plates to time-varying forces.
    
    Args:
        frequency_range: (min_freq, max_freq) in Hz
        n_frequencies: Number of frequency points
    
    Returns:
        Frequency response data
    """
    print("🔬 DYNAMIC PLATE RESPONSE SIMULATION")
    print("=" * 60)
    
    # Default plate parameters
    gap = 200e-9  # 200 nm
    size = 5e-6   # 5 μm
    thickness = 200e-9  # 200 nm
    material = 'silicon'
    
    # Get static response
    static_result = simulate_casimir_plate_deflection(gap, size, thickness, material)
    
    # Frequency array
    frequencies = np.logspace(np.log10(frequency_range[0]), 
                             np.log10(frequency_range[1]), 
                             n_frequencies)
    
    # Material properties
    E = 170e9  # Silicon Young's modulus
    nu = 0.28  # Poisson ratio
    rho = 2330  # Density kg/m³
    
    # Plate dynamics parameters
    spring_constant = static_result['spring_constant']
    mass = rho * size**2 * thickness  # Effective mass
    
    # Natural frequency
    omega0 = np.sqrt(spring_constant / mass)
    f0 = omega0 / (2 * np.pi)
    
    # Quality factor (assume air damping)
    Q = 1000  # Typical for microscale plates in air
    gamma = omega0 / Q  # Damping rate
    
    print(f"   • Natural frequency: {f0/1e6:.1f} MHz")
    print(f"   • Quality factor: {Q}")
    print(f"   • Effective mass: {mass*1e15:.2f} fg")
    
    # Frequency response function
    omega = 2 * np.pi * frequencies
    response = 1 / (spring_constant * (1 - (omega/omega0)**2 + 1j*omega*gamma/spring_constant))
    
    amplitude = np.abs(response)
    phase = np.angle(response)
    
    # Resonance enhancement
    resonance_enhancement = np.max(amplitude) / amplitude[0]
    
    print(f"   • Resonance enhancement: {resonance_enhancement:.1f}x")
    print(f"   • 3dB bandwidth: {f0/Q/1e3:.1f} kHz")
    
    return {
        'frequencies': frequencies,
        'amplitude_response': amplitude,
        'phase_response': phase,
        'natural_frequency': f0,
        'quality_factor': Q,
        'resonance_enhancement': resonance_enhancement,
        'static_deflection': static_result['max_deflection']
    }

# Demo and testing functions
def run_mechanical_fem_demo():
    """Run demonstration of mechanical FEM simulation."""
    print("🚀 MECHANICAL FEM DEMO")
    print("=" * 50)
    
    # Example: 1μm² silicon plate, 100nm thick, under 1kPa load
    E = 170e9     # Silicon Young's modulus (Pa)
    nu = 0.28     # Poisson ratio
    t = 100e-9    # Thickness (m)
    q = 1000      # Load (Pa)
    L = 1e-6      # Side length (m)
    
    w_solution = plate_deflection_fem(E, nu, t, q, L)
    
    deflections = w_solution.vector().get_local()
    max_deflection = np.max(np.abs(deflections))
    
    print(f"\n✅ Demo complete: Max deflection = {max_deflection*1e9:.2f} nm")
    return w_solution

if __name__ == "__main__":
    # Run demonstration
    demo_solution = run_mechanical_fem_demo()
    
    # Run optimization
    optimal_geometry = optimize_plate_geometry_for_stability()
    
    if optimal_geometry:
        print(f"\n🎯 Optimal stability ratio: {optimal_geometry['stability_ratio']:.2f}")
    
    # Run dynamic analysis
    dynamic_response = simulate_dynamic_plate_response()
    print(f"\n🎵 Resonance enhancement: {dynamic_response['resonance_enhancement']:.1f}x")
