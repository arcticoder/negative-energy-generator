"""
Example usage of the Negative Energy Generator framework.

This example demonstrates how to calculate negative energy requirements,
optimize parameters, and analyze the theoretical framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from theoretical.exotic_matter import ExoticMatterCalculator
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Run example calculations for negative energy generation."""
    
    print("=== Negative Energy Generator Example ===\n")
    
    # Initialize the exotic matter calculator
    calculator = ExoticMatterCalculator()
    
    # Calculate negative energy density for different radii
    radii = np.logspace(-12, -6, 50)  # From 1 pm to 1 μm
    warp_velocity = 0.1  # 10% speed of light
    
    results = []
    for radius in radii:
        result = calculator.compute_negative_density(radius, warp_velocity)
        results.append(result)
    
    # Extract energy densities
    energy_densities = [r['T00'] for r in results]
    
    # Print some key results
    print(f"Warp velocity: {warp_velocity * 100:.1f}% speed of light")
    print(f"Radius range: {radii[0]:.2e} m to {radii[-1]:.2e} m")
    print(f"Energy density range: {min(energy_densities):.2e} to {max(energy_densities):.2e} J/m³")
    
    # Find optimal radius (minimum absolute energy density)
    optimal_idx = np.argmin(np.abs(energy_densities))
    optimal_radius = radii[optimal_idx]
    optimal_density = energy_densities[optimal_idx]
    
    print(f"\nOptimal configuration:")
    print(f"  Radius: {optimal_radius:.2e} m")
    print(f"  Negative energy density: {optimal_density:.2e} J/m³")
    
    # Stability analysis
    density_array = np.array(energy_densities)
    stability = calculator.stability_analysis(density_array)
    
    print(f"\nStability Analysis:")
    print(f"  Stable: {stability['stable']}")
    print(f"  Max gradient: {stability['max_gradient']:.2e}")
    print(f"  Causality safe: {stability['causality_safe']}")
    
    # Calculate violation strength
    violation = calculator.violation_strength(optimal_density)
    print(f"  Energy condition violation strength: {violation:.2e}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.loglog(radii, np.abs(energy_densities))
    plt.xlabel('Radius (m)')
    plt.ylabel('|Energy Density| (J/m³)')
    plt.title('Negative Energy Density vs Radius')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    violation_strengths = [calculator.violation_strength(ed) for ed in energy_densities]
    plt.loglog(radii, violation_strengths)
    plt.xlabel('Radius (m)')
    plt.ylabel('Violation Strength')
    plt.title('Energy Condition Violation Strength')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    # Show stress-energy tensor at optimal point
    x_range = np.linspace(-optimal_radius*2, optimal_radius*2, 100)
    T00_profile = []
    for x in x_range:
        T = calculator.stress_energy_tensor(x, 0, 0, 0)
        T00_profile.append(T[0, 0])
    
    plt.plot(x_range/optimal_radius, T00_profile)
    plt.xlabel('Distance / Optimal Radius')
    plt.ylabel('T⁰⁰ (J/m³)')
    plt.title('Energy Density Profile')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # Energy requirement scaling
    total_energies = [abs(ed) * (4/3) * np.pi * r**3 for ed, r in zip(energy_densities, radii)]
    plt.loglog(radii, total_energies)
    plt.xlabel('Radius (m)')
    plt.ylabel('Total Negative Energy (J)')
    plt.title('Total Energy Requirements')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('negative_energy_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'negative_energy_analysis.png'")
    
    # Show theoretical implications
    print(f"\n=== Theoretical Implications ===")
    print(f"This analysis shows that negative energy generation requires:")
    print(f"1. Extremely small spatial scales (< {optimal_radius:.0e} m)")
    print(f"2. Energy densities of order {optimal_density:.0e} J/m³")
    print(f"3. Careful control of spacetime geometry")
    print(f"4. Violation of classical energy conditions")
    
    planck_energy = 1.956e9  # Planck energy in Joules
    total_optimal_energy = abs(optimal_density) * (4/3) * np.pi * optimal_radius**3
    energy_ratio = total_optimal_energy / planck_energy
    
    print(f"5. Total energy ~ {energy_ratio:.2e} × Planck energy")
    
    if energy_ratio < 1:
        print("   → This is sub-Planckian, potentially achievable!")
    else:
        print("   → This exceeds Planck scale, major theoretical challenge")

if __name__ == "__main__":
    main()
