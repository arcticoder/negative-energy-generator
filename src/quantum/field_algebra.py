"""
Polymer Field Algebra for Loop Quantum Gravity

Implements the basic field algebra operations needed for polymer quantization
in the context of ANEC violations and warp bubble physics.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class PolymerField:
    """
    Polymer field representation for LQG-modified scalar fields.
    
    In polymer quantization, the momentum operator is replaced by:
    sin(μπ̂)/(μ) instead of π̂
    
    This leads to modified dispersion relations and energy conditions.
    """
    
    def __init__(self, mu: float, field_values: np.ndarray, momentum_values: np.ndarray):
        """
        Initialize polymer field.
        
        Args:
            mu: Polymer scale parameter
            field_values: φ(x) field configuration
            momentum_values: π(x) momentum configuration
        """
        self.mu = mu
        self.phi = np.array(field_values)
        self.pi = np.array(momentum_values)
        
    def kinetic_energy(self) -> np.ndarray:
        """
        Compute polymer-modified kinetic energy density.
        
        Classical: ½π²
        Polymer: ½[sin(μπ)/(μ)]²
        """
        if self.mu == 0.0:
            # Classical limit
            return 0.5 * self.pi**2
        else:
            # Polymer-modified kinetic term
            return 0.5 * (np.sin(self.mu * self.pi) / self.mu)**2
    
    def gradient_energy(self, dx: float) -> np.ndarray:
        """
        Compute gradient energy density: ½(∇φ)²
        """
        # Use central differences with periodic boundary conditions
        grad_phi = (np.roll(self.phi, -1) - np.roll(self.phi, 1)) / (2 * dx)
        return 0.5 * grad_phi**2
    
    def total_energy_density(self, dx: float, mass: float = 0.0) -> np.ndarray:
        """
        Total energy density: kinetic + gradient + mass
        """
        kinetic = self.kinetic_energy()
        gradient = self.gradient_energy(dx)
        mass_term = 0.5 * mass**2 * self.phi**2
        
        return kinetic + gradient + mass_term
    
    def anec_integrand(self, dx: float, sampling_function: np.ndarray) -> np.ndarray:
        """
        Compute ANEC integrand: T_μν k^μ k^ν for null geodesic.
        
        For scalar field: T_00 = energy density
        """
        energy_density = self.total_energy_density(dx)
        return energy_density * sampling_function
    
    def evolve_step(self, dt: float, dx: float) -> 'PolymerField':
        """
        Evolve field one time step using polymer-modified dynamics.
        
        Equations of motion:
        ∂φ/∂t = sin(μπ)/μ  (polymer-modified)
        ∂π/∂t = ∇²φ - m²φ
        """
        # Update field using polymer-modified momentum
        if self.mu == 0.0:
            dphi_dt = self.pi
        else:
            dphi_dt = np.sin(self.mu * self.pi) / self.mu
        
        # Update momentum using field equation
        laplacian_phi = (np.roll(self.phi, -1) - 2*self.phi + np.roll(self.phi, 1)) / dx**2
        dpi_dt = laplacian_phi  # Assuming m=0 for simplicity
        
        # Simple Euler step (could use RK4 for better accuracy)
        new_phi = self.phi + dt * dphi_dt
        new_pi = self.pi + dt * dpi_dt
        
        return PolymerField(self.mu, new_phi, new_pi)


def create_gaussian_initial_conditions(N: int, dx: float, mu: float, 
                                     amplitude: float = 1.0, 
                                     sigma: float = 1.0,
                                     center: float = None) -> PolymerField:
    """
    Create Gaussian initial conditions for polymer field.
    
    Args:
        N: Number of grid points
        dx: Grid spacing
        mu: Polymer parameter
        amplitude: Amplitude of initial Gaussian
        sigma: Width of Gaussian
        center: Center position (default: middle of domain)
    """
    if center is None:
        center = N * dx / 2
    
    x = np.arange(N) * dx
    
    # Gaussian field profile
    phi_0 = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    
    # Initial momentum (can be chosen for specific dynamics)
    pi_0 = np.zeros_like(phi_0)  # Start at rest
    
    return PolymerField(mu, phi_0, pi_0)


def create_oscillating_initial_conditions(N: int, dx: float, mu: float,
                                        amplitude: float = 1.0,
                                        omega: float = 1.0,
                                        sigma: float = 1.0) -> PolymerField:
    """
    Create oscillating Gaussian initial conditions for negative energy generation.
    
    This creates the type of field configuration that can lead to ANEC violations
    when μ parameter is chosen appropriately.
    """
    x = np.arange(N) * dx
    center = N * dx / 2
    
    # Gaussian envelope with oscillation
    envelope = np.exp(-(x - center)**2 / (2 * sigma**2))
    phi_0 = amplitude * envelope * np.cos(omega * x)
    
    # Initial momentum for oscillation
    pi_0 = -amplitude * omega * envelope * np.sin(omega * x)
    
    # Scale momentum to optimize for polymer effects
    if mu > 0:
        # Choose amplitude such that μπ is in the optimal range for sin(μπ)/μ
        optimal_pi_scale = np.pi / (2 * mu)  # sin(π/2) = 1 maximizes sin(μπ)/μ
        pi_0 = pi_0 * optimal_pi_scale / (amplitude * omega)
    
    return PolymerField(mu, phi_0, pi_0)


def compute_classical_comparison(polymer_field: PolymerField, dx: float) -> Dict:
    """
    Compare polymer field with classical field (μ=0) for the same initial conditions.
    
    Returns energy differences and violation metrics.
    """
    # Create classical version
    classical_field = PolymerField(0.0, polymer_field.phi.copy(), polymer_field.pi.copy())
    
    # Compute energy densities
    polymer_energy = polymer_field.total_energy_density(dx)
    classical_energy = classical_field.total_energy_density(dx)
    
    # Integrated energies
    polymer_total = np.sum(polymer_energy) * dx
    classical_total = np.sum(classical_energy) * dx
    
    # Violation metrics
    energy_difference = polymer_total - classical_total
    max_negative_density = np.min(polymer_energy)
    
    return {
        'polymer_energy': polymer_total,
        'classical_energy': classical_total,
        'energy_difference': energy_difference,
        'max_negative_density': max_negative_density,
        'polymer_density': polymer_energy,
        'classical_density': classical_energy,
        'violation_strength': -energy_difference if energy_difference < 0 else 0.0
    }


if __name__ == "__main__":
    # Test the polymer field algebra
    print("Testing Polymer Field Algebra...")
    
    # Create test field
    N = 64
    dx = 0.1
    mu = 0.1
    
    field = create_oscillating_initial_conditions(N, dx, mu, amplitude=2.0)
    
    print(f"Created polymer field with μ={mu}")
    print(f"Field range: [{np.min(field.phi):.4f}, {np.max(field.phi):.4f}]")
    print(f"Momentum range: [{np.min(field.pi):.4f}, {np.max(field.pi):.4f}]")
    
    # Compute energies
    kinetic = np.sum(field.kinetic_energy()) * dx
    gradient = np.sum(field.gradient_energy(dx)) * dx
    total = np.sum(field.total_energy_density(dx)) * dx
    
    print(f"Total kinetic energy: {kinetic:.6f}")
    print(f"Total gradient energy: {gradient:.6f}")
    print(f"Total energy: {total:.6f}")
    
    # Compare with classical
    comparison = compute_classical_comparison(field, dx)
    print(f"Energy difference (polymer - classical): {comparison['energy_difference']:.6f}")
    print(f"Maximum negative density: {comparison['max_negative_density']:.6f}")
    
    print("Polymer field algebra test completed!")
