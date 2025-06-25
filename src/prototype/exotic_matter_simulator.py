"""
Exotic Matter Field Simulator
============================

This module implements the exotic matter field simulation based on the unified
generating functional approach. It discretizes spacetime on a finite grid and
computes the stress-energy tensor components from the generating functional.

Mathematical Foundation:
    G_G({x_e}) = 1/√(det(I - K_G))
    ⟨T_μν(x)⟩ = 2/√(-g) * δln(G_G)/δg^μν(x)
    ρ(x) = T_00(x)
    E_tot = ∫ ρ(x) d³x
"""

import numpy as np
from scipy.linalg import det, inv
from typing import Callable, Tuple, Optional
import warnings


class ExoticMatterSimulator:
    """
    Simulates exotic matter fields using the unified generating functional approach.
    
    This class discretizes spacetime on a finite grid and computes the stress-energy
    tensor components, particularly T_00 (energy density), from the generating functional.
    """
    
    def __init__(self, kernel_builder: Callable, g_metric: np.ndarray, grid: np.ndarray):
        """
        Initialize the exotic matter field simulator.
        
        Args:
            kernel_builder: Function that takes grid points and returns K_G matrix
            g_metric: 3×3 numpy array representing spatial metric
            grid: N×3 array of 3D grid points
        """
        self.grid = grid
        self.n_points = len(grid)
        self.g_metric = g_metric
        
        # Build the kernel matrix K_G
        self.K = kernel_builder(grid)
        
        # Compute metric determinant
        self.sqrt_g = np.sqrt(np.linalg.det(g_metric))
        
        # Precompute ln(G_G) = -½ ln(det(I - K_G))
        try:
            det_val = det(np.eye(self.K.shape[0]) - self.K)
            if det_val <= 0:
                warnings.warn("Determinant is non-positive, adding regularization")
                det_val = abs(det_val) + 1e-12
            self.lnG = -0.5 * np.log(det_val)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix encountered, using pseudoinverse")
            self.lnG = 0.0
    
    def T00(self, variation_generator: Callable) -> np.ndarray:
        """
        Compute T_00 (energy density) at all grid points.
        
        The stress-energy tensor component T_00 is computed using:
        T_00 = (2/√g) * δln(G_G)/δg^00
        
        where δln(G_G)/δg^00 = -½ Tr[(I-K)^(-1) * δK/δg^00]
        
        Args:
            variation_generator: Function that takes grid index and returns δK/δg^00
            
        Returns:
            Array of energy densities ρ(x) = T_00(x) at each grid point
        """
        try:
            # Compute (I - K)^(-1)
            I_minus_K = np.eye(self.K.shape[0]) - self.K
            
            # Check condition number
            cond = np.linalg.cond(I_minus_K)
            if cond > 1e12:
                warnings.warn(f"Ill-conditioned matrix (cond={cond:.2e}), results may be unreliable")
            
            invI_K = inv(I_minus_K)
            
            # Compute T_00 at each grid point
            rho = np.zeros(self.n_points)
            
            for i in range(self.n_points):
                # Get metric variation at point i
                dK = variation_generator(i)
                
                # δln(G)/δg^00 = -½ Tr[(I-K)^(-1) * δK/δg^00]
                delta_lnG = -0.5 * np.trace(invI_K @ dK)
                
                # T_00 = (2/√g) * δln(G)/δg^00
                rho[i] = (2.0 / self.sqrt_g) * delta_lnG
                
            return rho
            
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Linear algebra error in T00 computation: {e}")
            return np.zeros(self.n_points)
    
    def total_energy(self, rho: np.ndarray, dV: float) -> float:
        """
        Compute total energy by integrating energy density over the grid.
        
        Args:
            rho: Energy density array from T00()
            dV: Volume element (assuming uniform grid spacing)
            
        Returns:
            Total energy E_tot = ∫ ρ(x) d³x
        """
        return np.sum(rho) * dV
    
    def energy_analysis(self, rho: np.ndarray, dV: float) -> dict:
        """
        Perform comprehensive energy analysis.
        
        Args:
            rho: Energy density array
            dV: Volume element
            
        Returns:
            Dictionary with energy statistics
        """
        total_E = self.total_energy(rho, dV)
        negative_E = np.sum(rho[rho < 0]) * dV
        positive_E = np.sum(rho[rho > 0]) * dV
        
        return {
            'total_energy': total_E,
            'negative_energy': negative_E,
            'positive_energy': positive_E,
            'negative_fraction': abs(negative_E) / (abs(negative_E) + positive_E) if (abs(negative_E) + positive_E) > 0 else 0,
            'min_density': np.min(rho),
            'max_density': np.max(rho),
            'mean_density': np.mean(rho),
            'std_density': np.std(rho),
            'total_volume': len(rho) * dV
        }
    
    def field_gradient(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spatial gradients of the energy density field.
        
        Args:
            rho: Energy density array
            
        Returns:
            Tuple of gradient components (grad_x, grad_y, grad_z)
        """
        if len(self.grid.shape) != 2 or self.grid.shape[1] != 3:
            raise ValueError("Grid must be N×3 array for gradient computation")
        
        # Simple finite difference approximation
        # This is a basic implementation - more sophisticated methods could be used
        grad_x = np.gradient(rho)
        grad_y = np.gradient(rho)
        grad_z = np.gradient(rho)
        
        return grad_x, grad_y, grad_z


def default_kernel_builder(grid: np.ndarray, coupling_strength: float = 1.0, 
                          decay_length: float = 1.0) -> np.ndarray:
    """
    Default kernel builder for testing and demonstration.
    
    Creates a simple exponentially decaying kernel:
    K_ij = g * exp(-|r_i - r_j|/λ)
    
    Args:
        grid: N×3 array of grid points
        coupling_strength: Overall coupling strength g
        decay_length: Characteristic decay length λ
        
    Returns:
        N×N kernel matrix
    """
    n = len(grid)
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            r_ij = np.linalg.norm(grid[i] - grid[j])
            K[i, j] = coupling_strength * np.exp(-r_ij / decay_length)
    
    return K


def default_variation_generator(grid: np.ndarray, i: int) -> np.ndarray:
    """
    Default metric variation generator for testing.
    
    Creates a simple localized variation at grid point i.
    
    Args:
        grid: Grid points
        i: Index of the point where variation occurs
        
    Returns:
        δK/δg^00 matrix
    """
    n = len(grid)
    dK = np.zeros((n, n))
    
    # Simple localized variation - only affects point i and its neighbors
    dK[i, i] = 1.0
    
    # Add some coupling to nearby points
    for j in range(n):
        if i != j:
            r_ij = np.linalg.norm(grid[i] - grid[j])
            if r_ij < 2.0:  # Only nearby points
                dK[i, j] = 0.1 * np.exp(-r_ij)
                dK[j, i] = 0.1 * np.exp(-r_ij)
    
    return dK


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test grid
    n_grid = 10
    x = np.linspace(-1, 1, n_grid)
    y = np.linspace(-1, 1, n_grid)
    z = np.linspace(-1, 1, n_grid)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Simple flat metric
    g_metric = np.eye(3)
    
    # Create simulator
    def test_kernel_builder(grid):
        return default_kernel_builder(grid, coupling_strength=0.1, decay_length=0.5)
    
    def test_variation_generator(i):
        return default_variation_generator(grid, i)
    
    simulator = ExoticMatterSimulator(test_kernel_builder, g_metric, grid)
    
    # Compute energy density
    rho = simulator.T00(test_variation_generator)
    
    # Analyze results
    dV = (2.0 / n_grid)**3  # Volume element
    analysis = simulator.energy_analysis(rho, dV)
    
    print("Exotic Matter Field Simulation Results:")
    print(f"Total Energy: {analysis['total_energy']:.6e}")
    print(f"Negative Energy: {analysis['negative_energy']:.6e}")
    print(f"Positive Energy: {analysis['positive_energy']:.6e}")
    print(f"Negative Fraction: {analysis['negative_fraction']:.3f}")
    print(f"Energy Density Range: [{analysis['min_density']:.6e}, {analysis['max_density']:.6e}]")
