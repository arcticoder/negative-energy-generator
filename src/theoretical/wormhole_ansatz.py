#!/usr/bin/env python3
"""
Traversable Wormhole Ansatz for Negative Energy Generation
=========================================================

Implementation of Morris-Thorne and Krasnikov wormhole geometries
designed to produce negative ANEC integrals through exotic matter
configurations and modified spacetime geometries.

This module provides the mathematical framework for wormhole-based
negative energy generation, including shape functions, redshift 
functions, and stress-energy tensor calculations.

Author: Negative Energy Generator Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import scipy.integrate as integrate
import scipy.optimize as optimize

@dataclass
class WormholeConfig:
    """Configuration for wormhole ansatz calculations."""
    
    # Wormhole geometry parameters
    throat_radius: float = 1e-15        # Throat radius r₀ (m)
    shell_thickness: float = 1e-14      # Shell thickness R (m)
    redshift_param: float = 0.1         # Redshift parameter τ
    shape_param: float = 2.0            # Shape function parameter s
    exotic_strength: float = 1e-3       # Exotic matter strength Δ
    
    # Numerical parameters
    grid_points: int = 1000             # Spatial grid resolution
    time_steps: int = 500               # Temporal resolution
    integration_method: str = 'simpson' # Integration method
    
    # Physical constants
    c: float = 2.998e8                  # Speed of light (m/s)
    hbar: float = 1.055e-34            # Reduced Planck constant
    G: float = 6.674e-11               # Gravitational constant

class TraversableWormholeAnsatz:
    """
    Implementation of traversable wormhole geometries for negative energy generation.
    
    Based on Morris-Thorne wormhole solutions with modifications for:
    - Enhanced exotic matter concentrations
    - Optimized throat geometries
    - Negative energy flux maximization
    """
    
    def __init__(self, config: WormholeConfig = None):
        self.config = config or WormholeConfig()
        
        # Geometry cache
        self._shape_function_cache = {}
        self._redshift_function_cache = {}
        self._metric_cache = {}
        
        print(f"🌀 Traversable Wormhole Ansatz initialized")
        print(f"   Throat radius: {self.config.throat_radius:.2e} m")
        print(f"   Shell thickness: {self.config.shell_thickness:.2e} m")
        print(f"   Exotic matter strength: {self.config.exotic_strength:.2e}")
    
    def morris_thorne_shape_function(self, r: np.ndarray) -> np.ndarray:
        """
        Morris-Thorne shape function b(r) with exotic matter enhancement.
        
        b(r) = r₀ * (1 + (r₀/r)^s)^(-1/s) + Δ * exp(-(r-r₀)/R)
        
        Args:
            r: Radial coordinate array
            
        Returns:
            Shape function values
        """
        r0 = self.config.throat_radius
        R = self.config.shell_thickness
        s = self.config.shape_param
        delta = self.config.exotic_strength
        
        # Standard Morris-Thorne shape
        standard_shape = r0 * np.power(1 + np.power(r0/r, s), -1/s)
        
        # Exotic matter enhancement (shell around throat)
        exotic_enhancement = delta * np.exp(-(r - r0)/R)
        
        return standard_shape + exotic_enhancement
    
    def redshift_function(self, r: np.ndarray) -> np.ndarray:
        """
        Redshift function Φ(r) for wormhole metric.
        
        Φ(r) = τ * ln(r/r₀) with regularization near throat
        
        Args:
            r: Radial coordinate array
            
        Returns:
            Redshift function values
        """
        r0 = self.config.throat_radius
        tau = self.config.redshift_param
        
        # Regularized logarithm near throat
        r_reg = np.maximum(r, r0 * 1.001)  # Avoid singularity
        
        return tau * np.log(r_reg / r0)
    
    def metric_components(self, r: np.ndarray, t: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Calculate wormhole metric components in Morris-Thorne coordinates.
        
        ds² = -e^(2Φ(r)) dt² + dr²/(1-b(r)/r) + r²(dθ² + sin²θ dφ²)
        
        Args:
            r: Radial coordinate array
            t: Time coordinate
            
        Returns:
            Dictionary of metric components
        """
        b_r = self.morris_thorne_shape_function(r)
        phi_r = self.redshift_function(r)
        
        # Metric components
        g_tt = -np.exp(2 * phi_r)                    # g₀₀
        g_rr = 1.0 / (1.0 - b_r / r)                # g₁₁  
        g_theta = r**2                               # g₂₂
        g_phi = r**2                                 # g₃₃ (sin²θ factor separate)
        
        return {
            'g_tt': g_tt,
            'g_rr': g_rr,
            'g_theta': g_theta,
            'g_phi': g_phi,
            'shape_function': b_r,
            'redshift_function': phi_r
        }
    
    def stress_energy_tensor(self, r: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate stress-energy tensor components for wormhole geometry.
        
        Uses Einstein field equations: G_μν = 8πG T_μν
        
        Args:
            r: Radial coordinate array
            
        Returns:
            Stress-energy tensor components
        """
        b_r = self.morris_thorne_shape_function(r)
        phi_r = self.redshift_function(r)
        
        # Derivatives
        dr = r[1] - r[0] if len(r) > 1 else 1e-15
        
        b_prime = np.gradient(b_r, dr)
        phi_prime = np.gradient(phi_r, dr)
        
        # Stress-energy components (in geometric units G=c=1)
        # T₀₀ (energy density)
        T_00 = -b_prime / (8 * np.pi * r**2)
        
        # T₁₁ (radial pressure)  
        T_11 = (b_r * phi_prime) / (8 * np.pi * r**3) - T_00
        
        # T₂₂ = T₃₃ (tangential pressure)
        b_double_prime = np.gradient(b_prime, dr)
        T_22 = (1/(8*np.pi)) * (
            phi_prime**2 + 2*phi_prime/r + 
            b_double_prime/r**2 - b_prime/(2*r**3)
        )
        
        return {
            'energy_density': T_00,      # ρ = T₀₀
            'radial_pressure': T_11,     # p_r = T₁₁  
            'tangential_pressure': T_22, # p_t = T₂₂ = T₃₃
            'shape_derivative': b_prime,
            'redshift_derivative': phi_prime
        }
    
    def energy_density_profile(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate energy density ρ(r) for wormhole configuration.
        
        Args:
            r: Radial coordinate array
            
        Returns:
            Energy density profile
        """
        stress_energy = self.stress_energy_tensor(r)
        return stress_energy['energy_density']
    
    def calculate_anec_integrand(self, r: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculate ANEC integrand for null geodesics through wormhole.
        
        ANEC = ∫ T_μν k^μ k^ν dλ where k^μ is null vector
        
        Args:
            r: Radial coordinate array
            u: Null parameter array
            
        Returns:
            ANEC integrand values
        """
        metric = self.metric_components(r)
        stress_energy = self.stress_energy_tensor(r)
        
        # Null vector components (radial null geodesic)
        # k^t = E * e^(-Φ), k^r = ±E * sqrt(1-b/r)
        E = 1.0  # Energy normalization
        phi = metric['redshift_function']
        b = metric['shape_function']
        
        k_t = E * np.exp(-phi)
        k_r = E * np.sqrt(np.maximum(1 - b/r, 0.001))  # Regularized
        
        # ANEC integrand: T_μν k^μ k^ν
        T_00 = stress_energy['energy_density']
        T_11 = stress_energy['radial_pressure']
        
        g_tt = metric['g_tt']
        g_rr = metric['g_rr']
        
        anec_integrand = T_00 * k_t**2 * g_tt + T_11 * k_r**2 * g_rr
        
        return anec_integrand
    
    def compute_anec_integral(self, r_min: float = None, r_max: float = None) -> float:
        """
        Compute ANEC integral along null geodesic through wormhole.
        
        Args:
            r_min: Minimum radial coordinate
            r_max: Maximum radial coordinate
            
        Returns:
            ANEC integral value
        """
        if r_min is None:
            r_min = self.config.throat_radius * 1.01
        if r_max is None:
            r_max = self.config.throat_radius + 10 * self.config.shell_thickness
            
        # Create radial grid
        r_grid = np.linspace(r_min, r_max, self.config.grid_points)
        u_grid = np.linspace(0, 1, len(r_grid))  # Affine parameter
        
        # Calculate integrand
        integrand = self.calculate_anec_integrand(r_grid, u_grid)
        
        # Numerical integration
        if self.config.integration_method == 'simpson':
            anec_value = integrate.simpson(integrand, r_grid)
        elif self.config.integration_method == 'trapz':
            anec_value = integrate.trapz(integrand, r_grid)
        else:
            anec_value = integrate.quad(
                lambda r: np.interp(r, r_grid, integrand), 
                r_min, r_max
            )[0]
        
        return anec_value
    
    def optimize_for_negative_anec(self) -> Dict[str, float]:
        """
        Optimize wormhole parameters to minimize ANEC integral (maximize negativity).
        
        Returns:
            Optimization results and best parameters
        """
        print("🔧 Optimizing wormhole parameters for maximum ANEC violation...")
        
        def objective(params):
            """Objective function: minimize ANEC (maximize negativity)."""
            throat_radius, shell_thickness, redshift_param, shape_param, exotic_strength = params
            
            # Update configuration
            old_config = (
                self.config.throat_radius,
                self.config.shell_thickness, 
                self.config.redshift_param,
                self.config.shape_param,
                self.config.exotic_strength
            )
            
            self.config.throat_radius = throat_radius
            self.config.shell_thickness = shell_thickness
            self.config.redshift_param = redshift_param
            self.config.shape_param = shape_param
            self.config.exotic_strength = exotic_strength
            
            try:
                anec_value = self.compute_anec_integral()
                result = anec_value  # Minimize ANEC (want negative)
            except:
                result = 1e10  # Penalty for failed calculation
            
            # Restore configuration
            (self.config.throat_radius, self.config.shell_thickness, 
             self.config.redshift_param, self.config.shape_param,
             self.config.exotic_strength) = old_config
            
            return result
        
        # Parameter bounds
        bounds = [
            (1e-16, 1e-13),   # throat_radius
            (1e-15, 1e-12),   # shell_thickness  
            (0.01, 1.0),      # redshift_param
            (1.0, 5.0),       # shape_param
            (1e-6, 1e-1)      # exotic_strength
        ]
        
        # Initial guess
        x0 = [
            self.config.throat_radius,
            self.config.shell_thickness,
            self.config.redshift_param, 
            self.config.shape_param,
            self.config.exotic_strength
        ]
        
        # Optimization
        result = optimize.minimize(
            objective, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        
        # Apply best parameters
        if result.success:
            (self.config.throat_radius, self.config.shell_thickness,
             self.config.redshift_param, self.config.shape_param,
             self.config.exotic_strength) = result.x
        
        final_anec = self.compute_anec_integral()
        
        print("✅ Wormhole optimization complete!")
        print(f"   Final ANEC: {final_anec:.2e} J·s·m⁻³")
        print(f"   Negative ANEC: {'YES' if final_anec < 0 else 'NO'}")
        print(f"   Optimization success: {result.success}")
        
        return {
            'anec_value': final_anec,
            'optimization_success': result.success,
            'negative_anec_achieved': final_anec < 0,
            'best_parameters': {
                'throat_radius': self.config.throat_radius,
                'shell_thickness': self.config.shell_thickness,
                'redshift_param': self.config.redshift_param,
                'shape_param': self.config.shape_param,
                'exotic_strength': self.config.exotic_strength
            }
        }

def demo_wormhole_ansatz():
    """Demonstrate the traversable wormhole ansatz."""
    print("🌀 TRAVERSABLE WORMHOLE ANSATZ DEMO")
    print("=" * 50)
    
    # Create wormhole with enhanced parameters
    config = WormholeConfig(
        throat_radius=1e-14,
        shell_thickness=5e-14,
        exotic_strength=1e-2,
        grid_points=500
    )
    
    wormhole = TraversableWormholeAnsatz(config)
    
    # Test basic calculations
    print("\n=== Geometry Calculation Test ===")
    r_test = np.linspace(config.throat_radius * 1.1, 
                        config.throat_radius + 5 * config.shell_thickness, 
                        100)
    
    metric = wormhole.metric_components(r_test)
    stress_energy = wormhole.stress_energy_tensor(r_test)
    
    print(f"✓ Shape function range: [{metric['shape_function'].min():.2e}, {metric['shape_function'].max():.2e}]")
    print(f"✓ Energy density range: [{stress_energy['energy_density'].min():.2e}, {stress_energy['energy_density'].max():.2e}]")
    
    # Test ANEC calculation
    print("\n=== ANEC Integral Calculation ===")
    anec_value = wormhole.compute_anec_integral()
    print(f"✓ Initial ANEC: {anec_value:.2e} J·s·m⁻³")
    print(f"✓ Negative ANEC: {'YES' if anec_value < 0 else 'NO'}")
    
    # Optimization test
    print("\n=== Parameter Optimization ===")
    optimization_results = wormhole.optimize_for_negative_anec()
    
    print(f"✓ Optimized ANEC: {optimization_results['anec_value']:.2e} J·s·m⁻³")
    print(f"✓ Negative ANEC achieved: {optimization_results['negative_anec_achieved']}")
    
    return wormhole, optimization_results

if __name__ == "__main__":
    demo_wormhole_ansatz()
