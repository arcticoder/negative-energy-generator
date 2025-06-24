#!/usr/bin/env python3
"""
Generating Functional Approach for Closed-Form Stress-Energy Tensors
===================================================================

Implements the generating functional approach for analytic T₀₀ expressions:

G(J) = 1/√det(I-K) * exp(½ J† (I-K)⁻¹ J)

⟨T₀₀(x)⟩ = δ²G(J)/δJ(x)δJ(x) |_{J=0}

This yields closed-form expressions for stress-energy tensors in terms of
(I-K)⁻¹ and det(I-K), where K encodes the warp-bubble profile.

Author: Negative Energy Generator Framework
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import scipy.linalg as la
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

@dataclass 
class GeneratingFunctionalConfig:
    """Configuration for generating functional calculations."""
    
    # Discretization parameters
    grid_size: int = 100                    # Number of grid points
    spatial_extent: float = 1e-12           # Spatial extent (m)
    
    # Kernel parameters
    kernel_type: str = 'warp_bubble'        # Type of kernel operator
    kernel_strength: float = 1.0            # Overall kernel strength
    throat_radius: float = 1e-15           # Warp bubble throat radius
    shell_thickness: float = 1e-14         # Shell thickness
    
    # Field parameters
    field_mass: float = 0.0                # Field mass
    coupling_constant: float = 1.0         # Field coupling
    
    # Numerical parameters
    regularization: float = 1e-12          # Numerical regularization
    max_condition_number: float = 1e12     # Maximum condition number

class GeneratingFunctionalAnalysis:
    """
    Generating functional approach for closed-form stress-energy tensors.
    
    Uses functional differentiation of G(J) to obtain analytic ⟨T₀₀⟩ expressions
    in terms of operator kernels and their inverses.
    """
    
    def __init__(self, config: GeneratingFunctionalConfig = None):
        self.config = config or GeneratingFunctionalConfig()
        
        # Create spatial grid
        self.r_grid = np.linspace(1e-16, self.config.spatial_extent, self.config.grid_size)
        self.dr = self.r_grid[1] - self.r_grid[0]
        
        # Initialize kernel operator
        self.K_matrix = self._construct_kernel_matrix()
        
        print(f"📐 Generating Functional Analysis Initialized")
        print(f"   Grid size: {self.config.grid_size}")
        print(f"   Spatial extent: {self.config.spatial_extent:.2e} m")
        print(f"   Kernel type: {self.config.kernel_type}")
        print(f"   Kernel condition number: {np.linalg.cond(self.K_matrix):.2e}")
    
    def _construct_kernel_matrix(self) -> np.ndarray:
        """
        Construct the kernel operator matrix K encoding warp-bubble profile.
        
        K_{ij} represents the coupling between grid points i and j.
        """
        n = self.config.grid_size
        K = np.zeros((n, n))
        
        if self.config.kernel_type == 'warp_bubble':
            # Warp bubble kernel based on modified metric
            r0 = self.config.throat_radius
            R = self.config.shell_thickness
            
            for i in range(n):
                for j in range(n):
                    r_i, r_j = self.r_grid[i], self.r_grid[j]
                    
                    # Warp bubble profile function
                    f_i = self._warp_profile_function(r_i, r0, R)
                    f_j = self._warp_profile_function(r_j, r0, R)
                    
                    # Kernel coupling strength
                    distance = abs(r_i - r_j)
                    coupling = np.exp(-distance / R) * f_i * f_j
                    
                    K[i, j] = self.config.kernel_strength * coupling
        
        elif self.config.kernel_type == 'differential':
            # Differential operator kernel (discretized Laplacian)
            main_diag = -2 * np.ones(n)
            off_diag = np.ones(n-1)
            
            K = self.config.kernel_strength * (
                np.diag(main_diag) + 
                np.diag(off_diag, 1) + 
                np.diag(off_diag, -1)
            ) / self.dr**2
        
        elif self.config.kernel_type == 'gaussian':
            # Gaussian kernel for smooth coupling
            sigma = self.config.shell_thickness
            
            for i in range(n):
                for j in range(n):
                    r_i, r_j = self.r_grid[i], self.r_grid[j]
                    K[i, j] = self.config.kernel_strength * np.exp(-((r_i - r_j) / sigma)**2)
        
        # Add regularization for numerical stability
        K += self.config.regularization * np.eye(n)
        
        return K
    
    def _warp_profile_function(self, r: float, r0: float, R: float) -> float:
        """Warp bubble profile function f(r)."""
        if r <= r0:
            return 0.0  # Inside throat
        elif r <= r0 + R:
            # Transition region
            x = (r - r0) / R
            return np.tanh(5 * x) * np.exp(-x)
        else:
            # Asymptotic region
            return np.exp(-(r - r0 - R) / R)
    
    def compute_I_minus_K_inverse(self) -> Tuple[np.ndarray, float]:
        """
        Compute (I - K)⁻¹ and det(I - K).
        
        Returns:
            (I - K)⁻¹ matrix and determinant
        """
        n = self.config.grid_size
        I = np.eye(n)
        I_minus_K = I - self.K_matrix
        
        # Check condition number
        cond_num = np.linalg.cond(I_minus_K)
        if cond_num > self.config.max_condition_number:
            print(f"⚠️  Warning: High condition number {cond_num:.2e}")
            # Add more regularization
            I_minus_K += self.config.regularization * np.eye(n)
        
        try:
            # Compute inverse and determinant
            I_minus_K_inv = np.linalg.inv(I_minus_K)
            det_I_minus_K = np.linalg.det(I_minus_K)
            
            return I_minus_K_inv, det_I_minus_K
            
        except np.linalg.LinAlgError:
            print("❌ Matrix inversion failed, using pseudoinverse")
            I_minus_K_inv = np.linalg.pinv(I_minus_K)
            det_I_minus_K = np.linalg.det(I_minus_K + self.config.regularization * np.eye(n))
            
            return I_minus_K_inv, det_I_minus_K
    
    def compute_generating_functional_coefficient(self) -> Dict[str, float]:
        """
        Compute coefficients for the generating functional G(J).
        
        G(J) = C * exp(½ J† M⁻¹ J)  where C = 1/√det(I-K), M = I-K
        
        Returns:
            Generating functional coefficients
        """
        M_inv, det_M = self.compute_I_minus_K_inverse()
        
        # Coefficient C = 1/√det(I-K)
        if det_M <= 0:
            print(f"⚠️  Non-positive determinant: {det_M}")
            C = 0.0
        else:
            C = 1.0 / np.sqrt(abs(det_M))
        
        return {
            'coefficient_C': C,
            'determinant': det_M,
            'determinant_log': np.log(abs(det_M)) if det_M != 0 else -np.inf,
            'M_inverse_trace': np.trace(M_inv),
            'M_inverse_norm': np.linalg.norm(M_inv),
            'numerical_stability': abs(det_M) > self.config.regularization
        }
    
    def compute_vacuum_expectation_T00(self) -> Dict[str, np.ndarray]:
        """
        Compute ⟨T₀₀(x)⟩ via functional differentiation.
        
        ⟨T₀₀(i)⟩ = ∂²G/∂J(i)∂J(i) |_{J=0} = C * M⁻¹_{ii}
        
        Returns:
            Vacuum expectation values and components
        """
        print("🔄 Computing vacuum expectation ⟨T₀₀⟩...")
        
        # Get generating functional components
        gf_coeffs = self.compute_generating_functional_coefficient()
        M_inv, det_M = self.compute_I_minus_K_inverse()
        
        # Functional derivatives: ⟨T₀₀(i)⟩ = C * M⁻¹_{ii}
        C = gf_coeffs['coefficient_C']
        diagonal_elements = np.diag(M_inv)
        
        # Vacuum expectation values
        T00_vacuum = C * diagonal_elements
        
        # Off-diagonal correlations: ⟨T₀₀(i)T₀₀(j)⟩ ∝ M⁻¹_{ij}
        T00_correlations = C * M_inv
        
        # Additional stress-energy components from off-diagonal terms
        T00_enhanced = T00_vacuum + 0.1 * np.sum(T00_correlations, axis=1)
        
        print(f"   Coefficient C: {C:.2e}")
        print(f"   T₀₀ range: [{T00_vacuum.min():.2e}, {T00_vacuum.max():.2e}]")
        print(f"   Enhanced T₀₀ range: [{T00_enhanced.min():.2e}, {T00_enhanced.max():.2e}]")
        print(f"   Negative fraction: {(T00_enhanced < 0).sum()/len(T00_enhanced):.1%}")
        
        return {
            'T00_vacuum': T00_vacuum,
            'T00_enhanced': T00_enhanced,
            'T00_correlations': T00_correlations,
            'M_inverse': M_inv,
            'generating_coefficients': gf_coeffs,
            'spatial_grid': self.r_grid
        }
    
    def compute_closed_form_anec_integral(self) -> Dict[str, float]:
        """
        Compute ANEC integral using closed-form expressions.
        
        ANEC = ∫ ⟨T₀₀(r)⟩ dr = C * ∑ᵢ M⁻¹_{ii} * Δr
        
        Returns:
            Closed-form ANEC results
        """
        print("📊 Computing closed-form ANEC integral...")
        
        # Get vacuum expectation values
        T00_result = self.compute_vacuum_expectation_T00()
        
        # ANEC integrals
        anec_vacuum = np.trapz(T00_result['T00_vacuum'], self.r_grid)
        anec_enhanced = np.trapz(T00_result['T00_enhanced'], self.r_grid)
        
        # Analytical expression: ANEC = C * tr(M⁻¹) * (spatial extent)
        C = T00_result['generating_coefficients']['coefficient_C']
        trace_M_inv = T00_result['generating_coefficients']['M_inverse_trace']
        anec_analytical = C * trace_M_inv * self.config.spatial_extent / self.config.grid_size
        
        results = {
            'anec_vacuum': anec_vacuum,
            'anec_enhanced': anec_enhanced,
            'anec_analytical': anec_analytical,
            'coefficient_C': C,
            'trace_M_inverse': trace_M_inv,
            'negative_anec': anec_enhanced < 0,
            'enhancement_factor': anec_enhanced / anec_vacuum if anec_vacuum != 0 else 0
        }
        
        print(f"   ANEC (vacuum): {anec_vacuum:.2e} J·s·m⁻³")
        print(f"   ANEC (enhanced): {anec_enhanced:.2e} J·s·m⁻³")
        print(f"   ANEC (analytical): {anec_analytical:.2e} J·s·m⁻³")
        print(f"   Negative ANEC: {'YES' if results['negative_anec'] else 'NO'}")
        
        return results
    
    def optimize_kernel_parameters(self, target_anec: float = -1e5) -> Dict[str, any]:
        """
        Optimize kernel parameters for target ANEC value.
        
        Args:
            target_anec: Target ANEC integral value
            
        Returns:
            Optimization results
        """
        print(f"🎯 Optimizing kernel parameters for ANEC < {target_anec:.2e}")
        
        best_anec = float('inf')
        best_params = None
        best_result = None
        
        # Parameter search ranges
        strength_range = np.logspace(-2, 2, 20)  # 0.01 to 100
        throat_range = np.logspace(-16, -13, 15)  # 1e-16 to 1e-13 m
        thickness_range = np.logspace(-15, -12, 15)  # 1e-15 to 1e-12 m
        
        trial_count = 0
        
        for strength in strength_range:
            for throat in throat_range:
                for thickness in thickness_range:
                    if thickness <= throat:
                        continue  # Invalid configuration
                    
                    try:
                        # Update parameters
                        original_strength = self.config.kernel_strength
                        original_throat = self.config.throat_radius
                        original_thickness = self.config.shell_thickness
                        
                        self.config.kernel_strength = strength
                        self.config.throat_radius = throat
                        self.config.shell_thickness = thickness
                        
                        # Rebuild kernel matrix
                        self.K_matrix = self._construct_kernel_matrix()
                        
                        # Compute ANEC
                        anec_result = self.compute_closed_form_anec_integral()
                        anec_value = anec_result['anec_enhanced']
                        
                        # Check if this is better (more negative)
                        if anec_value < best_anec:
                            best_anec = anec_value
                            best_params = {
                                'kernel_strength': strength,
                                'throat_radius': throat,
                                'shell_thickness': thickness
                            }
                            best_result = anec_result
                            
                            if anec_value < target_anec:
                                print(f"   🎯 Target achieved! ANEC = {anec_value:.2e}")
                                print(f"      Strength: {strength:.2e}")
                                print(f"      Throat: {throat:.2e} m")
                                print(f"      Thickness: {thickness:.2e} m")
                        
                        trial_count += 1
                        
                        # Restore original parameters
                        self.config.kernel_strength = original_strength
                        self.config.throat_radius = original_throat
                        self.config.shell_thickness = original_thickness
                        
                    except Exception as e:
                        # Restore parameters on error
                        self.config.kernel_strength = original_strength
                        self.config.throat_radius = original_throat
                        self.config.shell_thickness = original_thickness
                        continue
        
        # Apply best parameters if found
        if best_params is not None:
            self.config.kernel_strength = best_params['kernel_strength']
            self.config.throat_radius = best_params['throat_radius']
            self.config.shell_thickness = best_params['shell_thickness']
            self.K_matrix = self._construct_kernel_matrix()
            
            print(f"\n✅ Optimization complete!")
            print(f"   Best ANEC: {best_anec:.2e} J·s·m⁻³")
            print(f"   Target achieved: {'YES' if best_anec < target_anec else 'NO'}")
            print(f"   Optimal parameters: {best_params}")
        
        return {
            'optimization_success': best_params is not None,
            'best_anec': best_anec,
            'target_achieved': best_anec < target_anec,
            'best_parameters': best_params,
            'best_result': best_result,
            'trials_completed': trial_count
        }
    
    def symbolic_analysis(self) -> Dict[str, any]:
        """
        Perform symbolic analysis of the generating functional.
        
        Returns symbolic expressions for small system sizes.
        """
        print("🔢 Performing symbolic analysis...")
        
        # Use smaller system for symbolic computation
        n_sym = min(4, self.config.grid_size)
        
        # Define symbolic variables
        K_sym = sp.Matrix([[sp.Symbol(f'K_{i}{j}') for j in range(n_sym)] for i in range(n_sym)])
        I_sym = sp.eye(n_sym)
        
        # Symbolic (I - K)
        I_minus_K_sym = I_sym - K_sym
        
        try:
            # Symbolic inverse and determinant
            I_minus_K_inv_sym = I_minus_K_sym.inv()
            det_sym = I_minus_K_sym.det()
            
            # Vacuum expectation (diagonal elements)
            T00_diagonal = [I_minus_K_inv_sym[i, i] for i in range(n_sym)]
            
            # Coefficient C = 1/√det(I-K)
            C_sym = 1 / sp.sqrt(det_sym)
            
            # Full T₀₀ expressions
            T00_expressions = [C_sym * diag_elem for diag_elem in T00_diagonal]
            
            print(f"   ✅ Symbolic analysis complete for {n_sym}×{n_sym} system")
            
            return {
                'symbolic_success': True,
                'system_size': n_sym,
                'I_minus_K_inverse': I_minus_K_inv_sym,
                'determinant': det_sym,
                'coefficient_C': C_sym,
                'T00_expressions': T00_expressions,
                'diagonal_elements': T00_diagonal
            }
            
        except Exception as e:
            print(f"   ❌ Symbolic analysis failed: {e}")
            return {
                'symbolic_success': False,
                'error': str(e)
            }

def demo_generating_functional():
    """Demonstrate generating functional approach."""
    print("📐 Generating Functional Approach Demo")
    print("=" * 50)
    
    # Create analysis system
    config = GeneratingFunctionalConfig(
        grid_size=50,
        spatial_extent=1e-12,
        kernel_type='warp_bubble',
        kernel_strength=0.5,
        throat_radius=2e-15,
        shell_thickness=1e-14
    )
    
    gf_analysis = GeneratingFunctionalAnalysis(config)
    
    # Compute vacuum expectation values
    T00_result = gf_analysis.compute_vacuum_expectation_T00()
    
    # Compute ANEC integral
    anec_result = gf_analysis.compute_closed_form_anec_integral()
    
    # Run optimization
    opt_result = gf_analysis.optimize_kernel_parameters(target_anec=-1e4)
    
    # Symbolic analysis
    sym_result = gf_analysis.symbolic_analysis()
    
    return gf_analysis, T00_result, anec_result, opt_result, sym_result

if __name__ == "__main__":
    demo_generating_functional()
