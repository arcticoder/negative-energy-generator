#!/usr/bin/env python3
"""
Machine Learning Ansatz Discovery Module
=======================================

Implements automated ansatz discovery using machine learning
to optimize warp-bubble geometries for maximum negative energy.

This tackles Bottleneck #4: Automated Ansatz Discovery via ML

Uses Bayesian optimization and neural networks to discover
optimal shape functions h(r,θ,φ) that maximize ∫ T₀₀ d⁴x.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

class MLAnsatzDiscovery:
    """
    Machine learning-based ansatz discovery for optimal negative energy geometries.
    
    Uses Bayesian optimization to explore the space of possible ansatz functions
    and discover those that maximize negative energy density.
    """
    
    def __init__(self, max_basis_functions: int = 20):
        """
        Initialize ML ansatz discovery.
        
        Args:
            max_basis_functions: maximum number of basis functions to use
        """
        self.max_basis_functions = max_basis_functions
        self.hbar = 1.054571817e-34  # Planck constant [J⋅s]
        self.c = 2.99792458e8        # Speed of light [m/s]
        
        # Basis function parameters
        self.basis_types = ['gaussian', 'polynomial', 'exponential', 'sinusoidal']
        
        # History tracking
        self.evaluation_history = []
        self.parameter_history = []
    
    def spherical_harmonics_basis(self, r: np.ndarray, theta: np.ndarray, 
                                 phi: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        Spherical harmonics basis for ansatz expansion.
        
        h(r,θ,φ) = Σᵢ aᵢ Rᵢ(r) Yₗᵐ(θ,φ)
        
        Args:
            r, theta, phi: spherical coordinates
            coeffs: expansion coefficients
            
        Returns:
            Basis function values
        """
        n_coeffs = len(coeffs)
        n_basis = min(n_coeffs, self.max_basis_functions)
        
        result = np.zeros_like(r)
        
        # Simplified basis functions
        for i in range(n_basis):
            # Radial part
            if i % 4 == 0:  # Gaussian
                R_i = np.exp(-(r - 1e-6)**2 / (0.5e-6)**2)
            elif i % 4 == 1:  # Polynomial
                R_i = (r / 1e-6)**2 * np.exp(-r / 1e-6)
            elif i % 4 == 2:  # Exponential
                R_i = np.exp(-abs(r - 1.5e-6) / 0.3e-6)
            else:  # Power law
                R_i = (r / 1e-6)**(-0.5) * np.exp(-r / 2e-6)
            
            # Angular part (simplified spherical harmonics)
            l = i // 4
            m = i % 4 - 2
            
            if l == 0:
                Y_lm = 1.0
            elif l == 1:
                if m == -1:
                    Y_lm = np.sin(theta) * np.sin(phi)
                elif m == 0:
                    Y_lm = np.cos(theta)
                else:
                    Y_lm = np.sin(theta) * np.cos(phi)
            else:
                # Higher order approximations
                Y_lm = np.cos(l * theta) * np.cos(m * phi)
            
            result += coeffs[i] * R_i * Y_lm
        
        return result
    
    def compute_stress_energy(self, coeffs: np.ndarray, 
                            mu: float = 0.5, tau: float = 1e-12) -> float:
        """
        Compute stress-energy tensor T₀₀ for given ansatz coefficients.
        
        This is the objective function that ML tries to minimize.
        
        Args:
            coeffs: ansatz expansion coefficients
            mu: coupling parameter
            tau: temporal scale
            
        Returns:
            Integrated T₀₀ value (target: make this as negative as possible)
        """
        # Spatial grid
        nr, ntheta, nphi = 50, 25, 25
        r = np.linspace(0.5e-6, 3e-6, nr)
        theta = np.linspace(0, np.pi, ntheta)
        phi = np.linspace(0, 2*np.pi, nphi)
        
        # Create meshgrid
        R, THETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')
        
        # Compute ansatz function
        h = self.spherical_harmonics_basis(R, THETA, PHI, coeffs)
        
        # Temporal derivative approximation
        # f(r,θ,φ,t) = 1 + μ g(t) h(r,θ,φ)
        # T₀₀ ∼ -μ ∂ₜ²g(t) h(r,θ,φ)
        
        # Gaussian pulse: g(t) = exp(-t²/2τ²)
        # ∂ₜ²g(0) = -1/τ²
        d2g_dt2 = -1 / tau**2
        
        T00 = -mu * d2g_dt2 * h
        
        # Volume element in spherical coordinates
        dr = r[1] - r[0]
        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        
        volume_element = R**2 * np.sin(THETA) * dr * dtheta * dphi
        
        # Integrate T₀₀ over space
        integral = np.sum(T00 * volume_element)
        
        # Convert to physical units
        return self.hbar * self.c**2 * integral
    
    def objective_function(self, coeffs: np.ndarray) -> float:
        """
        Objective function for optimization (minimize for most negative energy).
        
        Args:
            coeffs: ansatz coefficients to evaluate
            
        Returns:
            Objective value (negative of T₀₀ integral, so minimize = maximize |T₀₀|)
        """
        try:
            stress_energy = self.compute_stress_energy(coeffs)
            
            # Add regularization to prevent overfitting
            regularization = 0.01 * np.sum(coeffs**2)
            
            # We want to minimize this (maximize negative energy)
            objective = -stress_energy + regularization
            
            # Store history
            self.evaluation_history.append(stress_energy)
            self.parameter_history.append(coeffs.copy())
            
            return objective
            
        except Exception as e:
            # Return large positive value for failed evaluations
            return 1e10
    
    def bayesian_optimization(self, n_iterations: int = 50, 
                            n_initial: int = 10) -> Dict:
        """
        Bayesian optimization to find optimal ansatz coefficients.
        
        Args:
            n_iterations: number of optimization iterations
            n_initial: number of initial random samples
            
        Returns:
            Optimization results
        """
        print("🧠 BAYESIAN ANSATZ OPTIMIZATION")
        print("-" * 32)
        
        # Initialize with random coefficients
        bounds = [(-1.0, 1.0)] * self.max_basis_functions
        
        # Initial random sampling
        initial_X = []
        initial_y = []
        
        print(f"Initial sampling: {n_initial} points...")
        for i in range(n_initial):
            coeffs = np.random.uniform(-1, 1, self.max_basis_functions)
            objective_val = self.objective_function(coeffs)
            initial_X.append(coeffs)
            initial_y.append(objective_val)
            
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{n_initial}: Best so far = {np.min(initial_y):.3e}")
        
        X = np.array(initial_X)
        y = np.array(initial_y)
        
        # Set up Gaussian Process
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        print(f"\nBayesian optimization: {n_iterations} iterations...")
        
        for iteration in range(n_iterations):
            # Fit GP to current data
            gp.fit(X, y)
            
            # Acquisition function: Expected Improvement
            def acquisition(coeffs_flat):
                coeffs_2d = coeffs_flat.reshape(1, -1)
                mu, sigma = gp.predict(coeffs_2d, return_std=True)
                
                # Expected improvement
                improvement = np.min(y) - mu
                Z = improvement / (sigma + 1e-9)
                ei = improvement * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
                
                return -ei[0]  # Minimize negative EI
            
            # Optimize acquisition function
            best_acquisition = np.inf
            best_next_coeffs = None
            
            # Multiple random starts for acquisition optimization
            for _ in range(10):
                start_coeffs = np.random.uniform(-1, 1, self.max_basis_functions)
                
                try:
                    result = minimize(
                        acquisition,
                        start_coeffs,
                        method='L-BFGS-B',
                        bounds=bounds
                    )
                    
                    if result.success and result.fun < best_acquisition:
                        best_acquisition = result.fun
                        best_next_coeffs = result.x
                except:
                    continue
            
            if best_next_coeffs is not None:
                # Evaluate objective at next point
                next_objective = self.objective_function(best_next_coeffs)
                
                # Add to dataset
                X = np.vstack([X, best_next_coeffs])
                y = np.append(y, next_objective)
                
                if (iteration + 1) % 10 == 0:
                    best_so_far = np.min(y)
                    print(f"  Iteration {iteration+1}/{n_iterations}: Best = {best_so_far:.3e}")
        
        # Find best result
        best_idx = np.argmin(y)
        best_coeffs = X[best_idx]
        best_objective = y[best_idx]
        best_stress_energy = -best_objective  # Convert back
        
        print(f"\nOptimization complete!")
        print(f"Best T₀₀ integral: {best_stress_energy:.3e} J⋅s⋅m⁻³")
        print(f"Best coefficients: {best_coeffs}")
        
        return {
            'best_coeffs': best_coeffs,
            'best_objective': best_objective,
            'best_stress_energy': best_stress_energy,
            'all_X': X,
            'all_y': y,
            'gp_model': gp
        }
    
    def _normal_pdf(self, x):
        """Standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _normal_cdf(self, x):
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def genetic_algorithm_refinement(self, initial_coeffs: np.ndarray,
                                   population_size: int = 50,
                                   generations: int = 100) -> Dict:
        """
        Genetic algorithm refinement of ansatz coefficients.
        
        Follows up Bayesian optimization with genetic algorithm for fine-tuning.
        
        Args:
            initial_coeffs: starting coefficients from Bayesian optimization
            population_size: GA population size
            generations: number of generations
            
        Returns:
            Refined coefficients and results
        """
        print("🧬 GENETIC ALGORITHM REFINEMENT")
        print("-" * 31)
        
        # Initialize population around best coefficients
        population = []
        for _ in range(population_size):
            # Add noise to initial coefficients
            noise = np.random.normal(0, 0.1, len(initial_coeffs))
            individual = initial_coeffs + noise
            individual = np.clip(individual, -1, 1)  # Keep in bounds
            population.append(individual)
        
        population = np.array(population)
        
        best_fitness_history = []
        
        for generation in range(generations):
            # Evaluate fitness (negative objective function)
            fitness = []
            for individual in population:
                objective_val = self.objective_function(individual)
                fitness.append(-objective_val)  # Convert to maximization
            
            fitness = np.array(fitness)
            best_fitness_history.append(np.max(fitness))
            
            if (generation + 1) % 20 == 0:
                print(f"Generation {generation+1}/{generations}: "
                      f"Best fitness = {np.max(fitness):.3e}")
            
            # Selection (tournament selection)
            new_population = []
            
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 5
                tournament_indices = np.random.choice(population_size, tournament_size)
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            
            population = np.array(new_population)
            
            # Crossover and mutation
            for i in range(0, population_size - 1, 2):
                # Crossover
                if np.random.random() < 0.7:  # Crossover probability
                    alpha = np.random.random()
                    child1 = alpha * population[i] + (1 - alpha) * population[i + 1]
                    child2 = (1 - alpha) * population[i] + alpha * population[i + 1]
                    population[i] = child1
                    population[i + 1] = child2
                
                # Mutation
                for j in range(2):
                    if np.random.random() < 0.1:  # Mutation probability
                        mutation = np.random.normal(0, 0.05, len(population[i + j]))
                        population[i + j] += mutation
                        population[i + j] = np.clip(population[i + j], -1, 1)
        
        # Final evaluation
        final_fitness = []
        for individual in population:
            objective_val = self.objective_function(individual)
            final_fitness.append(-objective_val)
        
        final_fitness = np.array(final_fitness)
        best_idx = np.argmax(final_fitness)
        best_coeffs = population[best_idx]
        best_stress_energy = final_fitness[best_idx]
        
        print(f"\nGenetic algorithm complete!")
        print(f"Final best T₀₀: {best_stress_energy:.3e} J⋅s⋅m⁻³")
        
        return {
            'best_coeffs': best_coeffs,
            'best_stress_energy': best_stress_energy,
            'fitness_history': best_fitness_history,
            'final_population': population
        }
    
    def analyze_discovered_ansatz(self, coeffs: np.ndarray) -> Dict:
        """
        Analyze the discovered ansatz function.
        
        Args:
            coeffs: optimal coefficients
            
        Returns:
            Analysis of the ansatz geometry and properties
        """
        print("📊 DISCOVERED ANSATZ ANALYSIS")
        print("-" * 29)
        
        # Evaluate ansatz on grid
        nr, ntheta, nphi = 30, 15, 15
        r = np.linspace(0.5e-6, 3e-6, nr)
        theta = np.linspace(0, np.pi, ntheta)
        phi = np.linspace(0, 2*np.pi, nphi)
        
        R, THETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')
        h = self.spherical_harmonics_basis(R, THETA, PHI, coeffs)
        
        # Compute properties
        h_min = np.min(h)
        h_max = np.max(h)
        h_mean = np.mean(h)
        h_std = np.std(h)
        
        # Find negative regions
        negative_fraction = np.sum(h < 0) / h.size
        
        # Symmetry analysis
        # Check spherical symmetry by comparing θ=0 and θ=π profiles
        h_north = h[:, 0, 0]  # θ=0
        h_south = h[:, -1, 0]  # θ=π
        symmetry_measure = np.mean(np.abs(h_north - h_south))
        
        analysis = {
            'h_min': h_min,
            'h_max': h_max,
            'h_mean': h_mean,
            'h_std': h_std,
            'negative_fraction': negative_fraction,
            'symmetry_measure': symmetry_measure,
            'dominant_coeffs': np.argsort(np.abs(coeffs))[-5:],  # Top 5 coefficients
            'ansatz_values': h
        }
        
        print(f"Ansatz range: [{h_min:.3f}, {h_max:.3f}]")
        print(f"Mean value: {h_mean:.3f}")
        print(f"Standard deviation: {h_std:.3f}")
        print(f"Negative fraction: {negative_fraction*100:.1f}%")
        print(f"Symmetry measure: {symmetry_measure:.3f}")
        print(f"Dominant coefficients: {analysis['dominant_coeffs']}")
        
        # Assessment
        if negative_fraction > 0.3 and symmetry_measure > 0.1:
            assessment = "✅ BREAKTHROUGH: Asymmetric geometry with large negative regions"
        elif negative_fraction > 0.1:
            assessment = "🎯 PROGRESS: Significant negative regions discovered"
        else:
            assessment = "⚠️ LIMITED: Small negative regions found"
        
        print(f"Assessment: {assessment}")
        
        return analysis

def demonstrate_ml_ansatz_discovery():
    """Demonstrate machine learning ansatz discovery."""
    
    print("🤖 MACHINE LEARNING ANSATZ DISCOVERY DEMONSTRATION")
    print("=" * 52)
    print()
    print("Automated discovery of optimal warp-bubble geometries")
    print("Goal: Find h(r,θ,φ) that maximizes ∫ T₀₀ d⁴x")
    print()
    
    # Initialize ML system
    ml_system = MLAnsatzDiscovery(max_basis_functions=15)
    
    # Step 1: Bayesian optimization
    bayesian_result = ml_system.bayesian_optimization(n_iterations=30, n_initial=8)
    print()
    
    # Step 2: Genetic algorithm refinement
    ga_result = ml_system.genetic_algorithm_refinement(
        bayesian_result['best_coeffs'],
        population_size=30,
        generations=50
    )
    print()
    
    # Step 3: Analyze discovered ansatz
    analysis = ml_system.analyze_discovered_ansatz(ga_result['best_coeffs'])
    print()
    
    # Compare results
    print("🏆 OPTIMIZATION COMPARISON")
    print("-" * 24)
    
    bayesian_energy = bayesian_result['best_stress_energy']
    ga_energy = ga_result['best_stress_energy']
    
    print(f"Bayesian optimization: {bayesian_energy:.3e} J⋅s⋅m⁻³")
    print(f"Genetic algorithm: {ga_energy:.3e} J⋅s⋅m⁻³")
    
    improvement = abs(ga_energy / bayesian_energy)
    print(f"GA improvement: {improvement:.2f}×")
    
    # Target assessment
    current_theory = -2.09e-6  # Current theory baseline
    target = -1e5
    
    ml_improvement = abs(ga_energy / current_theory)
    remaining_gap = abs(target / ga_energy)
    
    print(f"\nML improvement vs current theory: {ml_improvement:.1f}×")
    print(f"Remaining gap to target: {remaining_gap:.0f}×")
    
    # Overall assessment
    if ml_improvement > 50 and analysis['negative_fraction'] > 0.2:
        overall = "✅ MAJOR BREAKTHROUGH: ML discovered superior geometry"
    elif ml_improvement > 10:
        overall = "🎯 SIGNIFICANT PROGRESS: Substantial improvement found"
    else:
        overall = "⚠️ INCREMENTAL: Modest gains achieved"
    
    print(f"Overall assessment: {overall}")
    
    return {
        'ml_system': ml_system,
        'bayesian_result': bayesian_result,
        'ga_result': ga_result,
        'analysis': analysis,
        'ml_improvement': ml_improvement,
        'remaining_gap': remaining_gap
    }

if __name__ == "__main__":
    demonstrate_ml_ansatz_discovery()
