"""
Surrogate-Model Bayesian Optimization of Exotic Matter Ansätze
============================================================

This module implements ML-powered optimization of exotic matter configurations
using Gaussian Process surrogates and Expected Improvement acquisition.

Mathematical Foundation:
    J(θ) = ∫ T₀₀(x;θ) d³x  (we want J(θ) as negative as possible)
    BO builds GP surrogate Ĵ(θ) and maximizes Expected Improvement:
    EI(θ) = E[max(0, J_min - Ĵ(θ))]
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path
import json
from datetime import datetime

try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.acquisition import gaussian_ei
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    warnings.warn("scikit-optimize not available. Install with: pip install scikit-optimize")
    SKOPT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class ExoticMatterBayesianOptimizer:
    """
    Bayesian optimization for exotic matter ansatz parameters.
    
    Uses Gaussian Process regression to build a cheap surrogate for expensive
    T₀₀ calculations and intelligently explores parameter space.
    """
    
    def __init__(self, energy_evaluator: Callable, parameter_space: List, 
                 acquisition_func: str = "EI", random_seed: int = 42):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            energy_evaluator: Function that takes parameter vector θ and returns J(θ)
            parameter_space: List of skopt space dimensions (Real, Integer, Categorical)
            acquisition_func: Acquisition function ("EI", "PI", "LCB")
            random_seed: Random seed for reproducibility
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        self.energy_evaluator = energy_evaluator
        self.parameter_space = parameter_space
        self.acquisition_func = acquisition_func
        self.random_seed = random_seed
        self.optimization_history = []
        self.best_params = None
        self.best_energy = np.inf  # We want most negative, so start with +inf
        
    def objective_wrapper(self, params: List) -> float:
        """
        Wrapper for the objective function that handles minimization convention.
        
        Since we want to maximize negative energy (minimize positive values),
        we return -J(θ) so that minimization finds the most negative J.
        """
        try:
            # Evaluate the exotic matter configuration
            J = self.energy_evaluator(params)
            
            # Store in history
            self.optimization_history.append({
                'params': params.copy(),
                'energy_J': J,
                'objective_value': -J,  # What we're minimizing
                'timestamp': datetime.now().isoformat()
            })
            
            # Update best if this is more negative
            if J < self.best_energy:
                self.best_energy = J
                self.best_params = params.copy()
            
            # Return negative for minimization (most negative J = smallest -J)
            return -J
            
        except Exception as e:
            warnings.warn(f"Evaluation failed for params {params}: {e}")
            return 1e6  # Large positive penalty
    
    def optimize(self, n_calls: int = 100, n_initial_points: int = 10,
                verbose: bool = True) -> Dict:
        """
        Run Bayesian optimization to find optimal exotic matter parameters.
        
        Args:
            n_calls: Total number of objective function evaluations
            n_initial_points: Number of random initial points
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization results
        """
        if verbose:
            print(f"🚀 Starting Bayesian Optimization for Exotic Matter Ansätze")
            print(f"   • Parameter space: {len(self.parameter_space)} dimensions")
            print(f"   • Budget: {n_calls} evaluations")
            print(f"   • Initial random points: {n_initial_points}")
        
        # Run optimization
        result = gp_minimize(
            func=self.objective_wrapper,
            dimensions=self.parameter_space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acquisition=self.acquisition_func,
            random_state=self.random_seed,
            verbose=verbose
        )
        
        # Extract results
        optimal_params = result.x
        optimal_objective = result.fun
        optimal_energy = -optimal_objective  # Convert back to energy
        
        optimization_result = {
            'success': True,
            'optimal_parameters': optimal_params,
            'optimal_energy_J': optimal_energy,
            'n_evaluations': len(result.func_vals),
            'convergence_trace': (-np.array(result.func_vals)).tolist(),  # Energy values
            'parameter_names': [dim.name for dim in self.parameter_space],
            'optimization_history': self.optimization_history,
            'acquisition_function': self.acquisition_func,
            'gp_result': {
                'best_objective': result.fun,
                'best_params': result.x,
                'func_vals': result.func_vals,
                'x_iters': result.x_iters
            }
        }
        
        if verbose:
            print(f"✅ Optimization completed!")
            print(f"   • Best energy: {optimal_energy:.3e} J")
            print(f"   • Best parameters: {optimal_params}")
            print(f"   • Total evaluations: {len(result.func_vals)}")
        
        return optimization_result
    
    def analyze_parameter_importance(self, n_samples: int = 1000) -> Dict:
        """
        Analyze which parameters are most important for negative energy.
        
        Uses the trained GP to sample parameter space and compute sensitivities.
        """
        if len(self.optimization_history) < 10:
            warnings.warn("Need at least 10 evaluations for parameter analysis")
            return {}
        
        # This would require access to the trained GP model
        # For now, return a basic analysis based on optimization history
        
        history = self.optimization_history
        params_array = np.array([h['params'] for h in history])
        energies = np.array([h['energy_J'] for h in history])
        
        # Simple correlation analysis
        correlations = {}
        for i, dim in enumerate(self.parameter_space):
            if hasattr(dim, 'name') and dim.name:
                param_name = dim.name
            else:
                param_name = f"param_{i}"
            
            if len(params_array) > 1:
                corr = np.corrcoef(params_array[:, i], energies)[0, 1]
                correlations[param_name] = corr
        
        return {
            'parameter_correlations': correlations,
            'most_important': max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None,
            'analysis_note': "Based on linear correlation with energy values"
        }
    
    def plot_optimization_trace(self, save_path: Optional[str] = None):
        """Plot the optimization convergence trace."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("matplotlib not available for plotting")
            return
        
        if not self.optimization_history:
            warnings.warn("No optimization history to plot")
            return
        
        energies = [h['energy_J'] for h in self.optimization_history]
        cumulative_best = np.minimum.accumulate(energies)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(energies, 'b-', alpha=0.6, label='Individual evaluations')
        plt.plot(cumulative_best, 'r-', linewidth=2, label='Best so far')
        plt.xlabel('Evaluation')
        plt.ylabel('Energy (J)')
        plt.title('Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(-np.array(energies), 'g-', alpha=0.7, label='|Energy| (log scale)')
        plt.xlabel('Evaluation')
        plt.ylabel('|Energy| (J)')
        plt.title('Energy Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization trace saved to {save_path}")
        else:
            plt.show()


def create_ansatz_parameter_space(ansatz_type: str = "warp_bubble") -> List:
    """
    Create parameter space for different exotic matter ansätze.
    
    Args:
        ansatz_type: Type of ansatz ("warp_bubble", "spin_network", "polymer_qft")
        
    Returns:
        List of skopt dimension objects
    """
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize required")
    
    if ansatz_type == "warp_bubble":
        # Warp bubble parameters: shape coefficients, throat size, etc.
        return [
            Real(0.1, 10.0, name="throat_radius"),
            Real(0.1, 5.0, name="expansion_rate"),
            Real(0.01, 1.0, name="wall_thickness"),
            Real(0.1, 2.0, name="shape_coeff_1"),
            Real(0.1, 2.0, name="shape_coeff_2"),
            Real(0.1, 2.0, name="shape_coeff_3"),
            Real(-1.0, 1.0, name="asymmetry_param"),
            Real(0.5, 3.0, name="smoothness_param")
        ]
    
    elif ansatz_type == "spin_network":
        # Spin network parameters: couplings, vertex weights, etc.
        return [
            Real(0.01, 1.0, name="coupling_strength"),
            Real(0.1, 10.0, name="decay_length"),
            Integer(3, 15, name="max_spin"),
            Real(0.1, 2.0, name="vertex_weight_1"),
            Real(0.1, 2.0, name="vertex_weight_2"),
            Real(0.1, 2.0, name="vertex_weight_3"),
            Real(0.01, 1.0, name="recoupling_param"),
            Categorical(['SU2', 'SU3', 'SO3'], name="gauge_group")
        ]
    
    elif ansatz_type == "polymer_qft":
        # Polymer QFT parameters: discretization, field couplings
        return [
            Real(1e-9, 1e-6, name="polymer_scale"),
            Real(0.1, 10.0, name="field_coupling"),
            Integer(5, 50, name="grid_size"),
            Real(0.01, 1.0, name="nonlinearity_param"),
            Real(0.1, 5.0, name="backreaction_strength"),
            Real(0.01, 2.0, name="quantum_correction_1"),
            Real(0.01, 2.0, name="quantum_correction_2")
        ]
    
    else:
        # Generic parameter space
        return [Real(0.1, 10.0, name=f"coeff_{i}") for i in range(10)]


# Example usage and testing
if __name__ == "__main__":
    print("=== Bayesian Optimization of Exotic Matter Ansätze ===")
    
    if not SKOPT_AVAILABLE:
        print("❌ scikit-optimize not available. Install with: pip install scikit-optimize")
        exit(1)
    
    # Mock energy evaluator for testing
    def mock_energy_evaluator(params):
        """
        Mock function that simulates exotic matter energy calculation.
        
        This would be replaced with actual T₀₀ computation from your
        exotic matter simulator.
        """
        # Simulate some complex energy landscape with a global minimum
        x = np.array(params)
        
        # Multiple local minima with one global minimum around [2, 1, 0.5, ...]
        energy = (
            np.sum((x - [2, 1, 0.5, 1, 1, 1, 0, 1.5])[:len(x)]**2) +  # Main quadratic
            0.5 * np.sum(np.sin(5 * x)**2) +  # Local oscillations
            0.1 * np.sum(x**4) +  # Quartic term
            np.random.normal(0, 0.01)  # Small noise
        )
        
        # Make it negative (exotic matter should have negative energy)
        return -energy - 10
    
    # Create parameter space for warp bubble
    space = create_ansatz_parameter_space("warp_bubble")
    
    # Initialize optimizer
    optimizer = ExoticMatterBayesianOptimizer(
        energy_evaluator=mock_energy_evaluator,
        parameter_space=space,
        acquisition_func="EI"
    )
    
    # Run optimization
    result = optimizer.optimize(n_calls=50, n_initial_points=10, verbose=True)
    
    # Analyze results
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"   • Best energy found: {result['optimal_energy_J']:.3e} J")
    print(f"   • Best parameters: {result['optimal_parameters']}")
    
    # Parameter importance analysis
    importance = optimizer.analyze_parameter_importance()
    if importance:
        print(f"\n🔍 PARAMETER IMPORTANCE:")
        for param, corr in importance['parameter_correlations'].items():
            print(f"   • {param}: correlation = {corr:.3f}")
        
        if importance['most_important']:
            name, corr = importance['most_important']
            print(f"   • Most important: {name} (|corr| = {abs(corr):.3f})")
    
    # Save results
    output_file = "bo_ansatz_optimization_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Plot if available
    try:
        optimizer.plot_optimization_trace("bo_optimization_trace.png")
    except:
        print("   (Plotting not available)")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Replace mock_energy_evaluator with real T₀₀ computation")
    print("2. Adjust parameter space for your specific ansatz")
    print("3. Run with larger budget (n_calls=200-1000)")
    print("4. Use best parameters in your exotic matter prototype")
    print("5. Iterate with refined parameter bounds based on results")
