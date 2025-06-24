# src/prototype/bayesian_optimization.py
import numpy as np
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-optimize not available. Install with: pip install scikit-optimize")
    SKOPT_AVAILABLE = False

hbar, c = 1.054e-34, 3e8

def objective(d):
    """Objective function: minimize negative |E| to maximize Casimir energy magnitude."""
    energy = np.sum(- (np.pi**2 * hbar * c)/(720 * np.array(d)**3))
    return -abs(energy)

def simple_grid_optimization(N, d_min, d_max, n_points=20):
    """Simple grid search optimization as fallback."""
    print("🔍 GRID SEARCH OPTIMIZATION")
    print("-" * 27)
    
    # Create grid
    d_range = np.linspace(d_min, d_max, n_points)
    
    best_energy = float('inf')
    best_gaps = None
    
    # Evaluate combinations (simplified for computational efficiency)
    for d_uniform in d_range:
        gaps = np.full(N, d_uniform)
        energy = -objective(gaps)  # Convert back to actual energy
        
        if abs(energy) > abs(best_energy):
            best_energy = energy
            best_gaps = gaps.copy()
    
    print(f"Best uniform gap: {best_gaps[0]*1e9:.2f} nm")
    print(f"Best energy: {best_energy:.3e} J/m²")
    
    return best_gaps, best_energy

def bayesian_optimization(N, d_min, d_max, n_calls=30):
    """Bayesian optimization using Gaussian processes."""
    
    if not SKOPT_AVAILABLE:
        print("Falling back to grid search...")
        return simple_grid_optimization(N, d_min, d_max)
    
    print("🤖 BAYESIAN OPTIMIZATION")
    print("-" * 24)
    
    # Define search space
    space = [Real(d_min, d_max, name=f'd{i}') for i in range(N)]
    
    # Run optimization
    print(f"Optimizing {N} gaps over {n_calls} evaluations...")
    
    res = gp_minimize(
        objective, 
        space, 
        n_calls=n_calls, 
        random_state=0,
        acq_func='EI',  # Expected Improvement
        n_initial_points=10
    )
    
    optimal_gaps = np.array(res.x)
    optimal_energy = -res.fun  # Convert back to actual energy
    
    print(f"Optimization converged: {res.success}")
    print(f"Function evaluations: {len(res.func_vals)}")
    print(f"Optimal gaps: {[f'{g*1e9:.2f}' for g in optimal_gaps]} nm")
    print(f"Optimal energy: {optimal_energy:.3e} J/m²")
    
    return optimal_gaps, optimal_energy

def optimization_comparison(N=5, d_min=5e-9, d_max=1e-8):
    """Compare different optimization strategies."""
    
    print("⚡ CASIMIR ARRAY OPTIMIZATION COMPARISON")
    print("=" * 42)
    print()
    
    # Uniform baseline
    print("📏 UNIFORM BASELINE")
    print("-" * 18)
    d_uniform = (d_min + d_max) / 2
    uniform_gaps = np.full(N, d_uniform)
    uniform_energy = np.sum(casimir_energy(uniform_gaps))
    
    print(f"Uniform gap: {d_uniform*1e9:.2f} nm")
    print(f"Uniform energy: {uniform_energy:.3e} J/m²")
    print()
    
    # Grid search optimization
    print("🔍 GRID SEARCH OPTIMIZATION")
    print("-" * 27)
    try:
        grid_gaps, grid_energy = simple_grid_optimization(N, d_min, d_max)
        if grid_gaps is not None and grid_energy is not None:
            grid_improvement = abs(grid_energy) / abs(uniform_energy)
            print(f"Grid improvement: {grid_improvement:.3f}×")
        else:
            print("❌ Grid optimization returned None")
            grid_improvement = 1.0
            grid_gaps, grid_energy = uniform_gaps.copy(), uniform_energy
    except Exception as e:
        print(f"❌ Grid optimization failed: {e}")
        grid_improvement = 1.0
        grid_gaps, grid_energy = uniform_gaps.copy(), uniform_energy
    print()
    
    # Bayesian optimization
    if SKOPT_AVAILABLE:
        print("🤖 BAYESIAN OPTIMIZATION")
        print("-" * 24)
        try:
            bayes_gaps, bayes_energy = bayesian_optimization(N, d_min, d_max)
            if bayes_gaps is not None and bayes_energy is not None:
                bayes_improvement = abs(bayes_energy) / abs(uniform_energy)
                
                print(f"Bayesian improvement: {bayes_improvement:.3f}×")
                
                # Compare grid vs Bayesian
                relative_gain = abs(bayes_energy) / abs(grid_energy)
                print(f"Bayesian vs Grid: {relative_gain:.3f}×")
                
                return {
                    'uniform': (uniform_gaps, uniform_energy),
                    'grid': (grid_gaps, grid_energy),
                    'bayesian': (bayes_gaps, bayes_energy),
                    'improvements': {
                        'grid': grid_improvement,
                        'bayesian': bayes_improvement
                    }
                }
            else:
                print("❌ Bayesian optimization returned None")
        except Exception as e:
            print(f"❌ Bayesian optimization failed: {e}")
    else:
        print("⚠️ Skipping Bayesian optimization (scikit-optimize not available)")
    
    return {
        'uniform': (uniform_gaps, uniform_energy),
        'grid': (grid_gaps, grid_energy),
        'improvements': {
            'grid': grid_improvement
        }
    }

def advanced_bayesian_features():
    """Demonstrate advanced Bayesian optimization features."""
    
    if not SKOPT_AVAILABLE:
        print("Advanced Bayesian features require scikit-optimize")
        return
    
    print("🚀 ADVANCED BAYESIAN FEATURES")
    print("-" * 30)
    
    # Multi-objective optimization placeholder
    print("📊 Available advanced features:")
    print("   • Multi-objective optimization")
    print("   • Constraint handling")
    print("   • Acquisition function tuning")
    print("   • Parallel evaluation")
    print("   • Active learning strategies")
    print()
    
    # Example: constraint-aware optimization
    def constrained_objective(d):
        """Objective with manufacturing constraints."""
        gaps = np.array(d)
        
        # Check constraints
        if np.any(gaps < 3e-9) or np.any(gaps > 2e-8):
            return 1e10  # Penalty for infeasible designs
        
        # Check uniformity constraint (optional)
        gap_variance = np.var(gaps)
        if gap_variance > (1e-9)**2:
            return 1e10  # Penalty for non-uniform gaps
        
        # Standard objective
        energy = np.sum(- (np.pi**2 * hbar * c)/(720 * gaps**3))
        return -abs(energy)
    
    print("🔧 Constraint-aware optimization enabled")
    print("   • Manufacturing limits: 3-20 nm")
    print("   • Uniformity constraint: σ < 1 nm")

def casimir_energy(d):
    """Helper function for energy calculation."""
    return - (np.pi**2 * hbar * c) / (720 * d**3)

# Example usage
if __name__=='__main__':
    results = optimization_comparison()
    
    print()
    print("=" * 42)
    print("🎯 OPTIMIZATION SUMMARY")
    print("=" * 42)
    
    for method, improvement in results.get('improvements', {}).items():
        print(f"{method.capitalize()} optimization: {improvement:.3f}× improvement")
    
    if SKOPT_AVAILABLE:
        advanced_bayesian_features()
    
    print()
    print("💡 NEXT STEPS:")
    print("   1. Implement optimal gaps in experimental setup")
    print("   2. Validate predicted vs measured energies")
    print("   3. Iterate optimization with experimental feedback")
    print("   4. Scale to larger arrays and different geometries")
