# File: src/analysis/jpa_bayes_opt.py
"""
Bayesian Optimization for JPA Squeezed Vacuum Parameters

Mathematical Foundation:
r(ε,Δ,Q) = ε√(Q/10⁶) / (1 + 4Δ²)
Squeezing(dB) = 20·log₁₀(e^r)

Gaussian Process surrogate optimization to maximize squeezing.
"""

import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.plots import plot_convergence, plot_objective
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("⚠️  scikit-optimize not available - using random search fallback")

try:
    from src.analysis.in_silico_stack_and_squeeze import simulate_jpa_squeezed_vacuum
    INSILICO_AVAILABLE = True
except ImportError:
    INSILICO_AVAILABLE = False
    print("⚠️  in_silico_stack_and_squeeze not available - using fallback")

def simulate_jpa_squeezed_vacuum_fallback(signal_freq: float, pump_power: float, temperature: float):
    """Fallback JPA simulation if main module unavailable."""
    # Physical constants
    hbar = 1.054571817e-34
    k_B = 1.380649e-23
    
    # JPA parameters
    josephson_energy = 25e6  # Hz
    charging_energy = 1e6   # Hz
    optimal_pump = 0.15
    
    # Thermal effects
    if temperature > 0:
        thermal_photons = 1 / (np.exp(hbar * signal_freq / (k_B * temperature)) - 1)
    else:
        thermal_photons = 0
    thermal_factor = 1 / (1 + 2 * thermal_photons)
    
    # Pump efficiency
    pump_detuning = abs(pump_power - optimal_pump)
    pump_efficiency = 1 / (1 + 10 * pump_detuning**2)
    
    # Squeezing calculation (fixed formula)
    r_max = 2.0
    r_effective = r_max * thermal_factor * pump_efficiency
    squeezing_dB = -20 * np.log10(np.exp(-r_effective)) if r_effective > 0 else 0
    
    # Energy density
    mode_volume = 1e-18
    rho_squeezed = -np.sinh(r_effective)**2 * hbar * signal_freq
    total_energy = rho_squeezed * mode_volume
    
    return {
        'squeezing_parameter': r_effective,
        'squeezing_db': squeezing_dB,
        'total_energy': total_energy,
        'thermal_factor': thermal_factor,
        'pump_efficiency': pump_efficiency
    }

# Use available JPA function
if INSILICO_AVAILABLE:
    jpa_func = simulate_jpa_squeezed_vacuum
else:
    jpa_func = simulate_jpa_squeezed_vacuum_fallback

if SKOPT_AVAILABLE:
    # Real Bayesian optimization with scikit-optimize
    print("✅ Using scikit-optimize for Bayesian optimization")
    
    def run_bayesian_optimization(n_calls=30, signal_freq=6e9, temperature=0.015):
        """Run Bayesian optimization for JPA parameters."""
        print(f"🧠 Running Bayesian optimization: {n_calls} evaluations")
        print(f"   Signal frequency: {signal_freq/1e9:.1f} GHz")
        print(f"   Temperature: {temperature*1000:.1f} mK")
        
        # Decision variables: pump_power (ε), detuning parameter
        space = [
            Real(0.01, 0.3, name='epsilon'),
            Real(-0.5, 0.5, name='delta_norm')  # Normalized detuning
        ]
        
        def objective(x):
            """Objective function for Bayesian optimization."""
            eps, delta_norm = x
            
            # Convert normalized detuning to actual parameter
            # For this model, we use delta_norm as a pump efficiency modifier
            pump_power_effective = eps * (1 - 0.5 * abs(delta_norm))
            pump_power_effective = np.clip(pump_power_effective, 0.01, 0.3)
            
            try:
                if INSILICO_AVAILABLE:
                    res = jpa_func(signal_freq=signal_freq, pump_power=pump_power_effective, 
                                 temperature=temperature)
                else:
                    res = jpa_func(signal_freq, pump_power_effective, temperature)
                
                squeezing_db = res['squeezing_db']
                
                # We want to maximize dB, so minimize negative dB
                return -squeezing_db
            except Exception as e:
                print(f"   ⚠️  Evaluation error: {e}")
                return 1000  # Penalty for failed evaluation
        
        # Run optimization
        print("   🔍 Starting Gaussian Process optimization...")
        res = gp_minimize(
            objective, 
            space,
            acq_func="EI",  # Expected Improvement
            n_calls=n_calls,
            n_initial_points=5,
            random_state=42,
            verbose=False
        )
        
        # Extract optimal parameters
        best_eps, best_delta = res.x
        best_db = -res.fun
        
        # Evaluate optimal configuration
        pump_effective = best_eps * (1 - 0.5 * abs(best_delta))
        pump_effective = np.clip(pump_effective, 0.01, 0.3)
        
        if INSILICO_AVAILABLE:
            final_result = jpa_func(signal_freq=signal_freq, pump_power=pump_effective, 
                                  temperature=temperature)
        else:
            final_result = jpa_func(signal_freq, pump_effective, temperature)
        
        print(f"✅ Bayesian optimization complete!")
        print(f"   • Optimal ε = {best_eps:.3f}")
        print(f"   • Optimal δ = {best_delta:.3f}")
        print(f"   • Effective pump = {pump_effective:.3f}")
        print(f"   • Maximum squeezing = {best_db:.2f} dB")
        print(f"   • Squeezing parameter r = {final_result['squeezing_parameter']:.3f}")
        print(f"   • Total energy = {final_result['total_energy']:.2e} J")
        print(f"   • Thermal factor = {final_result['thermal_factor']:.3f}")
        
        # Convergence analysis
        convergence_values = res.func_vals
        best_so_far = np.minimum.accumulate(convergence_values)
        improvement_rate = (best_so_far[0] - best_so_far[-1]) / len(best_so_far)
        
        print(f"   • Convergence rate = {improvement_rate:.4f} dB/iteration")
        print(f"   • Function evaluations = {len(res.func_vals)}")
        
        return {
            'optimization_result': res,
            'optimal_params': {
                'epsilon': best_eps,
                'delta_normalized': best_delta,
                'pump_effective': pump_effective
            },
            'optimal_performance': final_result,
            'convergence_analysis': {
                'values': convergence_values,
                'best_so_far': best_so_far,
                'improvement_rate': improvement_rate
            }
        }
    
    def parameter_sensitivity_analysis():
        """Analyze parameter sensitivity around optimal point."""
        print("\n🔬 PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 50)
        
        # Run optimization to find optimal point
        opt_result = run_bayesian_optimization(n_calls=25)
        best_eps = opt_result['optimal_params']['epsilon']
        best_delta = opt_result['optimal_params']['delta_normalized']
        
        # Sensitivity sweep
        eps_range = np.linspace(max(0.01, best_eps - 0.05), min(0.3, best_eps + 0.05), 11)
        delta_range = np.linspace(max(-0.5, best_delta - 0.2), min(0.5, best_delta + 0.2), 11)
        
        print(f"   Analyzing ε ∈ [{eps_range[0]:.3f}, {eps_range[-1]:.3f}]")
        print(f"   Analyzing δ ∈ [{delta_range[0]:.3f}, {delta_range[-1]:.3f}]")
        
        sensitivity_data = []
        
        # Epsilon sensitivity
        for eps in eps_range:
            pump_eff = eps * (1 - 0.5 * abs(best_delta))
            pump_eff = np.clip(pump_eff, 0.01, 0.3)
            res = jpa_func(6e9, pump_eff, 0.015)
            sensitivity_data.append({
                'type': 'epsilon',
                'value': eps,
                'squeezing_db': res['squeezing_db'],
                'energy': res['total_energy']
            })
        
        # Delta sensitivity  
        for delta in delta_range:
            pump_eff = best_eps * (1 - 0.5 * abs(delta))
            pump_eff = np.clip(pump_eff, 0.01, 0.3)
            res = jpa_func(6e9, pump_eff, 0.015)
            sensitivity_data.append({
                'type': 'delta',
                'value': delta,
                'squeezing_db': res['squeezing_db'],
                'energy': res['total_energy']
            })
        
        # Calculate sensitivity metrics
        eps_data = [d for d in sensitivity_data if d['type'] == 'epsilon']
        delta_data = [d for d in sensitivity_data if d['type'] == 'delta']
        
        eps_sensitivity = np.std([d['squeezing_db'] for d in eps_data])
        delta_sensitivity = np.std([d['squeezing_db'] for d in delta_data])
        
        print(f"   • ε sensitivity: {eps_sensitivity:.3f} dB std")
        print(f"   • δ sensitivity: {delta_sensitivity:.3f} dB std")
        print(f"   • More sensitive to: {'epsilon' if eps_sensitivity > delta_sensitivity else 'delta'}")
        
        return {
            'optimization_result': opt_result,
            'sensitivity_data': sensitivity_data,
            'sensitivity_metrics': {
                'epsilon_std': eps_sensitivity,
                'delta_std': delta_sensitivity
            }
        }

else:
    # Fallback random search
    print("⚠️  Using random search fallback")
    
    def run_random_search(n_calls=30):
        """Random search fallback."""
        print(f"🔍 Running random search: {n_calls} evaluations")
        
        best_eps, best_delta = None, None
        best_db = -float('inf')
        
        results = []
        
        for i in range(n_calls):
            eps = np.random.uniform(0.01, 0.3)
            delta = np.random.uniform(-0.5, 0.5)
            
            pump_eff = eps * (1 - 0.5 * abs(delta))
            pump_eff = np.clip(pump_eff, 0.01, 0.3)
            
            res = jpa_func(6e9, pump_eff, 0.015)
            db = res['squeezing_db']
            
            results.append({
                'epsilon': eps,
                'delta': delta,
                'pump_effective': pump_eff,
                'squeezing_db': db,
                'energy': res['total_energy']
            })
            
            if db > best_db:
                best_db = db
                best_eps = eps
                best_delta = delta
        
        print(f"✅ Random search complete!")
        print(f"   • Best ε = {best_eps:.3f}")
        print(f"   • Best δ = {best_delta:.3f}")
        print(f"   • Best squeezing = {best_db:.2f} dB")
        
        return {
            'optimal_params': {'epsilon': best_eps, 'delta_normalized': best_delta},
            'all_results': results,
            'best_squeezing': best_db
        }
    
    def parameter_sensitivity_analysis():
        """Simplified sensitivity analysis."""
        return run_random_search(50)

def main():
    """Main JPA optimization execution."""
    print("\n⚡ JPA SQUEEZED VACUUM BAYESIAN OPTIMIZATION")
    print("=" * 60)
    
    if SKOPT_AVAILABLE:
        # Full Bayesian optimization with sensitivity analysis
        result = parameter_sensitivity_analysis()
        
        # Display key results
        opt_params = result['optimization_result']['optimal_params']
        opt_perf = result['optimization_result']['optimal_performance']
        sens_metrics = result['sensitivity_metrics']
        
        print(f"\n🎯 OPTIMIZATION SUMMARY:")
        print(f"   • Optimal pump amplitude: ε = {opt_params['epsilon']:.3f}")
        print(f"   • Optimal detuning: δ = {opt_params['delta_normalized']:.3f}")
        print(f"   • Maximum squeezing: {opt_perf['squeezing_db']:.2f} dB")
        print(f"   • Squeezing parameter: r = {opt_perf['squeezing_parameter']:.3f}")
        print(f"   • Negative energy: {opt_perf['total_energy']:.2e} J")
        
        print(f"\n🔬 SENSITIVITY ANALYSIS:")
        print(f"   • Epsilon std: {sens_metrics['epsilon_std']:.3f} dB")
        print(f"   • Delta std: {sens_metrics['delta_std']:.3f} dB")
        
        return result
    else:
        # Simplified random search
        result = parameter_sensitivity_analysis()
        print(f"\n🎯 BEST RESULT: {result['best_squeezing']:.2f} dB squeezing")
        return result

if __name__ == "__main__":
    result = main()
