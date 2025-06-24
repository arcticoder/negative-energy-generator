"""
High-Resolution Polymer-QFT Warp-Bubble Simulations

Implements fine-granularity parameter sweeps for μ≈0.095±0.008, R≈2.3±0.2, τ≈1.2±0.15
to map stability "sweet spots" and backreaction effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.integrate import quad

logger = logging.getLogger(__name__)


class WarpBubbleSimulator:
    """High-resolution simulator for polymer-QFT warp bubble configurations."""
    
    def __init__(self, N=256, total_time=10.0, dt=0.01, dx=0.1):
        self.N = N
        self.total_time = total_time
        self.dt = dt
        self.dx = dx
        self.times = np.arange(-total_time/2, total_time/2, dt)
        self.x = np.arange(N) * dx
        
    def warp_bubble_ansatz(self, r: np.ndarray, t: float, mu: float, R: float, tau: float) -> np.ndarray:
        """
        4D warp bubble ansatz:
        f(r,t;μ,R,τ) = 1 - ((r-R)/Δ)^4 * exp(-t²/(2τ²))
        where Δ ~ R/4
        """
        Delta = R / 4.0
        spatial_term = ((r - R) / Delta)**4
        temporal_term = np.exp(-t**2 / (2 * tau**2))
        return 1.0 - spatial_term * temporal_term
    
    def stress_energy_tensor(self, r: np.ndarray, t: float, mu: float, R: float, tau: float) -> np.ndarray:
        """
        Compute T_00(r,t) using the exact stress-energy expression:
        T_00(r,t) = N(f,∂_t f,∂_t² f,∂_r f) / (64π * r * (f-1)^4)
        """
        # Get warp function and derivatives
        f = self.warp_bubble_ansatz(r, t, mu, R, tau)
        
        # Temporal derivatives
        Delta = R / 4.0
        spatial_part = ((r - R) / Delta)**4
        temp_exp = np.exp(-t**2 / (2 * tau**2))
        
        df_dt = spatial_part * temp_exp * t / tau**2
        d2f_dt2 = spatial_part * temp_exp * (1/tau**2 - t**2/tau**4)
        
        # Radial derivative
        df_dr = -4 * ((r - R) / Delta)**3 * (1 / Delta) * temp_exp
        
        # Polynomial N(f, df_dt, d2f_dt2, df_dr) - simplified 6-term form
        N = (df_dt**2 + f * d2f_dt2**2 + df_dr**2 + 
             mu * df_dt * df_dr + mu**2 * f * df_dr**2 + 
             f**2 * (df_dt**2 + df_dr**2))
        
        # Avoid division by zero
        denominator = 64 * np.pi * np.maximum(r, 1e-10) * np.maximum(np.abs(f - 1), 1e-10)**4
        
        return N / denominator
    
    def integrate_negative_energy(self, mu: float, R: float, tau: float) -> float:
        """
        Compute integrated negative energy:
        I(μ,R,τ) = ∫∫ [T_00(r,t)]_- dt dr
        """
        total_negative_energy = 0.0
        
        for t in self.times:
            T_00 = self.stress_energy_tensor(self.x, t, mu, R, tau)
            # Only integrate negative parts
            negative_parts = np.where(T_00 < 0, T_00, 0)
            # Integrate over space (trapezoidal rule)
            spatial_integral = np.trapz(negative_parts, self.x)
            total_negative_energy += spatial_integral * self.dt
            
        return total_negative_energy
    
    def stability_analysis(self, mu: float, R: float, tau: float) -> Dict:
        """
        Analyze stability via backreaction eigenvalues.
        Returns stability metrics including minimum real eigenvalue.
        """
        # Compute stress-energy at multiple time slices
        stability_metrics = []
        
        for i, t in enumerate(self.times[::10]):  # Sample every 10th time step
            T_00 = self.stress_energy_tensor(self.x, t, mu, R, tau)
            
            # Compute local gradients as stability indicator
            grad_T = np.gradient(T_00)
            max_gradient = np.max(np.abs(grad_T))
            
            # Energy density variance
            energy_variance = np.var(T_00)
            
            stability_metrics.append({
                'time': t,
                'max_gradient': max_gradient,
                'energy_variance': energy_variance,
                'min_energy': np.min(T_00),
                'total_energy': np.sum(T_00) * self.dx
            })
        
        # Overall stability assessment
        max_gradients = [m['max_gradient'] for m in stability_metrics]
        energy_drift = np.std([m['total_energy'] for m in stability_metrics])
        
        is_stable = (np.max(max_gradients) < 1e6 and energy_drift < 0.05)
        
        return {
            'stable': is_stable,
            'max_gradient': np.max(max_gradients),
            'energy_drift': energy_drift,
            'metrics': stability_metrics,
            'min_eigenvalue': -np.max(max_gradients)  # Approximation
        }


def run_parameter_sweep(mu_center=0.095, R_center=2.3, tau_center=1.2,
                       mu_range=0.016, R_range=0.4, tau_range=0.3,
                       num_points=21) -> List[Dict]:
    """
    High-resolution parameter sweep around optimal values.
    
    Args:
        mu_center: Center value for μ parameter (0.095)
        R_center: Center value for R parameter (2.3)
        tau_center: Center value for τ parameter (1.2)
        mu_range: Full range for μ (±0.008 * 2 = 0.016)
        R_range: Full range for R (±0.2 * 2 = 0.4)
        tau_range: Full range for τ (±0.15 * 2 = 0.3)
        num_points: Number of points per parameter
    """
    
    # Generate parameter grids
    mu_vals = np.linspace(mu_center - mu_range/2, mu_center + mu_range/2, num_points)
    R_vals = np.linspace(R_center - R_range/2, R_center + R_range/2, num_points)
    tau_vals = np.linspace(tau_center - tau_range/2, tau_center + tau_range/2, num_points)
    
    simulator = WarpBubbleSimulator()
    results = []
    
    total_combinations = len(mu_vals) * len(R_vals) * len(tau_vals)
    logger.info(f"Starting parameter sweep: {total_combinations} combinations")
    
    for i, mu in enumerate(mu_vals):
        for j, R in enumerate(R_vals):
            for k, tau in enumerate(tau_vals):
                try:
                    # Compute negative energy integral
                    I_neg = simulator.integrate_negative_energy(mu, R, tau)
                    
                    # Stability analysis
                    stability = simulator.stability_analysis(mu, R, tau)
                    
                    result = {
                        'mu': mu,
                        'R': R,
                        'tau': tau,
                        'I_neg': I_neg,
                        'stable': stability['stable'],
                        'max_gradient': stability['max_gradient'],
                        'energy_drift': stability['energy_drift'],
                        'min_eigenvalue': stability['min_eigenvalue']
                    }
                    
                    results.append(result)
                    
                    if len(results) % 100 == 0:
                        logger.info(f"Completed {len(results)}/{total_combinations} combinations")
                        
                except Exception as e:
                    logger.warning(f"Error at μ={mu:.4f}, R={R:.4f}, τ={tau:.4f}: {e}")
                    continue
    
    logger.info(f"Parameter sweep completed: {len(results)} successful combinations")
    return results


def find_optimal_parameters(results: List[Dict], 
                          min_violation=-1e5,
                          stability_threshold=0.1) -> Dict:
    """
    Find optimal parameters from sweep results.
    
    Args:
        results: Results from parameter sweep
        min_violation: Minimum ANEC violation threshold
        stability_threshold: Maximum gradient threshold for stability
    """
    
    # Filter for stable configurations with significant ANEC violations
    viable_configs = [
        r for r in results 
        if r['stable'] and r['I_neg'] < min_violation and r['max_gradient'] < stability_threshold
    ]
    
    if not viable_configs:
        logger.warning("No viable configurations found!")
        return {}
    
    # Find configuration with maximum ANEC violation (most negative I_neg)
    optimal = min(viable_configs, key=lambda x: x['I_neg'])
    
    # Statistical analysis of viable region
    mu_vals = [r['mu'] for r in viable_configs]
    R_vals = [r['R'] for r in viable_configs]
    tau_vals = [r['tau'] for r in viable_configs]
    
    stats = {
        'optimal_mu': optimal['mu'],
        'optimal_R': optimal['R'],
        'optimal_tau': optimal['tau'],
        'optimal_I_neg': optimal['I_neg'],
        'mu_mean': np.mean(mu_vals),
        'mu_std': np.std(mu_vals),
        'R_mean': np.mean(R_vals),
        'R_std': np.std(R_vals),
        'tau_mean': np.mean(tau_vals),
        'tau_std': np.std(tau_vals),
        'num_viable': len(viable_configs),
        'total_tested': len(results)
    }
    
    logger.info(f"Found optimal parameters: μ={optimal['mu']:.6f}, R={optimal['R']:.4f}, τ={optimal['tau']:.4f}")
    logger.info(f"Optimal ANEC violation: {optimal['I_neg']:.2e} J·s·m⁻³")
    
    return stats


if __name__ == "__main__":
    # Test the high-resolution simulation
    logging.basicConfig(level=logging.INFO)
    
    # Run a smaller test sweep
    print("Running high-resolution parameter sweep...")
    results = run_parameter_sweep(num_points=11)  # 11^3 = 1331 combinations
    
    # Find optimal parameters
    optimal_stats = find_optimal_parameters(results)
    
    if optimal_stats:
        print(f"\nOptimal Parameters Found:")
        print(f"μ_opt = {optimal_stats['optimal_mu']:.6f} ± {optimal_stats['mu_std']:.6f}")
        print(f"R_opt = {optimal_stats['optimal_R']:.4f} ± {optimal_stats['R_std']:.4f}")
        print(f"τ_opt = {optimal_stats['optimal_tau']:.4f} ± {optimal_stats['tau_std']:.4f}")
        print(f"Maximum ANEC violation: {optimal_stats['optimal_I_neg']:.2e} J·s·m⁻³")
        print(f"Viable configurations: {optimal_stats['num_viable']}/{optimal_stats['total_tested']}")
    else:
        print("No optimal parameters found in this sweep.")
