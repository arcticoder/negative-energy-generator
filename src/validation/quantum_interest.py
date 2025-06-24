"""
Quantum-Interest Trade-Off Optimization

Quantifies positive-energy "repayment" pulses and identifies pulse-shaping
protocols that minimize net overhead in negative energy generation.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class QuantumInterestPulse:
    """Represents a quantum interest pulse configuration."""
    A_minus: float      # Negative pulse amplitude
    A_plus: float       # Positive pulse amplitude  
    sigma: float        # Pulse width
    delay: float        # Time delay between pulses
    net_energy: float   # Net energy after repayment
    efficiency: float   # Energy efficiency ratio


def quantum_interest_bound(A_minus: float, sigma: float, kappa: float = np.pi**2/12) -> float:
    """
    Compute quantum interest bound for negative energy pulse.
    
    For a negative pulse ρ₋(t) = -A₋ exp[-t²/(2σ²)], the minimum positive
    repayment energy is: E₊ ≥ (π²/12σ²) |∫ ρ₋(t) dt|
    """
    negative_integral = A_minus * sigma * np.sqrt(2 * np.pi)
    return (kappa / sigma**2) * negative_integral


def negative_pulse(t: np.ndarray, A_minus: float, sigma: float, t_center: float = 0.0) -> np.ndarray:
    """Generate negative energy pulse: ρ₋(t) = -A₋ exp[-(t-t₀)²/(2σ²)]"""
    return -A_minus * np.exp(-(t - t_center)**2 / (2 * sigma**2))


def positive_pulse(t: np.ndarray, A_plus: float, sigma: float, delay: float) -> np.ndarray:
    """Generate positive repayment pulse: ρ₊(t) = A₊ exp[-(t-Δ)²/(2σ²)]"""
    return A_plus * np.exp(-(t - delay)**2 / (2 * sigma**2))


def optimize_quantum_interest_simple(A_minus: float, sigma: float, 
                                   kappa: float = np.pi**2/12) -> QuantumInterestPulse:
    """
    Find minimal positive pulse amplitude and optimal delay for quantum interest repayment.
    """
    # Minimum required positive amplitude from QI bound
    A_plus_min = quantum_interest_bound(A_minus, sigma, kappa)
    
    def cost_function(x):
        A_plus, delay = x
        
        # Penalty for violating QI bound
        qi_penalty = max(0, A_plus_min - A_plus) * 1e6
        
        # Minimize total positive energy (efficiency)
        positive_integral = A_plus * sigma * np.sqrt(2 * np.pi)
        
        # Penalty for very long delays (causality concerns)
        delay_penalty = 0.1 * delay**2 if delay > 5*sigma else 0
        
        return positive_integral + qi_penalty + delay_penalty
    
    # Initial guess
    x0 = [A_plus_min * 1.1, 2*sigma]
    bounds = [(A_plus_min, A_plus_min * 10), (0, 10*sigma)]
    
    result = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        A_plus_opt, delay_opt = result.x
        negative_integral = A_minus * sigma * np.sqrt(2 * np.pi)
        positive_integral = A_plus_opt * sigma * np.sqrt(2 * np.pi)
        net_energy = positive_integral - negative_integral
        efficiency = negative_integral / positive_integral
        
        return QuantumInterestPulse(
            A_minus=A_minus,
            A_plus=A_plus_opt,
            sigma=sigma,
            delay=delay_opt,
            net_energy=net_energy,
            efficiency=efficiency
        )
    else:
        logger.warning(f"Optimization failed for A_minus={A_minus}, sigma={sigma}")
        return None


def optimize_pulse_shape_advanced(A_minus: float, sigma: float, 
                                 n_pulses: int = 3) -> Dict:
    """
    Advanced optimization using multiple positive pulses for better efficiency.
    """
    def multi_pulse_positive(t, amplitudes, delays, widths):
        """Sum of multiple positive pulses"""
        total = np.zeros_like(t)
        for A, delay, width in zip(amplitudes, delays, widths):
            total += A * np.exp(-(t - delay)**2 / (2 * width**2))
        return total
    
    def cost_function(params):
        # params = [A1, A2, ..., An, delay1, delay2, ..., delayn, width1, width2, ..., widthn]
        n = n_pulses
        amplitudes = params[:n]
        delays = params[n:2*n]  
        widths = params[2*n:3*n]
        
        # Quantum interest constraint
        negative_integral = A_minus * sigma * np.sqrt(2 * np.pi)
        qi_bound = quantum_interest_bound(A_minus, sigma)
        
        # Total positive energy
        positive_integral = sum(A * w * np.sqrt(2 * np.pi) for A, w in zip(amplitudes, widths))
        
        # Penalty for violating QI bound
        qi_penalty = max(0, qi_bound - positive_integral) * 1e6
        
        # Efficiency term (minimize positive energy)
        efficiency_cost = positive_integral
        
        # Smoothness penalty (prefer similar widths)
        smoothness_penalty = 0.1 * np.sum((np.array(widths) - sigma)**2)
        
        return efficiency_cost + qi_penalty + smoothness_penalty
    
    # Parameter bounds
    bounds = []
    for i in range(n_pulses):
        bounds.append((0, quantum_interest_bound(A_minus, sigma) * 2))  # Amplitudes
    for i in range(n_pulses):
        bounds.append((0, 10*sigma))  # Delays
    for i in range(n_pulses):
        bounds.append((sigma/2, sigma*2))  # Widths
    
    # Use differential evolution for global optimization
    result = differential_evolution(cost_function, bounds, seed=42, maxiter=100)
    
    if result.success:
        n = n_pulses
        amplitudes = result.x[:n]
        delays = result.x[n:2*n]
        widths = result.x[2*n:3*n]
        
        negative_integral = A_minus * sigma * np.sqrt(2 * np.pi)
        positive_integral = sum(A * w * np.sqrt(2 * np.pi) for A, w in zip(amplitudes, widths))
        
        return {
            'success': True,
            'amplitudes': amplitudes,
            'delays': delays,
            'widths': widths,
            'net_energy': positive_integral - negative_integral,
            'efficiency': negative_integral / positive_integral,
            'qi_satisfied': positive_integral >= quantum_interest_bound(A_minus, sigma)
        }
    else:
        return {'success': False, 'message': 'Advanced optimization failed'}


def quantum_interest_parameter_sweep(A_minus_range: Tuple[float, float],
                                   sigma_range: Tuple[float, float],
                                   n_points: int = 20) -> List[QuantumInterestPulse]:
    """
    Sweep over amplitude and width parameters to map efficiency landscape.
    """
    A_vals = np.linspace(A_minus_range[0], A_minus_range[1], n_points)
    sigma_vals = np.linspace(sigma_range[0], sigma_range[1], n_points)
    
    results = []
    total_points = len(A_vals) * len(sigma_vals)
    count = 0
    
    for A_minus in A_vals:
        for sigma in sigma_vals:
            count += 1
            if count % 50 == 0:
                logger.info(f"QI optimization progress: {count}/{total_points}")
            
            pulse = optimize_quantum_interest_simple(A_minus, sigma)
            if pulse:
                results.append(pulse)
    
    return results


def analyze_warp_bubble_quantum_interest(mu: float, R: float, tau: float,
                                       characteristic_energy: float = 1e5) -> Dict:
    """
    Analyze quantum interest requirements for specific warp bubble parameters.
    """
    # Estimate negative energy characteristics from warp bubble
    # Use tau as the characteristic time scale
    sigma_characteristic = tau
    A_minus_characteristic = characteristic_energy / (sigma_characteristic * np.sqrt(2 * np.pi))
    
    # Optimize quantum interest
    pulse = optimize_quantum_interest_simple(A_minus_characteristic, sigma_characteristic)
    
    if pulse:
        # Try advanced optimization too
        advanced = optimize_pulse_shape_advanced(A_minus_characteristic, sigma_characteristic)
        
        return {
            'warp_params': {'mu': mu, 'R': R, 'tau': tau},
            'simple_optimization': pulse,
            'advanced_optimization': advanced,
            'qi_bound': quantum_interest_bound(A_minus_characteristic, sigma_characteristic),
            'sigma_used': sigma_characteristic,
            'A_minus_used': A_minus_characteristic
        }
    else:
        return {'success': False, 'message': 'QI optimization failed'}


def plot_quantum_interest_analysis(results: List[QuantumInterestPulse], 
                                 save_path: Optional[str] = None):
    """
    Visualize quantum interest optimization results.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    A_minus_vals = [r.A_minus for r in results]
    sigma_vals = [r.sigma for r in results] 
    efficiency_vals = [r.efficiency for r in results]
    net_energy_vals = [r.net_energy for r in results]
    delay_vals = [r.delay for r in results]
    
    # Plot 1: Efficiency vs amplitude
    ax1.scatter(A_minus_vals, efficiency_vals, c=sigma_vals, alpha=0.7)
    ax1.set_xlabel('Negative Amplitude A₋')
    ax1.set_ylabel('Efficiency (E₋/E₊)')
    ax1.set_title('Quantum Interest Efficiency')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Net energy vs pulse width  
    ax2.scatter(sigma_vals, net_energy_vals, c=A_minus_vals, alpha=0.7)
    ax2.set_xlabel('Pulse Width σ')
    ax2.set_ylabel('Net Energy Cost')
    ax2.set_title('Energy Overhead')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Optimal delay vs parameters
    ax3.scatter(A_minus_vals, delay_vals, c=efficiency_vals, alpha=0.7)
    ax3.set_xlabel('Negative Amplitude A₋')
    ax3.set_ylabel('Optimal Delay')
    ax3.set_title('Pulse Timing')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency distribution
    ax4.hist(efficiency_vals, bins=20, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Efficiency')
    ax4.set_ylabel('Count')
    ax4.set_title('Efficiency Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"QI analysis plot saved to {save_path}")
    
    plt.show()


def demonstrate_pulse_shaping(A_minus: float = 100.0, sigma: float = 1.0):
    """
    Demonstrate optimal pulse shaping for quantum interest repayment.
    """
    # Simple optimization
    pulse_simple = optimize_quantum_interest_simple(A_minus, sigma)
    
    # Advanced optimization
    advanced_result = optimize_pulse_shape_advanced(A_minus, sigma, n_pulses=3)
    
    # Plot pulse shapes
    t = np.linspace(-3*sigma, 10*sigma, 1000)
    
    plt.figure(figsize=(12, 8))
    
    # Negative pulse
    rho_minus = negative_pulse(t, A_minus, sigma)
    plt.plot(t, rho_minus, 'r-', linewidth=2, label='Negative Energy Pulse')
    
    # Simple positive pulse
    if pulse_simple:
        rho_plus_simple = positive_pulse(t, pulse_simple.A_plus, sigma, pulse_simple.delay)
        plt.plot(t, rho_plus_simple, 'b--', linewidth=2, 
                label=f'Simple Repayment (η={pulse_simple.efficiency:.3f})')
    
    # Advanced positive pulses
    if advanced_result['success']:
        rho_plus_advanced = np.zeros_like(t)
        for A, delay, width in zip(advanced_result['amplitudes'], 
                                  advanced_result['delays'], 
                                  advanced_result['widths']):
            rho_plus_advanced += A * np.exp(-(t - delay)**2 / (2 * width**2))
        
        plt.plot(t, rho_plus_advanced, 'g-', linewidth=2,
                label=f'Advanced Repayment (η={advanced_result["efficiency"]:.3f})')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Energy Density')
    plt.title('Quantum Interest Pulse Shaping')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pulse_simple, advanced_result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Quantum Interest Trade-off Analysis")
    print("="*50)
    
    # Demonstrate pulse shaping
    pulse_simple, advanced = demonstrate_pulse_shaping(A_minus=100.0, sigma=1.0)
    
    if pulse_simple:
        print(f"Simple optimization:")
        print(f"  A₊ = {pulse_simple.A_plus:.2f}")
        print(f"  Delay = {pulse_simple.delay:.2f}")
        print(f"  Efficiency = {pulse_simple.efficiency:.3f}")
        print(f"  Net energy = {pulse_simple.net_energy:.2f}")
    
    if advanced['success']:
        print(f"\\nAdvanced optimization:")
        print(f"  Efficiency = {advanced['efficiency']:.3f}")
        print(f"  QI satisfied = {advanced['qi_satisfied']}")
    
    # Analyze for optimal warp bubble parameters
    print("\\nAnalyzing optimal warp bubble parameters...")
    warp_analysis = analyze_warp_bubble_quantum_interest(mu=0.095, R=2.3, tau=1.2)
    
    if 'simple_optimization' in warp_analysis:
        opt = warp_analysis['simple_optimization']
        print(f"Warp bubble QI analysis:")
        print(f"  Efficiency = {opt.efficiency:.3f}")
        print(f"  Net energy cost = {opt.net_energy:.2e}")
        print(f"  Optimal delay = {opt.delay:.3f}")
