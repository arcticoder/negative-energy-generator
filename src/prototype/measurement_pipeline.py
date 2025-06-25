"""
Measurement & Data Analysis Pipeline
===================================

This module provides a comprehensive pipeline for ingesting real-time sensor data
from prototype exotic matter experiments and fitting the data to theoretical models.
It handles force measurements, frequency shifts, and other observables.

Mathematical Foundation:
    Casimir Force: F(d) = -π²ℏc A_eff/(240 d⁴)
    Time-series fitting: signal(t) = model(t, *params) + noise
    Chi-squared fitting: χ² = Σ[(data - model)²/σ²]
"""

import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize
from scipy.signal import savgol_filter, periodogram
from scipy.stats import chi2
from typing import Callable, Dict, List, Tuple, Optional, Any
import warnings
import json
from datetime import datetime


# Physical constants
HBAR = 1.054571817e-34  # J⋅s
C = 2.99792458e8        # m/s
PI = np.pi


def casimir_force_model(d: np.ndarray, A_eff: float) -> np.ndarray:
    """
    Theoretical Casimir force model.
    
    F(d) = -π²ℏc A_eff/(240 d⁴)
    
    Args:
        d: Array of gap distances [m]
        A_eff: Effective area parameter [m²]
        
    Returns:
        Array of Casimir forces [N]
    """
    return -PI**2 * HBAR * C * A_eff / (240 * d**4)


def enhanced_casimir_model(d: np.ndarray, A_eff: float, enhancement: float, 
                          d_cutoff: float = 1e-9) -> np.ndarray:
    """
    Enhanced Casimir force model with modifications.
    
    Args:
        d: Gap distances [m]
        A_eff: Effective area [m²]
        enhancement: Enhancement factor
        d_cutoff: Cutoff distance for regularization [m]
        
    Returns:
        Enhanced Casimir forces [N]
    """
    # Add cutoff to prevent divergence
    d_reg = np.maximum(d, d_cutoff)
    base_force = casimir_force_model(d_reg, A_eff)
    return enhancement * base_force


def fit_casimir_force(gaps: np.ndarray, forces: np.ndarray, 
                     uncertainties: Optional[np.ndarray] = None,
                     enhanced_model: bool = False) -> Dict:
    """
    Fit Casimir force data to theoretical model.
    
    Args:
        gaps: Array of gap distances [m]
        forces: Array of measured forces [N]
        uncertainties: Optional force uncertainties [N]
        enhanced_model: Whether to use enhanced model
        
    Returns:
        Dictionary with fit results and statistics
    """
    try:
        if enhanced_model:
            model_func = lambda d, A, enh: enhanced_casimir_model(d, A, enh)
            p0 = [1e-8, 1.0]  # Initial guess: A_eff, enhancement
            param_names = ['A_eff', 'enhancement_factor']
        else:
            model_func = lambda d, A: casimir_force_model(d, A)
            p0 = [1e-8]  # Initial guess: A_eff
            param_names = ['A_eff']
        
        # Perform fit
        if uncertainties is not None:
            popt, pcov = curve_fit(model_func, gaps, forces, p0=p0, sigma=uncertainties, absolute_sigma=True)
        else:
            popt, pcov = curve_fit(model_func, gaps, forces, p0=p0)
        
        # Calculate uncertainties
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate fit statistics
        fitted_forces = model_func(gaps, *popt)
        residuals = forces - fitted_forces
        
        # Chi-squared
        if uncertainties is not None:
            chi2_val = np.sum((residuals / uncertainties)**2)
        else:
            chi2_val = np.sum(residuals**2) / np.var(forces)
        
        dof = len(forces) - len(popt)
        reduced_chi2 = chi2_val / dof if dof > 0 else np.inf
        
        # P-value
        p_value = 1 - chi2.cdf(chi2_val, dof) if dof > 0 else 0
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((forces - np.mean(forces))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'success': True,
            'parameters': dict(zip(param_names, popt)),
            'uncertainties': dict(zip(param_names, perr)),
            'covariance_matrix': pcov.tolist(),
            'fitted_forces': fitted_forces.tolist(),
            'residuals': residuals.tolist(),
            'chi2': chi2_val,
            'reduced_chi2': reduced_chi2,
            'degrees_of_freedom': dof,
            'p_value': p_value,
            'r_squared': r_squared,
            'model_type': 'enhanced' if enhanced_model else 'standard'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'model_type': 'enhanced' if enhanced_model else 'standard'
        }


def analyze_time_series(time: np.ndarray, signal: np.ndarray, 
                       model_fn: Callable, p0: List[float],
                       signal_uncertainties: Optional[np.ndarray] = None) -> Dict:
    """
    Generic time-series fitting and analysis.
    
    Args:
        time: Time array
        signal: Signal array
        model_fn: Model function model_fn(t, *params)
        p0: Initial parameter guess
        signal_uncertainties: Optional signal uncertainties
        
    Returns:
        Dictionary with fit results and time-series analysis
    """
    try:
        # Fit the model
        if signal_uncertainties is not None:
            popt, pcov = curve_fit(model_fn, time, signal, p0=p0, sigma=signal_uncertainties)
        else:
            popt, pcov = curve_fit(model_fn, time, signal, p0=p0)
        
        perr = np.sqrt(np.diag(pcov))
        fitted_signal = model_fn(time, *popt)
        residuals = signal - fitted_signal
        
        # Time-series specific analysis
        
        # Trend analysis
        trend_slope = np.polyfit(time, signal, 1)[0]
        
        # Autocorrelation of residuals
        autocorr = np.correlate(residuals, residuals, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Power spectral density
        if len(signal) > 4:
            freqs, psd = periodogram(signal, fs=1/np.mean(np.diff(time)))
            dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
            dominant_frequency = freqs[dominant_freq_idx]
        else:
            freqs, psd, dominant_frequency = None, None, None
        
        # Stationarity test (simple)
        n_segments = min(4, len(signal) // 10)
        if n_segments >= 2:
            segment_size = len(signal) // n_segments
            segment_means = [np.mean(signal[i*segment_size:(i+1)*segment_size]) 
                           for i in range(n_segments)]
            stationarity_p = np.var(segment_means) / np.var(signal) if np.var(signal) > 0 else 0
        else:
            stationarity_p = np.nan
        
        return {
            'success': True,
            'fitted_parameters': popt.tolist(),
            'parameter_uncertainties': perr.tolist(),
            'covariance_matrix': pcov.tolist(),
            'fitted_signal': fitted_signal.tolist(),
            'residuals': residuals.tolist(),
            'rms_residual': np.sqrt(np.mean(residuals**2)),
            'trend_slope': trend_slope,
            'autocorrelation': autocorr[:min(20, len(autocorr))].tolist(),
            'dominant_frequency': dominant_frequency,
            'stationarity_parameter': stationarity_p,
            'power_spectral_density': {
                'frequencies': freqs.tolist() if freqs is not None else None,
                'psd': psd.tolist() if psd is not None else None
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def frequency_shift_analysis(frequencies: np.ndarray, gaps: np.ndarray,
                           baseline_freq: float) -> Dict:
    """
    Analyze frequency shifts in resonant systems.
    
    Useful for cavity QED experiments or mechanical resonators
    affected by Casimir forces.
    
    Args:
        frequencies: Measured resonant frequencies [Hz]
        gaps: Corresponding gap distances [m]
        baseline_freq: Reference frequency [Hz]
        
    Returns:
        Analysis of frequency shifts vs gap
    """
    # Relative frequency shifts
    delta_f = (frequencies - baseline_freq) / baseline_freq
    
    # Expected scaling: Δf ∝ 1/d⁴ for Casimir effect
    def freq_shift_model(d, amplitude):
        return amplitude / d**4
    
    try:
        # Fit frequency shift data
        popt, pcov = curve_fit(freq_shift_model, gaps, delta_f)
        fitted_shifts = freq_shift_model(gaps, *popt)
        
        return {
            'success': True,
            'relative_shifts': delta_f.tolist(),
            'amplitude_parameter': popt[0],
            'amplitude_uncertainty': np.sqrt(pcov[0, 0]),
            'fitted_shifts': fitted_shifts.tolist(),
            'correlation_coefficient': np.corrcoef(delta_f, fitted_shifts)[0, 1],
            'scaling_verification': {
                'expected_scaling': 'd^(-4)',
                'fit_quality': np.sqrt(np.mean((delta_f - fitted_shifts)**2)) / np.std(delta_f)
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'relative_shifts': delta_f.tolist()
        }


def real_time_data_processor():
    """
    Real-time data processing class for live experiments.
    """
    class DataProcessor:
        def __init__(self, buffer_size: int = 1000):
            self.buffer_size = buffer_size
            self.time_buffer = []
            self.signal_buffer = []
            self.processed_results = []
            
        def add_data_point(self, timestamp: float, signal_value: float):
            """Add new data point to buffer."""
            self.time_buffer.append(timestamp)
            self.signal_buffer.append(signal_value)
            
            # Maintain buffer size
            if len(self.time_buffer) > self.buffer_size:
                self.time_buffer.pop(0)
                self.signal_buffer.pop(0)
        
        def process_current_buffer(self, model_fn: Callable, p0: List[float]) -> Dict:
            """Process current buffer contents."""
            if len(self.time_buffer) < 10:  # Need minimum data
                return {'success': False, 'error': 'Insufficient data'}
            
            time_array = np.array(self.time_buffer)
            signal_array = np.array(self.signal_buffer)
            
            # Smooth data
            if len(signal_array) > 5:
                signal_smoothed = savgol_filter(signal_array, 5, 2)
            else:
                signal_smoothed = signal_array
            
            # Analyze
            result = analyze_time_series(time_array, signal_smoothed, model_fn, p0)
            result['timestamp'] = datetime.now().isoformat()
            result['buffer_size'] = len(self.time_buffer)
            
            self.processed_results.append(result)
            return result
        
        def get_recent_trends(self, n_recent: int = 10) -> Dict:
            """Analyze trends in recent processing results."""
            if len(self.processed_results) < 2:
                return {'success': False, 'error': 'Insufficient processed data'}
            
            recent = self.processed_results[-n_recent:]
            successful = [r for r in recent if r.get('success', False)]
            
            if len(successful) < 2:
                return {'success': False, 'error': 'Too few successful fits'}
            
            # Track parameter evolution
            param_evolution = {}
            for i, result in enumerate(successful):
                for j, param in enumerate(result.get('fitted_parameters', [])):
                    if j not in param_evolution:
                        param_evolution[j] = []
                    param_evolution[j].append(param)
            
            # Calculate trends
            trends = {}
            for param_idx, values in param_evolution.items():
                if len(values) > 1:
                    trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                    trends[f'parameter_{param_idx}_trend'] = trend_slope
            
            return {
                'success': True,
                'n_recent_successful': len(successful),
                'parameter_trends': trends,
                'recent_rms_residuals': [r.get('rms_residual', np.nan) for r in successful],
                'fit_quality_trend': np.mean([r.get('rms_residual', np.nan) for r in successful[-3:]]) if len(successful) >= 3 else np.nan
            }
    
    return DataProcessor


def experimental_design_optimizer(target_precision: float, available_area_cm2: float,
                                 measurement_time_s: float = 3600) -> Dict:
    """
    Optimize experimental design for target measurement precision.
    
    Args:
        target_precision: Target relative precision (e.g., 0.01 for 1%)
        available_area_cm2: Available experimental area
        measurement_time_s: Available measurement time
        
    Returns:
        Optimal experimental parameters
    """
    # Estimate optimal parameters
    area_m2 = available_area_cm2 * 1e-4
    
    # For Casimir force measurements, precision scales with area and measurement time
    # Basic noise model: σ_F ∝ 1/√(A × t)
    noise_scaling = 1 / np.sqrt(area_m2 * measurement_time_s)
    
    # Find optimal gap for best signal-to-noise
    gaps_test = np.logspace(-8, -6, 50)  # 10 nm to 1 μm
    forces_test = np.abs(casimir_force_model(gaps_test, area_m2))
    snr = forces_test / (noise_scaling * np.sqrt(forces_test))  # Shot noise limit
    
    optimal_gap_idx = np.argmax(snr)
    optimal_gap = gaps_test[optimal_gap_idx]
    optimal_force = forces_test[optimal_gap_idx]
    
    # Estimate achievable precision
    estimated_precision = noise_scaling / optimal_force
    
    return {
        'optimal_gap_m': optimal_gap,
        'optimal_gap_nm': optimal_gap * 1e9,
        'expected_force_N': optimal_force,
        'estimated_precision': estimated_precision,
        'meets_target': estimated_precision <= target_precision,
        'recommended_area_cm2': available_area_cm2,
        'recommended_measurement_time_s': measurement_time_s,
        'design_notes': [
            f"Optimal gap: {optimal_gap*1e9:.1f} nm",
            f"Expected force: {optimal_force:.2e} N",
            f"Estimated precision: {estimated_precision:.1%}",
            "Use force feedback for gap stabilization",
            "Consider temperature stabilization ±0.1 K"
        ]
    }


# Example usage and testing
if __name__ == "__main__":
    print("=== Casimir Force Fitting Example ===")
    
    # Generate synthetic data
    gaps_nm = np.array([10, 15, 20, 30, 50, 80, 120, 200])
    gaps_m = gaps_nm * 1e-9
    true_area = 1e-8  # 0.01 mm²
    
    # True forces with noise
    true_forces = casimir_force_model(gaps_m, true_area)
    noise_level = 0.05  # 5% noise
    noisy_forces = true_forces * (1 + noise_level * np.random.randn(len(true_forces)))
    force_uncertainties = abs(true_forces) * noise_level
    
    # Fit the data
    fit_result = fit_casimir_force(gaps_m, noisy_forces, force_uncertainties)
    
    if fit_result['success']:
        print(f"Fitted area: {fit_result['parameters']['A_eff']:.2e} m²")
        print(f"True area: {true_area:.2e} m²")
        print(f"Relative error: {abs(fit_result['parameters']['A_eff'] - true_area)/true_area:.1%}")
        print(f"Reduced χ²: {fit_result['reduced_chi2']:.2f}")
        print(f"R²: {fit_result['r_squared']:.3f}")
    
    print("\n=== Time Series Analysis Example ===")
    
    # Generate time series data
    t = np.linspace(0, 10, 100)
    def test_model(t, A, f, phi):
        return A * np.sin(2 * np.pi * f * t + phi)
    
    true_params = [2.0, 0.5, 0.3]  # amplitude, frequency, phase
    signal = test_model(t, *true_params) + 0.1 * np.random.randn(len(t))
    
    ts_result = analyze_time_series(t, signal, test_model, [1.0, 1.0, 0.0])
    
    if ts_result['success']:
        print(f"Fitted parameters: {ts_result['fitted_parameters']}")
        print(f"True parameters: {true_params}")
        print(f"RMS residual: {ts_result['rms_residual']:.3f}")
        if ts_result['dominant_frequency']:
            print(f"Dominant frequency: {ts_result['dominant_frequency']:.3f} Hz")
    
    print("\n=== Experimental Design Optimization ===")
    
    design_result = experimental_design_optimizer(
        target_precision=0.01,  # 1% precision
        available_area_cm2=0.01,  # 0.01 cm²
        measurement_time_s=3600   # 1 hour
    )
    
    print(f"Optimal gap: {design_result['optimal_gap_nm']:.1f} nm")
    print(f"Expected precision: {design_result['estimated_precision']:.1%}")
    print(f"Meets target: {design_result['meets_target']}")
    
    print("\n=== Real-time Data Processor Demo ===")
    
    # Demonstrate real-time processing
    processor = real_time_data_processor()()
    
    # Simulate adding data points
    for i in range(50):
        t_point = i * 0.1
        signal_point = 2.0 * np.sin(2 * np.pi * 0.5 * t_point) + 0.1 * np.random.randn()
        processor.add_data_point(t_point, signal_point)
    
    # Process current buffer
    rt_result = processor.process_current_buffer(test_model, [1.0, 1.0, 0.0])
    
    if rt_result['success']:
        print(f"Real-time fitted parameters: {rt_result['fitted_parameters']}")
        print(f"Buffer size: {rt_result['buffer_size']}")
        print(f"Processing timestamp: {rt_result['timestamp']}")
    
    # Add more data and check trends
    for i in range(50, 100):
        t_point = i * 0.1
        signal_point = 2.0 * np.sin(2 * np.pi * 0.5 * t_point) + 0.1 * np.random.randn()
        processor.add_data_point(t_point, signal_point)
    
    processor.process_current_buffer(test_model, [1.0, 1.0, 0.0])
    trends = processor.get_recent_trends()
    
    if trends['success']:
        print(f"Parameter trends: {trends['parameter_trends']}")
        print(f"Fit quality trend: {trends['fit_quality_trend']:.4f}")
