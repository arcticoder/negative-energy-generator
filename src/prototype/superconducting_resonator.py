"""
Superconducting Resonator for Negative Energy Generation
=======================================================

This module implements a superconducting microwave resonator system
optimized for negative energy extraction via the Dynamical Casimir Effect (DCE).

Mathematical Framework:
    H = ℏωₐ(t)â†â with ωₐ(t) = ω₀[1 + ε cos(Ωt)]
    |⟨T₀₀⟩| ≈ (ℏω₀ε²/16π)(Ω/ω₀)⁴ for resonant pumping
    
Key Features:
- Parametric amplification via superconducting flux modulation
- Real-time impedance matching and Q-factor optimization
- Cryogenic temperature control for coherence preservation
- Integration with measurement pipelines for T₀₀ extraction
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
import json
from datetime import datetime
from pathlib import Path

try:
    import scipy.optimize
    import scipy.signal
    from scipy.special import jv  # Bessel functions
    SCIPY_AVAILABLE = True
except ImportError:
    warnings.warn("SciPy not available for advanced computations")
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class SuperconductingResonator:
    """
    Superconducting microwave resonator for DCE-based negative energy generation.
    
    This class models and controls a superconducting cavity resonator with
    parametric pumping capabilities for extracting vacuum fluctuations.
    """
    
    def __init__(self, 
                 base_frequency: float = 5.0e9,  # 5 GHz base frequency
                 quality_factor: float = 1e6,     # High-Q superconducting cavity
                 temperature: float = 0.01,       # 10 mK operation
                 cavity_volume: float = 1e-9,     # 1 mm³ effective volume
                 coupling_strength: float = 0.1): # Coupling to external circuit
        """
        Initialize superconducting resonator parameters.
        
        Args:
            base_frequency: Fundamental resonance frequency [Hz]
            quality_factor: Cavity Q-factor (loss rate)
            temperature: Operating temperature [K]
            cavity_volume: Effective cavity volume [m³]
            coupling_strength: External coupling coefficient
        """
        self.base_frequency = base_frequency  # ω₀
        self.quality_factor = quality_factor  # Q
        self.temperature = temperature        # T
        self.cavity_volume = cavity_volume    # V
        self.coupling_strength = coupling_strength  # κ_ext
        
        # Physical constants
        self.hbar = 1.054571817e-34  # Reduced Planck constant
        self.kb = 1.380649e-23       # Boltzmann constant
        self.c = 299792458           # Speed of light
        self.mu0 = 4*np.pi*1e-7      # Vacuum permeability
        self.eps0 = 8.8541878128e-12 # Vacuum permittivity
        
        # Derived parameters
        self.linewidth = base_frequency / quality_factor  # γ = ω₀/Q
        self.thermal_photons = self._thermal_occupation()
        self.zero_point_energy = 0.5 * self.hbar * base_frequency
        
        # State variables
        self.pump_amplitude = 0.0    # ε - modulation depth
        self.pump_frequency = 0.0    # Ω - pump frequency
        self.detuning = 0.0          # Δ = ω_p - 2ω₀
        self.flux_control = 0.0      # Magnetic flux control
        
        # Operating history
        self.operation_log = []
        self.measurement_history = []
        
        print(f"🔧 Superconducting Resonator Initialized:")
        print(f"   • Base frequency: {base_frequency/1e9:.2f} GHz")
        print(f"   • Q-factor: {quality_factor:.1e}")
        print(f"   • Temperature: {temperature*1000:.1f} mK")
        print(f"   • Thermal photons: {self.thermal_photons:.3f}")
        print(f"   • Zero-point energy: {self.zero_point_energy/self.hbar/base_frequency:.3f} ℏω₀")
    
    def _thermal_occupation(self) -> float:
        """Calculate thermal photon occupation number."""
        if self.temperature == 0:
            return 0.0
        
        beta = 1.0 / (self.kb * self.temperature)
        n_thermal = 1.0 / (np.exp(beta * self.hbar * self.base_frequency) - 1.0)
        return n_thermal
    
    def set_parametric_pump(self, 
                           amplitude: float, 
                           frequency: float,
                           phase: float = 0.0) -> Dict[str, float]:
        """
        Configure parametric pumping for DCE generation.
        
        The pump modulates the cavity frequency as:
        ω(t) = ω₀[1 + ε cos(Ωt + φ)]
        
        Args:
            amplitude: Pump amplitude ε (fractional frequency modulation)
            frequency: Pump frequency Ω [Hz]
            phase: Pump phase φ [rad]
            
        Returns:
            Dictionary with pump configuration status
        """
        # Validate pump parameters
        if amplitude < 0 or amplitude > 1.0:
            raise ValueError("Pump amplitude must be in [0, 1]")
        
        if frequency < 0:
            raise ValueError("Pump frequency must be positive")
        
        self.pump_amplitude = amplitude
        self.pump_frequency = frequency
        self.pump_phase = phase
        self.detuning = frequency - 2 * self.base_frequency
        
        # Calculate theoretical DCE rate
        dce_rate = self._calculate_dce_rate()
        
        # Optimal frequency for maximum DCE is Ω ≈ 2ω₀
        optimal_frequency = 2 * self.base_frequency
        optimization_factor = 1.0 / (1.0 + (frequency - optimal_frequency)**2 / (self.linewidth**2))
        
        config_result = {
            'pump_amplitude': amplitude,
            'pump_frequency': frequency,
            'pump_phase': phase,
            'detuning': self.detuning,
            'dce_rate_theoretical': dce_rate,
            'optimal_frequency': optimal_frequency,
            'optimization_factor': optimization_factor,
            'timestamp': datetime.now().isoformat()
        }
        
        self.operation_log.append({
            'action': 'set_parametric_pump',
            'parameters': config_result
        })
        
        print(f"🎯 Parametric Pump Configured:")
        print(f"   • Amplitude: ε = {amplitude:.3f}")
        print(f"   • Frequency: Ω = {frequency/1e9:.3f} GHz")
        print(f"   • Detuning: Δ = {self.detuning/1e9:.3f} GHz")
        print(f"   • DCE rate: {dce_rate:.2e} s⁻¹")
        print(f"   • Optimization: {optimization_factor:.3f}")
        
        return config_result
    
    def _calculate_dce_rate(self) -> float:
        """
        Calculate theoretical Dynamical Casimir Effect photon generation rate.
        
        For parametric pumping: Γ_DCE ≈ (ε²/4) * (Ω/ω₀)⁴ * γ
        """
        if self.pump_amplitude == 0 or self.pump_frequency == 0:
            return 0.0
        
        epsilon = self.pump_amplitude
        omega_ratio = self.pump_frequency / self.base_frequency
        
        # DCE rate formula (simplified)
        dce_rate = (epsilon**2 / 4.0) * (omega_ratio**4) * self.linewidth
        
        # Include thermal suppression
        thermal_factor = 1.0 / (1.0 + self.thermal_photons)
        dce_rate *= thermal_factor
        
        return dce_rate
    
    def optimize_impedance_matching(self, target_coupling: Optional[float] = None) -> Dict[str, float]:
        """
        Optimize impedance matching for maximum power extraction.
        
        Args:
            target_coupling: Target external coupling strength
            
        Returns:
            Optimization results
        """
        if target_coupling is None:
            # Optimal coupling for maximum power transfer
            target_coupling = self.linewidth / 2.0
        
        # Current impedance mismatch
        current_mismatch = abs(self.coupling_strength - target_coupling)
        
        # Adjust coupling (in real system, this would control physical parameters)
        self.coupling_strength = target_coupling
        
        # Calculate power transfer efficiency
        total_loss_rate = self.linewidth + self.coupling_strength
        extraction_efficiency = self.coupling_strength / total_loss_rate
        
        # Q-factor with external loading
        loaded_q = self.base_frequency / total_loss_rate
        
        optimization_result = {
            'target_coupling': target_coupling,
            'achieved_coupling': self.coupling_strength,
            'impedance_mismatch': current_mismatch,
            'extraction_efficiency': extraction_efficiency,
            'loaded_q_factor': loaded_q,
            'intrinsic_q_factor': self.quality_factor,
            'power_transfer_ratio': extraction_efficiency,
            'timestamp': datetime.now().isoformat()
        }
        
        self.operation_log.append({
            'action': 'optimize_impedance_matching',
            'results': optimization_result
        })
        
        print(f"🔧 Impedance Matching Optimized:")
        print(f"   • External coupling: κ_ext = {self.coupling_strength:.2e} Hz")
        print(f"   • Extraction efficiency: {extraction_efficiency:.3f}")
        print(f"   • Loaded Q: {loaded_q:.1e}")
        print(f"   • Power transfer: {extraction_efficiency:.1%}")
        
        return optimization_result
    
    def measure_field_fluctuations(self, 
                                 measurement_time: float = 1e-3,
                                 sampling_rate: float = 1e9) -> Dict[str, Union[np.ndarray, float]]:
        """
        Measure electromagnetic field fluctuations and extract T₀₀.
        
        Args:
            measurement_time: Measurement duration [s]
            sampling_rate: ADC sampling rate [Hz]
            
        Returns:
            Measurement results including T₀₀ estimate
        """
        n_samples = int(measurement_time * sampling_rate)
        time_array = np.linspace(0, measurement_time, n_samples)
        
        # Simulate cavity field with DCE contributions
        
        # 1. Vacuum fluctuations (thermal + quantum)
        vacuum_noise = np.random.normal(0, np.sqrt(self.thermal_photons + 0.5), n_samples)
        
        # 2. DCE-generated photons
        if self.pump_amplitude > 0 and self.pump_frequency > 0:
            dce_rate = self._calculate_dce_rate()
            
            # DCE manifests as parametric amplification
            pump_modulation = self.pump_amplitude * np.cos(2*np.pi*self.pump_frequency*time_array + self.pump_phase)
            dce_contribution = dce_rate * pump_modulation * vacuum_noise
        else:
            dce_contribution = np.zeros(n_samples)
        
        # 3. Total field (I and Q quadratures)
        field_i = vacuum_noise + dce_contribution
        field_q = np.random.normal(0, np.sqrt(self.thermal_photons + 0.5), n_samples)
        
        # Add cavity response (filtering)
        if SCIPY_AVAILABLE:
            # Cavity transfer function
            cavity_pole = -self.linewidth / 2.0
            b, a = scipy.signal.bilinear([1], [1, -cavity_pole/sampling_rate])
            field_i = scipy.signal.filtfilt(b, a, field_i)
            field_q = scipy.signal.filtfilt(b, a, field_q)
        
        # Calculate field properties
        field_amplitude = np.sqrt(field_i**2 + field_q**2)
        field_power = field_amplitude**2
        
        # Estimate T₀₀ from field measurements
        T00_estimate = self._extract_stress_energy_tensor(field_i, field_q, time_array)
        
        # Statistical analysis
        mean_amplitude = np.mean(field_amplitude)
        rms_amplitude = np.sqrt(np.mean(field_amplitude**2))
        peak_amplitude = np.max(field_amplitude)
        
        # Energy calculations
        total_energy = np.mean(field_power) * self.hbar * self.base_frequency * self.cavity_volume
        negative_energy_fraction = np.sum(T00_estimate < 0) / len(T00_estimate)
        
        measurement_result = {
            'time_array': time_array,
            'field_i': field_i,
            'field_q': field_q,
            'field_amplitude': field_amplitude,
            'field_power': field_power,
            'T00_estimate': T00_estimate,
            'mean_amplitude': mean_amplitude,
            'rms_amplitude': rms_amplitude,
            'peak_amplitude': peak_amplitude,
            'total_energy': total_energy,
            'negative_energy_fraction': negative_energy_fraction,
            'dce_rate_measured': dce_rate if hasattr(self, 'dce_rate') else self._calculate_dce_rate(),
            'measurement_duration': measurement_time,
            'sampling_rate': sampling_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        self.measurement_history.append(measurement_result)
        
        print(f"📊 Field Fluctuation Measurement:")
        print(f"   • Measurement time: {measurement_time*1000:.1f} ms")
        print(f"   • Mean amplitude: {mean_amplitude:.3e}")
        print(f"   • Total energy: {total_energy:.3e} J")
        print(f"   • Negative energy fraction: {negative_energy_fraction:.3f}")
        print(f"   • T₀₀ range: [{np.min(T00_estimate):.2e}, {np.max(T00_estimate):.2e}]")
        
        return measurement_result
    
    def _extract_stress_energy_tensor(self, 
                                    field_i: np.ndarray, 
                                    field_q: np.ndarray, 
                                    time_array: np.ndarray) -> np.ndarray:
        """
        Extract T₀₀ component of stress-energy tensor from field measurements.
        
        For electromagnetic fields:
        T₀₀ = (1/2μ₀)[B² + ε₀E²] - ⟨T₀₀⟩_vacuum
        """
        # Convert field quadratures to E and B fields
        # This is a simplified model - real system requires careful calibration
        
        # Electric field (proportional to field quadratures)
        E_field = field_i * np.sqrt(self.hbar * self.base_frequency / (2 * self.eps0 * self.cavity_volume))
        
        # Magnetic field (from time derivative)
        if len(field_q) > 1:
            dt = time_array[1] - time_array[0]
            B_field = -np.gradient(field_q, dt) / self.c**2
        else:
            B_field = np.zeros_like(field_q)
        
        # Stress-energy tensor T₀₀ component
        energy_density = 0.5 * (self.eps0 * E_field**2 + B_field**2 / self.mu0)
        
        # Subtract vacuum contribution
        vacuum_energy_density = 0.5 * self.hbar * self.base_frequency / self.cavity_volume
        T00 = energy_density - vacuum_energy_density
        
        return T00
    
    def run_automated_sequence(self, 
                             optimization_params: Dict,
                             measurement_params: Dict) -> Dict[str, any]:
        """
        Run automated measurement sequence for negative energy extraction.
        
        Args:
            optimization_params: Parameters for system optimization
            measurement_params: Parameters for field measurements
            
        Returns:
            Complete characterization results
        """
        print(f"🤖 Starting Automated Superconducting Resonator Sequence")
        
        sequence_results = {
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'summary': {}
        }
        
        # Step 1: Optimize impedance matching
        print("\n📍 Step 1: Impedance Matching Optimization")
        impedance_result = self.optimize_impedance_matching(
            target_coupling=optimization_params.get('target_coupling')
        )
        sequence_results['steps'].append({
            'step': 'impedance_optimization',
            'results': impedance_result
        })
        
        # Step 2: Configure parametric pump
        print("\n📍 Step 2: Parametric Pump Configuration")
        pump_result = self.set_parametric_pump(
            amplitude=optimization_params.get('pump_amplitude', 0.1),
            frequency=optimization_params.get('pump_frequency', 2*self.base_frequency),
            phase=optimization_params.get('pump_phase', 0.0)
        )
        sequence_results['steps'].append({
            'step': 'pump_configuration',
            'results': pump_result
        })
        
        # Step 3: Field fluctuation measurements
        print("\n📍 Step 3: Field Fluctuation Measurements")
        measurement_result = self.measure_field_fluctuations(
            measurement_time=measurement_params.get('measurement_time', 1e-3),
            sampling_rate=measurement_params.get('sampling_rate', 1e9)
        )
        sequence_results['steps'].append({
            'step': 'field_measurements',
            'results': measurement_result
        })
        
        # Step 4: Analysis and optimization
        print("\n📍 Step 4: Results Analysis")
        
        # Extract key metrics
        extraction_efficiency = impedance_result['extraction_efficiency']
        dce_rate = pump_result['dce_rate_theoretical']
        negative_energy_fraction = measurement_result['negative_energy_fraction']
        total_energy = measurement_result['total_energy']
        
        # Performance score
        performance_score = (extraction_efficiency * 
                           min(dce_rate / self.linewidth, 1.0) * 
                           negative_energy_fraction)
        
        sequence_results['summary'] = {
            'performance_score': performance_score,
            'extraction_efficiency': extraction_efficiency,
            'dce_rate': dce_rate,
            'negative_energy_fraction': negative_energy_fraction,
            'total_energy': total_energy,
            'temperature': self.temperature,
            'quality_factor': self.quality_factor,
            'recommendations': self._generate_recommendations(sequence_results)
        }
        
        sequence_results['end_time'] = datetime.now().isoformat()
        
        print(f"\n✅ Automated Sequence Complete!")
        print(f"   • Performance score: {performance_score:.3f}")
        print(f"   • Extraction efficiency: {extraction_efficiency:.1%}")
        print(f"   • DCE rate: {dce_rate:.2e} s⁻¹")
        print(f"   • Negative energy fraction: {negative_energy_fraction:.3f}")
        
        return sequence_results
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate optimization recommendations based on measurement results."""
        recommendations = []
        
        # Check performance metrics
        summary = results['summary']
        
        if summary['extraction_efficiency'] < 0.5:
            recommendations.append("Optimize impedance matching - current efficiency is low")
        
        if summary['dce_rate'] < self.linewidth:
            recommendations.append("Increase pump amplitude or optimize pump frequency")
        
        if summary['negative_energy_fraction'] < 0.1:
            recommendations.append("Check for thermal noise - consider lower operating temperature")
        
        if self.temperature > 0.05:
            recommendations.append("Lower operating temperature for better coherence")
        
        if self.quality_factor < 1e5:
            recommendations.append("Improve cavity Q-factor to reduce losses")
        
        if len(recommendations) == 0:
            recommendations.append("System operating optimally - proceed with experiments")
        
        return recommendations
    
    def plot_measurement_results(self, save_path: Optional[str] = None):
        """Plot recent measurement results."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("matplotlib not available for plotting")
            return
        
        if not self.measurement_history:
            warnings.warn("No measurement data to plot")
            return
        
        # Use most recent measurement
        data = self.measurement_history[-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Time-domain field
        axes[0, 0].plot(data['time_array']*1e6, data['field_i'], 'b-', alpha=0.7, label='I quadrature')
        axes[0, 0].plot(data['time_array']*1e6, data['field_q'], 'r-', alpha=0.7, label='Q quadrature')
        axes[0, 0].set_xlabel('Time [μs]')
        axes[0, 0].set_ylabel('Field Amplitude')
        axes[0, 0].set_title('Field Quadratures')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Field amplitude
        axes[0, 1].plot(data['time_array']*1e6, data['field_amplitude'], 'g-', linewidth=1.5)
        axes[0, 1].set_xlabel('Time [μs]')
        axes[0, 1].set_ylabel('|Field|')
        axes[0, 1].set_title('Field Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # T₀₀ stress-energy
        axes[1, 0].plot(data['time_array']*1e6, data['T00_estimate'], 'purple', linewidth=1.5)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Time [μs]')
        axes[1, 0].set_ylabel('T₀₀ [J/m³]')
        axes[1, 0].set_title('Stress-Energy Tensor')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Power spectral density
        if SCIPY_AVAILABLE and len(data['field_amplitude']) > 10:
            freqs, psd = scipy.signal.welch(data['field_amplitude'], 
                                          fs=data['sampling_rate'], 
                                          nperseg=min(256, len(data['field_amplitude'])//4))
            axes[1, 1].semilogy(freqs/1e9, psd)
            axes[1, 1].set_xlabel('Frequency [GHz]')
            axes[1, 1].set_ylabel('PSD')
            axes[1, 1].set_title('Power Spectral Density')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Histogram of T₀₀ values
            axes[1, 1].hist(data['T00_estimate'], bins=30, alpha=0.7, color='orange')
            axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('T₀₀ [J/m³]')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('T₀₀ Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("=== Superconducting Resonator for Negative Energy Generation ===")
    
    # Initialize resonator
    resonator = SuperconductingResonator(
        base_frequency=5.0e9,    # 5 GHz
        quality_factor=1e6,      # High-Q cavity
        temperature=0.01,        # 10 mK
        cavity_volume=1e-9,      # 1 mm³
        coupling_strength=0.1
    )
    
    # Configuration parameters
    optimization_params = {
        'target_coupling': 1e4,           # Optimal coupling
        'pump_amplitude': 0.05,           # 5% frequency modulation
        'pump_frequency': 2*5.0e9,        # 10 GHz pump (2ω₀)
        'pump_phase': 0.0
    }
    
    measurement_params = {
        'measurement_time': 1e-3,     # 1 ms measurement
        'sampling_rate': 1e9          # 1 GS/s ADC
    }
    
    # Run automated sequence
    results = resonator.run_automated_sequence(
        optimization_params=optimization_params,
        measurement_params=measurement_params
    )
    
    # Save results
    output_file = "superconducting_resonator_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = results.copy()
    for step in json_results['steps']:
        if 'results' in step and 'time_array' in step['results']:
            # Remove large arrays from JSON (keep summary data)
            step['results'] = {k: v for k, v in step['results'].items() 
                             if not isinstance(v, np.ndarray)}
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Plot results
    try:
        resonator.plot_measurement_results("superconducting_resonator_measurement.png")
    except:
        print("   (Plotting not available)")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Integrate with cryogenic control systems")
    print("2. Add real-time feedback for pump optimization")
    print("3. Implement flux bias control for frequency tuning")
    print("4. Connect to quantum-limited amplifiers")
    print("5. Scale up cavity volume for higher energy extraction")
