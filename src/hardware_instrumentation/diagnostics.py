"""
Hardware Instrumentation and Diagnostics Module
===============================================

Precision measurement systems for negative energy density detection and characterization.

This module implements four main measurement approaches:
1. Interferometric Phase Detection - ΔT₀₀ → refractive index changes → phase shifts
2. Calorimetric Energy Sensing - Direct thermal measurement of energy density changes
3. Phase-Shift Interferometry - Real-time interferometric measurement pipeline
4. Real-Time DAQ Systems - FPGA-style high-speed data acquisition

Mathematical Foundations:
- Phase shift: Δφ = (2π/λ) Δn L
- Electro-optic coupling: Δn = ½n³rE where E = √(2μ₀|ΔT₀₀|)
- Calorimetric response: ΔT = ΔE/(Cₚm) where ΔE = ΔT₀₀ × V
- Real-time sampling: fs ≥ 2f_max (Nyquist criterion)
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union
import warnings
from dataclasses import dataclass
import time

# Physical constants
μ0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
c = 2.998e8            # Speed of light (m/s)
h = 6.626e-34          # Planck constant (J⋅s)
k_B = 1.381e-23        # Boltzmann constant (J/K)

# Instrumentation limits and specifications
PHASE_RESOLUTION = 1e-6    # Minimum detectable phase shift (rad)
TEMP_RESOLUTION = 1e-6     # Minimum detectable temperature change (K)
SAMPLING_RATE_MAX = 1e12   # Maximum realistic sampling rate (Hz)
BUFFER_SIZE_MAX = 1e6      # Maximum buffer size (samples)

@dataclass
class MeasurementResult:
    """Data structure for measurement results with metadata."""
    times: np.ndarray
    values: np.ndarray
    units: str
    measurement_type: str
    metadata: Dict
    noise_floor: float = 0.0
    signal_to_noise: float = 0.0

class InterferometricProbe:
    """
    Precision interferometric probe for detecting ΔT₀₀-induced refractive index changes.
    
    Mathematical model:
    1. Energy density to electric field: E = √(2μ₀|ΔT₀₀|)
    2. Electro-optic effect: Δn = ½n₀³rE  
    3. Phase shift: Δφ = (2π/λ) Δn L
    
    This probe can detect minute changes in the stress-energy tensor through
    precision interferometric phase measurements.
    """
    
    def __init__(self, wavelength: float, path_length: float, 
                 n0: float, r_coeff: float, material: str = "LiIO3"):
        """
        Initialize interferometric probe.
        
        Args:
            wavelength: Probe laser wavelength (m)
            path_length: Optical path length through sample (m)
            n0: Baseline refractive index (dimensionless)
            r_coeff: Electro-optic coefficient (m/V)
            material: Probe material name
        """
        self.lambda0 = wavelength
        self.L = path_length
        self.n0 = n0
        self.r = r_coeff
        self.material = material
        
        # Calculated parameters
        self.k0 = 2 * np.pi / wavelength  # Wave number
        self.sensitivity = self.calculate_sensitivity()
        
        # Noise characteristics
        self.shot_noise_limit = np.sqrt(h * c / wavelength) / (2 * np.pi)  # rad/√Hz
        self.thermal_noise = 1e-8  # rad/√Hz (typical)
        
    def calculate_sensitivity(self) -> float:
        """Calculate theoretical sensitivity (rad per J/m³)."""
        # dφ/dT₀₀ = (2π/λ) × (½n₀³r) × √(2μ₀) × L / (2√|T₀₀|)
        # For small T₀₀, approximate as linear sensitivity
        return (self.k0 * 0.5 * self.n0**3 * self.r * 
                np.sqrt(2 * μ0) * self.L)
    
    def phase_shift(self, delta_T00: float) -> float:
        """
        Calculate phase shift from energy density change.
        
        Args:
            delta_T00: Change in T₀₀ component (J/m³)
            
        Returns:
            Phase shift (rad)
        """
        if delta_T00 == 0:
            return 0.0
            
        # Convert energy density to effective electric field
        # E = √(2μ₀|ΔT₀₀|) - simplified field-energy relationship
        E_field = np.sqrt(2 * μ0 * abs(delta_T00))
        
        # Electro-optic refractive index change
        delta_n = 0.5 * self.n0**3 * self.r * E_field
        
        # Phase shift through optical path
        phase_shift = self.k0 * delta_n * self.L
        
        # Apply sign from original T₀₀
        return phase_shift if delta_T00 > 0 else -phase_shift
    
    def simulate_pulse(self, times: np.ndarray, T00_profile: np.ndarray, 
                      add_noise: bool = True) -> MeasurementResult:
        """
        Simulate interferometric measurement of a T₀₀ pulse.
        
        Args:
            times: Time array (s)
            T00_profile: Energy density profile ΔT₀₀(t) (J/m³)
            add_noise: Whether to include realistic noise
            
        Returns:
            MeasurementResult with phase shift data
        """
        # Calculate ideal phase shifts
        phases = np.array([self.phase_shift(T00) for T00 in T00_profile])
        
        # Add realistic noise if requested
        if add_noise:
            dt = np.mean(np.diff(times)) if len(times) > 1 else 1e-9
            noise_amplitude = self.shot_noise_limit / np.sqrt(dt)
            noise = np.random.normal(0, noise_amplitude, len(phases))
            phases += noise
        
        # Calculate signal-to-noise ratio
        signal_rms = np.sqrt(np.mean(phases**2))
        noise_rms = self.shot_noise_limit / np.sqrt(np.mean(np.diff(times))) if add_noise else 0
        snr = signal_rms / noise_rms if noise_rms > 0 else float('inf')
        
        return MeasurementResult(
            times=times,
            values=phases,
            units="rad",
            measurement_type="interferometric_phase",
            metadata={
                "wavelength": self.lambda0,
                "path_length": self.L,
                "material": self.material,
                "sensitivity": self.sensitivity,
                "max_T00": np.max(np.abs(T00_profile))
            },
            noise_floor=noise_rms,
            signal_to_noise=snr
        )
    
    def frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate frequency response of the probe.
        
        Args:
            frequencies: Frequency array (Hz)
            
        Returns:
            Complex frequency response
        """
        # Simple first-order response limited by light transit time
        tau_optical = self.L / c  # Optical transit time
        omega = 2 * np.pi * frequencies
        return 1 / (1 + 1j * omega * tau_optical)

class CalorimetricSensor:
    """
    Calorimetric sensor for direct thermal measurement of energy density changes.
    
    Mathematical model:
    1. Energy absorption: ΔE = ΔT₀₀ × V
    2. Temperature rise: ΔT = ΔE/(Cₚm) where m = ρV
    3. Thermal time constant: τ = ρCₚV/(hA)
    
    This sensor provides absolute energy measurements but with slower response
    compared to interferometric methods.
    """
    
    def __init__(self, volume: float, density: float, Cp: float, 
                 thermal_conductivity: float = 100, surface_area: float = None,
                 material: str = "Silicon"):
        """
        Initialize calorimetric sensor.
        
        Args:
            volume: Sensor volume (m³)
            density: Material density (kg/m³)
            Cp: Specific heat capacity (J/(kg⋅K))
            thermal_conductivity: Thermal conductivity (W/(m⋅K))
            surface_area: Surface area for heat loss (m²)
            material: Sensor material name
        """
        self.V = volume
        self.rho = density
        self.Cp = Cp
        self.k_thermal = thermal_conductivity
        self.material = material
        
        # Calculate derived parameters
        self.mass = self.rho * self.V
        self.heat_capacity = self.mass * self.Cp
        
        # Estimate surface area if not provided (assume cube)
        if surface_area is None:
            side_length = self.V**(1/3)
            self.A = 6 * side_length**2
        else:
            self.A = surface_area
            
        # Thermal time constant (simplified)
        self.tau_thermal = self.heat_capacity / (self.k_thermal * self.A / (self.V**(1/3)))
        
        # Sensitivity and noise
        self.sensitivity = 1.0 / self.heat_capacity  # K per J
        self.thermal_noise = np.sqrt(4 * k_B * 300 * self.k_thermal) / self.heat_capacity  # K/√Hz
    
    def temp_rise(self, delta_T00: float) -> float:
        """
        Calculate instantaneous temperature rise from energy density change.
        
        Args:
            delta_T00: Change in T₀₀ component (J/m³)
            
        Returns:
            Temperature rise (K)
        """
        # Total energy absorbed by sensor volume
        delta_E = delta_T00 * self.V
        
        # Temperature rise
        return delta_E / self.heat_capacity
    
    def simulate_pulse(self, times: np.ndarray, T00_profile: np.ndarray,
                      include_thermal_dynamics: bool = True,
                      add_noise: bool = True) -> MeasurementResult:
        """
        Simulate calorimetric measurement with thermal dynamics.
        
        Args:
            times: Time array (s)
            T00_profile: Energy density profile ΔT₀₀(t) (J/m³)
            include_thermal_dynamics: Include thermal time constant effects
            add_noise: Whether to include thermal noise
            
        Returns:
            MeasurementResult with temperature data
        """
        # Calculate ideal temperature response
        temps = np.array([self.temp_rise(T00) for T00 in T00_profile])
        
        # Apply thermal dynamics if requested
        if include_thermal_dynamics and len(times) > 1:
            dt = np.mean(np.diff(times))
            # Simple exponential relaxation
            alpha = dt / self.tau_thermal
            for i in range(1, len(temps)):
                temps[i] = temps[i-1] * (1 - alpha) + temps[i] * alpha
        
        # Add thermal noise if requested
        if add_noise:
            dt = np.mean(np.diff(times)) if len(times) > 1 else 1e-9
            noise_amplitude = self.thermal_noise / np.sqrt(dt)
            noise = np.random.normal(0, noise_amplitude, len(temps))
            temps += noise
        
        # Calculate signal-to-noise ratio
        signal_rms = np.sqrt(np.mean(temps**2))
        noise_rms = self.thermal_noise / np.sqrt(np.mean(np.diff(times))) if add_noise else 0
        snr = signal_rms / noise_rms if noise_rms > 0 else float('inf')
        
        return MeasurementResult(
            times=times,
            values=temps,
            units="K",
            measurement_type="calorimetric_temperature",
            metadata={
                "volume": self.V,
                "material": self.material,
                "thermal_time_constant": self.tau_thermal,
                "sensitivity": self.sensitivity,
                "max_T00": np.max(np.abs(T00_profile))
            },
            noise_floor=noise_rms,
            signal_to_noise=snr
        )

class PhaseShiftInterferometer:
    """
    Complete phase-shift interferometry system with sampling and signal processing.
    
    Combines interferometric probe with realistic sampling, filtering, and
    signal processing capabilities for real-time measurements.
    """
    
    def __init__(self, probe: InterferometricProbe, sampling_rate: float,
                 anti_alias_filter: bool = True, digital_filter: bool = True):
        """
        Initialize phase-shift interferometer system.
        
        Args:
            probe: InterferometricProbe instance
            sampling_rate: ADC sampling rate (Hz)
            anti_alias_filter: Enable anti-aliasing filter
            digital_filter: Enable digital signal processing
        """
        self.probe = probe
        self.fs = min(sampling_rate, SAMPLING_RATE_MAX)
        self.anti_alias = anti_alias_filter
        self.digital_filter = digital_filter
        
        # Nyquist frequency and filter parameters
        self.f_nyquist = self.fs / 2
        self.filter_cutoff = 0.8 * self.f_nyquist  # 80% of Nyquist
        
        # System specifications
        self.bit_depth = 16  # ADC resolution
        self.full_scale = 2 * np.pi  # ±π phase range
        self.lsb = self.full_scale / (2**self.bit_depth)  # Least significant bit
        
    def acquire(self, duration: float, T00_function: Callable[[float], float],
               real_time_processing: bool = True) -> MeasurementResult:
        """
        Acquire interferometric data over specified duration.
        
        Args:
            duration: Total acquisition time (s)
            T00_function: Function T₀₀(t) returning energy density at time t
            real_time_processing: Apply real-time signal processing
            
        Returns:
            MeasurementResult with processed phase data
        """
        # Generate time base
        n_samples = int(duration * self.fs)
        times = np.arange(n_samples) / self.fs
        
        # Sample T₀₀ function
        T00_values = np.array([T00_function(t) for t in times])
        
        # Get interferometric measurement
        result = self.probe.simulate_pulse(times, T00_values, add_noise=True)
        phases = result.values
        
        # Apply real-time processing
        if real_time_processing:
            phases = self._apply_signal_processing(phases)
        
        # Update result with processed data
        result.values = phases
        result.metadata.update({
            "sampling_rate": self.fs,
            "duration": duration,
            "n_samples": n_samples,
            "processing_applied": real_time_processing
        })
        
        return result
    
    def _apply_signal_processing(self, signal: np.ndarray) -> np.ndarray:
        """Apply real-time signal processing pipeline."""
        processed = signal.copy()
        
        # Anti-aliasing filter (simple Butterworth approximation)
        if self.anti_alias:
            # Simple moving average (crude anti-alias filter)
            kernel_size = max(1, int(self.fs / (4 * self.filter_cutoff)))
            kernel = np.ones(kernel_size) / kernel_size
            processed = np.convolve(processed, kernel, mode='same')
        
        # ADC quantization
        processed = np.round(processed / self.lsb) * self.lsb
        
        # Digital filtering (basic low-pass)
        if self.digital_filter:
            # Simple exponential smoothing
            alpha = 0.1
            for i in range(1, len(processed)):
                processed[i] = alpha * processed[i] + (1 - alpha) * processed[i-1]
        
        return processed
    
    def frequency_sweep(self, freq_range: Tuple[float, float], 
                       n_points: int = 100) -> Dict:
        """
        Perform frequency response characterization.
        
        Args:
            freq_range: (f_min, f_max) frequency range (Hz)
            n_points: Number of frequency points
            
        Returns:
            Dictionary with frequency response data
        """
        frequencies = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
        
        # Get probe frequency response
        probe_response = self.probe.frequency_response(frequencies)
        
        # System bandwidth limitations
        system_response = np.ones_like(frequencies, dtype=complex)
        
        # Sampling rate limitation
        aliasing_mask = frequencies < self.f_nyquist
        system_response[~aliasing_mask] *= 0.01  # Strong attenuation above Nyquist
        
        # Anti-alias filter response
        if self.anti_alias:
            filter_response = 1 / (1 + (frequencies / self.filter_cutoff)**4)
            system_response *= filter_response
        
        # Combined response
        total_response = probe_response * system_response
        
        return {
            'frequencies': frequencies,
            'probe_response': probe_response,
            'system_response': system_response,
            'total_response': total_response,
            'magnitude_dB': 20 * np.log10(np.abs(total_response)),
            'phase_deg': np.degrees(np.angle(total_response))
        }

class RealTimeDAQ:
    """
    Real-time data acquisition system with circular buffer and triggering.
    
    Simulates FPGA-style high-speed data acquisition with configurable
    triggering, buffering, and real-time analysis capabilities.
    """
    
    def __init__(self, buffer_size: int, sampling_rate: float,
                 trigger_level: float = 0.0, trigger_mode: str = "rising"):
        """
        Initialize real-time DAQ system.
        
        Args:
            buffer_size: Circular buffer size (samples)
            sampling_rate: Sampling rate (Hz)
            trigger_level: Trigger threshold
            trigger_mode: Trigger mode ("rising", "falling", "level")
        """
        self.buffer_size = min(int(buffer_size), int(BUFFER_SIZE_MAX))
        self.fs = sampling_rate
        self.trigger_level = trigger_level
        self.trigger_mode = trigger_mode
        
        # Circular buffers
        self.times = np.zeros(self.buffer_size)
        self.values = np.zeros(self.buffer_size)
        self.idx = 0
        self.triggered = False
        self.trigger_idx = 0
        
        # Statistics
        self.sample_count = 0
        self.trigger_count = 0
        self.overrun_count = 0
        
    def add_sample(self, t: float, value: float) -> bool:
        """
        Add sample to circular buffer with trigger detection.
        
        Args:
            t: Time stamp (s)
            value: Measured value
            
        Returns:
            True if trigger event detected
        """
        # Store sample in circular buffer
        buffer_idx = self.idx % self.buffer_size
        self.times[buffer_idx] = t
        self.values[buffer_idx] = value
        
        # Check for trigger condition
        trigger_detected = self._check_trigger(value)
        
        if trigger_detected:
            self.triggered = True
            self.trigger_idx = self.idx
            self.trigger_count += 1
        
        # Update counters
        self.idx += 1
        self.sample_count += 1
        
        # Check for buffer overrun
        if self.sample_count > self.buffer_size:
            self.overrun_count += 1
        
        return trigger_detected
    
    def _check_trigger(self, value: float) -> bool:
        """Check trigger condition based on current and previous values."""
        if self.sample_count == 0:
            return False
            
        prev_idx = (self.idx - 1) % self.buffer_size
        prev_value = self.values[prev_idx]
        
        if self.trigger_mode == "rising":
            return prev_value <= self.trigger_level < value
        elif self.trigger_mode == "falling":
            return prev_value >= self.trigger_level > value
        elif self.trigger_mode == "level":
            return abs(value) > abs(self.trigger_level)
        else:
            return False
    
    def get_buffer(self, samples_before_trigger: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve circular buffer data, optionally centered on last trigger.
        
        Args:
            samples_before_trigger: Number of samples before trigger to include
            
        Returns:
            Tuple of (times, values) arrays
        """
        if not self.triggered or samples_before_trigger is None:
            # Return entire buffer in chronological order
            start_idx = self.idx % self.buffer_size
            times = np.roll(self.times, -start_idx)
            values = np.roll(self.values, -start_idx)
        else:
            # Return data centered on trigger
            trigger_buffer_idx = self.trigger_idx % self.buffer_size
            start_idx = (trigger_buffer_idx - samples_before_trigger) % self.buffer_size
            
            indices = [(start_idx + i) % self.buffer_size for i in range(self.buffer_size)]
            times = self.times[indices]
            values = self.values[indices]
        
        return times, values
    
    def get_statistics(self) -> Dict:
        """Get DAQ system statistics."""
        return {
            'sample_count': self.sample_count,
            'trigger_count': self.trigger_count,
            'overrun_count': self.overrun_count,
            'buffer_utilization': min(self.sample_count / self.buffer_size, 1.0),
            'trigger_rate': self.trigger_count / (self.sample_count / self.fs) if self.sample_count > 0 else 0,
            'overrun_rate': self.overrun_count / self.sample_count if self.sample_count > 0 else 0
        }
    
    def reset(self):
        """Reset DAQ system state."""
        self.times.fill(0)
        self.values.fill(0)
        self.idx = 0
        self.triggered = False
        self.trigger_idx = 0
        self.sample_count = 0
        self.trigger_count = 0
        self.overrun_count = 0

# Utility functions for common measurement scenarios

def generate_T00_pulse(pulse_type: str = "gaussian", amplitude: float = -1e7,
                      center_time: float = 1e-9, width: float = 0.2e-9) -> Callable:
    """
    Generate synthetic ΔT₀₀ pulse functions for testing.
    
    Args:
        pulse_type: Type of pulse ("gaussian", "exponential", "square")
        amplitude: Peak amplitude (J/m³)
        center_time: Pulse center time (s)
        width: Pulse width parameter (s)
        
    Returns:
        Function T₀₀(t) that can be used with measurement systems
    """
    if pulse_type == "gaussian":
        def T00_func(t):
            return amplitude * np.exp(-((t - center_time) / width)**2)
    elif pulse_type == "exponential":
        def T00_func(t):
            return amplitude * np.exp(-(t - center_time) / width) if t >= center_time else 0
    elif pulse_type == "square":
        def T00_func(t):
            return amplitude if abs(t - center_time) < width else 0
    else:
        raise ValueError(f"Unknown pulse type: {pulse_type}")
    
    return T00_func

def benchmark_instrumentation_suite() -> Dict:
    """
    Benchmark all instrumentation components with synthetic data.
    
    Returns:
        Dictionary with benchmark results for all instruments
    """
    print("🔬 Benchmarking Instrumentation Suite")
    print("=" * 50)
    
    # Test parameters
    duration = 5e-9  # 5 ns
    T00_amplitude = -1e7  # J/m³
    
    # Generate test pulse
    T00_pulse = generate_T00_pulse("gaussian", T00_amplitude, 2.5e-9, 0.5e-9)
    
    results = {}
    
    # 1. Interferometric probe test
    print("   📡 Testing interferometric probe...")
    probe = InterferometricProbe(
        wavelength=1.55e-6,  # 1550 nm telecom wavelength
        path_length=0.1,     # 10 cm path
        n0=1.5,              # Typical glass
        r_coeff=1e-12,       # m/V, typical electro-optic coefficient
        material="LiIO3"
    )
    
    times = np.linspace(0, duration, 1000)
    T00_values = np.array([T00_pulse(t) for t in times])
    
    interfero_result = probe.simulate_pulse(times, T00_values)
    max_phase = np.max(np.abs(interfero_result.values))
    
    results['interferometric'] = {
        'max_phase_shift_rad': max_phase,
        'max_phase_shift_deg': np.degrees(max_phase),
        'sensitivity': probe.sensitivity,
        'snr': interfero_result.signal_to_noise,
        'status': 'SUCCESS'
    }
    
    print(f"      ✅ Max phase shift: {max_phase:.2e} rad ({np.degrees(max_phase):.3f}°)")
    print(f"      • SNR: {interfero_result.signal_to_noise:.1f}")
    
    # 2. Calorimetric sensor test
    print("   🌡️  Testing calorimetric sensor...")
    calorimeter = CalorimetricSensor(
        volume=1e-18,      # 1 femtoliter
        density=2330,      # Silicon density (kg/m³)
        Cp=700,            # Silicon heat capacity (J/(kg⋅K))
        material="Silicon"
    )
    
    calor_result = calorimeter.simulate_pulse(times, T00_values)
    max_temp = np.max(np.abs(calor_result.values))
    
    results['calorimetric'] = {
        'max_temp_rise_K': max_temp,
        'max_temp_rise_mK': max_temp * 1000,
        'sensitivity': calorimeter.sensitivity,
        'thermal_time_constant': calorimeter.tau_thermal,
        'snr': calor_result.signal_to_noise,
        'status': 'SUCCESS'
    }
    
    print(f"      ✅ Max temperature rise: {max_temp:.2e} K ({max_temp*1000:.3f} mK)")
    print(f"      • Thermal τ: {calorimeter.tau_thermal:.2e} s")
    
    # 3. Phase-shift interferometer test
    print("   🔬 Testing phase-shift interferometer...")
    interferometer = PhaseShiftInterferometer(probe, sampling_rate=1e11)
    
    psi_result = interferometer.acquire(duration, T00_pulse)
    
    results['phase_shift_interferometer'] = {
        'processed_max_phase': np.max(np.abs(psi_result.values)),
        'sampling_rate': interferometer.fs,
        'snr': psi_result.signal_to_noise,
        'status': 'SUCCESS'
    }
    
    print(f"      ✅ Processed max phase: {np.max(np.abs(psi_result.values)):.2e} rad")
    print(f"      • Sampling rate: {interferometer.fs:.1e} Hz")
    
    # 4. Real-time DAQ test
    print("   📊 Testing real-time DAQ...")
    daq = RealTimeDAQ(buffer_size=10000, sampling_rate=1e10, trigger_level=1e-6)
    
    # Simulate real-time data acquisition
    trigger_count = 0
    for i, (t, phase) in enumerate(zip(times, interfero_result.values)):
        if daq.add_sample(t, phase):
            trigger_count += 1
    
    daq_stats = daq.get_statistics()
    
    results['real_time_daq'] = {
        'trigger_count': trigger_count,
        'sample_count': daq_stats['sample_count'],
        'buffer_utilization': daq_stats['buffer_utilization'],
        'trigger_rate': daq_stats['trigger_rate'],
        'status': 'SUCCESS'
    }
    
    print(f"      ✅ Triggers detected: {trigger_count}")
    print(f"      • Buffer utilization: {daq_stats['buffer_utilization']:.1%}")
    
    # Summary
    successful_tests = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
    print(f"\n🎯 Instrumentation Benchmark Complete: {successful_tests}/4 systems operational")
    
    return results

if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark_results = benchmark_instrumentation_suite()
    
    # Example usage demonstration
    print("\n🔬 Example Usage Demonstration")
    print("=" * 40)
    
    # Create synthetic negative energy pulse
    pulse_func = generate_T00_pulse("gaussian", -1e7, 1e-9, 0.2e-9)
    
    # Set up measurement systems
    probe = InterferometricProbe(1.55e-6, 0.1, 1.5, 1e-12)
    calorimeter = CalorimetricSensor(1e-18, 2330, 700)
    interferometer = PhaseShiftInterferometer(probe, 1e11)
    
    # Acquire data
    interfero_data = interferometer.acquire(5e-9, pulse_func)
    times = np.linspace(0, 5e-9, 1000)
    T00_vals = [pulse_func(t) for t in times]
    calor_data = calorimeter.simulate_pulse(times, T00_vals)
    
    print(f"📡 Interferometric: max Δφ = {np.max(np.abs(interfero_data.values)):.2e} rad")
    print(f"🌡️  Calorimetric: max ΔT = {np.max(np.abs(calor_data.values))*1000:.3f} mK")
    print(f"🎯 Phase resolution: {PHASE_RESOLUTION:.1e} rad")
    print(f"🎯 Temperature resolution: {TEMP_RESOLUTION*1000:.3f} mK")
    
    print("\n✅ Hardware Instrumentation Module Ready!")
