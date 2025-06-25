"""
Josephson Parametric Amplifier (JPA) Squeezed Vacuum System
===========================================================

This module implements a Josephson Parametric Amplifier system for generating
squeezed vacuum states with negative energy density through quantum interference.

Mathematical Framework:
    â_out = μâ_in + νâ†_in where |μ|² - |ν|² = 1
    ⟨T₀₀⟩ = -ℏω₀|ν|²/2V for perfect squeezing
    Squeezing parameter: ξ = ln(μ + ν) = 2r where r is squeezing strength
    
Key Features:
- Josephson junction-based parametric amplification
- Quantum-limited squeezing with real-time phase control
- Multi-mode squeezed state generation
- Integration with homodyne detection for state verification
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

try:
    import scipy.optimize
    import scipy.signal
    from scipy.special import hermite, factorial
    SCIPY_AVAILABLE = True
except ImportError:
    warnings.warn("SciPy not available for advanced computations")
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


@dataclass
class SqueezingParameters:
    """Parameters describing a squeezed quantum state."""
    squeezing_strength: float      # r - squeezing parameter
    squeezing_angle: float         # φ - squeezing angle
    displacement: complex          # α - coherent displacement
    frequency: float               # ω - mode frequency
    
    @property
    def mu(self) -> complex:
        """Parametric amplifier gain coefficient μ."""
        return np.cosh(self.squeezing_strength)
    
    @property
    def nu(self) -> complex:
        """Parametric amplifier gain coefficient ν."""
        return np.sinh(self.squeezing_strength) * np.exp(1j * self.squeezing_angle)
    
    @property
    def variance_reduction(self) -> float:
        """Variance reduction factor in squeezed quadrature."""
        return np.exp(-2 * self.squeezing_strength)
    
    @property
    def variance_increase(self) -> float:
        """Variance increase factor in anti-squeezed quadrature."""
        return np.exp(2 * self.squeezing_strength)


class JosephsonParametricAmplifier:
    """
    Josephson Parametric Amplifier for squeezed vacuum generation.
    
    This class models a JPA circuit based on a nonlinear Josephson junction
    driven by a strong pump tone to generate squeezed vacuum states.
    """
    
    def __init__(self,
                 signal_frequency: float = 6.0e9,     # 6 GHz signal frequency
                 pump_frequency: float = 12.0e9,      # 12 GHz pump (2ω_s)
                 josephson_energy: float = 20e-24,    # 20 μeV Josephson energy
                 charging_energy: float = 1e-24,     # 1 μeV charging energy
                 pump_power: float = -100,            # -100 dBm pump power
                 temperature: float = 0.01):          # 10 mK operation
        """
        Initialize Josephson Parametric Amplifier.
        
        Args:
            signal_frequency: Signal frequency [Hz]
            pump_frequency: Pump frequency [Hz] (typically 2ω_s)
            josephson_energy: EJ Josephson coupling energy [J]
            charging_energy: EC charging energy [J]
            pump_power: Pump power [dBm]
            temperature: Operating temperature [K]
        """
        self.signal_frequency = signal_frequency      # ω_s
        self.pump_frequency = pump_frequency          # ω_p
        self.josephson_energy = josephson_energy      # EJ
        self.charging_energy = charging_energy        # EC
        self.pump_power = pump_power                  # P_pump
        self.temperature = temperature                # T
        
        # Physical constants
        self.hbar = 1.054571817e-34
        self.kb = 1.380649e-23
        self.e = 1.602176634e-19      # Elementary charge
        self.flux_quantum = self.hbar / (2 * self.e)  # Φ₀ = h/2e
        
        # Derived parameters
        self.plasma_frequency = np.sqrt(8 * self.josephson_energy * self.charging_energy) / self.hbar
        self.anharmonicity = -self.charging_energy / self.hbar  # α = -EC/ℏ
        self.thermal_photons = self._calculate_thermal_photons()
        
        # JPA operating parameters
        self.pump_amplitude = self._power_to_amplitude(pump_power)
        self.detuning = pump_frequency - 2 * signal_frequency
        self.phase_bias = 0.0         # φ_bias - flux bias phase
        self.pump_phase = 0.0         # φ_pump - pump phase
        
        # Squeezing state parameters
        self.squeezing_params = SqueezingParameters(
            squeezing_strength=0.0,
            squeezing_angle=0.0,
            displacement=0.0+0j,
            frequency=signal_frequency
        )
        
        # Operating history
        self.operation_log = []
        self.squeezing_history = []
        
        print(f"⚡ Josephson Parametric Amplifier Initialized:")
        print(f"   • Signal frequency: {signal_frequency/1e9:.3f} GHz")
        print(f"   • Pump frequency: {pump_frequency/1e9:.3f} GHz")
        print(f"   • Josephson energy: EJ = {josephson_energy/1.602e-25:.1f} μeV")
        print(f"   • Charging energy: EC = {charging_energy/1.602e-25:.1f} μeV")
        print(f"   • Plasma frequency: {self.plasma_frequency/1e9:.3f} GHz")
        print(f"   • Anharmonicity: α = {self.anharmonicity/2/np.pi/1e6:.1f} MHz")
        print(f"   • Temperature: {temperature*1000:.1f} mK")
        print(f"   • Thermal photons: {self.thermal_photons:.3f}")
    
    def _calculate_thermal_photons(self) -> float:
        """Calculate thermal photon occupation."""
        if self.temperature == 0:
            return 0.0
        
        beta = 1.0 / (self.kb * self.temperature)
        n_thermal = 1.0 / (np.exp(beta * self.hbar * self.signal_frequency) - 1.0)
        return n_thermal
    
    def _power_to_amplitude(self, power_dbm: float) -> float:
        """Convert pump power [dBm] to normalized amplitude."""
        # Convert dBm to Watts
        power_watts = 10**(power_dbm/10) * 1e-3
        
        # Convert to photon number (simplified model)
        photon_energy = self.hbar * self.pump_frequency
        pump_photons = power_watts / photon_energy
        
        # Normalize by junction critical current equivalent
        # This is a simplified model - real calculation requires circuit details
        normalized_amplitude = np.sqrt(pump_photons / 1e6)  # Normalize by typical scale
        
        return normalized_amplitude
    
    def configure_squeezing(self,
                          target_squeezing_db: float = 10.0,
                          squeezing_angle: float = 0.0,
                          optimize_pump: bool = True) -> Dict[str, float]:
        """
        Configure JPA for optimal squeezing generation.
        
        Args:
            target_squeezing_db: Target squeezing in dB (S = 10 log₁₀(variance_reduction))
            squeezing_angle: Squeezing angle [rad]
            optimize_pump: Whether to optimize pump parameters
            
        Returns:
            Configuration results
        """
        # Convert squeezing dB to squeezing parameter r
        squeezing_db_linear = 10**(target_squeezing_db/10)
        target_squeezing_r = 0.5 * np.log(squeezing_db_linear)
        
        if optimize_pump:
            # Optimize pump amplitude for target squeezing
            optimal_pump = self._optimize_pump_for_squeezing(target_squeezing_r)
            self.pump_amplitude = optimal_pump['amplitude']
            self.pump_phase = optimal_pump['phase']
        
        # Calculate achievable squeezing with current parameters
        achieved_squeezing_r = self._calculate_squeezing_strength()
        achieved_squeezing_db = 10 * np.log10(np.exp(2 * achieved_squeezing_r))
        
        # Update squeezing parameters
        self.squeezing_params.squeezing_strength = achieved_squeezing_r
        self.squeezing_params.squeezing_angle = squeezing_angle
        
        # Calculate negative energy density
        negative_energy_density = self._calculate_negative_energy()
        
        config_result = {
            'target_squeezing_db': target_squeezing_db,
            'achieved_squeezing_db': achieved_squeezing_db,
            'squeezing_parameter_r': achieved_squeezing_r,
            'squeezing_angle': squeezing_angle,
            'pump_amplitude': self.pump_amplitude,
            'pump_phase': self.pump_phase,
            'variance_reduction': self.squeezing_params.variance_reduction,
            'variance_increase': self.squeezing_params.variance_increase,
            'negative_energy_density': negative_energy_density,
            'mu_coefficient': abs(self.squeezing_params.mu),
            'nu_coefficient': abs(self.squeezing_params.nu),
            'timestamp': datetime.now().isoformat()
        }
        
        self.operation_log.append({
            'action': 'configure_squeezing',
            'parameters': config_result
        })
        
        print(f"🎯 JPA Squeezing Configuration:")
        print(f"   • Target squeezing: {target_squeezing_db:.1f} dB")
        print(f"   • Achieved squeezing: {achieved_squeezing_db:.1f} dB")
        print(f"   • Squeezing parameter: r = {achieved_squeezing_r:.3f}")
        print(f"   • Variance reduction: {self.squeezing_params.variance_reduction:.3f}")
        print(f"   • Negative energy density: {negative_energy_density:.2e} J/m³")
        print(f"   • μ coefficient: {abs(self.squeezing_params.mu):.3f}")
        print(f"   • ν coefficient: {abs(self.squeezing_params.nu):.3f}")
        
        return config_result
    
    def _optimize_pump_for_squeezing(self, target_r: float) -> Dict[str, float]:
        """Optimize pump parameters for target squeezing strength."""
        
        def squeezing_objective(pump_params):
            """Objective function for pump optimization."""
            pump_amp, pump_phase = pump_params
            
            # Set temporary parameters
            old_amp = self.pump_amplitude
            old_phase = self.pump_phase
            
            self.pump_amplitude = pump_amp
            self.pump_phase = pump_phase
            
            # Calculate squeezing
            achieved_r = self._calculate_squeezing_strength()
            
            # Restore parameters
            self.pump_amplitude = old_amp
            self.pump_phase = old_phase
            
            # Return squared error
            return (achieved_r - target_r)**2
        
        # Optimization bounds
        amp_bounds = (0.0, 2.0 * self.pump_amplitude)
        phase_bounds = (0.0, 2*np.pi)
        
        if SCIPY_AVAILABLE:
            result = scipy.optimize.minimize(
                squeezing_objective,
                x0=[self.pump_amplitude, self.pump_phase],
                bounds=[amp_bounds, phase_bounds],
                method='L-BFGS-B'
            )
            
            optimal_amp = result.x[0]
            optimal_phase = result.x[1]
            success = result.success
        else:
            # Simple grid search if scipy not available
            amp_values = np.linspace(amp_bounds[0], amp_bounds[1], 20)
            phase_values = np.linspace(phase_bounds[0], phase_bounds[1], 20)
            
            best_error = np.inf
            optimal_amp = self.pump_amplitude
            optimal_phase = self.pump_phase
            
            for amp in amp_values:
                for phase in phase_values:
                    error = squeezing_objective([amp, phase])
                    if error < best_error:
                        best_error = error
                        optimal_amp = amp
                        optimal_phase = phase
            
            success = True
        
        return {
            'amplitude': optimal_amp,
            'phase': optimal_phase,
            'optimization_success': success
        }
    
    def _calculate_squeezing_strength(self) -> float:
        """
        Calculate squeezing strength from JPA parameters.
        
        For a driven Josephson junction:
        r ≈ g₂|α_p|/4Δ where g₂ is the nonlinearity, α_p is pump amplitude, Δ is detuning
        """
        # Nonlinear coupling strength (simplified model)
        g2 = self.anharmonicity  # Second-order nonlinearity from charging energy
        
        # Effective pump amplitude at the junction
        alpha_pump = self.pump_amplitude * np.exp(1j * self.pump_phase)
        
        # Detuning from parametric resonance
        effective_detuning = abs(self.detuning) + self.anharmonicity  # Include nonlinear shift
        
        if effective_detuning == 0:
            effective_detuning = abs(self.anharmonicity) / 10  # Avoid division by zero
        
        # Squeezing parameter (simplified formula)
        squeezing_r = abs(g2) * abs(alpha_pump) / (4 * abs(effective_detuning))
        
        # Limit maximum squeezing (physical constraints)
        max_squeezing = 2.0  # ~17 dB maximum
        squeezing_r = min(squeezing_r, max_squeezing)
        
        # Include thermal degradation
        thermal_factor = 1.0 / (1.0 + self.thermal_photons)
        squeezing_r *= thermal_factor
        
        return squeezing_r
    
    def _calculate_negative_energy(self) -> float:
        """Calculate negative energy density from squeezed vacuum."""
        # For squeezed vacuum: ⟨T₀₀⟩ = -ℏω|ν|²/(2V)
        # This is the negative energy contribution
        
        nu_squared = abs(self.squeezing_params.nu)**2
        
        # Assume unit volume for density calculation
        volume = 1.0  # m³
        
        negative_energy_density = -(self.hbar * self.signal_frequency * nu_squared) / (2 * volume)
        
        return negative_energy_density
    
    def generate_squeezed_vacuum(self, 
                               duration: float = 1e-3,
                               sampling_rate: float = 1e9) -> Dict[str, Union[np.ndarray, float]]:
        """
        Generate squeezed vacuum state measurements.
        
        Args:
            duration: Measurement duration [s]
            sampling_rate: Sampling rate [Hz]
            
        Returns:
            Dictionary with squeezed vacuum properties
        """
        n_samples = int(duration * sampling_rate)
        time_array = np.linspace(0, duration, n_samples)
        
        # Generate squeezed vacuum quadratures
        # Start with vacuum fluctuations
        X_vacuum = np.random.normal(0, 1, n_samples)  # Position quadrature
        P_vacuum = np.random.normal(0, 1, n_samples)  # Momentum quadrature
        
        # Apply squeezing transformation
        r = self.squeezing_params.squeezing_strength
        phi = self.squeezing_params.squeezing_angle
        
        # Squeezing matrix
        S11 = np.exp(-r) * np.cos(phi)**2 + np.exp(r) * np.sin(phi)**2
        S12 = (np.exp(r) - np.exp(-r)) * np.sin(phi) * np.cos(phi)
        S21 = S12
        S22 = np.exp(r) * np.cos(phi)**2 + np.exp(-r) * np.sin(phi)**2
        
        # Apply squeezing
        X_squeezed = S11 * X_vacuum + S12 * P_vacuum
        P_squeezed = S21 * X_vacuum + S22 * P_vacuum
        
        # Add thermal noise
        thermal_noise_X = np.random.normal(0, np.sqrt(self.thermal_photons), n_samples)
        thermal_noise_P = np.random.normal(0, np.sqrt(self.thermal_photons), n_samples)
        
        X_total = X_squeezed + thermal_noise_X
        P_total = P_squeezed + thermal_noise_P
        
        # Calculate field amplitudes and phases
        field_amplitude = np.sqrt(X_total**2 + P_total**2)
        field_phase = np.arctan2(P_total, X_total)
        
        # Calculate instantaneous energy density
        energy_density = self._calculate_energy_density_from_quadratures(X_total, P_total)
        
        # Statistical analysis
        X_variance = np.var(X_total)
        P_variance = np.var(P_total)
        squeezing_measured = -10 * np.log10(min(X_variance, P_variance))  # Measured squeezing in dB
        
        # Energy properties
        mean_energy = np.mean(energy_density)
        negative_energy_fraction = np.sum(energy_density < 0) / len(energy_density)
        
        generation_result = {
            'time_array': time_array,
            'X_quadrature': X_total,
            'P_quadrature': P_total,
            'field_amplitude': field_amplitude,
            'field_phase': field_phase,
            'energy_density': energy_density,
            'X_variance': X_variance,
            'P_variance': P_variance,
            'squeezing_measured_db': squeezing_measured,
            'mean_energy': mean_energy,
            'negative_energy_fraction': negative_energy_fraction,
            'theoretical_squeezing_db': 10 * np.log10(self.squeezing_params.variance_reduction),
            'duration': duration,
            'sampling_rate': sampling_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        self.squeezing_history.append(generation_result)
        
        print(f"🌌 Squeezed Vacuum Generated:")
        print(f"   • Duration: {duration*1000:.1f} ms")
        print(f"   • X variance: {X_variance:.3f}")
        print(f"   • P variance: {P_variance:.3f}")
        print(f"   • Measured squeezing: {squeezing_measured:.1f} dB")
        print(f"   • Mean energy: {mean_energy:.3e} J/m³")
        print(f"   • Negative energy fraction: {negative_energy_fraction:.3f}")
        
        return generation_result
    
    def _calculate_energy_density_from_quadratures(self, X: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Calculate energy density from field quadratures."""
        # Energy density for quantum field
        # ρ = ℏω(X² + P²)/2 - ⟨vacuum⟩
        
        vacuum_energy_density = 0.5 * self.hbar * self.signal_frequency
        
        # Total energy density
        energy_density = 0.5 * self.hbar * self.signal_frequency * (X**2 + P**2) - vacuum_energy_density
        
        return energy_density
    
    def perform_homodyne_measurement(self, 
                                   measurement_angle: float = 0.0,
                                   integration_time: float = 1e-6) -> Dict[str, float]:
        """
        Perform homodyne measurement to verify squeezing.
        
        Args:
            measurement_angle: LO phase angle [rad]
            integration_time: Integration time [s]
            
        Returns:
            Homodyne measurement results
        """
        # Generate short measurement for analysis
        measurement_data = self.generate_squeezed_vacuum(
            duration=integration_time,
            sampling_rate=1e9
        )
        
        # Rotate quadratures to measurement angle
        X = measurement_data['X_quadrature']
        P = measurement_data['P_quadrature']
        
        # Measured quadrature
        X_measured = X * np.cos(measurement_angle) + P * np.sin(measurement_angle)
        P_measured = -X * np.sin(measurement_angle) + P * np.cos(measurement_angle)
        
        # Calculate variances
        variance_measured = np.var(X_measured)
        variance_orthogonal = np.var(P_measured)
        
        # Squeezing calculation
        squeezing_db = -10 * np.log10(variance_measured)
        antisqueezing_db = 10 * np.log10(variance_orthogonal)
        
        homodyne_result = {
            'measurement_angle': measurement_angle,
            'integration_time': integration_time,
            'variance_measured': variance_measured,
            'variance_orthogonal': variance_orthogonal,
            'squeezing_db': squeezing_db,
            'antisqueezing_db': antisqueezing_db,
            'uncertainty_product': variance_measured * variance_orthogonal,
            'heisenberg_limit': 1.0,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"🔬 Homodyne Measurement Results:")
        print(f"   • Measurement angle: {measurement_angle:.3f} rad")
        print(f"   • Squeezing: {squeezing_db:.1f} dB")
        print(f"   • Anti-squeezing: {antisqueezing_db:.1f} dB")
        print(f"   • Uncertainty product: {variance_measured * variance_orthogonal:.3f}")
        
        return homodyne_result
    
    def run_squeezed_vacuum_protocol(self, protocol_params: Dict) -> Dict[str, any]:
        """
        Run complete squeezed vacuum generation and characterization protocol.
        
        Args:
            protocol_params: Protocol configuration parameters
            
        Returns:
            Complete protocol results
        """
        print(f"🤖 Starting JPA Squeezed Vacuum Protocol")
        
        protocol_results = {
            'start_time': datetime.now().isoformat(),
            'protocol_params': protocol_params,
            'steps': [],
            'summary': {}
        }
        
        # Step 1: Configure squeezing
        print("\n📍 Step 1: Configure Squeezing Parameters")
        squeezing_config = self.configure_squeezing(
            target_squeezing_db=protocol_params.get('target_squeezing_db', 10.0),
            squeezing_angle=protocol_params.get('squeezing_angle', 0.0),
            optimize_pump=protocol_params.get('optimize_pump', True)
        )
        protocol_results['steps'].append({
            'step': 'squeezing_configuration',
            'results': squeezing_config
        })
        
        # Step 2: Generate squeezed vacuum
        print("\n📍 Step 2: Generate Squeezed Vacuum")
        vacuum_generation = self.generate_squeezed_vacuum(
            duration=protocol_params.get('generation_time', 1e-3),
            sampling_rate=protocol_params.get('sampling_rate', 1e9)
        )
        protocol_results['steps'].append({
            'step': 'vacuum_generation',
            'results': vacuum_generation
        })
        
        # Step 3: Homodyne verification
        print("\n📍 Step 3: Homodyne Verification")
        homodyne_results = []
        
        # Measure at multiple angles to map full squeezing ellipse
        angles = np.linspace(0, np.pi, protocol_params.get('n_angles', 8))
        for angle in angles:
            homodyne_result = self.perform_homodyne_measurement(
                measurement_angle=angle,
                integration_time=protocol_params.get('integration_time', 1e-6)
            )
            homodyne_results.append(homodyne_result)
        
        protocol_results['steps'].append({
            'step': 'homodyne_verification',
            'results': homodyne_results
        })
        
        # Step 4: Analysis and summary
        print("\n📍 Step 4: Protocol Analysis")
        
        # Extract key metrics
        achieved_squeezing = squeezing_config['achieved_squeezing_db']
        negative_energy_density = squeezing_config['negative_energy_density']
        negative_energy_fraction = vacuum_generation['negative_energy_fraction']
        
        # Find maximum squeezing from homodyne measurements
        max_squeezing = max([h['squeezing_db'] for h in homodyne_results])
        min_antisqueezing = min([h['antisqueezing_db'] for h in homodyne_results])
        
        # Performance metrics
        squeezing_efficiency = max_squeezing / protocol_params.get('target_squeezing_db', 10.0)
        energy_extraction_rate = abs(negative_energy_density) * protocol_params.get('generation_time', 1e-3)
        
        protocol_results['summary'] = {
            'achieved_squeezing_db': achieved_squeezing,
            'max_measured_squeezing_db': max_squeezing,
            'negative_energy_density': negative_energy_density,
            'negative_energy_fraction': negative_energy_fraction,
            'squeezing_efficiency': squeezing_efficiency,
            'energy_extraction_rate': energy_extraction_rate,
            'protocol_success': squeezing_efficiency > 0.8,
            'recommendations': self._generate_protocol_recommendations(protocol_results)
        }
        
        protocol_results['end_time'] = datetime.now().isoformat()
        
        print(f"\n✅ JPA Protocol Complete!")
        print(f"   • Achieved squeezing: {achieved_squeezing:.1f} dB")
        print(f"   • Maximum measured: {max_squeezing:.1f} dB")
        print(f"   • Negative energy density: {negative_energy_density:.2e} J/m³")
        print(f"   • Efficiency: {squeezing_efficiency:.1%}")
        
        return protocol_results
    
    def _generate_protocol_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on protocol results."""
        recommendations = []
        
        summary = results['summary']
        config = results['steps'][0]['results']
        
        if summary['squeezing_efficiency'] < 0.8:
            recommendations.append("Optimize pump power and phase for better squeezing")
        
        if config['achieved_squeezing_db'] < 5.0:
            recommendations.append("Increase Josephson energy or reduce charging energy")
        
        if summary['negative_energy_fraction'] < 0.3:
            recommendations.append("Check for thermal noise and environmental coupling")
        
        if self.temperature > 0.02:
            recommendations.append("Lower operating temperature for improved coherence")
        
        if abs(self.detuning) > self.signal_frequency / 10:
            recommendations.append("Adjust pump frequency closer to 2ω_s for optimal parametric gain")
        
        if len(recommendations) == 0:
            recommendations.append("Protocol operating optimally - proceed with experiments")
        
        return recommendations
    
    def plot_squeezing_results(self, save_path: Optional[str] = None):
        """Plot recent squeezing measurement results."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("matplotlib not available for plotting")
            return
        
        if not self.squeezing_history:
            warnings.warn("No squeezing data to plot")
            return
        
        # Use most recent measurement
        data = self.squeezing_history[-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Quadrature time series
        axes[0, 0].plot(data['time_array']*1e6, data['X_quadrature'], 'b-', alpha=0.7, label='X quadrature')
        axes[0, 0].plot(data['time_array']*1e6, data['P_quadrature'], 'r-', alpha=0.7, label='P quadrature')
        axes[0, 0].set_xlabel('Time [μs]')
        axes[0, 0].set_ylabel('Quadrature Amplitude')
        axes[0, 0].set_title('Squeezed Vacuum Quadratures')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phase space plot
        axes[0, 1].scatter(data['X_quadrature'], data['P_quadrature'], 
                          alpha=0.3, s=1, c=data['time_array'], cmap='viridis')
        axes[0, 1].set_xlabel('X Quadrature')
        axes[0, 1].set_ylabel('P Quadrature')
        axes[0, 1].set_title('Phase Space Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axis('equal')
        
        # Energy density time series
        axes[1, 0].plot(data['time_array']*1e6, data['energy_density'], 'purple', linewidth=1)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Time [μs]')
        axes[1, 0].set_ylabel('Energy Density [J/m³]')
        axes[1, 0].set_title('Energy Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Variance analysis
        x_var = np.var(data['X_quadrature'])
        p_var = np.var(data['P_quadrature'])
        
        angles = np.linspace(0, 2*np.pi, 100)
        variances = []
        for angle in angles:
            rotated_quad = (data['X_quadrature'] * np.cos(angle) + 
                           data['P_quadrature'] * np.sin(angle))
            variances.append(np.var(rotated_quad))
        
        axes[1, 1].plot(angles, variances, 'g-', linewidth=2)
        axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Vacuum limit')
        axes[1, 1].set_xlabel('Measurement Angle [rad]')
        axes[1, 1].set_ylabel('Quadrature Variance')
        axes[1, 1].set_title('Squeezing vs. Measurement Angle')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Squeezing plot saved to {save_path}")
        else:
            plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("=== Josephson Parametric Amplifier Squeezed Vacuum System ===")
    
    # Initialize JPA
    jpa = JosephsonParametricAmplifier(
        signal_frequency=6.0e9,      # 6 GHz signal
        pump_frequency=12.0e9,       # 12 GHz pump
        josephson_energy=20e-24,     # 20 μeV
        charging_energy=1e-24,       # 1 μeV
        pump_power=-100,             # -100 dBm
        temperature=0.01             # 10 mK
    )
    
    # Protocol parameters
    protocol_params = {
        'target_squeezing_db': 12.0,     # Target 12 dB squeezing
        'squeezing_angle': 0.0,          # Squeeze X quadrature
        'optimize_pump': True,           # Optimize pump parameters
        'generation_time': 1e-3,         # 1 ms generation
        'sampling_rate': 1e9,            # 1 GS/s
        'n_angles': 16,                  # Homodyne angles
        'integration_time': 1e-6         # 1 μs integration
    }
    
    # Run protocol
    results = jpa.run_squeezed_vacuum_protocol(protocol_params)
    
    # Save results
    output_file = "jpa_squeezed_vacuum_results.json"
    
    # Convert numpy arrays to lists for JSON
    json_results = results.copy()
    for step in json_results['steps']:
        if 'results' in step:
            step_results = step['results']
            if isinstance(step_results, list):
                # Homodyne results
                continue
            elif isinstance(step_results, dict):
                # Remove large arrays
                step['results'] = {k: v for k, v in step_results.items() 
                                 if not isinstance(v, np.ndarray)}
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Plot results
    try:
        jpa.plot_squeezing_results("jpa_squeezing_measurement.png")
    except:
        print("   (Plotting not available)")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Integrate with flux bias control for frequency tuning")
    print("2. Add real-time feedback for pump optimization")
    print("3. Implement multi-mode squeezing for broadband operation")
    print("4. Connect to quantum error correction protocols")
    print("5. Scale up for continuous squeezed vacuum generation")
