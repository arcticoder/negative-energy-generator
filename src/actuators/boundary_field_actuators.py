"""
Actuator Interface for Real-Time Boundary Field Control
======================================================

Translates control commands into physical actuator signals for:
1. High-voltage boundary field modulators
2. High-current magnetic coil drivers  
3. Laser intensity/phase modulators
4. Polymer insert field manipulators

Mathematical Foundation:
- Control command u → Physical actuator command
- Transfer functions: G_actuator(s) = output/input
- Bandwidth limitations and dynamics
- Safety interlocks and protection

Actuator Types:
1. Voltage Modulators: V_out = G_v(s) * u_v(t)
2. Current Drivers: I_out = G_i(s) * u_i(t)  
3. Laser Modulators: P_laser = G_l(s) * u_l(t)
4. Field Shapers: E_field = G_f(s) * u_f(t)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# Actuator specifications and limits
VOLTAGE_MODULATOR_MAX = 1e6      # 1 MV maximum output
CURRENT_DRIVER_MAX = 1000        # 1 kA maximum output  
LASER_POWER_MAX = 1e15          # 1 PW maximum peak power
FIELD_SHAPER_MAX = 1e14         # 100 TV/m maximum field

# Bandwidth specifications (Hz)
VOLTAGE_MODULATOR_BW = 1e9      # 1 GHz voltage modulator
CURRENT_DRIVER_BW = 1e8         # 100 MHz current driver
LASER_MODULATOR_BW = 1e12       # 1 THz laser modulator
FIELD_SHAPER_BW = 1e10          # 10 GHz field shaper

# Safety margins
SAFETY_MARGIN = 0.9             # 90% of maximum ratings
SLEW_RATE_LIMIT = 1e15          # Maximum rate of change (units/s)

class ActuatorBase:
    """
    Base class for all actuator types.
    
    Provides common functionality:
    - Transfer function modeling
    - Bandwidth limitations
    - Safety interlocks
    - Command filtering
    """
    
    def __init__(self, max_output: float, bandwidth: float, 
                 name: str, transfer_function: Optional[signal.TransferFunction] = None):
        """
        Initialize base actuator.
        
        Args:
            max_output: Maximum actuator output
            bandwidth: 3dB bandwidth (Hz)
            name: Actuator identifier
            transfer_function: Custom transfer function (default: first-order)
        """
        self.max_output = max_output
        self.bandwidth = bandwidth
        self.name = name
        self.safe_max = max_output * SAFETY_MARGIN
        
        # Default first-order transfer function: G(s) = K / (τs + 1)
        if transfer_function is None:
            tau = 1.0 / (2 * np.pi * bandwidth)  # Time constant
            self.tf = signal.TransferFunction([1], [tau, 1])
        else:
            self.tf = transfer_function
        
        # State tracking
        self.current_output = 0.0
        self.command_history = []
        self.output_history = []
        self.safety_violations = 0
        self.total_commands = 0
        
    def apply_command(self, command: float, dt: float) -> float:
        """
        Apply command with transfer function dynamics and safety checks.
        
        Args:
            command: Desired actuator command
            dt: Time step (seconds)
            
        Returns:
            Actual actuator output
        """
        self.total_commands += 1
        
        # Safety checks
        if abs(command) > self.safe_max:
            self.safety_violations += 1
            command = np.sign(command) * self.safe_max
        
        # Slew rate limiting
        if len(self.output_history) > 0:
            max_delta = SLEW_RATE_LIMIT * dt
            prev_output = self.output_history[-1]
            if abs(command - prev_output) > max_delta:
                command = prev_output + np.sign(command - prev_output) * max_delta
        
        # Apply transfer function (simplified first-order response)
        tau = 1.0 / (2 * np.pi * self.bandwidth)
        alpha = dt / (tau + dt)  # Digital filter coefficient
        
        # Update output with dynamics
        self.current_output = (1 - alpha) * self.current_output + alpha * command
        
        # Log data
        self.command_history.append(command)
        self.output_history.append(self.current_output)
        
        return self.current_output
    
    def get_frequency_response(self, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get magnitude and phase response."""
        w, h = signal.freqresp(self.tf, 2*np.pi*frequencies)
        magnitude_db = 20 * np.log10(np.abs(h))
        phase_deg = np.degrees(np.angle(h))
        return magnitude_db, phase_deg
    
    def get_statistics(self) -> Dict:
        """Get actuator performance statistics."""
        if self.total_commands == 0:
            return {'status': 'No commands received'}
        
        return {
            'total_commands': self.total_commands,
            'safety_violations': self.safety_violations,
            'safety_violation_rate': self.safety_violations / self.total_commands,
            'current_output': self.current_output,
            'max_output_reached': max(np.abs(self.output_history)) if self.output_history else 0,
            'average_output': np.mean(np.abs(self.output_history)) if self.output_history else 0,
            'utilization_rate': (max(np.abs(self.output_history)) / self.safe_max 
                               if self.output_history else 0)
        }

class VoltageModulator(ActuatorBase):
    """
    High-voltage boundary field modulator.
    
    Mathematical Model:
    V_out(t) = G_v(s) * u_v(t)
    G_v(s) = K_v / (τ_v s + 1)
    
    Physical Implementation:
    - High-voltage amplifier with capacitive load
    - Bandwidth limited by RC time constant
    - Arcing protection and voltage monitoring
    """
    
    def __init__(self, max_voltage: float = VOLTAGE_MODULATOR_MAX, 
                 bandwidth: float = VOLTAGE_MODULATOR_BW):
        """Initialize voltage modulator."""
        # Second-order transfer function for voltage amplifier
        # G(s) = K / (s²/ωn² + 2ζs/ωn + 1)
        wn = 2 * np.pi * bandwidth  # Natural frequency
        zeta = 0.7  # Damping ratio (slightly underdamped)
        
        tf = signal.TransferFunction([wn**2], [1, 2*zeta*wn, wn**2])
        
        super().__init__(max_voltage, bandwidth, "Voltage_Modulator", tf)
        
        # Voltage-specific parameters
        self.capacitance = 1e-9  # 1 nF load capacitance
        self.resistance = 1e6    # 1 MΩ load resistance
        self.rc_constant = self.resistance * self.capacitance
        
    def apply_voltage_command(self, voltage_command: float, dt: float) -> float:
        """
        Apply voltage command with electrical dynamics.
        
        Args:
            voltage_command: Desired output voltage (V)
            dt: Time step (seconds)
            
        Returns:
            Actual output voltage (V)
        """
        # Additional safety check for breakdown
        breakdown_voltage = 0.8 * self.max_output  # Conservative breakdown limit
        if abs(voltage_command) > breakdown_voltage:
            print(f"⚠️  {self.name}: Breakdown risk at {voltage_command/1e6:.1f} MV")
            voltage_command = np.sign(voltage_command) * breakdown_voltage
        
        return self.apply_command(voltage_command, dt)

class CurrentDriver(ActuatorBase):
    """
    High-current magnetic coil driver.
    
    Mathematical Model:
    I_out(t) = G_i(s) * u_i(t)
    G_i(s) = K_i / (L/R s + 1)
    
    Physical Implementation:
    - High-current switching amplifier
    - Inductive load (magnetic coils)
    - Current feedback control
    """
    
    def __init__(self, max_current: float = CURRENT_DRIVER_MAX,
                 bandwidth: float = CURRENT_DRIVER_BW):
        """Initialize current driver."""
        # First-order transfer function for inductive load
        # τ = L/R where L is inductance, R is resistance
        
        super().__init__(max_current, bandwidth, "Current_Driver")
        
        # Current driver specific parameters
        self.inductance = 1e-6      # 1 μH coil inductance
        self.resistance = 1e-3      # 1 mΩ coil resistance
        self.l_over_r = self.inductance / self.resistance
        
    def apply_current_command(self, current_command: float, dt: float) -> float:
        """
        Apply current command with magnetic dynamics.
        
        Args:
            current_command: Desired output current (A)
            dt: Time step (seconds)
            
        Returns:
            Actual output current (A)
        """
        # Thermal protection
        power_dissipated = (self.current_output**2) * self.resistance
        max_thermal_power = 1e6  # 1 MW thermal limit
        
        if power_dissipated > max_thermal_power:
            thermal_limit = np.sqrt(max_thermal_power / self.resistance)
            current_command = np.sign(current_command) * min(abs(current_command), thermal_limit)
        
        return self.apply_command(current_command, dt)

class LaserModulator(ActuatorBase):
    """
    High-intensity laser modulator.
    
    Mathematical Model:
    P_laser(t) = G_l(s) * u_l(t)
    G_l(s) = K_l / (τ_l s + 1)
    
    Physical Implementation:
    - Electro-optic modulator or pump diode
    - Optical amplifier chain
    - Thermal management system
    """
    
    def __init__(self, max_power: float = LASER_POWER_MAX,
                 bandwidth: float = LASER_MODULATOR_BW):
        """Initialize laser modulator."""
        super().__init__(max_power, bandwidth, "Laser_Modulator")
        
        # Laser-specific parameters
        self.wavelength = 1.55e-6    # 1550 nm operating wavelength
        self.beam_area = 1e-6        # 1 mm² beam area
        self.max_intensity = max_power / self.beam_area  # W/m²
        
    def apply_laser_command(self, power_command: float, dt: float) -> float:
        """
        Apply laser power command with optical dynamics.
        
        Args:
            power_command: Desired laser power (W)
            dt: Time step (seconds)
            
        Returns:
            Actual laser power (W)
        """
        # Optical damage threshold check
        intensity = power_command / self.beam_area
        damage_threshold = 1e16  # W/m² (conservative)
        
        if intensity > damage_threshold:
            print(f"⚠️  {self.name}: Optical damage risk at {intensity/1e16:.1f} × threshold")
            power_command = damage_threshold * self.beam_area
        
        return self.apply_command(power_command, dt)

class FieldShaper(ActuatorBase):
    """
    Electromagnetic field shaper for polymer inserts.
    
    Mathematical Model:
    E_field(t) = G_f(s) * u_f(t)
    G_f(s) = K_f * exp(-s*τ_delay) / (τ_f s + 1)
    
    Physical Implementation:
    - Multi-electrode array
    - Field focusing optics
    - Real-time field mapping
    """
    
    def __init__(self, max_field: float = FIELD_SHAPER_MAX,
                 bandwidth: float = FIELD_SHAPER_BW):
        """Initialize field shaper."""
        super().__init__(max_field, bandwidth, "Field_Shaper")
        
        # Field shaper parameters
        self.electrode_spacing = 1e-6    # 1 μm electrode spacing
        self.dielectric_constant = 5.0   # Relative permittivity
        self.breakdown_field = 1e14      # V/m breakdown strength
        
    def apply_field_command(self, field_command: float, dt: float) -> float:
        """
        Apply field shaping command with electrostatic dynamics.
        
        Args:
            field_command: Desired electric field (V/m)
            dt: Time step (seconds)
            
        Returns:
            Actual electric field (V/m)
        """
        # Dielectric breakdown protection
        if abs(field_command) > self.breakdown_field:
            print(f"⚠️  {self.name}: Breakdown risk at {field_command/1e14:.1f} × 10¹⁴ V/m")
            field_command = np.sign(field_command) * self.breakdown_field
        
        return self.apply_command(field_command, dt)

class ActuatorNetwork:
    """
    Manages multiple actuators as an integrated system.
    
    Provides:
    - Centralized command distribution
    - Inter-actuator coordination
    - System-level safety monitoring
    - Performance optimization
    """
    
    def __init__(self):
        """Initialize actuator network."""
        # Create actuator instances
        self.actuators = {
            'voltage_mod_1': VoltageModulator(),
            'voltage_mod_2': VoltageModulator(),
            'current_drv_1': CurrentDriver(),
            'laser_mod_1': LaserModulator(),
            'field_shaper_1': FieldShaper()
        }
        
        # Network state
        self.is_enabled = True
        self.emergency_stop = False
        self.coordination_matrix = self._build_coordination_matrix()
        
    def _build_coordination_matrix(self) -> np.ndarray:
        """Build inter-actuator coordination matrix."""
        n_actuators = len(self.actuators)
        
        # Simple coordination: prevent conflicting commands
        coord_matrix = np.eye(n_actuators)
        
        # Add coupling terms (example: voltage modulators should coordinate)
        coord_matrix[0, 1] = 0.1  # voltage_mod_1 affects voltage_mod_2
        coord_matrix[1, 0] = 0.1  # and vice versa
        
        return coord_matrix
    
    def apply_command_vector(self, command_vector: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply command vector to all actuators.
        
        Args:
            command_vector: Commands for each actuator
            dt: Time step (seconds)
            
        Returns:
            Actual actuator outputs
        """
        if self.emergency_stop:
            return np.zeros_like(command_vector)
        
        if not self.is_enabled:
            print("⚠️  Actuator network disabled")
            return np.zeros_like(command_vector)
        
        # Check command vector size
        if len(command_vector) != len(self.actuators):
            raise ValueError(f"Command vector size {len(command_vector)} != {len(self.actuators)}")
        
        # Apply coordination matrix
        coordinated_commands = self.coordination_matrix @ command_vector
        
        # Apply commands to individual actuators
        outputs = []
        actuator_names = list(self.actuators.keys())
        
        for i, (name, actuator) in enumerate(self.actuators.items()):
            command = coordinated_commands[i]
            
            # Apply type-specific command method
            if isinstance(actuator, VoltageModulator):
                output = actuator.apply_voltage_command(command, dt)
            elif isinstance(actuator, CurrentDriver):
                output = actuator.apply_current_command(command, dt)
            elif isinstance(actuator, LaserModulator):
                output = actuator.apply_laser_command(command, dt)
            elif isinstance(actuator, FieldShaper):
                output = actuator.apply_field_command(command, dt)
            else:
                output = actuator.apply_command(command, dt)
            
            outputs.append(output)
        
        return np.array(outputs)
    
    def get_network_status(self) -> Dict:
        """Get comprehensive network status."""
        status = {
            'network_enabled': self.is_enabled,
            'emergency_stop': self.emergency_stop,
            'actuator_count': len(self.actuators),
            'actuator_status': {}
        }
        
        # Individual actuator status
        total_violations = 0
        total_commands = 0
        
        for name, actuator in self.actuators.items():
            stats = actuator.get_statistics()
            status['actuator_status'][name] = stats
            
            if 'safety_violations' in stats:
                total_violations += stats['safety_violations']
            if 'total_commands' in stats:
                total_commands += stats['total_commands']
        
        # Network-level metrics
        status['network_safety_violation_rate'] = (total_violations / total_commands 
                                                  if total_commands > 0 else 0)
        status['network_total_commands'] = total_commands
        
        return status
    
    def emergency_shutdown(self):
        """Immediate emergency shutdown of all actuators."""
        print("🚨 EMERGENCY SHUTDOWN ACTIVATED")
        self.emergency_stop = True
        
        # Force all actuators to zero output
        for actuator in self.actuators.values():
            actuator.current_output = 0.0
    
    def reset_network(self):
        """Reset network to operational state."""
        print("🔄 Resetting actuator network")
        self.emergency_stop = False
        self.is_enabled = True
        
        # Reset individual actuators
        for actuator in self.actuators.values():
            actuator.current_output = 0.0
            actuator.command_history = []
            actuator.output_history = []
            actuator.safety_violations = 0
            actuator.total_commands = 0

def demonstrate_actuator_system():
    """Demonstrate actuator system with realistic commands."""
    print("🔧 ACTUATOR SYSTEM DEMONSTRATION")
    print("=" * 40)
    
    # Create actuator network
    network = ActuatorNetwork()
    
    print(f"📡 Network initialized with {len(network.actuators)} actuators:")
    for name, actuator in network.actuators.items():
        print(f"   • {name}: {actuator.name} (max: {actuator.max_output:.1e})")
    
    # Simulation parameters
    duration = 1e-6  # 1 microsecond
    dt = 1e-9       # 1 ns time step
    times = np.arange(0, duration, dt)
    n_steps = len(times)
    
    # Generate test command sequence
    print(f"\n⚡ Generating command sequence ({n_steps} steps)...")
    
    n_actuators = len(network.actuators)
    command_sequence = np.zeros((n_actuators, n_steps))
    
    # Create realistic command patterns
    for i in range(n_actuators):
        actuator_name = list(network.actuators.keys())[i]
        actuator = network.actuators[actuator_name]
        
        # Different patterns for different actuator types
        if isinstance(actuator, VoltageModulator):
            # Step + sinusoidal for voltage modulators
            amplitude = 0.5 * actuator.safe_max
            frequency = 1e6  # 1 MHz
            command_sequence[i, :] = amplitude * (
                0.5 + 0.5 * np.sin(2*np.pi*frequency*times)
            )
        elif isinstance(actuator, CurrentDriver):
            # Ramp + square wave for current drivers
            amplitude = 0.3 * actuator.safe_max
            period = duration / 4
            command_sequence[i, :] = amplitude * signal.square(2*np.pi*times/period)
        elif isinstance(actuator, LaserModulator):
            # Gaussian pulses for laser
            amplitude = 0.1 * actuator.safe_max
            pulse_width = 100e-9  # 100 ns pulses
            for pulse_time in [0.2e-6, 0.5e-6, 0.8e-6]:
                pulse_mask = np.abs(times - pulse_time) < pulse_width
                command_sequence[i, pulse_mask] += amplitude * np.exp(
                    -((times[pulse_mask] - pulse_time) / (pulse_width/4))**2
                )
        else:
            # Chirp signal for field shaper
            amplitude = 0.2 * actuator.safe_max
            f0, f1 = 1e6, 10e6  # 1-10 MHz chirp
            command_sequence[i, :] = amplitude * signal.chirp(times, f0, duration, f1)
    
    # Run simulation
    print("🔄 Running actuator simulation...")
    output_sequence = np.zeros((n_actuators, n_steps))
    
    for t in range(n_steps):
        commands = command_sequence[:, t]
        outputs = network.apply_command_vector(commands, dt)
        output_sequence[:, t] = outputs
    
    # Analyze results
    network_status = network.get_network_status()
    
    print(f"\n📊 SIMULATION RESULTS")
    print("=" * 25)
    print(f"   • Total commands processed: {network_status['network_total_commands']:,}")
    print(f"   • Network safety violation rate: {network_status['network_safety_violation_rate']:.1%}")
    print(f"   • Emergency stops: {'Yes' if network_status['emergency_stop'] else 'No'}")
    
    # Individual actuator performance
    print(f"\n🔧 Individual Actuator Performance:")
    for name, stats in network_status['actuator_status'].items():
        if 'utilization_rate' in stats:
            print(f"   • {name}: {stats['utilization_rate']:.1%} utilization, "
                  f"{stats['safety_violation_rate']:.1%} violations")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Command vs Output for first actuator
    ax1 = axes[0, 0]
    times_us = times * 1e6
    actuator_idx = 0
    ax1.plot(times_us, command_sequence[actuator_idx, :]/1e6, 'b-', 
             label='Command', linewidth=2)
    ax1.plot(times_us, output_sequence[actuator_idx, :]/1e6, 'r--', 
             label='Output', linewidth=2)
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Voltage (MV)')
    ax1.set_title('Voltage Modulator Response')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All actuator outputs (normalized)
    ax2 = axes[0, 1]
    actuator_names = list(network.actuators.keys())
    for i, name in enumerate(actuator_names):
        actuator = network.actuators[name]
        normalized_output = output_sequence[i, :] / actuator.safe_max
        ax2.plot(times_us, normalized_output, label=name.replace('_', ' ').title(), 
                linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Normalized Output')
    ax2.set_title('All Actuator Outputs (Normalized)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Frequency response comparison
    ax3 = axes[1, 0]
    frequencies = np.logspace(5, 12, 100)  # 100 kHz to 1 THz
    
    for name, actuator in network.actuators.items():
        mag_db, phase_deg = actuator.get_frequency_response(frequencies)
        ax3.semilogx(frequencies/1e9, mag_db, label=name.replace('_', ' ').title())
    
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_title('Actuator Frequency Response')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Utilization statistics
    ax4 = axes[1, 1]
    actuator_names = [name.replace('_', ' ').title() for name in network.actuators.keys()]
    utilizations = []
    violations = []
    
    for name, stats in network_status['actuator_status'].items():
        if 'utilization_rate' in stats:
            utilizations.append(stats['utilization_rate'])
            violations.append(stats['safety_violation_rate'])
        else:
            utilizations.append(0)
            violations.append(0)
    
    x_pos = np.arange(len(actuator_names))
    width = 0.35
    
    ax4.bar(x_pos - width/2, utilizations, width, label='Utilization Rate', alpha=0.8)
    ax4.bar(x_pos + width/2, violations, width, label='Violation Rate', alpha=0.8)
    
    ax4.set_xlabel('Actuator')
    ax4.set_ylabel('Rate')
    ax4.set_title('Actuator Performance Statistics')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(actuator_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('actuator_system_demonstration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Visualization saved to 'actuator_system_demonstration.png'")
    
    return {
        'network': network,
        'command_sequence': command_sequence,
        'output_sequence': output_sequence,
        'times': times,
        'network_status': network_status
    }

if __name__ == "__main__":
    print("🔧 Actuator Interface System")
    print("=" * 35)
    print("Translating control commands into physical actuator signals")
    
    # Run demonstration
    demo_results = demonstrate_actuator_system()
    
    print(f"\n✅ Actuator Interface System Ready!")
    print(f"   • {len(demo_results['network'].actuators)} actuators operational")
    print(f"   • Command processing rate: {CONTROL_FREQUENCY/1e9:.0f} GHz")
    print(f"   • Safety violation rate: {demo_results['network_status']['network_safety_violation_rate']:.1%}")
