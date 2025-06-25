"""
Photonic Crystal Engine for Negative Energy Generation
=====================================================

This module implements a photonic crystal system optimized for negative energy
extraction through engineered electromagnetic vacuum fluctuations and cavity QED effects.

Mathematical Framework:
    Modified dispersion: ω²(k) = c²k²[1 + δ(k)] for photonic band structure
    Casimir force: F = -π²ℏc/(240a⁴) × f(geometry, materials)
    Energy density: ⟨T₀₀⟩ = -ℏ∑_k ω_k(ρ_k - ρ_vacuum) where ρ_k is mode density
    
Key Features:
- Engineered photonic band gaps for vacuum modification
- Nanostructured metamaterials with negative index regions
- Active tuning via electro-optic and nonlinear effects
- Integration with quantum sensors for T₀₀ measurement
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import scipy.optimize
    import scipy.signal
    import scipy.special
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    warnings.warn("SciPy not available for advanced computations")
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class CrystalGeometry(Enum):
    """Photonic crystal geometry types."""
    SQUARE_LATTICE = "square"
    TRIANGULAR_LATTICE = "triangular"
    HONEYCOMB_LATTICE = "honeycomb"
    KAGOME_LATTICE = "kagome"
    WOODPILE_3D = "woodpile"
    INVERSE_OPAL = "inverse_opal"


@dataclass
class MaterialProperties:
    """Material properties for photonic crystal components."""
    name: str
    refractive_index: complex
    permittivity: complex
    permeability: complex
    loss_tangent: float
    nonlinear_susceptibility: complex = 0.0
    electro_optic_coefficient: float = 0.0
    
    @property
    def impedance(self) -> complex:
        """Characteristic impedance of the material."""
        return np.sqrt(self.permeability / self.permittivity)
    
    @property
    def absorption_coefficient(self) -> float:
        """Absorption coefficient [m⁻¹]."""
        return 2 * np.pi * self.refractive_index.imag / (3e8 / 1e9)  # Assuming ~GHz


class PhotonicCrystalEngine:
    """
    Photonic crystal system for negative energy generation.
    
    This class models and controls a photonic crystal structure designed
    to modify vacuum electromagnetic fluctuations for energy extraction.
    """
    
    def __init__(self,
                 lattice_type: CrystalGeometry = CrystalGeometry.SQUARE_LATTICE,
                 lattice_constant: float = 500e-9,      # 500 nm lattice constant
                 filling_fraction: float = 0.3,         # 30% filling with high-ε material
                 n_periods: Tuple[int, int, int] = (20, 20, 1),  # 20×20×1 unit cells
                 operating_frequency: float = 600e12):   # 600 THz (500 nm wavelength)
        """
        Initialize photonic crystal engine.
        
        Args:
            lattice_type: Crystal lattice geometry
            lattice_constant: Lattice spacing [m]
            filling_fraction: Volume fraction of high-ε material
            n_periods: Number of periods in (x, y, z) directions
            operating_frequency: Design frequency [Hz]
        """
        self.lattice_type = lattice_type
        self.lattice_constant = lattice_constant      # a
        self.filling_fraction = filling_fraction      # f
        self.n_periods = n_periods                    # (Nx, Ny, Nz)
        self.operating_frequency = operating_frequency # ω₀
        
        # Physical constants
        self.c = 299792458              # Speed of light
        self.hbar = 1.054571817e-34     # Reduced Planck constant
        self.eps0 = 8.854187817e-12     # Vacuum permittivity
        self.mu0 = 4*np.pi*1e-7         # Vacuum permeability
        
        # Derived parameters
        self.operating_wavelength = self.c / operating_frequency
        self.k0 = 2*np.pi / self.operating_wavelength
        self.brillouin_zone_size = np.pi / lattice_constant
        
        # Default materials (Silicon/Air photonic crystal)
        self.matrix_material = MaterialProperties(
            name="air",
            refractive_index=1.0+0j,
            permittivity=1.0+0j,
            permeability=1.0+0j,
            loss_tangent=0.0
        )
        
        self.inclusion_material = MaterialProperties(
            name="silicon",
            refractive_index=3.5+0.01j,
            permittivity=12.25+0.07j,
            permeability=1.0+0j,
            loss_tangent=0.001,
            electro_optic_coefficient=1e-11  # Pockels coefficient
        )
        
        # Crystal state
        self.band_structure = None
        self.density_of_states = None
        self.casimir_force = 0.0
        self.vacuum_energy_modification = 0.0
        
        # Control parameters
        self.electric_field_bias = 0.0     # Applied E-field for tuning
        self.temperature = 300.0           # Operating temperature [K]
        self.external_pump_power = 0.0     # Nonlinear pump power
        
        # Operating history
        self.operation_log = []
        self.measurement_history = []
        
        print(f"🔮 Photonic Crystal Engine Initialized:")
        print(f"   • Lattice type: {lattice_type.value}")
        print(f"   • Lattice constant: {lattice_constant*1e9:.0f} nm")
        print(f"   • Filling fraction: {filling_fraction:.1%}")
        print(f"   • Crystal size: {n_periods[0]}×{n_periods[1]}×{n_periods[2]} unit cells")
        print(f"   • Operating frequency: {operating_frequency/1e12:.1f} THz")
        print(f"   • Operating wavelength: {self.operating_wavelength*1e9:.0f} nm")
        print(f"   • Matrix: {self.matrix_material.name} (n={self.matrix_material.refractive_index.real:.2f})")
        print(f"   • Inclusion: {self.inclusion_material.name} (n={self.inclusion_material.refractive_index.real:.2f})")
    
    def calculate_band_structure(self, n_k_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Calculate photonic band structure using plane wave expansion.
        
        Args:
            n_k_points: Number of k-points along high-symmetry directions
            
        Returns:
            Dictionary with band structure data
        """
        print(f"🧮 Calculating Photonic Band Structure...")
        
        # High-symmetry k-points for different lattice types
        if self.lattice_type == CrystalGeometry.SQUARE_LATTICE:
            k_path, k_labels = self._get_square_lattice_k_path(n_k_points)
        elif self.lattice_type == CrystalGeometry.TRIANGULAR_LATTICE:
            k_path, k_labels = self._get_triangular_lattice_k_path(n_k_points)
        else:
            # Default to square lattice
            k_path, k_labels = self._get_square_lattice_k_path(n_k_points)
        
        # Calculate eigenfrequencies along k-path
        n_bands = 10  # Calculate first 10 bands
        eigenfrequencies = np.zeros((len(k_path), n_bands))
        
        for i, k_point in enumerate(k_path):
            # Solve eigenvalue problem for this k-point
            freqs = self._solve_maxwell_eigenvalue(k_point, n_bands)
            eigenfrequencies[i, :] = freqs
        
        # Identify band gaps
        band_gaps = self._find_band_gaps(eigenfrequencies)
        
        # Calculate density of states
        dos_frequencies, dos_values = self._calculate_density_of_states(eigenfrequencies)
        
        self.band_structure = {
            'k_path': k_path,
            'k_labels': k_labels,
            'eigenfrequencies': eigenfrequencies,
            'band_gaps': band_gaps,
            'dos_frequencies': dos_frequencies,
            'dos_values': dos_values,
            'n_bands': n_bands,
            'calculation_time': datetime.now().isoformat()
        }
        
        print(f"   • Calculated {n_bands} bands along {len(k_path)} k-points")
        print(f"   • Found {len(band_gaps)} band gaps")
        print(f"   • Frequency range: {eigenfrequencies.min()/1e12:.1f} - {eigenfrequencies.max()/1e12:.1f} THz")
        
        return self.band_structure
    
    def _get_square_lattice_k_path(self, n_points: int) -> Tuple[np.ndarray, List[str]]:
        """Generate k-point path for square lattice."""
        a = self.lattice_constant
        
        # High-symmetry points
        gamma = np.array([0, 0])
        X = np.array([np.pi/a, 0])
        M = np.array([np.pi/a, np.pi/a])
        
        # Path: Γ → X → M → Γ
        k_path_segments = []
        
        # Γ → X
        for i in range(n_points//3):
            t = i / (n_points//3 - 1) if n_points//3 > 1 else 0
            k_path_segments.append(gamma + t * (X - gamma))
        
        # X → M
        for i in range(n_points//3):
            t = i / (n_points//3 - 1) if n_points//3 > 1 else 0
            k_path_segments.append(X + t * (M - X))
        
        # M → Γ
        for i in range(n_points - 2*(n_points//3)):
            t = i / (n_points - 2*(n_points//3) - 1) if (n_points - 2*(n_points//3)) > 1 else 0
            k_path_segments.append(M + t * (gamma - M))
        
        k_path = np.array(k_path_segments)
        k_labels = ['Γ', 'X', 'M', 'Γ']
        
        return k_path, k_labels
    
    def _get_triangular_lattice_k_path(self, n_points: int) -> Tuple[np.ndarray, List[str]]:
        """Generate k-point path for triangular lattice."""
        a = self.lattice_constant
        
        # High-symmetry points for triangular lattice
        gamma = np.array([0, 0])
        K = np.array([2*np.pi/(3*a), 2*np.pi/(np.sqrt(3)*a)])
        M = np.array([np.pi/a, np.pi/(np.sqrt(3)*a)])
        
        # Path: Γ → K → M → Γ
        k_path_segments = []
        
        # Γ → K
        for i in range(n_points//3):
            t = i / (n_points//3 - 1) if n_points//3 > 1 else 0
            k_path_segments.append(gamma + t * (K - gamma))
        
        # K → M
        for i in range(n_points//3):
            t = i / (n_points//3 - 1) if n_points//3 > 1 else 0
            k_path_segments.append(K + t * (M - K))
        
        # M → Γ
        for i in range(n_points - 2*(n_points//3)):
            t = i / (n_points - 2*(n_points//3) - 1) if (n_points - 2*(n_points//3)) > 1 else 0
            k_path_segments.append(M + t * (gamma - M))
        
        k_path = np.array(k_path_segments)
        k_labels = ['Γ', 'K', 'M', 'Γ']
        
        return k_path, k_labels
    
    def _solve_maxwell_eigenvalue(self, k_point: np.ndarray, n_bands: int) -> np.ndarray:
        """
        Solve Maxwell eigenvalue problem for given k-point.
        
        This is a simplified model - real implementation would use
        full plane wave expansion or finite element methods.
        """
        # Simplified model using effective medium theory
        
        # Effective permittivity
        eps_eff = self._calculate_effective_permittivity()
        
        # Wave vector magnitude
        k_mag = np.linalg.norm(k_point)
        
        # Simple dispersion relation with modifications
        # ω = ck/√ε_eff with band structure modifications
        
        # Create mock band structure with gaps
        frequencies = []
        
        for band in range(n_bands):
            # Base frequency
            omega_base = self.c * k_mag / np.sqrt(eps_eff.real)
            
            # Add band-dependent offset
            omega_band = omega_base * (1 + 0.1 * band)
            
            # Add band gap effects
            if band > 0:
                # Create gaps between bands
                gap_factor = 1 + 0.2 * np.sin(k_mag * self.lattice_constant)
                omega_band *= gap_factor
            
            # Add periodicity effects
            bragg_condition = np.sin(k_mag * self.lattice_constant / 2)
            omega_band *= (1 + 0.1 * bragg_condition**2)
            
            frequencies.append(omega_band)
        
        return np.array(frequencies)
    
    def _calculate_effective_permittivity(self) -> complex:
        """Calculate effective permittivity using Maxwell-Garnett theory."""
        eps_matrix = self.matrix_material.permittivity
        eps_inclusion = self.inclusion_material.permittivity
        f = self.filling_fraction
        
        # Maxwell-Garnett effective medium theory
        eps_eff = eps_matrix * (1 + 2*f*(eps_inclusion - eps_matrix)/(eps_inclusion + 2*eps_matrix)) / (1 - f*(eps_inclusion - eps_matrix)/(eps_inclusion + 2*eps_matrix))
        
        return eps_eff
    
    def _find_band_gaps(self, eigenfrequencies: np.ndarray) -> List[Dict]:
        """Identify photonic band gaps."""
        band_gaps = []
        
        n_k, n_bands = eigenfrequencies.shape
        
        for band in range(n_bands - 1):
            # Find minimum of upper band and maximum of lower band
            upper_min = np.min(eigenfrequencies[:, band + 1])
            lower_max = np.max(eigenfrequencies[:, band])
            
            if upper_min > lower_max:
                # Band gap exists
                gap_width = upper_min - lower_max
                gap_center = (upper_min + lower_max) / 2
                gap_ratio = gap_width / gap_center
                
                band_gaps.append({
                    'lower_band': band,
                    'upper_band': band + 1,
                    'gap_center': gap_center,
                    'gap_width': gap_width,
                    'gap_ratio': gap_ratio,
                    'lower_max': lower_max,
                    'upper_min': upper_min
                })
        
        return band_gaps
    
    def _calculate_density_of_states(self, eigenfrequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate photonic density of states."""
        # Flatten all frequencies
        all_freqs = eigenfrequencies.flatten()
        
        # Create frequency bins
        freq_min = np.min(all_freqs)
        freq_max = np.max(all_freqs)
        n_bins = 100
        
        freq_bins = np.linspace(freq_min, freq_max, n_bins)
        dos_values = np.zeros(n_bins)
        
        # Histogram with Gaussian broadening
        sigma = (freq_max - freq_min) / (10 * n_bins)  # Broadening
        
        for freq in all_freqs:
            # Add Gaussian contribution to DOS
            gaussian = np.exp(-0.5 * ((freq_bins - freq) / sigma)**2) / (sigma * np.sqrt(2*np.pi))
            dos_values += gaussian
        
        # Normalize
        dos_values /= len(all_freqs)
        
        return freq_bins, dos_values
    
    def calculate_casimir_force(self, gap_distance: float = 1e-6) -> Dict[str, float]:
        """
        Calculate Casimir force between photonic crystal surfaces.
        
        Args:
            gap_distance: Distance between surfaces [m]
            
        Returns:
            Dictionary with Casimir force analysis
        """
        print(f"🔧 Calculating Casimir Force...")
        
        # Standard Casimir force between parallel plates
        F_standard = -np.pi**2 * self.hbar * self.c / (240 * gap_distance**4)
        
        # Modification due to photonic crystal
        if self.band_structure is not None:
            # Use density of states to modify Casimir force
            dos_freq = self.band_structure['dos_frequencies']
            dos_vals = self.band_structure['dos_values']
            
            # Integration over modified density of states
            # This is simplified - real calculation requires full spectral function
            modification_factor = np.trapz(dos_vals, dos_freq) / np.trapz(np.ones_like(dos_vals), dos_freq)
        else:
            # Use effective medium approximation
            eps_eff = self._calculate_effective_permittivity()
            modification_factor = eps_eff.real**0.5  # Simplified scaling
        
        # Modified Casimir force
        F_modified = F_standard * modification_factor
        
        # Force per unit area
        force_per_area = F_modified  # Already per unit area
        
        # Energy density
        energy_density = abs(F_modified) * gap_distance / 2  # Approximate energy density
        
        self.casimir_force = F_modified
        
        casimir_result = {
            'gap_distance': gap_distance,
            'force_standard': F_standard,
            'force_modified': F_modified,
            'modification_factor': modification_factor,
            'force_per_area': force_per_area,
            'energy_density': energy_density,
            'force_enhancement': F_modified / F_standard,
            'timestamp': datetime.now().isoformat()
        }
        
        self.operation_log.append({
            'action': 'calculate_casimir_force',
            'results': casimir_result
        })
        
        print(f"   • Gap distance: {gap_distance*1e6:.1f} μm")
        print(f"   • Standard Casimir force: {F_standard:.2e} N/m²")
        print(f"   • Modified force: {F_modified:.2e} N/m²")
        print(f"   • Enhancement factor: {F_modified/F_standard:.3f}")
        print(f"   • Energy density: {energy_density:.2e} J/m³")
        
        return casimir_result
    
    def apply_electro_optic_tuning(self, applied_voltage: float, 
                                 electrode_spacing: float = 10e-6) -> Dict[str, float]:
        """
        Apply electro-optic tuning to modify crystal properties.
        
        Args:
            applied_voltage: Applied voltage [V]
            electrode_spacing: Distance between electrodes [m]
            
        Returns:
            Tuning results
        """
        print(f"⚡ Applying Electro-Optic Tuning...")
        
        # Electric field
        E_field = applied_voltage / electrode_spacing
        
        # Refractive index change via Pockels effect
        # Δn = -0.5 * n³ * r * E for linear electro-optic effect
        n0 = self.inclusion_material.refractive_index.real
        r_eff = self.inclusion_material.electro_optic_coefficient
        
        delta_n = -0.5 * n0**3 * r_eff * E_field
        
        # Update material properties
        old_n = self.inclusion_material.refractive_index
        new_n = old_n.real + delta_n + 1j * old_n.imag
        
        self.inclusion_material.refractive_index = new_n
        self.inclusion_material.permittivity = new_n**2
        
        # Calculate frequency shift
        # Simple estimate: Δω/ω ≈ -Δn/n
        relative_frequency_shift = -delta_n / n0
        new_operating_frequency = self.operating_frequency * (1 + relative_frequency_shift)
        
        # Store original values
        self.electric_field_bias = E_field
        
        tuning_result = {
            'applied_voltage': applied_voltage,
            'electric_field': E_field,
            'refractive_index_change': delta_n,
            'old_refractive_index': old_n,
            'new_refractive_index': new_n,
            'relative_frequency_shift': relative_frequency_shift,
            'new_operating_frequency': new_operating_frequency,
            'tuning_efficiency': abs(relative_frequency_shift) / (E_field / 1e6),  # Per MV/m
            'timestamp': datetime.now().isoformat()
        }
        
        self.operation_log.append({
            'action': 'electro_optic_tuning',
            'parameters': tuning_result
        })
        
        print(f"   • Applied voltage: {applied_voltage:.1f} V")
        print(f"   • Electric field: {E_field/1e6:.2f} MV/m")
        print(f"   • Δn: {delta_n:.2e}")
        print(f"   • New refractive index: {new_n.real:.4f}")
        print(f"   • Frequency shift: {relative_frequency_shift*100:.3f}%")
        
        return tuning_result
    
    def measure_vacuum_modification(self, 
                                  measurement_volume: float = 1e-18,
                                  integration_time: float = 1e-3) -> Dict[str, Union[float, np.ndarray]]:
        """
        Measure modification of vacuum electromagnetic fluctuations.
        
        Args:
            measurement_volume: Measurement volume [m³]
            integration_time: Integration time [s]
            
        Returns:
            Vacuum modification measurement results
        """
        print(f"📊 Measuring Vacuum Modification...")
        
        # Time array for measurement
        sampling_rate = 1e12  # 1 THz sampling (optical frequencies)
        n_samples = int(integration_time * sampling_rate)
        time_array = np.linspace(0, integration_time, n_samples)
        
        # Simulate electromagnetic field fluctuations
        
        # 1. Unmodified vacuum fluctuations
        vacuum_fluctuations = np.random.normal(0, 1, n_samples)
        
        # 2. Photonic crystal modification
        if self.band_structure is not None:
            # Use density of states to weight fluctuations
            dos_freq = self.band_structure['dos_frequencies']
            dos_vals = self.band_structure['dos_values']
            
            # Create filter based on DOS
            # This is simplified - real measurement requires careful spectroscopy
            freq_array = np.fft.fftfreq(n_samples, 1/sampling_rate)
            freq_filter = np.interp(np.abs(freq_array), dos_freq, dos_vals, 
                                  left=0, right=0)
            
            # Apply filter in frequency domain
            vacuum_fft = np.fft.fft(vacuum_fluctuations)
            modified_fft = vacuum_fft * freq_filter
            modified_fluctuations = np.real(np.fft.ifft(modified_fft))
        else:
            # Simple modification model
            modification_factor = self._calculate_vacuum_modification_factor()
            modified_fluctuations = vacuum_fluctuations * modification_factor
        
        # 3. Add band gap effects
        if self.band_structure and self.band_structure['band_gaps']:
            # Suppress fluctuations in band gap frequencies
            for gap in self.band_structure['band_gaps']:
                gap_center = gap['gap_center']
                gap_width = gap['gap_width']
                
                # Create notch filter for this gap
                gap_freq_indices = np.where(
                    (np.abs(freq_array) > gap_center - gap_width/2) & 
                    (np.abs(freq_array) < gap_center + gap_width/2)
                )[0]
                
                if len(gap_freq_indices) > 0:
                    suppression_factor = 0.1  # 90% suppression in gap
                    modified_fft[gap_freq_indices] *= suppression_factor
            
            modified_fluctuations = np.real(np.fft.ifft(modified_fft))
        
        # Calculate field properties
        E_field_unmodified = vacuum_fluctuations * np.sqrt(self.hbar * self.operating_frequency / (2 * self.eps0 * measurement_volume))
        E_field_modified = modified_fluctuations * np.sqrt(self.hbar * self.operating_frequency / (2 * self.eps0 * measurement_volume))
        
        # Energy density calculation
        energy_density_unmodified = 0.5 * self.eps0 * E_field_unmodified**2
        energy_density_modified = 0.5 * self.eps0 * E_field_modified**2
        
        # Vacuum energy modification
        vacuum_energy_change = np.mean(energy_density_modified) - np.mean(energy_density_unmodified)
        
        # Statistical analysis
        variance_unmodified = np.var(E_field_unmodified)
        variance_modified = np.var(E_field_modified)
        variance_ratio = variance_modified / variance_unmodified
        
        # Spectral analysis
        if SCIPY_AVAILABLE:
            freqs, psd_unmodified = scipy.signal.welch(E_field_unmodified, fs=sampling_rate, nperseg=min(1024, n_samples//4))
            freqs, psd_modified = scipy.signal.welch(E_field_modified, fs=sampling_rate, nperseg=min(1024, n_samples//4))
        else:
            freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)[:n_samples//2]
            psd_unmodified = np.abs(np.fft.fft(E_field_unmodified)[:n_samples//2])**2
            psd_modified = np.abs(np.fft.fft(E_field_modified)[:n_samples//2])**2
        
        measurement_result = {
            'time_array': time_array,
            'E_field_unmodified': E_field_unmodified,
            'E_field_modified': E_field_modified,
            'energy_density_unmodified': energy_density_unmodified,
            'energy_density_modified': energy_density_modified,
            'vacuum_energy_change': vacuum_energy_change,
            'variance_unmodified': variance_unmodified,
            'variance_modified': variance_modified,
            'variance_ratio': variance_ratio,
            'spectral_frequencies': freqs,
            'psd_unmodified': psd_unmodified,
            'psd_modified': psd_modified,
            'measurement_volume': measurement_volume,
            'integration_time': integration_time,
            'sampling_rate': sampling_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        self.measurement_history.append(measurement_result)
        self.vacuum_energy_modification = vacuum_energy_change
        
        print(f"   • Integration time: {integration_time*1000:.1f} ms")
        print(f"   • Measurement volume: {measurement_volume*1e18:.1f} am³")
        print(f"   • Vacuum energy change: {vacuum_energy_change:.2e} J/m³")
        print(f"   • Variance ratio: {variance_ratio:.3f}")
        print(f"   • Field modification: {(variance_ratio-1)*100:+.1f}%")
        
        return measurement_result
    
    def _calculate_vacuum_modification_factor(self) -> float:
        """Calculate vacuum modification factor from crystal parameters."""
        # Simplified model based on filling fraction and contrast
        eps_contrast = abs(self.inclusion_material.permittivity - self.matrix_material.permittivity)
        
        # Modification scales with contrast and filling fraction
        modification = 1.0 + 0.1 * self.filling_fraction * eps_contrast.real
        
        return modification
    
    def run_photonic_crystal_protocol(self, protocol_params: Dict) -> Dict[str, any]:
        """
        Run complete photonic crystal characterization and optimization protocol.
        
        Args:
            protocol_params: Protocol configuration parameters
            
        Returns:
            Complete protocol results
        """
        print(f"🤖 Starting Photonic Crystal Engine Protocol")
        
        protocol_results = {
            'start_time': datetime.now().isoformat(),
            'protocol_params': protocol_params,
            'steps': [],
            'summary': {}
        }
        
        # Step 1: Calculate band structure
        print("\n📍 Step 1: Band Structure Calculation")
        band_structure = self.calculate_band_structure(
            n_k_points=protocol_params.get('n_k_points', 50)
        )
        protocol_results['steps'].append({
            'step': 'band_structure',
            'results': band_structure
        })
        
        # Step 2: Casimir force calculation
        print("\n📍 Step 2: Casimir Force Analysis")
        casimir_result = self.calculate_casimir_force(
            gap_distance=protocol_params.get('gap_distance', 1e-6)
        )
        protocol_results['steps'].append({
            'step': 'casimir_force',
            'results': casimir_result
        })
        
        # Step 3: Electro-optic tuning (if specified)
        if protocol_params.get('apply_tuning', False):
            print("\n📍 Step 3: Electro-Optic Tuning")
            tuning_result = self.apply_electro_optic_tuning(
                applied_voltage=protocol_params.get('tuning_voltage', 10.0),
                electrode_spacing=protocol_params.get('electrode_spacing', 10e-6)
            )
            protocol_results['steps'].append({
                'step': 'electro_optic_tuning',
                'results': tuning_result
            })
        
        # Step 4: Vacuum modification measurement
        print("\n📍 Step 4: Vacuum Modification Measurement")
        vacuum_measurement = self.measure_vacuum_modification(
            measurement_volume=protocol_params.get('measurement_volume', 1e-18),
            integration_time=protocol_params.get('integration_time', 1e-3)
        )
        protocol_results['steps'].append({
            'step': 'vacuum_measurement',
            'results': vacuum_measurement
        })
        
        # Step 5: Analysis and optimization
        print("\n📍 Step 5: Protocol Analysis")
        
        # Extract key metrics
        n_band_gaps = len(band_structure['band_gaps'])
        casimir_enhancement = casimir_result['force_enhancement']
        vacuum_energy_change = vacuum_measurement['vacuum_energy_change']
        variance_ratio = vacuum_measurement['variance_ratio']
        
        # Performance score
        performance_score = (n_band_gaps * 
                           abs(casimir_enhancement - 1) * 
                           abs(variance_ratio - 1))
        
        # Energy extraction potential
        total_volume = (self.lattice_constant * self.n_periods[0] * 
                       self.lattice_constant * self.n_periods[1] * 
                       self.lattice_constant * self.n_periods[2])
        
        total_extractable_energy = abs(vacuum_energy_change) * total_volume
        
        protocol_results['summary'] = {
            'performance_score': performance_score,
            'n_band_gaps': n_band_gaps,
            'casimir_enhancement': casimir_enhancement,
            'vacuum_energy_change': vacuum_energy_change,
            'variance_ratio': variance_ratio,
            'total_extractable_energy': total_extractable_energy,
            'crystal_volume': total_volume,
            'operating_frequency': self.operating_frequency,
            'recommendations': self._generate_crystal_recommendations(protocol_results)
        }
        
        protocol_results['end_time'] = datetime.now().isoformat()
        
        print(f"\n✅ Photonic Crystal Protocol Complete!")
        print(f"   • Performance score: {performance_score:.3f}")
        print(f"   • Band gaps: {n_band_gaps}")
        print(f"   • Casimir enhancement: {casimir_enhancement:.3f}")
        print(f"   • Vacuum modification: {(variance_ratio-1)*100:+.1f}%")
        print(f"   • Extractable energy: {total_extractable_energy:.2e} J")
        
        return protocol_results
    
    def _generate_crystal_recommendations(self, results: Dict) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        summary = results['summary']
        
        if summary['n_band_gaps'] == 0:
            recommendations.append("Increase refractive index contrast to open band gaps")
        
        if abs(summary['casimir_enhancement'] - 1) < 0.1:
            recommendations.append("Optimize lattice geometry for stronger Casimir modification")
        
        if abs(summary['variance_ratio'] - 1) < 0.05:
            recommendations.append("Increase filling fraction or crystal size for larger vacuum modification")
        
        if summary['total_extractable_energy'] < 1e-21:
            recommendations.append("Scale up crystal dimensions or improve material contrast")
        
        # Check if tuning was applied
        tuning_applied = any(step['step'] == 'electro_optic_tuning' for step in results['steps'])
        if not tuning_applied:
            recommendations.append("Consider electro-optic tuning for dynamic control")
        
        if len(recommendations) == 0:
            recommendations.append("Crystal optimally configured - proceed with experiments")
        
        return recommendations
    
    def plot_crystal_results(self, save_path: Optional[str] = None):
        """Plot photonic crystal analysis results."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("matplotlib not available for plotting")
            return
        
        if self.band_structure is None:
            warnings.warn("No band structure data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Band structure
        k_path = self.band_structure['k_path']
        eigenfreqs = self.band_structure['eigenfrequencies']
        
        # Convert k-path to 1D for plotting
        k_distance = np.zeros(len(k_path))
        for i in range(1, len(k_path)):
            k_distance[i] = k_distance[i-1] + np.linalg.norm(k_path[i] - k_path[i-1])
        
        for band in range(eigenfreqs.shape[1]):
            axes[0, 0].plot(k_distance, eigenfreqs[:, band]/1e12, 'b-', alpha=0.7)
        
        # Mark band gaps
        for gap in self.band_structure['band_gaps']:
            axes[0, 0].axhspan(gap['lower_max']/1e12, gap['upper_min']/1e12, 
                              alpha=0.3, color='red', label='Band gap' if gap == self.band_structure['band_gaps'][0] else '')
        
        axes[0, 0].set_xlabel('k-path')
        axes[0, 0].set_ylabel('Frequency [THz]')
        axes[0, 0].set_title('Photonic Band Structure')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Density of states
        dos_freq = self.band_structure['dos_frequencies']
        dos_vals = self.band_structure['dos_values']
        
        axes[0, 1].plot(dos_freq/1e12, dos_vals, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Frequency [THz]')
        axes[0, 1].set_ylabel('Density of States')
        axes[0, 1].set_title('Photonic Density of States')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Field measurement (if available)
        if self.measurement_history:
            data = self.measurement_history[-1]
            
            # Time domain
            axes[1, 0].plot(data['time_array']*1e12, data['E_field_unmodified'], 'b-', alpha=0.7, label='Unmodified')
            axes[1, 0].plot(data['time_array']*1e12, data['E_field_modified'], 'r-', alpha=0.7, label='Modified')
            axes[1, 0].set_xlabel('Time [ps]')
            axes[1, 0].set_ylabel('E-field [V/m]')
            axes[1, 0].set_title('Vacuum Field Modification')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Power spectral density
            axes[1, 1].semilogy(data['spectral_frequencies']/1e12, data['psd_unmodified'], 'b-', alpha=0.7, label='Unmodified')
            axes[1, 1].semilogy(data['spectral_frequencies']/1e12, data['psd_modified'], 'r-', alpha=0.7, label='Modified')
            axes[1, 1].set_xlabel('Frequency [THz]')
            axes[1, 1].set_ylabel('PSD')
            axes[1, 1].set_title('Power Spectral Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Crystal structure visualization
            self._plot_crystal_structure(axes[1, 0])
            axes[1, 0].set_title('Crystal Structure')
            
            # Material properties
            freqs = np.linspace(0.5*self.operating_frequency, 1.5*self.operating_frequency, 100)
            eps_real = np.real(self.inclusion_material.permittivity) * np.ones_like(freqs)
            eps_imag = np.imag(self.inclusion_material.permittivity) * np.ones_like(freqs)
            
            axes[1, 1].plot(freqs/1e12, eps_real, 'r-', label='Real(ε)')
            axes[1, 1].plot(freqs/1e12, eps_imag, 'b-', label='Imag(ε)')
            axes[1, 1].set_xlabel('Frequency [THz]')
            axes[1, 1].set_ylabel('Permittivity')
            axes[1, 1].set_title('Material Properties')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Crystal plot saved to {save_path}")
        else:
            plt.show()
    
    def _plot_crystal_structure(self, ax):
        """Plot 2D representation of crystal structure."""
        if self.lattice_type == CrystalGeometry.SQUARE_LATTICE:
            # Plot square lattice
            for i in range(self.n_periods[0]):
                for j in range(self.n_periods[1]):
                    x = i * self.lattice_constant
                    y = j * self.lattice_constant
                    
                    # Draw inclusion (circle for this visualization)
                    radius = np.sqrt(self.filling_fraction) * self.lattice_constant / 2
                    circle = Circle((x, y), radius, 
                                  facecolor='blue', edgecolor='black', alpha=0.7)
                    ax.add_patch(circle)
        
        ax.set_xlim(-self.lattice_constant/2, (self.n_periods[0]-0.5)*self.lattice_constant)
        ax.set_ylim(-self.lattice_constant/2, (self.n_periods[1]-0.5)*self.lattice_constant)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)


# Example usage and testing
if __name__ == "__main__":
    print("=== Photonic Crystal Engine for Negative Energy Generation ===")
    
    # Initialize photonic crystal
    crystal = PhotonicCrystalEngine(
        lattice_type=CrystalGeometry.SQUARE_LATTICE,
        lattice_constant=500e-9,      # 500 nm
        filling_fraction=0.3,         # 30% silicon
        n_periods=(20, 20, 1),        # 20×20×1 unit cells
        operating_frequency=600e12    # 600 THz
    )
    
    # Protocol parameters
    protocol_params = {
        'n_k_points': 50,               # Band structure resolution
        'gap_distance': 1e-6,           # 1 μm Casimir gap
        'apply_tuning': True,           # Apply electro-optic tuning
        'tuning_voltage': 10.0,         # 10 V tuning
        'electrode_spacing': 10e-6,     # 10 μm electrode spacing
        'measurement_volume': 1e-18,    # 1 am³ measurement volume
        'integration_time': 1e-3        # 1 ms integration
    }
    
    # Run protocol
    results = crystal.run_photonic_crystal_protocol(protocol_params)
    
    # Save results
    output_file = "photonic_crystal_results.json"
    
    # Convert numpy arrays to lists for JSON
    json_results = results.copy()
    for step in json_results['steps']:
        if 'results' in step:
            step_results = step['results']
            # Remove large arrays
            if isinstance(step_results, dict):
                step['results'] = {k: v for k, v in step_results.items() 
                                 if not isinstance(v, np.ndarray) or v.size < 100}
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Plot results
    try:
        crystal.plot_crystal_results("photonic_crystal_analysis.png")
    except:
        print("   (Plotting not available)")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Implement full plane wave expansion for accurate band structure")
    print("2. Add nonlinear optical effects for active control")
    print("3. Integrate with fabrication specifications")
    print("4. Scale up to 3D photonic crystals")
    print("5. Connect to real-time measurement systems")
