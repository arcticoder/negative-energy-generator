#!/usr/bin/env python3
"""
Photon-Photon Scattering Lorentz Invariance Violation Simulator
==============================================================

Simulates LIV modifications to γγ → e⁺e⁻ threshold and cross-section
for high-energy photons from blazars and GRBs interacting with EBL.

Mathematical Foundation:
- Modified photon dispersion: E² = p²c² ± ξ(p^n c^n)/(E_LIV^(n-2))
- γγ threshold: E₁E₂(1 - cosθ) ≥ (2mₑc²)²
- LIV threshold shift: s_min = 4m²c⁴[1 ± δ(E/E_LIV)^(n-2)]
- Cross-section modification: σ(s) = σ₀(s) × F_LIV(s, ξ, n)

Physical Parameters:
- Electron mass: mₑ = 0.511 MeV
- Fine structure constant: α = 1/137.036
- EBL photon energies: 0.1-10 eV (NIR-optical)
- VHE γ-ray energies: 10 GeV - 100 TeV

Observational Signatures:
- Spectral cutoffs in blazar/GRB spectra
- Energy-dependent attenuation horizons  
- Modified pair-production opacity
- TeV transparency windows
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Physical constants
c = 2.99792458e8           # Speed of light (m/s)
h_bar = 1.054571817e-34    # Reduced Planck constant (J⋅s)
alpha = 1/137.036          # Fine structure constant
eV = 1.602176634e-19       # Electron volt (J)

# Particle masses and energies (eV)
m_e = 0.5109989e6          # Electron mass
m_e_squared = m_e**2       # Convenient shorthand

# Energy scales
E_PLANCK = 1.22e19 * 1e9   # Planck energy (eV)
E_LIV_DEFAULT = 1e28       # Default LIV scale (eV)

# Photon energy ranges
EBL_ENERGY_MIN = 0.1       # Minimum EBL photon energy (eV)
EBL_ENERGY_MAX = 10.0      # Maximum EBL photon energy (eV)
VHE_ENERGY_MIN = 1e10      # Minimum VHE γ-ray energy (eV) = 10 GeV
VHE_ENERGY_MAX = 1e14      # Maximum VHE γ-ray energy (eV) = 100 TeV

class PhotonPhotonLIVSimulator:
    """
    Simulator for photon-photon scattering with Lorentz Invariance Violation.
    
    Models γγ → e⁺e⁻ pair production with modified dispersion relations
    and calculates LIV signatures in high-energy gamma-ray astronomy.
    """
    
    def __init__(self, E_LIV: float = E_LIV_DEFAULT, order: int = 2, 
                 xi: float = 1.0, enable_ebl_model: bool = True):
        """
        Initialize photon-photon LIV simulator.
        
        Args:
            E_LIV: LIV energy scale (eV)
            order: LIV correction order (n)
            xi: Dimensionless LIV coefficient
            enable_ebl_model: Whether to include EBL density model
        """
        self.E_LIV = E_LIV
        self.n = order
        self.xi = xi
        self.enable_ebl = enable_ebl_model
        
        # Pre-calculate threshold parameters
        self.threshold_standard = 4 * m_e_squared  # Standard threshold (eV²)
        
        # Initialize EBL model if enabled
        if self.enable_ebl:
            self._initialize_ebl_model()
        
        print(f"🔬 Photon-Photon LIV Simulator Initialized")
        print(f"   • LIV scale: E_LIV = {E_LIV:.2e} eV ({E_LIV/E_PLANCK:.1e} × E_Planck)")
        print(f"   • LIV order: n = {order}")
        print(f"   • LIV strength: ξ = {xi}")
        print(f"   • EBL model: {'Enabled' if self.enable_ebl else 'Disabled'}")
        print(f"   • Standard threshold: s_min = {self.threshold_standard:.2e} eV²")
    
    def _initialize_ebl_model(self):
        """
        Initialize simplified extragalactic background light (EBL) model.
        
        Uses generic EBL spectrum for photon density calculation.
        """
        # Simplified EBL spectral energy distribution
        # Based on Franceschini et al. (2008) model at z=0
        ebl_energies = np.logspace(-1, 1, 50)  # 0.1 to 10 eV
        
        # EBL intensity: I(E) ∝ E^(-1) exp(-E/E₀) with E₀ ≈ 1 eV
        E_0 = 1.0  # eV
        ebl_intensities = ebl_energies**(-1) * np.exp(-ebl_energies / E_0)
        
        # Normalize to realistic EBL density ~ 1 photon/cm³
        normalization = 1.0 / np.trapz(ebl_intensities, ebl_energies)
        ebl_intensities *= normalization
        
        # Create interpolation function
        self.ebl_spectrum = interp1d(ebl_energies, ebl_intensities, 
                                   bounds_error=False, fill_value=0.0)
        
        print(f"   • EBL model: {len(ebl_energies)} energy points, "
              f"peak at {E_0} eV")
    
    def modified_threshold(self, E_gamma: float, xi_sign: int = 1) -> float:
        """
        Calculate LIV-modified pair production threshold.
        
        Modified threshold condition:
        s_min = 4m²c⁴[1 ± δ(E_γ/E_LIV)^(n-2)]
        
        Args:
            E_gamma: High-energy photon energy (eV)
            xi_sign: +1 for superluminal, -1 for subluminal LIV
            
        Returns:
            Modified threshold in invariant mass squared (eV²)
        """
        # LIV correction
        if self.E_LIV > 0:
            delta = xi_sign * self.xi * (E_gamma / self.E_LIV)**(self.n - 2)
        else:
            delta = 0
        
        # Modified threshold
        threshold_modified = self.threshold_standard * (1 + delta)
        
        return max(threshold_modified, 0)  # Ensure positive threshold
    
    def threshold_photon_energy(self, E_gamma: float, angle: float = np.pi, 
                              xi_sign: int = 1) -> float:
        """
        Calculate minimum EBL photon energy for pair production.
        
        Threshold condition: E₁E₂(1 - cosθ) ≥ s_min
        
        Args:
            E_gamma: High-energy photon energy (eV)
            angle: Scattering angle (radians, default: head-on collision)
            xi_sign: LIV sign
            
        Returns:
            Minimum EBL photon energy (eV)
        """
        s_min = self.modified_threshold(E_gamma, xi_sign)
        geometric_factor = 1 - np.cos(angle)
        
        if geometric_factor <= 0:
            return np.inf  # No interaction possible
        
        return s_min / (E_gamma * geometric_factor)
    
    def cross_section_standard(self, s: float) -> float:
        """
        Calculate standard γγ → e⁺e⁻ cross-section.
        
        Breit-Wheeler formula:
        σ₀(s) = (πr₀²/2)[β(3-β⁴)ln((1+β)/(1-β)) - 2β(2-β²)]
        where β = √(1 - 4m²c⁴/s) and r₀ = e²/(4πε₀mc²)
        
        Args:
            s: Invariant mass squared (eV²)
            
        Returns:
            Cross-section in barns (10⁻²⁴ cm²)
        """
        if s < self.threshold_standard:
            return 0.0
        
        # Classical electron radius in cm
        r_0 = 2.818e-13  # cm
        
        # Velocity parameter
        beta = np.sqrt(1 - self.threshold_standard / s)
        
        if beta <= 0:
            return 0.0
        
        # Breit-Wheeler cross-section
        term1 = beta * (3 - beta**4) * np.log((1 + beta) / (1 - beta))
        term2 = 2 * beta * (2 - beta**2)
        
        sigma = (np.pi * r_0**2 / 2) * (term1 - term2)
        
        # Convert cm² to barns
        return sigma * 1e24
    
    def cross_section_modified(self, s: float, E_gamma: float, 
                             xi_sign: int = 1) -> float:
        """
        Calculate LIV-modified γγ → e⁺e⁻ cross-section.
        
        Simple phenomenological model:
        σ_LIV(s) = σ₀(s) × F_LIV(s, E_γ, ξ, n)
        
        Args:
            s: Invariant mass squared (eV²)
            E_gamma: High-energy photon energy (eV)
            xi_sign: LIV sign
            
        Returns:
            Modified cross-section in barns
        """
        # Standard cross-section
        sigma_0 = self.cross_section_standard(s)
        
        if sigma_0 <= 0:
            return 0.0
        
        # LIV modification factor (phenomenological)
        if self.E_LIV > 0:
            liv_factor = 1 + xi_sign * self.xi * 0.1 * (E_gamma / self.E_LIV)**(self.n - 2)
        else:
            liv_factor = 1.0
        
        return sigma_0 * max(liv_factor, 0.1)  # Ensure reasonable bounds
    
    def opacity_integrand(self, eps_ebl: float, E_gamma: float, 
                         xi_sign: int = 1) -> float:
        """
        Calculate opacity integrand for pair production.
        
        dτ/dε = ∫ n(ε,z) σ(s) (1-cosθ) d(cosθ)
        
        Args:
            eps_ebl: EBL photon energy (eV)
            E_gamma: High-energy photon energy (eV)
            xi_sign: LIV sign
            
        Returns:
            Opacity integrand value
        """
        if not self.enable_ebl:
            return 0.0
        
        # EBL photon density
        n_ebl = self.ebl_spectrum(eps_ebl)
        
        if n_ebl <= 0:
            return 0.0
        
        # Angular integration over collision geometry
        def angle_integrand(cos_theta):
            s = 2 * E_gamma * eps_ebl * (1 - cos_theta)
            sigma = self.cross_section_modified(s, E_gamma, xi_sign)
            return sigma * (1 - cos_theta)
        
        # Integrate over scattering angles
        cos_min = max(-1, 1 - self.modified_threshold(E_gamma, xi_sign) / 
                      (E_gamma * eps_ebl))
        
        if cos_min >= 1:
            return 0.0
        
        integral, _ = quad(angle_integrand, cos_min, 1, limit=20)
        
        return n_ebl * integral
    
    def optical_depth(self, E_gamma: float, xi_sign: int = 1, 
                     distance: float = 100) -> float:
        """
        Calculate optical depth for γγ pair production.
        
        τ = ∫₀ᴰ dl ∫ dε n(ε) σ(E,ε) ⟨1-cosθ⟩
        
        Args:
            E_gamma: High-energy photon energy (eV)
            xi_sign: LIV sign
            distance: Propagation distance (Mpc)
            
        Returns:
            Optical depth (dimensionless)
        """
        if not self.enable_ebl:
            return 0.0
        
        # EBL energy range for integration
        eps_min = max(EBL_ENERGY_MIN, 
                     self.threshold_photon_energy(E_gamma, np.pi, xi_sign))
        eps_max = EBL_ENERGY_MAX
        
        if eps_min >= eps_max:
            return 0.0
        
        # Integrate over EBL spectrum
        def integrand(eps):
            return self.opacity_integrand(eps, E_gamma, xi_sign)
        
        try:
            integral, _ = quad(integrand, eps_min, eps_max, 
                             limit=50, epsabs=1e-10)
        except:
            integral = 0.0
        
        # Convert distance to cm and apply
        distance_cm = distance * 3.086e24  # Mpc to cm
        
        return integral * distance_cm
    
    def attenuation_factor(self, E_gamma: float, xi_sign: int = 1, 
                          distance: float = 100) -> float:
        """
        Calculate γ-ray attenuation factor due to pair production.
        
        Attenuation: A = exp(-τ)
        
        Args:
            E_gamma: High-energy photon energy (eV)
            xi_sign: LIV sign
            distance: Distance (Mpc)
            
        Returns:
            Attenuation factor (0 to 1)
        """
        tau = self.optical_depth(E_gamma, xi_sign, distance)
        return np.exp(-tau)

def scan_photon_energies(energy_range: Tuple[float, float], 
                        n_points: int = 50) -> np.ndarray:
    """
    Generate logarithmic energy scan for γ-ray analysis.
    
    Args:
        energy_range: (E_min, E_max) in eV
        n_points: Number of energy points
        
    Returns:
        Energy array in eV
    """
    return np.logspace(np.log10(energy_range[0]), 
                      np.log10(energy_range[1]), n_points)

def benchmark_blazar_observations(simulator: PhotonPhotonLIVSimulator) -> Dict:
    """
    Benchmark LIV predictions against blazar observations.
    
    Args:
        simulator: Configured LIV simulator
        
    Returns:
        Dictionary with benchmark results
    """
    print("🔭 Benchmarking Blazar Observations")
    print("=" * 35)
    
    # Representative blazar sources and distances
    blazars = {
        'Mrk421': {'distance': 134, 'z': 0.031},      # Mpc, redshift
        'Mrk501': {'distance': 456, 'z': 0.034},
        'PKS2155': {'distance': 1290, 'z': 0.116},
        '3C279': {'distance': 5000, 'z': 0.536}
    }
    
    # VHE energy range for analysis
    energies = scan_photon_energies((1e11, 1e14), 30)  # 100 GeV to 100 TeV
    
    benchmark_results = {}
    
    for blazar_name, properties in blazars.items():
        print(f"\n   🌟 {blazar_name} (d = {properties['distance']} Mpc)")
        
        distance = properties['distance']
        results = {'energies': energies}
        
        # Calculate attenuation for different LIV scenarios
        for xi_sign, label in [(0, 'standard'), (+1, 'superluminal'), (-1, 'subluminal')]:
            attenuations = []
            optical_depths = []
            
            for E in energies:
                if xi_sign == 0:
                    # Standard case: use xi=0 effectively
                    sim_std = PhotonPhotonLIVSimulator(E_LIV=1e50, xi=0)
                    att = sim_std.attenuation_factor(E, 1, distance)
                    tau = sim_std.optical_depth(E, 1, distance)
                else:
                    att = simulator.attenuation_factor(E, xi_sign, distance)
                    tau = simulator.optical_depth(E, xi_sign, distance)
                
                attenuations.append(att)
                optical_depths.append(tau)
            
            results[f'{label}_attenuation'] = attenuations
            results[f'{label}_optical_depth'] = optical_depths
        
        # Find characteristic energies
        standard_att = np.array(results['standard_attenuation'])
        cutoff_indices = np.where(standard_att < 0.1)[0]
        
        if len(cutoff_indices) > 0:
            cutoff_energy = energies[cutoff_indices[0]]
            results['cutoff_energy_eV'] = cutoff_energy
            print(f"      Standard cutoff: {cutoff_energy:.2e} eV ({cutoff_energy/1e12:.1f} TeV)")
        
        benchmark_results[blazar_name] = results
    
    return benchmark_results

def visualize_photon_photon_liv(simulator: PhotonPhotonLIVSimulator,
                               benchmark_data: Dict,
                               save_filename: str = 'photon_photon_liv_analysis.png'):
    """
    Create comprehensive visualization of photon-photon LIV effects.
    
    Args:
        simulator: Configured LIV simulator
        benchmark_data: Results from benchmark analysis
        save_filename: Output filename
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    fig.suptitle('Photon-Photon Scattering LIV Analysis', fontsize=16)
    
    # Plot 1: Threshold energy modification
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Pair Production Threshold')
    ax1.set_xlabel('γ-ray Energy (eV)')
    ax1.set_ylabel('Threshold EBL Energy (eV)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    gamma_energies = scan_photon_energies((1e11, 1e14), 30)
    
    for xi_sign, label, color in [(0, 'Standard', 'k'), 
                                  (+1, 'Superluminal', 'b'), 
                                  (-1, 'Subluminal', 'r')]:
        threshold_energies = []
        for E in gamma_energies:
            if xi_sign == 0:
                sim_std = PhotonPhotonLIVSimulator(E_LIV=1e50, xi=0)
                E_th = sim_std.threshold_photon_energy(E, np.pi, 1)
            else:
                E_th = simulator.threshold_photon_energy(E, np.pi, xi_sign)
            threshold_energies.append(E_th)
        
        # Filter out infinite values
        finite_mask = np.isfinite(threshold_energies)
        if np.any(finite_mask):
            ax1.loglog(gamma_energies[finite_mask], 
                      np.array(threshold_energies)[finite_mask], 
                      label=label, color=color, linewidth=2)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cross-section modification
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('γγ Cross-Section')
    ax2.set_xlabel('Invariant Mass² (eV²)')
    ax2.set_ylabel('Cross-Section (barns)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    s_values = np.logspace(12, 16, 50)  # Around threshold region
    E_test = 1e13  # 10 TeV test energy
    
    for xi_sign, label, color in [(0, 'Standard', 'k'), 
                                  (+1, 'Superluminal', 'b'), 
                                  (-1, 'Subluminal', 'r')]:
        cross_sections = []
        for s in s_values:
            if xi_sign == 0:
                sim_std = PhotonPhotonLIVSimulator(E_LIV=1e50, xi=0)
                sigma = sim_std.cross_section_standard(s)
            else:
                sigma = simulator.cross_section_modified(s, E_test, xi_sign)
            cross_sections.append(sigma)
        
        nonzero_mask = np.array(cross_sections) > 0
        if np.any(nonzero_mask):
            ax2.loglog(s_values[nonzero_mask], 
                      np.array(cross_sections)[nonzero_mask],
                      label=label, color=color, linewidth=2)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: EBL spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('EBL Photon Spectrum')
    ax3.set_xlabel('Photon Energy (eV)')
    ax3.set_ylabel('Intensity (arb. units)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    if simulator.enable_ebl:
        ebl_energies = np.logspace(-1, 1, 100)
        ebl_intensities = [simulator.ebl_spectrum(E) for E in ebl_energies]
        nonzero_mask = np.array(ebl_intensities) > 0
        
        if np.any(nonzero_mask):
            ax3.loglog(ebl_energies[nonzero_mask], 
                      np.array(ebl_intensities)[nonzero_mask],
                      'g-', linewidth=2, label='EBL Model')
            ax3.legend()
    
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-6: Blazar attenuation spectra
    blazar_names = list(benchmark_data.keys())[:3]  # First 3 blazars
    
    for i, blazar_name in enumerate(blazar_names):
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(f'{blazar_name} Attenuation')
        ax.set_xlabel('γ-ray Energy (eV)')
        ax.set_ylabel('Attenuation Factor')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        data = benchmark_data[blazar_name]
        energies = data['energies']
        
        for scenario, color in [('standard', 'k'), 
                               ('superluminal', 'b'), 
                               ('subluminal', 'r')]:
            key = f'{scenario}_attenuation'
            if key in data:
                attenuations = data[key]
                positive_mask = np.array(attenuations) > 1e-10
                
                if np.any(positive_mask):
                    ax.loglog(energies[positive_mask], 
                             np.array(attenuations)[positive_mask],
                             label=scenario.capitalize(), color=color, linewidth=2)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Optical depth comparison
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_title('Optical Depth vs Distance')
    ax7.set_xlabel('Distance (Mpc)')
    ax7.set_ylabel('Optical Depth')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    
    distances = np.logspace(1, 4, 20)  # 10 Mpc to 10 Gpc
    test_energy = 1e13  # 10 TeV
    
    for xi_sign, label, color in [(+1, 'Superluminal', 'b'), 
                                  (-1, 'Subluminal', 'r')]:
        optical_depths = []
        for d in distances:
            tau = simulator.optical_depth(test_energy, xi_sign, d)
            optical_depths.append(tau)
        
        positive_mask = np.array(optical_depths) > 1e-10
        if np.any(positive_mask):
            ax7.loglog(distances[positive_mask], 
                      np.array(optical_depths)[positive_mask],
                      label=f'{label} LIV', color=color, linewidth=2)
    
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: LIV signature strength
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_title('LIV Signature vs Energy Scale')
    ax8.set_xlabel('LIV Energy Scale (eV)')
    ax8.set_ylabel('Relative Threshold Shift')
    ax8.set_xscale('log')
    ax8.set_yscale('symlog', linthresh=1e-6)
    
    E_LIV_values = np.logspace(24, 30, 20)
    test_gamma_energy = 1e13
    
    threshold_shifts = []
    for E_LIV in E_LIV_values:
        sim_test = PhotonPhotonLIVSimulator(E_LIV=E_LIV, order=2, xi=1.0)
        
        threshold_std = sim_test.modified_threshold(test_gamma_energy, 0)
        threshold_liv = sim_test.modified_threshold(test_gamma_energy, 1)
        
        if threshold_std > 0:
            shift = (threshold_liv - threshold_std) / threshold_std
        else:
            shift = 0
        
        threshold_shifts.append(shift)
    
    ax8.semilogx(E_LIV_values, threshold_shifts, 'g-o', 
                markersize=4, linewidth=2, label='n=2, ξ=1')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_title('Blazar Cutoff Energies')
    ax9.set_ylabel('Cutoff Energy (TeV)')
    ax9.set_yscale('log')
    
    blazar_names_plot = []
    cutoff_energies = []
    
    for blazar_name, data in benchmark_data.items():
        if 'cutoff_energy_eV' in data:
            blazar_names_plot.append(blazar_name)
            cutoff_energies.append(data['cutoff_energy_eV'] / 1e12)  # Convert to TeV
    
    if blazar_names_plot:
        x_pos = range(len(blazar_names_plot))
        ax9.bar(x_pos, cutoff_energies, alpha=0.7, color='orange')
        ax9.set_xticks(x_pos)
        ax9.set_xticklabels(blazar_names_plot, rotation=45, ha='right')
    
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"   📊 Visualization saved: {save_filename}")
    
    return fig

def main():
    """Main execution function for photon-photon LIV analysis."""
    print("🌌 Photon-Photon Scattering LIV Simulator")
    print("=" * 45)
    
    # Initialize simulator with quantum gravity parameters
    print("\n1️⃣ Initializing LIV simulator...")
    simulator = PhotonPhotonLIVSimulator(
        E_LIV=1e28,      # Near Planck scale
        order=2,         # Quadratic corrections
        xi=1.0,          # Order unity coefficient
        enable_ebl_model=True
    )
    
    # Basic threshold calculations
    print("\n2️⃣ Basic threshold calculations...")
    test_energies = [1e12, 1e13, 1e14]  # 1, 10, 100 TeV
    
    for E_gamma in test_energies:
        print(f"\n   γ-ray energy: {E_gamma:.0e} eV ({E_gamma/1e12:.0f} TeV)")
        
        for xi_sign, label in [(+1, 'Superluminal'), (-1, 'Subluminal')]:
            E_th = simulator.threshold_photon_energy(E_gamma, np.pi, xi_sign)
            tau = simulator.optical_depth(E_gamma, xi_sign, 100)  # 100 Mpc
            att = simulator.attenuation_factor(E_gamma, xi_sign, 100)
            
            print(f"      {label}: E_th = {E_th:.2e} eV, "
                  f"τ(100 Mpc) = {tau:.3f}, A = {att:.3f}")
    
    # Benchmark against blazar observations
    print("\n3️⃣ Benchmarking blazar observations...")
    benchmark_results = benchmark_blazar_observations(simulator)
    
    # Generate comprehensive visualization
    print("\n4️⃣ Creating visualization...")
    fig = visualize_photon_photon_liv(simulator, benchmark_results)
    plt.close(fig)
    
    print("\n✅ Photon-Photon LIV Analysis Complete!")
    print(f"   • Analyzed {len(benchmark_results)} blazar sources")
    print(f"   • LIV scale: {simulator.E_LIV:.1e} eV")
    print(f"   • Generated comprehensive analysis plots")
    
    # Summary of key findings
    print("\n📊 Key Findings:")
    for blazar_name, data in benchmark_results.items():
        if 'cutoff_energy_eV' in data:
            cutoff = data['cutoff_energy_eV']
            print(f"   • {blazar_name}: Cutoff at {cutoff:.1e} eV ({cutoff/1e12:.1f} TeV)")
    
    return {
        'simulator': simulator,
        'benchmark_results': benchmark_results
    }

if __name__ == "__main__":
    results = main()
