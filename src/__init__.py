"""
Negative Energy Generator - Core Module

This module provides the main interface for negative energy generation
using advanced theoretical physics principles.
"""

from .theoretical.exotic_matter import ExoticMatterCalculator
from .quantum.anec_violations import ANECViolationFramework
from .optimization.energy_optimizer import EnergyOptimizer
from .practical.lv_energy_engine_fixed import LorentzViolatingEngine

__version__ = "0.1.0"
__author__ = "Negative Energy Research Team"

__all__ = [
    "ExoticMatterCalculator",
    "ANECViolationFramework", 
    "EnergyOptimizer",
    "LorentzViolatingEngine"
]

class NegativeEnergyGenerator:
    """
    Main class for negative energy generation combining all theoretical
    and practical components.
    """
    
    def __init__(self):
        self.exotic_matter = ExoticMatterCalculator()
        self.anec_framework = ANECViolationFramework()
        self.optimizer = EnergyOptimizer()
        self.engine = LorentzViolatingEngine()
    
    def calculate_requirements(self, target_energy_density):
        """Calculate the requirements for generating target negative energy density."""
        return self.exotic_matter.compute_negative_density(target_energy_density)
    
    def optimize_parameters(self, requirements):
        """Optimize parameters to minimize energy requirements."""
        return self.optimizer.minimize_energy_requirements(requirements)
    
    def generate_energy(self, optimized_params):
        """Generate negative energy using optimized parameters."""
        return self.engine.extract_energy(optimized_params)
