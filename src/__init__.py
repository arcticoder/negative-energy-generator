"""
Negative Energy Generator - Modernized Prototype Stack

This module provides the main interface for ML-powered negative energy generation
using modern prototype implementations and advanced ML optimization.
"""

# Import new prototype modules (safely)
try:
    from .prototype import *
    PROTOTYPE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Prototype modules not available: {e}")
    PROTOTYPE_AVAILABLE = False

# Import ML modules (safely)
try:
    from .ml import *
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML modules not available (optional): {e}")
    ML_AVAILABLE = False

# Legacy theoretical modules (safe imports)
try:
    from .theoretical.exotic_matter import ExoticMatterCalculator
    from .quantum.anec_violations import WarpBubble
    THEORETICAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Legacy theoretical modules not available: {e}")
    THEORETICAL_AVAILABLE = False

__version__ = "2.0.0"
__author__ = "Negative Energy Research Team"

__all__ = [
    "PROTOTYPE_AVAILABLE",
    "ML_AVAILABLE", 
    "THEORETICAL_AVAILABLE"
]

# Legacy compatibility class (optional)
class NegativeEnergyGenerator:
    """
    Legacy compatibility class for negative energy generation.
    For new development, use the prototype modules directly.
    """
    
    def __init__(self):
        if THEORETICAL_AVAILABLE:
            try:
                self.exotic_matter = ExoticMatterCalculator()
                self.warp_bubble = WarpBubble()
            except:
                self.exotic_matter = None
                self.warp_bubble = None
        else:
            self.exotic_matter = None
            self.warp_bubble = None
    
    def calculate_requirements(self, target_energy_density):
        """Calculate the requirements for generating target negative energy density."""
        if self.exotic_matter:
            return self.exotic_matter.compute_negative_density(target_energy_density)
        else:
            raise ImportError("Legacy theoretical modules not available. Use prototype modules instead.")
    
    def get_prototype_system(self):
        """Get the modern prototype system."""
        if PROTOTYPE_AVAILABLE:
            from .prototype import create_combined_system
            return create_combined_system()
        else:
            raise ImportError("Prototype modules not available.")
    
    def generate_energy(self, optimized_params):
        """Generate negative energy using optimized parameters."""
        return self.engine.extract_energy(optimized_params)
