"""
Machine Learning Optimization Package for Negative Energy Generation
==================================================================

This package provides advanced ML-based optimization algorithms for
exotic matter research and negative energy system optimization.
"""

# Import availability flags first
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import skopt
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import deap
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

# Core imports (always available)
__version__ = "1.0.0"
__author__ = "Negative Energy Research Team"

# Conditional imports based on available dependencies
available_modules = []

if SKOPT_AVAILABLE:
    try:
        from .bo_ansatz_opt import BayesianAnsatzOptimizer
        available_modules.append("BayesianAnsatzOptimizer")
    except ImportError as e:
        print(f"Warning: Could not import BayesianAnsatzOptimizer: {e}")

if DEAP_AVAILABLE:
    try:
        from .genetic_ansatz import GeneticAnsatzOptimizer
        available_modules.append("GeneticAnsatzOptimizer")
    except ImportError as e:
        print(f"Warning: Could not import GeneticAnsatzOptimizer: {e}")

if TORCH_AVAILABLE:
    try:
        from .pinn_exotic import ProfileNet, ExoticMatterPINN
        from .pinn_exotic import mock_warp_bubble_energy_computer, mock_quantum_field_energy_computer
        available_modules.extend(["ProfileNet", "ExoticMatterPINN"])
    except ImportError as e:
        print(f"Warning: Could not import PINN modules: {e}")

# Export all available classes and functions
__all__ = available_modules.copy()

# Add utility functions and constants
__all__.extend([
    "TORCH_AVAILABLE",
    "SKOPT_AVAILABLE", 
    "DEAP_AVAILABLE",
    "get_available_optimizers",
    "install_missing_dependencies"
])

def get_available_optimizers():
    """
    Get list of available optimization algorithms.
    
    Returns:
        Dict with available optimizers and their status
    """
    status = {
        "BayesianOptimization": {
            "available": SKOPT_AVAILABLE,
            "class": "BayesianAnsatzOptimizer" if SKOPT_AVAILABLE else None,
            "description": "Gaussian process-based optimization with uncertainty quantification",
            "install_cmd": "pip install scikit-optimize"
        },
        "GeneticAlgorithm": {
            "available": DEAP_AVAILABLE,
            "class": "GeneticAnsatzOptimizer" if DEAP_AVAILABLE else None,
            "description": "Evolutionary optimization for complex parameter landscapes",
            "install_cmd": "pip install deap"
        },
        "PhysicsInformedNN": {
            "available": TORCH_AVAILABLE,
            "class": "ExoticMatterPINN" if TORCH_AVAILABLE else None,
            "description": "Deep learning with physics constraints for direct optimization",
            "install_cmd": "pip install torch"
        }
    }
    
    return status

def install_missing_dependencies():
    """
    Print installation commands for missing dependencies.
    """
    print("=== Machine Learning Dependencies Status ===")
    
    status = get_available_optimizers()
    
    for name, info in status.items():
        if info["available"]:
            print(f"✅ {name}: Available ({info['class']})")
        else:
            print(f"❌ {name}: Missing - Install with: {info['install_cmd']}")
    
    print("\nTo install all ML dependencies:")
    print("pip install scikit-optimize deap torch matplotlib seaborn")
    
    print(f"\nCurrently available modules: {available_modules}")

# Print status on import
if __name__ != "__main__":
    missing_deps = []
    if not SKOPT_AVAILABLE:
        missing_deps.append("scikit-optimize")
    if not DEAP_AVAILABLE:
        missing_deps.append("deap") 
    if not TORCH_AVAILABLE:
        missing_deps.append("torch")
    
    if missing_deps:
        print(f"ML package loaded with missing dependencies: {missing_deps}")
        print("Run `from src.ml import install_missing_dependencies; install_missing_dependencies()` for details")
    else:
        print(f"ML package loaded successfully with {len(available_modules)} optimizers available")
