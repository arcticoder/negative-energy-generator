"""
In-Silico Negative Energy Research Simulation Suite
==================================================

This package provides five integrated simulation modules for high-fidelity 
negative energy research using open-source computational physics tools.

Modules:
    electromagnetic_fdtd: FDTD simulation for vacuum-mode sculpting (MEEP)
    quantum_circuit_sim: Quantum circuit DCE & JPA simulation (QuTiP)
    mechanical_fem: Mechanical FEM for virtual plate deflection (FEniCS)
    photonic_crystal_band: Photonic band structure calculations (MPB)
    surrogate_model: ML-accelerated surrogate for optimization (PyTorch)

Each module encapsulates core PDE/Hamiltonian mathematics and provides
Python stubs for simulation, optimization, and surrogate modeling.
"""

__version__ = "1.0.0"
__author__ = "Negative Energy Research Team"

# Import main simulation modules
try:
    from .electromagnetic_fdtd import (
        run_fdtd_simulation,
        compute_casimir_energy_shift,
        optimize_cavity_geometry,
        run_electromagnetic_demo
    )
except ImportError:
    def run_fdtd_simulation(*args, **kwargs):
        raise ImportError("electromagnetic_fdtd not available")
    def compute_casimir_energy_shift(*args, **kwargs):
        raise ImportError("electromagnetic_fdtd not available")
    def optimize_cavity_geometry(*args, **kwargs):
        raise ImportError("electromagnetic_fdtd not available")
    def run_electromagnetic_demo(*args, **kwargs):
        raise ImportError("electromagnetic_fdtd not available")

try:
    from .quantum_circuit_sim import (
        simulate_quantum_circuit,
        analyze_negative_energy_extraction,
        optimize_jpa_protocol,
        run_quantum_demo
    )
except ImportError:
    def simulate_quantum_circuit(*args, **kwargs):
        raise ImportError("quantum_circuit_sim not available")
    def analyze_negative_energy_extraction(*args, **kwargs):
        raise ImportError("quantum_circuit_sim not available")
    def optimize_jpa_protocol(*args, **kwargs):
        raise ImportError("quantum_circuit_sim not available")
    def run_quantum_demo(*args, **kwargs):
        raise ImportError("quantum_circuit_sim not available")

try:
    from .mechanical_fem import (
        solve_plate_fem,
        compute_casimir_force,
        optimize_plate_geometry,
        run_mechanical_demo
    )
except ImportError:
    def solve_plate_fem(*args, **kwargs):
        raise ImportError("mechanical_fem not available")
    def compute_casimir_force(*args, **kwargs):
        raise ImportError("mechanical_fem not available")
    def optimize_plate_geometry(*args, **kwargs):
        raise ImportError("mechanical_fem not available")
    def run_mechanical_demo(*args, **kwargs):
        raise ImportError("mechanical_fem not available")

try:
    from .photonic_crystal_band import (
        compute_bandstructure,
        optimize_photonic_crystal_for_negative_energy,
        simulate_photonic_crystal_cavity,
        run_photonic_band_demo
    )
except ImportError:
    def compute_bandstructure(*args, **kwargs):
        raise ImportError("photonic_crystal_band not available")
    def optimize_photonic_crystal_for_negative_energy(*args, **kwargs):
        raise ImportError("photonic_crystal_band not available")
    def simulate_photonic_crystal_cavity(*args, **kwargs):
        raise ImportError("photonic_crystal_band not available")
    def run_photonic_band_demo(*args, **kwargs):
        raise ImportError("photonic_crystal_band not available")

try:
    from .surrogate_model import (
        MultiPhysicsSurrogate,
        bayesian_optimization,
        multi_domain_optimization,
        run_surrogate_demo
    )
except ImportError:
    class MultiPhysicsSurrogate:
        def __init__(self, *args, **kwargs):
            raise ImportError("surrogate_model not available")
    def bayesian_optimization(*args, **kwargs):
        raise ImportError("surrogate_model not available")
    def multi_domain_optimization(*args, **kwargs):
        raise ImportError("surrogate_model not available")
    def run_surrogate_demo(*args, **kwargs):
        raise ImportError("surrogate_model not available")

# Module metadata
MODULES = {
    'electromagnetic_fdtd': {
        'description': 'FDTD simulation for vacuum-mode sculpting',
        'backend': 'MEEP (mock)',
        'physics': 'Maxwell equations, zero-point energy',
        'applications': ['Casimir cavity design', 'Metamaterial optimization']
    },
    'quantum_circuit_sim': {
        'description': 'Quantum circuit DCE & JPA simulation',
        'backend': 'QuTiP (mock)',
        'physics': 'Lindblad master equation, quantum optics',
        'applications': ['Dynamic Casimir Effect', 'Josephson parametric amplifier']
    },
    'mechanical_fem': {
        'description': 'Mechanical FEM for virtual plate deflection',
        'backend': 'FEniCS (mock)',
        'physics': 'Kirchhoff-Love plate theory, Casimir force',
        'applications': ['Plate stability', 'Force measurement']
    },
    'photonic_crystal_band': {
        'description': 'Photonic band structure calculations',
        'backend': 'MPB (mock)',
        'physics': 'Plane-wave expansion, photonic band gaps',
        'applications': ['Metamaterial design', 'Vacuum mode engineering']
    },
    'surrogate_model': {
        'description': 'ML-accelerated surrogate for optimization',
        'backend': 'PyTorch + scikit-learn (mock)',
        'physics': 'Bayesian optimization, Gaussian processes',
        'applications': ['Parameter optimization', 'Multi-physics coupling']
    }
}

def print_module_info():
    """Print information about available simulation modules."""
    print("🔬 IN-SILICO NEGATIVE ENERGY SIMULATION SUITE")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Modules: {len(MODULES)}")
    print()
    
    for module_name, info in MODULES.items():
        print(f"📁 {module_name}")
        print(f"   • Description: {info['description']}")
        print(f"   • Backend: {info['backend']}")
        print(f"   • Physics: {info['physics']}")
        print(f"   • Applications: {', '.join(info['applications'])}")
        print()

if __name__ == "__main__":
    print_module_info()
