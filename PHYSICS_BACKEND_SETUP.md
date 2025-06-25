# Real Physics Backend Installation Guide

This guide helps you install the real computational physics backends for the simulation suite.

## Overview

The simulation suite now uses **real computational physics** instead of mock implementations:

1. **MEEP** - Maxwell's equations FDTD solver
2. **QuTiP** - Quantum dynamics and circuit simulation
3. **FEniCS** - Finite element method for mechanical simulations
4. **MPB** - Photonic band structure calculations
5. **PyTorch** - Neural network surrogate models

## Installation Instructions

### 1. PyTorch (Already Included)
```bash
pip install torch torchvision
```

### 2. MEEP - Electromagnetic FDTD
```bash
# Option A: Conda (recommended)
conda install -c conda-forge pymeeus

# Option B: Build from source
# See: https://meep.readthedocs.io/en/latest/Installation/
```

### 3. QuTiP - Quantum Dynamics
```bash
pip install qutip
```

### 4. FEniCS - Finite Element Method
```bash
# Option A: Conda (recommended)
conda install -c conda-forge fenics

# Option B: Docker
# See: https://fenics.readthedocs.io/en/latest/installation.html
```

### 5. MPB - Photonic Band Structure
```bash
# Complex installation - recommend conda
conda install -c conda-forge mpb

# Or build from source:
# See: https://mpb.readthedocs.io/en/latest/Installation/
```

### 6. Optimization Libraries
```bash
pip install scikit-optimize deap
```

## Quick Start

1. Install basic requirements:
```bash
pip install -r requirements.txt
```

2. Install physics backends (choose your preferred method above)

3. Test the integrated workflow:
```bash
python advanced_ml_optimization_demo.py
```

## Fallback Behavior

If a physics backend is not installed, the simulation modules automatically fall back to lightweight mock implementations. You'll see informative messages like:

```
MEEP not available, using mock FDTD simulation
QuTiP not available, using mock quantum simulation
```

## Verification

Run this Python snippet to check which backends are available:

```python
def check_backends():
    backends = {}
    
    try:
        import meep
        backends['MEEP'] = f"✓ Available (v{meep.__version__})"
    except ImportError:
        backends['MEEP'] = "✗ Not installed"
    
    try:
        import qutip
        backends['QuTiP'] = f"✓ Available (v{qutip.__version__})"
    except ImportError:
        backends['QuTiP'] = "✗ Not installed"
    
    try:
        import fenics
        backends['FEniCS'] = "✓ Available"
    except ImportError:
        backends['FEniCS'] = "✗ Not installed"
    
    try:
        import meep  # MPB often comes with MEEP
        backends['MPB'] = "✓ Available (bundled with MEEP)"
    except ImportError:
        backends['MPB'] = "✗ Not installed"
    
    try:
        import torch
        backends['PyTorch'] = f"✓ Available (v{torch.__version__})"
    except ImportError:
        backends['PyTorch'] = "✗ Not installed"
    
    for name, status in backends.items():
        print(f"{name}: {status}")

if __name__ == "__main__":
    check_backends()
```

## Performance Notes

- **MEEP**: GPU acceleration available with CUDA
- **QuTiP**: Supports parallel computation for large systems
- **FEniCS**: MPI parallelization for large meshes
- **PyTorch**: GPU acceleration for neural networks

## Troubleshooting

### Common Issues

1. **FEniCS installation fails**: Use conda instead of pip
2. **MPB complex dependencies**: Try installing MEEUS first, which often includes MPB
3. **MEEP build errors**: Use conda-forge channel

### Docker Alternative

If installation is problematic, consider using the official Docker images:

```bash
# FEniCS
docker pull quay.io/fenicsproject/stable

# MEEP
docker pull meep/meep
```

## Production Deployment

For production use:
1. Install all backends for maximum accuracy
2. Use GPU-enabled versions where available
3. Configure MPI for parallel computation
4. Monitor memory usage for large simulations

## Support

- MEEP: https://meep.readthedocs.io/
- QuTiP: https://qutip.org/docs/latest/
- FEniCS: https://fenicsproject.org/documentation/
- MPB: https://mpb.readthedocs.io/
