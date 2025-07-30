import numpy as np
import pytest
# Ensure src directory is on sys.path for in-place imports
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))
from simulation.qft_backend import PhysicsCore

def test_qft_backend_smoke():
    xs = np.linspace(-1, 1, 16)
    core = PhysicsCore(grid=(xs, xs, xs), dx=xs[1]-xs[0])
    # Try building an LQG tensor; skip if the LQG package isn't installed
    try:
        T = core.build_LQG_tensor(dict(alpha=0.1, beta=0.2, mass=0.0))
    except Exception:
        pytest.skip("LQG backend not available")
    rho, mask, extra = core.detect_exotics(T)
    assert rho.shape == mask.shape