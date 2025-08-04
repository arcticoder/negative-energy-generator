import numpy as np
import pytest
from simulation.qft_backend import PhysicsCore

def test_evolve_qft_fallback_identity():
    """
    V&V: PhysicsCore.evolve_QFT should return a history list of length steps+1,
    with all entries equal to the initial field when QuantumFieldOperator unavailable.
    """
    # Create simple initial field
    N = 10
    x = np.linspace(0, 1, N)
    core = PhysicsCore(grid=(x, x, x), dx=0.1)
    phi0 = np.arange(N)
    steps = 5
    history = core.evolve_QFT(phi0, steps=steps, dt=0.1)
    # Check type and length
    assert isinstance(history, list), "History should be a list"
    assert len(history) == steps + 1, f"Expected {steps+1} states, got {len(history)}"
    # Check that each entry equals phi0
    for state in history:
        assert np.array_equal(state, phi0), "Each state should equal the initial field"
