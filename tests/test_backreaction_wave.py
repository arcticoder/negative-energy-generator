import numpy as np
from simulation.backreaction import solve_semiclassical_metric

def test_zero_source_remains_zero():
    """
    With zero T00 source, initial zero conditions should produce zero metric perturbations over time.
    """
    N = 50
    x = np.linspace(0, 1, N)
    T00 = np.zeros(N)
    dt = 0.1
    steps = 10

    h_final, history = solve_semiclassical_metric(x, T00, dt=dt, steps=steps, G=1.0)
    # All history and final should be zero
    assert np.allclose(history, 0), "History should be all zeros for zero source"
    assert np.allclose(h_final, 0), "Final h should be zero for zero source"
