import numpy as np
from simulation.backreaction import solve_semiclassical_metric

def test_solve_semiclassical_metric_shapes_and_initial_step():
    """
    Test that the solver returns correct shapes and initial update for constant T00.
    """
    N = 50
    x = np.linspace(0, 1, N)
    T00 = np.ones(N)
    dt = 0.1
    steps = 10
    # Solve backreaction
    h_final, history = solve_semiclassical_metric(x, T00, dt=dt, steps=steps, G=1.0)
    # History shape should be (steps+1, N)
    assert history.shape == (steps + 1, N)
    # Final perturbation shape
    assert h_final.shape == (N,)
    # Compute expected source contribution for first step
    source = 8 * np.pi * 1.0 * T00
    expected_first = dt**2 * source
    # history[1] = initial h_next since h_prev = 0 and h = 0
    assert np.allclose(history[1], expected_first)
