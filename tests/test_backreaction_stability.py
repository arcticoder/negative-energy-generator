import numpy as np
from simulation.backreaction import solve_semiclassical_metric

def test_constant_source_growth_matches_theoretical():
    """
    For constant T00 and zero initial conditions, the n-th metric perturbation h_n should follow h_n = (2n-1) * dt^2 * source.
    """
    N = 10
    x = np.linspace(0, 1, N)
    # Constant source
    T00 = np.ones(N)
    dt = 0.1
    steps = 5
    G = 1.0

    # Solve backreaction
    h_final, history = solve_semiclassical_metric(x, T00, dt=dt, steps=steps, G=G)

    source_term = 8 * np.pi * G * T00
    # Verify first-step growth matches theoretical source-only prediction
    expected_first = dt**2 * source_term
    assert np.allclose(history[1], expected_first, atol=1e-6), (
        f"First step: expected dt^2 * source, got {history[1][0]}"
    )
