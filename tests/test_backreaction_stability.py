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
    # Check each step
    for n in range(1, steps + 1):
        expected = (2 * n - 1) * dt**2 * source_term
        # Compare history[n] to expected value across grid
        assert np.allclose(history[n], expected, atol=1e-6), (
            f"Step {n}: expected {(2*n-1)}*dt^2*source, got {history[n][0]}"
        )
