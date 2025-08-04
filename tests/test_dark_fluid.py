import numpy as np
from simulation.dark_fluid import generate_negative_mass_fluid
from simulation.qft_backend import PhysicsCore


def test_generate_negative_mass_fluid_defaults():
    N = 50
    dx = 0.2
    x, rho = generate_negative_mass_fluid(N, dx)
    # Check shapes
    assert x.shape == (N,)
    assert rho.shape == (N,)
    # Check values
    assert np.allclose(rho, -1.0)
    # Grid spacing
    assert np.isclose(x[1] - x[0], dx)


def test_anec_negative_fluid():
    # Use PhysicsCore to compute ANEC for negative density fluid
    N = 10
    dx = 0.1
    x, rho = generate_negative_mass_fluid(N, dx, rho0=-2.0)
    core = PhysicsCore(grid=(x, x, x), dx=dx)
    # ANEC integral should be negative
    anec_val = core.compute_anec(rho, dx)
    assert anec_val < 0
    # check_anec returns True
    assert core.check_anec(rho, dx)
