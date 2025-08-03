#!/usr/bin/env python3
"""
Demo for dynamic evolution of 1+1D Klein-Gordon field showing energy tracking.
"""
import os
import sys
import pathlib
import numpy as np
import h5py
from simulation.lattice_qft import solve_klein_gordon, compute_energy_density

def main():
    # Demo parameters
    N = 100
    dx = 1.0
    dt = 0.1
    steps = 50
    alpha = 0.0  # massless case
    beta = 0.0   # no damping

    # Initial condition: single-mode sine wave
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    phi_init = np.sin(x)
    phi_dt_init = np.zeros(N)

    # Evolve and record states
    phi, phi_dt, phi_hist, phi_dt_hist = solve_klein_gordon(
        N, dx, dt, steps, alpha, beta, phi_init, phi_dt_init, record_states=True
    )

    # Compute total energy at each time step
    energies = [np.sum(compute_energy_density(phi_t, phi_dt_t, dx)) for phi_t, phi_dt_t in zip(phi_hist, phi_dt_hist)]

    # Save results to HDF5
    os.makedirs('results', exist_ok=True)
    output_file = 'results/dynamic_evolution.h5'
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('energies', data=np.array(energies))

    # Summary
    print(f"Dynamic evolution energies saved to {output_file}")
    print(f"Initial energy: {energies[0]:.6f}, final energy: {energies[-1]:.6f}")

if __name__ == '__main__':
    main()
