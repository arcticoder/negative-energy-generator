#!/usr/bin/env python3
"""
Command-line demo for semiclassical backreaction: couples lattice QFT output to metric evolution solver.
"""
import os
import numpy as np
import h5py
from simulation.lattice_qft import solve_klein_gordon, compute_energy_density
from simulation.backreaction import solve_semiclassical_metric


def main():
    # Demo parameters for lattice QFT
    N = 100
    dx = 0.01
    dt = 0.001
    steps = 50
    alpha = 0.5
    beta = 0.1
    x = np.linspace(0, N*dx, N)

    # Solve Klein-Gordon to get field and derivative
    phi, phi_dt = solve_klein_gordon(N, dx, dt, steps, alpha, beta)
    # Compute energy density T00 = rho
    rho = compute_energy_density(phi, phi_dt)

    # Solve backreaction metric evolution
    h_final, history = solve_semiclassical_metric(x, rho, dt=dt, steps=steps)

    # Prepare output
    output_file = "results/backreaction_demo.h5"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('rho', data=rho)
        f.create_dataset('h_final', data=h_final)
        f.create_dataset('h_history', data=history)

    print(f"Backreaction demo results saved to {output_file}")
    print(f"Final metric perturbation mean: {np.mean(h_final):.3e}")

if __name__ == "__main__":
    main()
