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
    parser = argparse.ArgumentParser(description="Backreaction demo CLI")
    parser.add_argument("--output", type=str,
                        default="results/backreaction_demo.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--N", type=int, default=100, help="Grid size N")
    parser.add_argument("--dx", type=float, default=0.01, help="Spatial step dx")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step dt")
    parser.add_argument("--steps", type=int, default=50, help="Number of time steps")
    parser.add_argument("--alpha", type=float, default=0.5, help="Ansatz alpha parameter")
    parser.add_argument("--beta", type=float, default=0.1, help="Ansatz beta parameter")
    args = parser.parse_args()

    # Demo parameters for lattice QFT
    N = args.N
    dx = args.dx
    dt = args.dt
    steps = args.steps
    alpha = args.alpha
    beta = args.beta
    x = np.linspace(0, N*dx, N)

    # Solve Klein-Gordon to get field and derivative
    phi, phi_dt = solve_klein_gordon(N, dx, dt, steps, alpha, beta)
    # Compute energy density T00 = rho
    rho = compute_energy_density(phi, phi_dt)

    # Solve backreaction metric evolution
    h_final, history = solve_semiclassical_metric(x, rho, dt=dt, steps=steps)

    # Prepare output
    output_file = args.output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('rho', data=rho)
        f.create_dataset('h_final', data=h_final)
        f.create_dataset('h_history', data=history)

    print(f"Backreaction demo results saved to {output_file}")
    print(f"Final metric perturbation mean: {np.mean(h_final):.3e}")

if __name__ == "__main__":
    main()
