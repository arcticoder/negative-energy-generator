#!/usr/bin/env python3
"""
Orchestrate parameter sweeps for 1+1D lattice QFT energy density calculations
"""
import argparse
import numpy as np
import h5py
from simulation.lattice_qft import solve_klein_gordon, compute_energy_density


def parameter_sweep(alpha_values, beta_values, grid_sizes, dx, dt, steps, output_file):
    """
    Run sweeps over ansatz parameters (alpha, beta) and grid sizes.
    Saves resulting energy densities into an HDF5 file.
    """
    with h5py.File(output_file, 'w') as f:
        for alpha in alpha_values:
            for beta in beta_values:
                for N in grid_sizes:
                    # Solve Klein-Gordon field evolution
                    phi, phi_dt = solve_klein_gordon(N, dx, dt, steps, alpha, beta)
                    # Compute energy density
                    rho = compute_energy_density(phi, phi_dt, dx)
                    dataset_name = f"alpha_{alpha}_beta_{beta}_N_{N}"
                    f.create_dataset(dataset_name, data=rho)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lattice QFT parameter sweep")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.1, 1.0], help="Alpha values")
    parser.add_argument("--beta", nargs="+", type=float, default=[0.1, 1.0], help="Beta values")
    parser.add_argument("--grid", nargs="+", type=int, default=[100, 200], help="Grid sizes (number of points)")
    parser.add_argument("--dx", type=float, default=0.01, help="Spatial step size")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step size")
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps")
    parser.add_argument("--output", type=str, default="results/parameter_sweep.h5", help="Output HDF5 file path")
    args = parser.parse_args()
    parameter_sweep(args.alpha, args.beta, args.grid, args.dx, args.dt, args.steps, args.output)
