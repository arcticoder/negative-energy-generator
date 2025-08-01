#!/usr/bin/env python3
"""
Command-line demo showcasing lattice QFT parameter sweep and result inspection.
"""
import os
import sys
import pathlib
# Ensure the 'src' directory is on the path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import h5py  # type: ignore
from simulation.parameter_sweep import parameter_sweep  # type: ignore


def main():
    # Demo parameters
    alpha_values = [0.5]
    beta_values = [0.1]
    grid_sizes = [100]
    dx = 0.01
    dt = 0.001
    steps = 50
    output_file = "results/demo_sweep.h5"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Running parameter sweep demo -> {output_file}")
    parameter_sweep(alpha_values, beta_values, grid_sizes, dx, dt, steps, output_file)

    # Inspect generated results
    with h5py.File(output_file, 'r') as f:
        print("Generated datasets:")
        for name in f.keys():
            data = f[name][:]
            print(f" - {name}: shape {data.shape}, mean energy density {data.mean():.3e}")

if __name__ == "__main__":
    main()
