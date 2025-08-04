#!/usr/bin/env python3
"""
Monte Carlo UQ for semiclassical backreaction metric response under varied QFT stresses.
"""
import os
import json
import argparse
import numpy as np
from simulation.lattice_qft import solve_klein_gordon, compute_energy_density
from simulation.backreaction import solve_semiclassical_metric

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo UQ for backreaction metric response")
    parser.add_argument("--samples", type=int, default=50, help="Number of Monte Carlo samples")
    parser.add_argument("--N", type=int, default=50, help="Number of lattice points for QFT")
    parser.add_argument("--dx", type=float, default=0.1, help="Spatial step size dx")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size dt")
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps for both QFT and backreaction")
    parser.add_argument("--output", type=str, default="results/backreaction_uq_metrics.json", help="Output JSON file for UQ metrics")
    args = parser.parse_args()

    N = args.N
    dx = args.dx
    dt = args.dt
    steps = args.steps
    x = np.linspace(0, N*dx, N, endpoint=False)

    metrics = []
    for i in range(args.samples):
        # Random QFT parameters
        alpha = np.random.uniform(0.0, 1.0)
        beta = np.random.uniform(0.0, 0.5)
        # QFT solve
        phi, phi_dt = solve_klein_gordon(N, dx, dt, steps, alpha, beta)
        rho = compute_energy_density(phi, phi_dt, dx)
        # Backreaction
        h_final, _ = solve_semiclassical_metric(x, rho, dt=dt, steps=steps)
        # Metric response metric: max absolute perturbation
        metrics.append(float(np.max(np.abs(h_final))))

    results = {
        "mean_max_h": float(np.mean(metrics)),
        "std_max_h": float(np.std(metrics)),
        "samples": args.samples
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Backreaction UQ metrics saved to {args.output}")

if __name__ == '__main__':
    main()
