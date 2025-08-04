#!/usr/bin/env python3
"""
Monte Carlo UQ for toy stress-energy ansatz parameters using PhysicsCore.
"""
import os
import json
import argparse
import numpy as np
from simulation.qft_backend import PhysicsCore


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo UQ for toy ansatz parameters α and β")
    parser.add_argument("--alpha-mean", type=float, default=1.0, help="Mean of α distribution")
    parser.add_argument("--alpha-std", type=float, default=0.1, help="Stddev of α distribution")
    parser.add_argument("--beta-mean", type=float, default=1.0, help="Mean of β distribution")
    parser.add_argument("--beta-std", type=float, default=0.1, help="Stddev of β distribution")
    parser.add_argument("--samples", type=int, default=100, help="Number of Monte Carlo samples")
    parser.add_argument("--grid-size", type=int, default=32, help="Number of grid points per dimension")
    parser.add_argument("--dx", type=float, default=0.1, help="Grid spacing dx")
    parser.add_argument("--output", type=str, default="results/qft_uq_metrics.json", help="Output JSON file for UQ metrics")
    args = parser.parse_args()

    # Define spatial grid
    x = np.linspace(-1.0, 1.0, args.grid_size)
    core = PhysicsCore(grid=(x, x, x), dx=args.dx)

    negatives = []
    for _ in range(args.samples):
        # sample ansatz parameters
        alpha = np.random.normal(args.alpha_mean, args.alpha_std)
        beta = np.random.normal(args.beta_mean, args.beta_std)
        T = core.build_toy_ansatz({'alpha': alpha, 'beta': beta})
        rho = core.local_energy_density(T)
        mask = core.find_negative(rho)
        fraction = mask.sum() / mask.size
        negatives.append(fraction)

    metrics = {
        'mean_negative_fraction': float(np.mean(negatives)),
        'std_negative_fraction': float(np.std(negatives)),
        'samples': args.samples
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"UQ metrics saved to {args.output}")


if __name__ == '__main__':
    main()
