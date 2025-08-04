#!/usr/bin/env python3
"""
CLI tool for detection threshold sensitivity for negative-energy detection in PhysicsCore.
"""
import os
import json
import argparse
import numpy as np
from simulation.qft_backend import PhysicsCore

def main():
    parser = argparse.ArgumentParser(description="Detection threshold sensitivity UQ for negative-energy detection")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ansatz alpha parameter")
    parser.add_argument("--beta", type=float, default=1.0, help="Ansatz beta parameter")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0], help="Threshold values for rho below which negative detected")
    parser.add_argument("--N", type=int, default=50, help="Grid size for each dimension")
    parser.add_argument("--L", type=float, default=10.0, help="Physical domain length")
    parser.add_argument("--output", type=str, default="results/detection_threshold_uq.json", help="Output JSON file path")
    args = parser.parse_args()

    # Prepare spatial grid
    xs = np.linspace(-args.L/2, args.L/2, args.N)
    core = PhysicsCore(grid=(xs, xs, xs), dx=xs[1] - xs[0])
    # Compute base energy density
    T = core.build_toy_ansatz({'alpha': args.alpha, 'beta': args.beta})
    rho = core.local_energy_density(T)

    fractions = []
    for thr in args.thresholds:
        mask = rho < thr
        frac = float(mask.sum()) / mask.size
        fractions.append(frac)

    results = {
        'alpha': args.alpha,
        'beta': args.beta,
        'thresholds': args.thresholds,
        'fractions': fractions,
        'N': args.N,
        'L': args.L
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detection threshold UQ saved to {args.output}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Negative fractions: {fractions}")

if __name__ == '__main__':
    main()
