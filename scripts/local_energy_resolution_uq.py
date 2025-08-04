#!/usr/bin/env python3
"""
CLI tool for grid resolution uncertainty propagation in local energy density.
"""
import os
import json
import numpy as np
import argparse
from simulation.qft_backend import PhysicsCore

def main():
    parser = argparse.ArgumentParser(description="Grid resolution UQ for local energy density")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ansatz alpha parameter")
    parser.add_argument("--beta", type=float, default=1.0, help="Ansatz beta parameter")
    parser.add_argument("--N", nargs="+", type=int, default=[50, 100, 200], help="Grid sizes to test")
    parser.add_argument("--L", type=float, default=10.0, help="Physical domain length")
    parser.add_argument("--output", type=str, default="results/local_energy_resolution_uq.json", help="Output JSON file")
    args = parser.parse_args()

    integrals = []
    dx_values = []
    for N in args.N:
        xs = np.linspace(-args.L/2, args.L/2, N)
        dx = xs[1] - xs[0]
        core = PhysicsCore(grid=(xs, xs, xs), dx=dx)
        T = core.build_toy_ansatz({'alpha': args.alpha, 'beta': args.beta})
        rho = core.local_energy_density(T)
        integral = core.compute_anec(rho, dx)
        integrals.append(integral)
        dx_values.append(dx)

    mean = float(np.mean(integrals))
    std = float(np.std(integrals))

    results = {
        "alpha": args.alpha,
        "beta": args.beta,
        "N_values": args.N,
        "dx_values": dx_values,
        "integrals": integrals,
        "mean_integral": mean,
        "std_integral": std
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Mean integral: {mean:.6f}, Std: {std:.6f}")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
