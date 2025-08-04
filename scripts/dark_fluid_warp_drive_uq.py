#!/usr/bin/env python3
"""
CLI tool for Warp Drive Parameter UQ using dark fluid warp bubble profiles.
"""
import os
import json
import argparse
import numpy as np
from simulation.dark_fluid import generate_warp_bubble_fluid
from simulation.qft_backend import PhysicsCore
from simulation.backreaction import solve_semiclassical_metric

def main():
    parser = argparse.ArgumentParser(description="Warp Drive Dark Fluid UQ")
    parser.add_argument("--rho0", type=float, default=-1.0, help="Amplitude of negative density")
    parser.add_argument("--R_values", nargs='+', type=float, default=[1.0], help="Bubble radius values for sampling")
    parser.add_argument("--sigma_values", nargs='+', type=float, default=[0.2], help="Width parameter values for sampling")
    parser.add_argument("--N", type=int, default=100, help="Grid size")
    parser.add_argument("--dx", type=float, default=0.1, help="Spatial step dx")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step dt for backreaction")
    parser.add_argument("--steps", type=int, default=20, help="Time steps for backreaction")
    parser.add_argument("--output", type=str, default="results/dark_fluid_warp_drive_uq.json", help="Output JSON file")
    args = parser.parse_args()

    metrics = []
    # Prepare grid for PhysicsCore
    xs = None
    for R in args.R_values:
        for sigma in args.sigma_values:
            # Generate fluid profile
            x, rho = generate_warp_bubble_fluid(args.N, args.dx, args.rho0, R, sigma)
            # Setup PhysicsCore for ANEC
            core = PhysicsCore(grid=(x, x, x), dx=args.dx)
            anec_val = core.compute_anec(rho, args.dx)
            violated = core.check_anec(rho, args.dx)
            # Simulate backreaction
            h_final, history = solve_semiclassical_metric(x, rho, dt=args.dt, steps=args.steps)
            max_h = float(np.max(np.abs(h_final)))
            # Aggregate
            metrics.append({
                'R': R,
                'sigma': sigma,
                'anec_integral': anec_val,
                'anec_violated': violated,
                'max_h': max_h
            })
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'metrics': metrics}, f, indent=2)
    print(f"Warp drive UQ metrics saved to {args.output}")

if __name__ == '__main__':
    main()
