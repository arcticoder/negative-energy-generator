#!/usr/bin/env python3
"""
Demo for Dark Fluid models: generate negative-mass fluid, compute ANEC, and simulate backreaction.
"""
import os
import json
import numpy as np
from simulation.dark_fluid import generate_negative_mass_fluid
from simulation.qft_backend import PhysicsCore
from simulation.backreaction import solve_semiclassical_metric

def main():
    # Demo parameters
    N = 100
    dx = 0.1
    rho0 = -1.0
    dt = 0.05
    steps = 20

    # Generate negative mass fluid profile
    x, rho = generate_negative_mass_fluid(N, dx, rho0)

    # Compute ANEC integral
    core = PhysicsCore(grid=(x, x, x), dx=dx)
    anec_val = core.compute_anec(rho, dx)
    violated = core.check_anec(rho, dx)

    # Simulate backreaction for metric perturbation
    h_final, history = solve_semiclassical_metric(x, rho, dt=dt, steps=steps)
    max_h = float(np.max(np.abs(h_final)))

    # Save results
    os.makedirs('results', exist_ok=True)
    output_file = 'results/dark_fluid_demo.json'
    results = {
        'anec_integral': anec_val,
        'anec_violated': violated,
        'max_h': max_h
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"ANEC integral: {anec_val:.6f} (violated: {violated})")
    print(f"Max metric perturbation |h|: {max_h:.6f}")
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
