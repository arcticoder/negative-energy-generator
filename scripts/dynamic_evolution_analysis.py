#!/usr/bin/env python3
"""
Compute uncertainty metrics for dynamic evolution energy time-series.
"""
import os
import sys
import pathlib
import h5py
import numpy as np
import json

def main():
    # Determine project root
    root = pathlib.Path(__file__).parent.parent
    data_file = root / 'results' / 'dynamic_evolution.h5'
    if not data_file.exists():
        print(f"Data file not found: {data_file}", file=sys.stderr)
        sys.exit(1)

    # Load energies
    with h5py.File(data_file, 'r') as f:
        energies = f['energies'][:]

    # Compute drift relative to initial
    E0 = energies[0]
    drift = np.abs(energies - E0)

    metrics = {
        'initial_energy': float(E0),
        'final_energy': float(energies[-1]),
        'mean_drift': float(drift.mean()),
        'max_drift': float(drift.max()),
        'std_drift': float(drift.std()),
        'n_steps': int(energies.size)
    }

    # Output JSON file
    out_dir = root / 'results'
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / 'dynamic_evolution_metrics.json'
    with open(out_file, 'w') as jf:
        json.dump(metrics, jf, indent=2)

    # Print summary
    print("Dynamic Evolution Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v}")

    print(f"Metrics saved to {out_file}")

if __name__ == '__main__':
    main()
