#!/usr/bin/env python3
"""
CLI tool for sensor-field conversion calibration: simulate sensor readouts from energy time-series with noise and gain.
"""
import os
import json
import argparse
import numpy as np
import h5py

def main():
    parser = argparse.ArgumentParser(description="Simulate sensor readout with noise and gain")
    parser.add_argument("--data", type=str,
                        default="results/dynamic_evolution.h5",
                        help="HDF5 file with 'energies' dataset")
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Standard deviation of Gaussian noise")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Sensor gain factor")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for noise generation")
    parser.add_argument("--output", type=str,
                        default="results/sensor_readout.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    # Load energy time-series
    with h5py.File(args.data, 'r') as f:
        if 'energies' not in f:
            print(f"Dataset 'energies' not found in {args.data}", file=sys.stderr)
            exit(1)
        energies = f['energies'][:]

    # Simulate readings
    rng = np.random.default_rng(args.seed)
    noise = rng.normal(loc=0.0, scale=args.noise_std, size=energies.shape)
    readings = args.gain * energies + noise

    # Metrics
    mean_reading = float(np.mean(readings))
    std_reading = float(np.std(readings))

    # Prepare output
    results = {
        'gain': args.gain,
        'noise_std': args.noise_std,
        'mean_reading': mean_reading,
        'std_reading': std_reading,
        'n_steps': int(energies.size)
    }
    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Sensor readout simulation saved to {args.output}")
    print(f"Mean reading: {mean_reading:.6f}, Std: {std_reading:.6f}")

if __name__ == '__main__':
    main()
