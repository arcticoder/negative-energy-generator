#!/usr/bin/env python3
"""
Generate a CLI report of dynamic evolution energy metrics.
"""
import json
import sys
import pathlib

def main():
    project_root = pathlib.Path(__file__).parent.parent
    metrics_file = project_root / 'results' / 'dynamic_evolution_metrics.json'
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}", file=sys.stderr)
        sys.exit(1)

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    print("Dynamic Evolution Report:")
    print(f"  Initial Energy   : {metrics.get('initial_energy')}")
    print(f"  Final Energy     : {metrics.get('final_energy')}")
    print(f"  Mean Drift       : {metrics.get('mean_drift')}")
    print(f"  Max Drift        : {metrics.get('max_drift')}")
    print(f"  Std Dev of Drift : {metrics.get('std_drift')}")
    print(f"  Number of Steps  : {metrics.get('n_steps')}")
    # Attempt to plot energy time-series
    try:
        import h5py
        import matplotlib.pyplot as plt
        h5_file = project_root / 'results' / 'dynamic_evolution.h5'
        with h5py.File(h5_file, 'r') as hf:
            energies = hf['energies'][:]
        plt.figure()
        plt.plot(energies)
        plt.xlabel('Time Step')
        plt.ylabel('Total Energy')
        plt.title('Dynamic Evolution Energy Drift')
        plot_file = project_root / 'results' / 'dynamic_evolution_plot.png'
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")
    except ImportError:
        print("Matplotlib or h5py not available; skipping plot generation.", file=sys.stderr)
    except Exception as e:
        print(f"Error generating plot: {e}", file=sys.stderr)
    sys.exit(0)

if __name__ == '__main__':
    main()
