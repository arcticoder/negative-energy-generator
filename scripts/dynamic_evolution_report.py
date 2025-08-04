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
    sys.exit(0)

if __name__ == '__main__':
    main()
