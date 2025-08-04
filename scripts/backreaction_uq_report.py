#!/usr/bin/env python3
"""
Generate report and plot for backreaction UQ metrics.
"""
import os
import json
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Report and plot for backreaction UQ metrics")
    parser.add_argument("--input", type=str, default="results/backreaction_uq_metrics.json", help="Input JSON file for UQ metrics")
    parser.add_argument("--output-plot", type=str, default="results/backreaction_uq_plot.png", help="Output PNG plot path")
    args = parser.parse_args()

    # Load metrics
    with open(args.input, 'r') as f:
        data = json.load(f)
    mean_h = data.get('mean_max_h')
    std_h = data.get('std_max_h')
    samples = data.get('samples')

    # Print report
    print(f"Backreaction UQ Report:")
    print(f"Samples: {samples}")
    print(f"Mean max |h|: {mean_h:.6f}")
    print(f"Std of max |h|: {std_h:.6f}")

    # Simple plot: bar chart
    metrics = [mean_h - std_h, mean_h, mean_h + std_h]
    labels = ['mean-std', 'mean', 'mean+std']
    plt.figure()
    plt.bar(labels, metrics, color=['red','blue','green'])
    plt.ylabel('Max |h|')
    plt.title('Backreaction UQ Metrics')

    # Save plot
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
    plt.savefig(args.output_plot)
    print(f"Plot saved to {args.output_plot}")

if __name__ == '__main__':
    main()
