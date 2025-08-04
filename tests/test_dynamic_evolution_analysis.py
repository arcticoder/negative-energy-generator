import json
import subprocess
import sys
import pathlib

def test_dynamic_evolution_analysis(tmp_path, monkeypatch):
    """Integration test: run dynamic_evolution_analysis and verify JSON metrics file contents."""
    project_root = pathlib.Path(__file__).parent.parent
    monkeypatch.chdir(project_root)

    # Ensure dynamic_evolution.h5 exists by running the demo if needed
    demo_script = project_root / 'scripts' / 'dynamic_evolution_demo.py'
    subprocess.run([sys.executable, str(demo_script)], check=True)

    # Run analysis script
    analysis_script = project_root / 'scripts' / 'dynamic_evolution_analysis.py'
    subprocess.run([sys.executable, str(analysis_script)], check=True)

    # Verify JSON metrics file
    metrics_file = project_root / 'results' / 'dynamic_evolution_metrics.json'
    assert metrics_file.exists(), f"Metrics file not found: {metrics_file}"

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Check required keys and reasonable values
    for key in ['initial_energy', 'final_energy', 'mean_drift', 'max_drift', 'std_drift', 'n_steps']:
        assert key in metrics, f"Missing metric: {key}"
    assert metrics['n_steps'] > 1, "Expected multiple time steps in metrics"
    assert metrics['max_drift'] >= metrics['mean_drift'], "Max drift should be at least as large as mean drift"
