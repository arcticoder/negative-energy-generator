import subprocess
import sys
import pathlib
import json

def test_dynamic_evolution_report(tmp_path, monkeypatch, capsys):
    """Integration test: run dynamic_evolution_report.py and verify CLI output and metrics."""
    # Prepare environment
    project_root = pathlib.Path(__file__).parent.parent
    monkeypatch.chdir(project_root)

    # Ensure metrics file exists by running analysis
    analysis_script = project_root / 'scripts' / 'dynamic_evolution_analysis.py'
    subprocess.run([sys.executable, str(analysis_script)], check=True)

    # Run the report script and capture output
    report_script = project_root / 'scripts' / 'dynamic_evolution_report.py'
    result = subprocess.run([sys.executable, str(report_script)], capture_output=True, text=True, check=True)

    # Check output contains expected report lines
    output = result.stdout
    assert "Dynamic Evolution Report:" in output
    for key in ["Initial Energy", "Final Energy", "Mean Drift", "Max Drift", "Std Dev of Drift", "Number of Steps"]:
        assert key in output, f"Expected '{key}' in report output"

    # Also verify that metrics file JSON matches CLI output values
    metrics_file = project_root / 'results' / 'dynamic_evolution_metrics.json'
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Check numerical values appear in output
    assert str(metrics.get('initial_energy')) in output
    assert str(metrics.get('n_steps')) in output
