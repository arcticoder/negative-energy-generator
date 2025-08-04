import subprocess
import sys
import pathlib
import pytest


def test_dynamic_evolution_plot(tmp_path, monkeypatch):
    """Integration test: run dynamic_evolution_report.py and verify plot file creation."""
    project_root = pathlib.Path(__file__).parent.parent
    monkeypatch.chdir(project_root)

    # Ensure metrics and HDF5 exist by running analysis and report
    analysis_script = project_root / 'scripts' / 'dynamic_evolution_analysis.py'
    subprocess.run([sys.executable, str(analysis_script)], check=True)
    report_script = project_root / 'scripts' / 'dynamic_evolution_report.py'
    subprocess.run([sys.executable, str(report_script)], check=True)

    # Verify plot file created
    plot_file = project_root / 'results' / 'dynamic_evolution_plot.png'
    assert plot_file.exists(), f"Expected plot file not found: {plot_file}"
