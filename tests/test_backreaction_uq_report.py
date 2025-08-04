import subprocess
import sys
import os
import json
import pytest
import pathlib

def test_backreaction_uq_report_script(tmp_path):
    """
    Integration test: run backreaction_uq_report.py and verify JSON report and plot PNG creation.
    """
    script = pathlib.Path(__file__).parent.parent / 'scripts' / 'backreaction_uq_report.py'
    # Prepare fake metrics JSON
    metrics = {'mean_max_h': 0.5, 'std_max_h': 0.1, 'samples': 3}
    input_file = tmp_path / 'metrics.json'
    with open(input_file, 'w') as f:
        json.dump(metrics, f)
    output_plot = tmp_path / 'uq_plot.png'
    cmd = [sys.executable, str(script), '--input', str(input_file), '--output-plot', str(output_plot)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Report script failed: {result.stderr}"
    # Verify plot file exists and is non-empty
    assert output_plot.exists(), "Plot PNG was not created"
    assert output_plot.stat().st_size > 0, "Plot PNG is empty"
