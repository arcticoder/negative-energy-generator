import json
import subprocess
import sys
import tempfile
import os
import pytest


def test_qft_toy_ansatz_uq_script(tmp_path):
    """
    Integration test: run qft_toy_ansatz_uq.py script and verify JSON metrics output.
    """
    script = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'qft_toy_ansatz_uq.py')
    output_file = tmp_path / 'uq_metrics.json'
    # Run script with minimal samples and grid size
    cmd = [sys.executable, script,
           '--samples', '5',
           '--grid-size', '4',
           '--dx', '0.5',
           '--output', str(output_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    # Verify output file exists
    assert output_file.exists(), "UQ metrics file was not created"
    # Verify JSON content
    data = json.loads(output_file.read_text())
    assert 'mean_negative_fraction' in data
    assert 'std_negative_fraction' in data
    assert data['samples'] == 5
    # Values should be between 0 and 1
    mean_frac = data['mean_negative_fraction']
    std_frac = data['std_negative_fraction']
    assert 0.0 <= mean_frac <= 1.0
    assert 0.0 <= std_frac <= 1.0
