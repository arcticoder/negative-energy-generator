import json
import subprocess
import sys
import pytest


def test_local_energy_resolution_uq(tmp_path):
    # Prepare output path
    output = tmp_path / "local_energy_uq_results.json"
    # Run the CLI tool
    cmd = [sys.executable, "scripts/local_energy_resolution_uq.py",
           "--alpha", "1.0", "--beta", "1.0",
           "--N", "10", "20",
           "--L", "5.0",
           "--output", str(output)]
    subprocess.check_call(cmd)
    # Load results
    data = json.loads(output.read_text())
    # Validate JSON structure
    assert data["alpha"] == 1.0
    assert data["beta"] == 1.0
    assert "N_values" in data and data["N_values"] == [10, 20]
    assert "dx_values" in data and len(data["dx_values"]) == 2
    assert "integrals" in data and len(data["integrals"]) == 2
    assert isinstance(data["mean_integral"], float)
    assert isinstance(data["std_integral"], float)
