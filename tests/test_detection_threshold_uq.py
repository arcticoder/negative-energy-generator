import json
import subprocess
import sys

def test_detection_threshold_uq(tmp_path):
    output = tmp_path / "threshold_uq.json"
    # Run the CLI tool with two thresholds
    cmd = [sys.executable, "scripts/detection_threshold_uq.py",
           "--alpha", "1.0", "--beta", "1.0",
           "--thresholds", "-0.5", "0.0", "0.5",
           "--N", "20", "--L", "4.0",
           "--output", str(output)]
    subprocess.check_call(cmd)
    data = json.loads(output.read_text())
    # Validate structure
    assert data["alpha"] == 1.0
    assert data["beta"] == 1.0
    assert data["thresholds"] == [-0.5, 0.0, 0.5]
    assert "fractions" in data and len(data["fractions"]) == 3
    # Fractions should be between 0 and 1
    for frac in data["fractions"]:
        assert 0.0 <= frac <= 1.0
