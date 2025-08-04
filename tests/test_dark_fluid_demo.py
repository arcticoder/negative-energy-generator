import subprocess
import sys
import json
import pytest
import pathlib

def test_dark_fluid_demo_script(tmp_path):
    """
    Integration test: run dark_fluid_demo.py and verify JSON output.
    """
    script = pathlib.Path(__file__).parent.parent / 'scripts' / 'dark_fluid_demo.py'
    # Change working directory to tmp_path
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, f"Demo failed: {result.stderr}"
    output_file = tmp_path / 'results' / 'dark_fluid_demo.json'
    assert output_file.exists(), "dark_fluid_demo.json not created"
    data = json.loads(output_file.read_text())
    assert 'anec_integral' in data
    assert 'anec_violated' in data
    assert 'max_h' in data
    assert isinstance(data['anec_integral'], float)
    assert isinstance(data['anec_violated'], bool)
    assert isinstance(data['max_h'], float)
