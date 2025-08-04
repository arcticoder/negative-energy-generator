import subprocess
import sys
import json
import pathlib

def test_backreaction_uq_script(tmp_path):
    """
    Integration test: run backreaction_uq.py script and verify JSON output metrics.
    """
    script = pathlib.Path(__file__).parent.parent / 'scripts' / 'backreaction_uq.py'
    output_file = tmp_path / 'uq_metrics.json'
    cmd = [sys.executable, str(script),
           '--samples', '5',
           '--N', '20',
           '--dx', '0.1',
           '--dt', '0.01',
           '--steps', '10',
           '--output', str(output_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_file.exists(), "Output JSON file not created"
    data = json.loads(output_file.read_text())
    assert 'mean_max_h' in data and 'std_max_h' in data and data['samples'] == 5
    # Validate value ranges
    assert 0.0 <= data['mean_max_h']
    assert 0.0 <= data['std_max_h']
