import json
import subprocess
import sys
import pathlib

def test_dark_fluid_warp_drive_uq(tmp_path, monkeypatch):
    # Run CLI tool for warp drive UQ
    output = tmp_path / 'warp_drive_uq.json'
    script = pathlib.Path(__file__).parent.parent / 'scripts' / 'dark_fluid_warp_drive_uq.py'
    cmd = [sys.executable, str(script),
           '--rho0', '-1.0',
           '--R_values', '1.0', '2.0',
           '--sigma_values', '0.2',
           '--N', '50',
           '--dx', '0.1',
           '--dt', '0.05',
           '--steps', '10',
           '--output', str(output)]
    subprocess.check_call(cmd)
    data = json.loads(output.read_text())
    assert 'metrics' in data
    assert isinstance(data['metrics'], list)
    assert len(data['metrics']) == 2
    for entry in data['metrics']:
        assert 'R' in entry and 'sigma' in entry
        assert 'anec_integral' in entry
        assert 'max_h' in entry
