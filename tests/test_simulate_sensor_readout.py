import json
import pytest
import subprocess
import sys
import pathlib

def test_simulate_sensor_readout(tmp_path, monkeypatch):
    # Create dummy HDF5 with energies dataset
    import h5py
    import numpy as np
    data_file = tmp_path / 'energies.h5'
    with h5py.File(data_file, 'w') as f:
        f.create_dataset('energies', data=np.linspace(0,1,10))
    # Run CLI tool
    output_file = tmp_path / 'sensor_readout.json'
    script = pathlib.Path(__file__).parent.parent / 'scripts' / 'simulate_sensor_readout.py'
    cmd = [sys.executable, str(script),
           '--data', str(data_file),
           '--noise-std', '0.0',
           '--gain', '2.0',
           '--seed', '0',
           '--output', str(output_file)]
    subprocess.check_call(cmd)
    data = json.loads(output_file.read_text())
    assert data['gain'] == 2.0
    assert data['noise_std'] == 0.0
    assert data['mean_reading'] == pytest.approx(np.mean(np.linspace(0,1,10))*2.0)
    assert data['n_steps'] == 10
