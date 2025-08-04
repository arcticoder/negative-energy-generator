import os
import sys
import pathlib
import subprocess
import h5py

def test_lattice_sweep_demo(tmp_path, monkeypatch):
    """
    Integration test: run lattice_sweep_demo.py and verify HDF5 output.
    """
    demo_script = pathlib.Path(__file__).parent.parent / 'scripts' / 'lattice_sweep_demo.py'
    # Change working directory to tmp_path to avoid polluting repo
    monkeypatch.chdir(tmp_path)
    # Run the demo script
    result = subprocess.run([sys.executable, str(demo_script)], capture_output=True, text=True)
    assert result.returncode == 0, f"Demo script failed: {result.stderr}"
    # Check HDF5 output
    output_file = tmp_path / 'results' / 'demo_sweep.h5'
    assert output_file.exists(), "Demo HDF5 file not created"
    # Verify dataset
    with h5py.File(output_file, 'r') as f:
        keys = list(f.keys())
        assert len(keys) == 1, "Expected one dataset"
        data = f[keys[0]][:]
        # Default grid size is 100
        assert data.shape == (100,), f"Unexpected data shape: {data.shape}"
