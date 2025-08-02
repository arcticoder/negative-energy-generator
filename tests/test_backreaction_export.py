import os
import sys
import pathlib
import subprocess
import h5py
import numpy as np

def test_backreaction_demo_export(tmp_path, monkeypatch):
    # Ensure scripts are importable
    repo_root = pathlib.Path(__file__).parent.parent
    # Run backreaction_demo.py with cwd set to tmp_path
    result = subprocess.run(
        [sys.executable, str(repo_root / 'scripts' / 'backreaction_demo.py')],
        cwd=str(tmp_path),
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    # Check output file
    output_file = tmp_path / 'results' / 'backreaction_demo.h5'
    assert output_file.exists(), "Backreaction demo output file not found"

    # Validate contents
    with h5py.File(output_file, 'r') as f:
        assert 'rho' in f and 'h_final' in f and 'h_history' in f
        rho = f['rho'][:]
        h_final = f['h_final'][:]
        h_history = f['h_history'][:]
        # Basic sanity checks
        assert isinstance(rho, np.ndarray)
        assert isinstance(h_final, np.ndarray)
        assert isinstance(h_history, np.ndarray)
        # h_history should have first row zeros and match h_final last row
        assert np.allclose(h_history[0], 0)
        assert np.allclose(h_history[-1], h_final)
