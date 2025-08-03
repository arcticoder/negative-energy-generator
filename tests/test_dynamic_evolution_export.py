import os
import subprocess
import h5py
import numpy as np
import pathlib
import pytest


def test_dynamic_evolution_demo_export(tmp_path, monkeypatch):
    """
    Integration test: run dynamic_evolution_demo and ensure HDF5 export contains energies dataset
    """
    # Change to project root directory
    project_root = pathlib.Path(__file__).parent.parent
    monkeypatch.chdir(project_root)

    # Clean up any existing results
    results_dir = project_root / 'results'
    if results_dir.exists():
        for f in results_dir.iterdir():
            f.unlink()
    
    # Run the demo script
    cmd = ['python', 'scripts/dynamic_evolution_demo.py']
    subprocess.run(cmd, check=True)

    # Verify output file
    output_file = results_dir / 'dynamic_evolution.h5'
    assert output_file.exists(), f"Expected HDF5 file not found: {output_file}"

    # Inspect dataset
    with h5py.File(output_file, 'r') as f:
        assert 'energies' in f, "Dataset 'energies' missing in HDF5 file"
        energies = f['energies'][:]
        # Should have steps+1 entries
        assert energies.ndim == 1 and energies.size > 1, "Energies array has unexpected shape"
        # Check energy variation is small
        E0 = energies[0]
        rel_var = np.abs(energies - E0) / E0
        assert np.all(rel_var < 2e-3), "Energy variation exceeds tolerance"
