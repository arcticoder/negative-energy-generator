import os
import sys
import pathlib
# Add 'src' directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import h5py  # type: ignore
from simulation.parameter_sweep import parameter_sweep  # type: ignore

def test_parameter_sweep_export(tmp_path):
    # Simple sweep with no evolution steps to produce zero energy density
    alpha_values = [0.1]
    beta_values = [0.0]
    grid_sizes = [10]
    dx = 1.0
    dt = 0.1
    steps = 0  # no time evolution
    output_file = tmp_path / "test_export.h5"

    # Run parameter sweep
    parameter_sweep(alpha_values, beta_values, grid_sizes, dx, dt, steps, str(output_file))

    # Check file created
    assert output_file.exists(), "HDF5 output file was not created."

    # Validate dataset content
    with h5py.File(output_file, 'r') as f:
        keys = list(f.keys())
        assert keys == ['alpha_0.1_beta_0.0_N_10']
        data = f[keys[0]][:]
        assert data.shape == (10,), "Dataset shape mismatch."
        # With zero evolution steps, energy density should be zero vector
        assert all(data == 0), "Energy density should be zero for zero evolution steps."
