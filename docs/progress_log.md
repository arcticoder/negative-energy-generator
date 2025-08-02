# Progress Log

## 2025-08-01

### Repository Status
- Negative-energy-generator framework repository active; core QFT backend and analysis modules implemented.

### Completed to Date
- Implemented core simulation methods: `local_energy_density`, `find_negative`, `evolve_QFT`, `build_quantum_op_tensor`, `detect_exotics`, and `verify_conservation`.
- Defined Verification & Validation (V&V) tasks in `VnV-TODO.ndjson` and Uncertainty Quantification (UQ) tasks in `UQ-TODO.ndjson` for QFT backend features.
- Created `docs/literature_review.md` summarizing key publications on macroscopic negative energy sources.

### In Progress
- Integration tests for stress-energy conservation (`verify_conservation`).
- UQ experiments: Monte Carlo sampling, time-step sensitivity, grid resolution analysis for QFT outputs.
- Implementing 1+1D lattice QFT solver in `src/simulation/lattice_qft.py` for real scalar fields (finite-difference Klein–Gordon).
- Developed parameter sweep orchestrator in `src/simulation/parameter_sweep.py` for ansatz and grid sweeps.
- Created unit tests in `tests/test_lattice_energy.py` validating energy density calculations and solver output shapes.
- Created command-line demo `scripts/lattice_sweep_demo.py` showcasing sweep execution and result summaries.
- Created discretization accuracy test in `tests/test_lattice_discretization.py` verifying discrete Laplacian against analytical solution.
- Scaffolding GitHub Actions CI workflow in `.github/workflows/ci.yml`.
- Added `pandas` to project dependencies and CI install step to satisfy data-frame tests.

### Next Tasks
1. Update V&V and UQ trackers with tasks for lattice solver validation and uncertainty quantification of lattice energy densities.
2. Implement finite-difference time-integration in `solve_klein_gordon` for dynamic field evolution.
3. Configure CI workflow (e.g., GitHub Actions) to run unit tests, integration tests, and CLI demo automatically.
4. Implement integration test for HDF5 export result validation (`test_parameter_sweep_export.py`).
5. Review and optimize time-integration accuracy in `solve_klein_gordon` based on discretization test outcomes.
6. Test and iterate on the CI workflow to ensure it runs unit tests and the CLI demo successfully.
7. Complete and validate the dynamic field evolution against discretization tests.
8. Add integration tests for HDF5 export result validation in `tests/test_parameter_sweep_export.py`.
- Re-run CI to confirm fix.
- Proceed with writing HDF5 export integration test.
- Run `pytest` locally under the updated environment to replicate CI and collect any remaining errors.
- Validate the CLI demo execution locally to ensure imports resolve correctly.
- Install `pandas` locally in the venv and run `pytest` to validate all tests pass.
- Verify CLI demo runs correctly under the updated environment.
- Push changes and trigger CI to confirm remote workflow success.
- Proceed with dynamic field evolution validation.

### Progress Update
- Changed CI workflow matrix `python-version` to block sequence to resolve YAML parsing errors.
- Updated CI install step to install core dependencies (`numpy`, `scipy`, `matplotlib`, `h5py`, `pytest`) before installing the package to resolve test import errors.
- All tests passed locally under updated environment.
- CLI demo executed successfully, generated HDF5 output as expected.
- Modified CI install commands to use `python -m pip install` consistently, ensuring correct interpreter usage.
- Bumped SciPy requirement to `>=1.11` to match available wheels for Python 3.12 and avoid install failures.
- Added `pytest.ini` to configure test discovery under `tests/` and set `pythonpath = src`, simplifying local and CI runs.

## 2025-08-06

### In Progress
- Added `test_time_integration_basic.py` to validate `solve_klein_gordon` output shapes and finite values.

### Next Tasks
- Run the new test locally and confirm CI compatibility.
- Refine time integration accuracy by comparing against analytical solution for small dt.

### Progress Update
- Enhanced `compute_energy_density` to accept `dx` and normalize spatial gradient; updated tests accordingly.
- Energy conservation test passed with proper gradient normalization.
- Analytical solution comparison test passed, confirming correct sign inversion at half period.

### Next Tasks
- Adjust solver parameters (e.g., reduce `dt`) to improve accuracy and reduce numerical dispersion.
- Integrate dynamic evolution accuracy benchmarks into CI as additional pytest jobs.
- Begin writing documentation for solver API and usage examples.

```latest-progress
Gradient normalization was added to `compute_energy_density`, and the energy conservation test now passes. The analytical solution inversion is confirmed. I updated the progress log. Next, we can refine solver parameters to improve accuracy, integrate these benchmarks into CI, or start writing solver API documentation. 
```
```progress
I added a pytest.ini to target tests under tests and automatically include src in `PYTHONPATH`, streamlining test discovery and import resolution. Updated the progress log. Next, I'll rerun both local and CI tests to confirm everything passes, then start on dynamic field evolution accuracy comparisons.
```
```progress
I updated the CI install steps to consistently use `python -m pip install`, and bumped SciPy to `>=1.11` for Python 3.12 wheel support. The progress log reflects these changes. Next, we should trigger CI to verify the remote workflow, then continue with dynamic field evolution accuracy testing.
```
```progress
The basic time integration test passed locally. I updated progress_log.md. Next, I’ll refine the solver’s accuracy like comparing to analytical solution for small `dt`.
```
```progress
Local tests and the CLI demo ran successfully. The `pandas` dependency resolved the test failures, and the demo generated expected output. I updated the progress log—next, I’ll push these changes so CI can verify the remote workflow, then proceed with dynamic field evolution validation. 
```
```progress
I added a CI step to export `PYTHONPATH` pointing to src, which should fix import errors for the `simulation` package in both tests and the CLI demo. Updated the progress log accordingly. Next, I’ll rerun CI and address any further packaging issues.
```
```progress
Logged the new HDF5 export test implementation in progress_log.md. Next, I'll run pytest locally, fix any import or packaging issues, and ensure the `simulation` module installs correctly so CI tests pass.
```
```progress
I switched the Python version matrix to a block sequence format to ensure YAML validity. Updated the progress log. Next up is rerunning CI to confirm the fix, then implementing the HDF5 export integration test.
```
```progress
I fixed the YAML indentation for the `strategy` block under `jobs.build` so the CI syntax is valid. Updated the progress log. Next, I'll implement the HDF5 export integration test at `tests/test_parameter_sweep_export.py`.
```
```progress
I’ve quoted the Python versions in the CI matrix to prevent YAML misinterpretation (3.10 became 3.1). The workflow now correctly picks up 3.10 and 3.12. I updated the progress log—next, I’ll write the HDF5 export integration test.
```
```progress
I’ve scaffolded the GitHub Actions CI workflow at ci.yml to install dependencies, run unit tests, and execute the CLI demo. The progress_log.md is updated accordingly. Next up, I’ll implement finite-difference time integration validation against discretization tests and write the HDF5 export integration test.
```
```progress
I’ll proceed to implement `solve_klein_gordon`’s time integration in lattice_qft.py. 
```
```progress
I’ve added a discretization unit test for the Laplacian and updated the progress log. Next, I recommend setting up a CI workflow to automatically run all tests and the CLI demo—ensuring stability—and writing the HDF5 export integration test. I can scaffold a GitHub Actions config if you’d like.
```
```progress
V&V and UQ trackers updated with high- and medium-priority tasks for the lattice QFT solver, HDF5 exports, and parameter sensitivities. The progress log and test/demo scripts are in place. Next, I recommend implementing the discretization unit test or refining the solver's accuracy. Let me know if you’d like to proceed with those, or I can set up a CI workflow next.
```
```progress
The CLI demo `lattice_sweep_demo.py` is in place and the progress log reflects that. The remaining tasks:

1. Add V&V and UQ tracker entries for lattice solver validation.
2. Ensure the `solve_klein_gordon` integration is fully implemented (already updated earlier).

Next, I’ll update the V&V and UQ ndjson files with validation and uncertainty tasks for the lattice solver. Let me know if that aligns!
```
```oldest-progress
Unit tests for the lattice QFT solver and energy density are in place, and the progress log reflects that. Next up:

1. Building the CLI demo at `scripts/lattice_sweep_demo.py`.
2. Updating V&V and UQ trackers with new lattice solver tasks.
3. Adding finite-difference time integration logic to `solve_klein_gordon`.
```

```file-history
~/Code/asciimath$ find . -path "./.venv" -prune -o -type f -regex '.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' -print | while read file; do stat -c '%Y %n' "$file"; done | sort -nr | while read timestamp file; do echo "$(date -d @$timestamp '+%Y-%m-%d %H:%M:%S') $file"; done | head -n 40
2025-08-01 19:47:03 ./docs/progress_log.md
2025-08-01 16:50:14 ./tests/test_energy_conservation.py
2025-08-01 16:50:14 ./tests/test_analytical_solution.py
2025-08-01 16:50:14 ./src/simulation/lattice_qft.py
2025-08-01 16:34:11 ./pytest.ini
2025-08-01 16:28:34 ./tests/test_zero_initial_condition.py
2025-08-01 16:28:34 ./pyproject.toml
2025-08-01 16:28:34 ./.github/workflows/ci.yml
2025-08-01 15:32:26 ./tests/test_time_integration_basic.py
2025-08-01 15:04:58 ./results/demo_sweep.h5
2025-08-01 14:59:52 ./validation_summary.json
2025-08-01 14:58:53 ./corrected_validation_results.json
2025-08-01 14:47:10 ./scripts/lattice_sweep_demo.py
2025-08-01 14:26:14 ./tests/test_parameter_sweep_export.py
2025-08-01 09:28:10 ./tests/test_qft_backend.py
2025-08-01 09:28:10 ./tests/test_lattice_energy.py
2025-08-01 09:28:10 ./tests/test_lattice_discretization.py
2025-08-01 09:28:10 ./src/simulation/qft_backend.py
2025-08-01 09:28:10 ./src/simulation/photonic_crystal_band.py
2025-08-01 09:28:10 ./src/simulation/parameter_sweep.py
2025-08-01 09:28:10 ./src/simulation/mechanical_fem.py
2025-08-01 09:28:10 ./src/simulation/electromagnetic_fdtd.py
2025-08-01 09:28:10 ./physics_driven_prototype_validation.py
2025-08-01 09:28:10 ./docs/literature_review.md
2025-08-01 09:28:10 ./docs/future-directions.md
2025-08-01 09:28:10 ./VnV-TODO.ndjson
2025-08-01 09:28:10 ./UQ-TODO.ndjson
2025-08-01 09:28:10 ./README.md
2025-08-01 09:28:10 ./.github/instructions/copilot-instructions.md
2025-07-31 13:22:17 ./working_validation_test.py
2025-07-31 13:22:17 ./working_negative_energy_generator.py
2025-07-31 13:22:17 ./verify_prototype_stack_fixed.py
2025-07-31 13:22:17 ./verify_prototype_stack.py
2025-07-31 13:22:17 ./unified_exotic_matter_sourcing_results.json
2025-07-31 13:22:17 ./unified_exotic_matter_sourcing.py
2025-07-31 13:22:17 ./two_phase_summary.py
2025-07-31 13:22:17 ./traversal-analysis.json
2025-07-31 13:22:17 ./theory_scan_results.json
2025-07-31 13:22:17 ./tests/test_diagnostics.py
2025-07-31 13:22:17 ./test_validation.py
````

```test-history
$ /home/sherri3/Code/asciimath/negative-energy-generator/.venv/bin/python -m pytest --maxfail=1
======================================= test session starts =======================================
platform linux -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/sherri3/Code/asciimath/negative-energy-generator
configfile: pytest.ini
testpaths: tests
collected 30 items                                                                                

tests/test_analytical_solution.py .                                                         [  3%]
tests/test_diagnostics.py .....................                                             [ 73%]
tests/test_energy_conservation.py .                                                         [ 76%]
tests/test_lattice_discretization.py .                                                      [ 80%]
tests/test_lattice_energy.py ..                                                             [ 86%]
tests/test_parameter_sweep_export.py .                                                      [ 90%]
tests/test_qft_backend.py .                                                                 [ 93%]
tests/test_time_integration_basic.py .                                                      [ 96%]
tests/test_zero_initial_condition.py .                                                      [100%]

======================================== warnings summary =========================================
src/simulation/quantum_circuit_sim.py:32
  /home/sherri3/Code/asciimath/negative-energy-generator/src/simulation/quantum_circuit_sim.py:32: UserWarning: QuTiP not available. Install with: pip install qutip
    warnings.warn("QuTiP not available. Install with: pip install qutip")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================== 30 passed, 1 warning in 0.31s =================================

$ gh run view 16689060234 --log-failed
build (3.12)    Install dependencies    ﻿2025-08-02T02:48:14.4841357Z ##[group]Run python -m pip install --upgrade pip
build (3.12)    Install dependencies    2025-08-02T02:48:14.4842220Z python -m pip install --upgrade pip
build (3.12)    Install dependencies    2025-08-02T02:48:14.4843009Z # Install core dependencies and test tools
build (3.12)    Install dependencies    2025-08-02T02:48:14.4844002Z python -m pip install numpy scipy matplotlib pandas h5py pytest
build (3.12)    Install dependencies    2025-08-02T02:48:14.4844844Z # Install the project package
build (3.12)    Install dependencies    2025-08-02T02:48:14.4845542Z python -m pip install .
build (3.12)    Install dependencies    2025-08-02T02:48:14.4888592Z shell: /usr/bin/bash -e {0}
build (3.12)    Install dependencies    2025-08-02T02:48:14.4889245Z env:
build (3.12)    Install dependencies    2025-08-02T02:48:14.4890059Z   PYTHONPATH: /home/runner/work/negative-energy-generator/negative-energy-generator/src
build (3.12)    Install dependencies    2025-08-02T02:48:14.4891087Z   pythonLocation: /opt/hostedtoolcache/Python/3.12.11/x64
build (3.12)    Install dependencies    2025-08-02T02:48:14.4892627Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.12.11/x64/lib/pkgconfig
build (3.12)    Install dependencies    2025-08-02T02:48:14.4894314Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.12.11/x64
build (3.12)    Install dependencies    2025-08-02T02:48:14.4895754Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.12.11/x64
build (3.12)    Install dependencies    2025-08-02T02:48:14.4897237Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.12.11/x64
build (3.12)    Install dependencies    2025-08-02T02:48:14.4898693Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.12.11/x64/lib
build (3.12)    Install dependencies    2025-08-02T02:48:14.4899728Z ##[endgroup]
build (3.12)    Install dependencies    2025-08-02T02:48:15.1457804Z Requirement already satisfied: pip in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (25.1.1)
build (3.12)    Install dependencies    2025-08-02T02:48:15.2324533Z Collecting pip
build (3.12)    Install dependencies    2025-08-02T02:48:15.2799019Z   Downloading pip-25.2-py3-none-any.whl.metadata (4.7 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:15.2923487Z Downloading pip-25.2-py3-none-any.whl (1.8 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:15.3652675Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 29.1 MB/s eta 0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:15.3817563Z Installing collected packages: pip
build (3.12)    Install dependencies    2025-08-02T02:48:15.3819440Z   Attempting uninstall: pip
build (3.12)    Install dependencies    2025-08-02T02:48:15.3839001Z     Found existing installation: pip 25.1.1
build (3.12)    Install dependencies    2025-08-02T02:48:16.1115501Z     Uninstalling pip-25.1.1:
build (3.12)    Install dependencies    2025-08-02T02:48:16.1174345Z       Successfully uninstalled pip-25.1.1
build (3.12)    Install dependencies    2025-08-02T02:48:16.9820641Z Successfully installed pip-25.2
build (3.12)    Install dependencies    2025-08-02T02:48:17.7076137Z Collecting numpy
build (3.12)    Install dependencies    2025-08-02T02:48:17.7508779Z   Downloading numpy-2.3.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:17.8956148Z Collecting scipy
build (3.12)    Install dependencies    2025-08-02T02:48:17.9032730Z   Downloading scipy-1.16.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (61 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.0515105Z Collecting matplotlib
build (3.12)    Install dependencies    2025-08-02T02:48:18.0589035Z   Downloading matplotlib-3.10.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.1739524Z Collecting pandas
build (3.12)    Install dependencies    2025-08-02T02:48:18.1821753Z   Downloading pandas-2.3.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.2617235Z Collecting h5py
build (3.12)    Install dependencies    2025-08-02T02:48:18.2690584Z   Downloading h5py-3.14.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.3095841Z Collecting pytest
build (3.12)    Install dependencies    2025-08-02T02:48:18.3174586Z   Downloading pytest-8.4.1-py3-none-any.whl.metadata (7.7 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.3961401Z Collecting contourpy>=1.0.1 (from matplotlib)
build (3.12)    Install dependencies    2025-08-02T02:48:18.4031503Z   Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.4193919Z Collecting cycler>=0.10 (from matplotlib)
build (3.12)    Install dependencies    2025-08-02T02:48:18.4292210Z   Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.5764635Z Collecting fonttools>=4.22.0 (from matplotlib)
build (3.12)    Install dependencies    2025-08-02T02:48:18.5840448Z   Downloading fonttools-4.59.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (107 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.6688701Z Collecting kiwisolver>=1.3.1 (from matplotlib)
build (3.12)    Install dependencies    2025-08-02T02:48:18.6761764Z   Downloading kiwisolver-1.4.8-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.2 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.7055502Z Collecting packaging>=20.0 (from matplotlib)
build (3.12)    Install dependencies    2025-08-02T02:48:18.7126500Z   Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.8815154Z Collecting pillow>=8 (from matplotlib)
build (3.12)    Install dependencies    2025-08-02T02:48:18.8888146Z   Downloading pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.9244423Z Collecting pyparsing>=2.3.1 (from matplotlib)
build (3.12)    Install dependencies    2025-08-02T02:48:18.9330922Z   Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:18.9552680Z Collecting python-dateutil>=2.7 (from matplotlib)
build (3.12)    Install dependencies    2025-08-02T02:48:18.9621022Z   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.0154934Z Collecting pytz>=2020.1 (from pandas)
build (3.12)    Install dependencies    2025-08-02T02:48:19.0223884Z   Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.0416182Z Collecting tzdata>=2022.7 (from pandas)
build (3.12)    Install dependencies    2025-08-02T02:48:19.0486442Z   Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.0743826Z Collecting iniconfig>=1 (from pytest)
build (3.12)    Install dependencies    2025-08-02T02:48:19.0814366Z   Downloading iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.0978515Z Collecting pluggy<2,>=1.5 (from pytest)
build (3.12)    Install dependencies    2025-08-02T02:48:19.1049814Z   Downloading pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.1296633Z Collecting pygments>=2.7.2 (from pytest)
build (3.12)    Install dependencies    2025-08-02T02:48:19.1372214Z   Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.1746371Z Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib)
build (3.12)    Install dependencies    2025-08-02T02:48:19.1815735Z   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.1941955Z Downloading numpy-2.3.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.6 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.2726841Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.6/16.6 MB 232.3 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:19.2801484Z Downloading scipy-1.16.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.2 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.4631037Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.2/35.2 MB 195.9 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:19.4706893Z Downloading matplotlib-3.10.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.5072835Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 247.8 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:19.5144045Z Downloading pandas-2.3.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.0 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.5654403Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.0/12.0 MB 241.9 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:19.5768179Z Downloading h5py-3.14.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.9 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.6099654Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 151.9 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:19.6186683Z Downloading pytest-8.4.1-py3-none-any.whl (365 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.6295633Z Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.6395207Z Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (362 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.6507400Z Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.6599282Z Downloading fonttools-4.59.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (4.9 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.6893897Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 176.3 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:19.6969692Z Downloading iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.7067840Z Downloading kiwisolver-1.4.8-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.5 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.7178704Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 142.1 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:19.7255904Z Downloading packaging-25.0-py3-none-any.whl (66 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.7356077Z Downloading pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.7692352Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 205.7 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:19.7768134Z Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.7851234Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 155.5 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-02T02:48:19.7925820Z Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.8018343Z Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.8121345Z Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.8234496Z Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:19.8330844Z Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
build (3.12)    Install dependencies    2025-08-02T02:48:20.0411218Z Installing collected packages: pytz, tzdata, six, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, scipy, python-dateutil, pytest, h5py, contourpy, pandas, matplotlib
build (3.12)    Install dependencies    2025-08-02T02:48:34.9143266Z
build (3.12)    Install dependencies    2025-08-02T02:48:34.9170249Z Successfully installed contourpy-1.3.3 cycler-0.12.1 fonttools-4.59.0 h5py-3.14.0 iniconfig-2.1.0 kiwisolver-1.4.8 matplotlib-3.10.5 numpy-2.3.2 packaging-25.0 pandas-2.3.1 pillow-11.3.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.3 pytest-8.4.1 python-dateutil-2.9.0.post0 pytz-2025.2 scipy-1.16.1 six-1.17.0 tzdata-2025.2
build (3.12)    Install dependencies    2025-08-02T02:48:35.4349543Z Processing /home/runner/work/negative-energy-generator/negative-energy-generator
build (3.12)    Install dependencies    2025-08-02T02:48:35.4372979Z   Installing build dependencies: started
build (3.12)    Install dependencies    2025-08-02T02:48:36.3721617Z   Installing build dependencies: finished with status 'done'
build (3.12)    Install dependencies    2025-08-02T02:48:36.3726586Z   Getting requirements to build wheel: started
build (3.12)    Install dependencies    2025-08-02T02:48:36.8534604Z   Getting requirements to build wheel: finished with status 'done'
build (3.12)    Install dependencies    2025-08-02T02:48:36.8543612Z   Preparing metadata (pyproject.toml): started
build (3.12)    Install dependencies    2025-08-02T02:48:37.0438118Z   Preparing metadata (pyproject.toml): finished with status 'done'
build (3.12)    Install dependencies    2025-08-02T02:48:37.0469117Z Requirement already satisfied: numpy>=1.24 in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (from negative_energy_generator==0.1.0) (2.3.2)
build (3.12)    Install dependencies    2025-08-02T02:48:37.0474602Z Requirement already satisfied: scipy>=1.11 in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (from negative_energy_generator==0.1.0) (1.16.1)
build (3.12)    Install dependencies    2025-08-02T02:48:37.0479420Z Requirement already satisfied: matplotlib>=3.6 in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (from negative_energy_generator==0.1.0) (3.10.5)
build (3.12)    Install dependencies    2025-08-02T02:48:37.0484349Z Requirement already satisfied: pandas>=1.5 in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (from negative_energy_generator==0.1.0) (2.3.1)
build (3.12)    Install dependencies    2025-08-02T02:48:37.1004119Z INFO: pip is looking at multiple versions of negative-energy-generator to determine which version is compatible with other requirements. This could take a while.
build (3.12)    Install dependencies    2025-08-02T02:48:37.1007998Z ERROR: Could not find a version that satisfies the requirement lqg_first_principles_gravitational_constant>=0.1.0 (from negative-energy-generator) (from versions: none)
build (3.12)    Install dependencies    2025-08-02T02:48:37.1047495Z ERROR: No matching distribution found for lqg_first_principles_gravitational_constant>=0.1.0
build (3.12)    Install dependencies    2025-08-02T02:48:37.1632839Z ##[error]Process completed with exit code 1.
build (3.10)    Install dependencies    ﻿2025-08-02T02:48:12.6146291Z ##[group]Run python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T02:48:12.6146845Z python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T02:48:12.6147324Z # Install core dependencies and test tools
build (3.10)    Install dependencies    2025-08-02T02:48:12.6147865Z python -m pip install numpy scipy matplotlib pandas h5py pytest
build (3.10)    Install dependencies    2025-08-02T02:48:12.6148381Z # Install the project package
build (3.10)    Install dependencies    2025-08-02T02:48:12.6148781Z python -m pip install .
build (3.10)    Install dependencies    2025-08-02T02:48:12.6249845Z shell: /usr/bin/bash -e {0}
build (3.10)    Install dependencies    2025-08-02T02:48:12.6250226Z env:
build (3.10)    Install dependencies    2025-08-02T02:48:12.6250740Z   PYTHONPATH: /home/runner/work/negative-energy-generator/negative-energy-generator/src
build (3.10)    Install dependencies    2025-08-02T02:48:12.6251541Z   pythonLocation: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T02:48:12.6252182Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib/pkgconfig
build (3.10)    Install dependencies    2025-08-02T02:48:12.6252754Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T02:48:12.6253275Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T02:48:12.6253823Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T02:48:12.6254336Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib
build (3.10)    Install dependencies    2025-08-02T02:48:12.6254771Z ##[endgroup]
build (3.10)    Install dependencies    2025-08-02T02:48:14.4394169Z Requirement already satisfied: pip in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (25.1.1)
build (3.10)    Install dependencies    2025-08-02T02:48:14.5369842Z Collecting pip
build (3.10)    Install dependencies    2025-08-02T02:48:14.5887526Z   Downloading pip-25.2-py3-none-any.whl.metadata (4.7 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:14.5998677Z Downloading pip-25.2-py3-none-any.whl (1.8 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:14.6454675Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 79.5 MB/s eta 0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:14.6826654Z Installing collected packages: pip
build (3.10)    Install dependencies    2025-08-02T02:48:14.6828235Z   Attempting uninstall: pip
build (3.10)    Install dependencies    2025-08-02T02:48:14.6834626Z     Found existing installation: pip 25.1.1
build (3.10)    Install dependencies    2025-08-02T02:48:14.7394430Z     Uninstalling pip-25.1.1:
build (3.10)    Install dependencies    2025-08-02T02:48:14.7457497Z       Successfully uninstalled pip-25.1.1
build (3.10)    Install dependencies    2025-08-02T02:48:15.5404895Z Successfully installed pip-25.2
build (3.10)    Install dependencies    2025-08-02T02:48:16.3467213Z Collecting numpy
build (3.10)    Install dependencies    2025-08-02T02:48:16.3918434Z   Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:16.5518976Z Collecting scipy
build (3.10)    Install dependencies    2025-08-02T02:48:16.5598368Z   Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:16.7344593Z Collecting matplotlib
build (3.10)    Install dependencies    2025-08-02T02:48:16.7423113Z   Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:16.9009122Z Collecting pandas
build (3.10)    Install dependencies    2025-08-02T02:48:16.9088336Z   Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.0012920Z Collecting h5py
build (3.10)    Install dependencies    2025-08-02T02:48:17.0090099Z   Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.0559973Z Collecting pytest
build (3.10)    Install dependencies    2025-08-02T02:48:17.0749877Z   Downloading pytest-8.4.1-py3-none-any.whl.metadata (7.7 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.1741637Z Collecting contourpy>=1.0.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T02:48:17.1821272Z   Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.1998311Z Collecting cycler>=0.10 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T02:48:17.2081583Z   Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.3861903Z Collecting fonttools>=4.22.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T02:48:17.3939977Z   Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (107 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.4755581Z Collecting kiwisolver>=1.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T02:48:17.4833896Z   Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.2 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.5178217Z Collecting packaging>=20.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T02:48:17.5254149Z   Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.7594253Z Collecting pillow>=8 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T02:48:17.7674211Z   Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.8054501Z Collecting pyparsing>=2.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T02:48:17.8131210Z   Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.8323846Z Collecting python-dateutil>=2.7 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T02:48:17.8398270Z   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.9004230Z Collecting pytz>=2020.1 (from pandas)
build (3.10)    Install dependencies    2025-08-02T02:48:17.9079768Z   Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.9277775Z Collecting tzdata>=2022.7 (from pandas)
build (3.10)    Install dependencies    2025-08-02T02:48:17.9352730Z   Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.9668836Z Collecting exceptiongroup>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T02:48:17.9744515Z   Downloading exceptiongroup-1.3.0-py3-none-any.whl.metadata (6.7 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:17.9905629Z Collecting iniconfig>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T02:48:17.9978015Z   Downloading iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.0157446Z Collecting pluggy<2,>=1.5 (from pytest)
build (3.10)    Install dependencies    2025-08-02T02:48:18.0231953Z   Downloading pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.0503362Z Collecting pygments>=2.7.2 (from pytest)
build (3.10)    Install dependencies    2025-08-02T02:48:18.0589497Z   Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.0795616Z Collecting tomli>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T02:48:18.0876389Z   Downloading tomli-2.2.1-py3-none-any.whl.metadata (10 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.1131545Z Collecting typing-extensions>=4.6.0 (from exceptiongroup>=1->pytest)
build (3.10)    Install dependencies    2025-08-02T02:48:18.1204575Z   Downloading typing_extensions-4.14.1-py3-none-any.whl.metadata (3.0 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.1446574Z Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib)
build (3.10)    Install dependencies    2025-08-02T02:48:18.1522494Z   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.1667894Z Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.5268558Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.8/16.8 MB 48.9 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:18.5348625Z Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.7180774Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 207.8 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:18.7262845Z Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.7641481Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 240.8 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:18.7719626Z Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.3 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.8277577Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.3/12.3 MB 229.7 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:18.8357836Z Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.8726149Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 129.1 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:18.8808524Z Downloading pytest-8.4.1-py3-none-any.whl (365 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.8920022Z Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.9020349Z Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.9126728Z Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.9220066Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.9315791Z Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.9560099Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 213.3 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:18.9634989Z Downloading iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.9731730Z Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:18.9840817Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 178.9 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:18.9915351Z Downloading packaging-25.0-py3-none-any.whl (66 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.0043317Z Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.0392573Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 201.3 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:19.0474763Z Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.0569438Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 153.1 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T02:48:19.0650583Z Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.0756026Z Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.0863284Z Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.0979760Z Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.1073605Z Downloading tomli-2.2.1-py3-none-any.whl (14 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.1211885Z Downloading typing_extensions-4.14.1-py3-none-any.whl (43 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.1307371Z Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
build (3.10)    Install dependencies    2025-08-02T02:48:19.3888951Z Installing collected packages: pytz, tzdata, typing-extensions, tomli, six, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, scipy, python-dateutil, h5py, exceptiongroup, contourpy, pytest, pandas, matplotlib
build (3.10)    Install dependencies    2025-08-02T02:48:32.5535949Z
build (3.10)    Install dependencies    2025-08-02T02:48:32.5582680Z Successfully installed contourpy-1.3.2 cycler-0.12.1 exceptiongroup-1.3.0 fonttools-4.59.0 h5py-3.14.0 iniconfig-2.1.0 kiwisolver-1.4.8 matplotlib-3.10.5 numpy-2.2.6 packaging-25.0 pandas-2.3.1 pillow-11.3.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.3 pytest-8.4.1 python-dateutil-2.9.0.post0 pytz-2025.2 scipy-1.15.3 six-1.17.0 tomli-2.2.1 typing-extensions-4.14.1 tzdata-2025.2
build (3.10)    Install dependencies    2025-08-02T02:48:33.1127262Z Processing /home/runner/work/negative-energy-generator/negative-energy-generator
build (3.10)    Install dependencies    2025-08-02T02:48:33.1153363Z   Installing build dependencies: started
build (3.10)    Install dependencies    2025-08-02T02:48:34.1046600Z   Installing build dependencies: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T02:48:34.1052822Z   Getting requirements to build wheel: started
build (3.10)    Install dependencies    2025-08-02T02:48:34.5835170Z   Getting requirements to build wheel: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T02:48:34.5844527Z   Preparing metadata (pyproject.toml): started
build (3.10)    Install dependencies    2025-08-02T02:48:34.7693742Z   Preparing metadata (pyproject.toml): finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T02:48:34.7731817Z Requirement already satisfied: numpy>=1.24 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.2.6)
build (3.10)    Install dependencies    2025-08-02T02:48:34.7739194Z Requirement already satisfied: scipy>=1.11 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (1.15.3)
build (3.10)    Install dependencies    2025-08-02T02:48:34.7745702Z Requirement already satisfied: matplotlib>=3.6 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (3.10.5)
build (3.10)    Install dependencies    2025-08-02T02:48:34.7751358Z Requirement already satisfied: pandas>=1.5 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.3.1)
build (3.10)    Install dependencies    2025-08-02T02:48:34.8962381Z INFO: pip is looking at multiple versions of negative-energy-generator to determine which version is compatible with other requirements. This could take a while.
build (3.10)    Install dependencies    2025-08-02T02:48:34.8965683Z ERROR: Could not find a version that satisfies the requirement lqg_first_principles_gravitational_constant>=0.1.0 (from negative-energy-generator) (from versions: none)
build (3.10)    Install dependencies    2025-08-02T02:48:34.8975277Z ERROR: No matching distribution found for lqg_first_principles_gravitational_constant>=0.1.0
build (3.10)    Install dependencies    2025-08-02T02:48:34.9537123Z ##[error]Process completed with exit code 1.
````