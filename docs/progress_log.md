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

```latest-progress
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
```progress
Unit tests for the lattice QFT solver and energy density are in place, and the progress log reflects that. Next up:

1. Building the CLI demo at `scripts/lattice_sweep_demo.py`.
2. Updating V&V and UQ trackers with new lattice solver tasks.
3. Adding finite-difference time integration logic to `solve_klein_gordon`.
```
```oldest-progress
I’ve scaffolded `parameter_sweep.py` for 1+1D lattice QFT sweeps and updated the progress log to reflect that. Next up:

1. Creating unit tests in `tests/test_lattice_energy.py`.
2. Developing a CLI demo script `scripts/lattice_sweep_demo.py`.
3. Adding corresponding V&V/UQ tasks.
```

```file-history
~/Code/asciimath$ find . -path "./.venv" -prune -o -type f -regex '.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' -print | while read file; do stat -c '%Y %n' "$file"; done | sort -nr | while read timestamp file; do echo "$(date -d @$timestamp '+%Y-%m-%d %H:%M:%S') $file"; done | head -n 40
2025-08-01 16:36:35 ./docs/progress_log.md
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
2025-08-01 09:28:10 ./src/simulation/lattice_qft.py
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
2025-07-31 13:22:17 ./test_progress_tracking.py
2025-07-31 13:22:17 ./test_multilayer.py
````

```test-history
$ export PYTHONPATH=src && /home/sherri3/Code/asciimath/negative-energy-generator/.venv/bin/python -m pytest tests/test_time_integration_basic.py --maxfail=1 --disable-warnings -q
.                                                      [100%]
1 passed, 1 warning in 0.28s

$ gh run view 16687272898 --log-failed
build (3.12)    Install dependencies    ﻿2025-08-01T23:39:15.6623806Z ##[group]Run python -m pip install --upgrade pip
build (3.12)    Install dependencies    2025-08-01T23:39:15.6625007Z python -m pip install --upgrade pip
build (3.12)    Install dependencies    2025-08-01T23:39:15.6626140Z # Install core dependencies and test tools
build (3.12)    Install dependencies    2025-08-01T23:39:15.6627637Z python -m pip install numpy scipy matplotlib pandas h5py pytest
build (3.12)    Install dependencies    2025-08-01T23:39:15.6628870Z # Install the project package
build (3.12)    Install dependencies    2025-08-01T23:39:15.6629804Z python -m pip install .
build (3.12)    Install dependencies    2025-08-01T23:39:15.6666900Z shell: /usr/bin/bash -e {0}
build (3.12)    Install dependencies    2025-08-01T23:39:15.6668152Z env:
build (3.12)    Install dependencies    2025-08-01T23:39:15.6669449Z   PYTHONPATH: /home/runner/work/negative-energy-generator/negative-energy-generator/src
build (3.12)    Install dependencies    2025-08-01T23:39:15.6671060Z   pythonLocation: /opt/hostedtoolcache/Python/3.12.11/x64
build (3.12)    Install dependencies    2025-08-01T23:39:15.6672415Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.12.11/x64/lib/pkgconfig
build (3.12)    Install dependencies    2025-08-01T23:39:15.6673753Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.12.11/x64
build (3.12)    Install dependencies    2025-08-01T23:39:15.6674971Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.12.11/x64
build (3.12)    Install dependencies    2025-08-01T23:39:15.6676427Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.12.11/x64
build (3.12)    Install dependencies    2025-08-01T23:39:15.6677674Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.12.11/x64/lib
build (3.12)    Install dependencies    2025-08-01T23:39:15.6678706Z ##[endgroup]
build (3.12)    Install dependencies    2025-08-01T23:39:16.1745622Z Requirement already satisfied: pip in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (25.1.1)
build (3.12)    Install dependencies    2025-08-01T23:39:16.2937991Z Collecting pip
build (3.12)    Install dependencies    2025-08-01T23:39:16.3780811Z   Downloading pip-25.2-py3-none-any.whl.metadata (4.7 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:16.3980762Z Downloading pip-25.2-py3-none-any.whl (1.8 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:16.4805927Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 39.0 MB/s eta 0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:16.4965878Z Installing collected packages: pip
build (3.12)    Install dependencies    2025-08-01T23:39:16.4968487Z   Attempting uninstall: pip
build (3.12)    Install dependencies    2025-08-01T23:39:16.4988364Z     Found existing installation: pip 25.1.1
build (3.12)    Install dependencies    2025-08-01T23:39:16.5384187Z     Uninstalling pip-25.1.1:
build (3.12)    Install dependencies    2025-08-01T23:39:16.5441649Z       Successfully uninstalled pip-25.1.1
build (3.12)    Install dependencies    2025-08-01T23:39:17.4159504Z Successfully installed pip-25.2
build (3.12)    Install dependencies    2025-08-01T23:39:18.2006494Z Collecting numpy
build (3.12)    Install dependencies    2025-08-01T23:39:18.2689939Z   Downloading numpy-2.3.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:18.4245604Z Collecting scipy
build (3.12)    Install dependencies    2025-08-01T23:39:18.4404694Z   Downloading scipy-1.16.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (61 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:18.5928898Z Collecting matplotlib
build (3.12)    Install dependencies    2025-08-01T23:39:18.6087152Z   Downloading matplotlib-3.10.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:18.7298839Z Collecting pandas
build (3.12)    Install dependencies    2025-08-01T23:39:18.7468755Z   Downloading pandas-2.3.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:18.8363309Z Collecting h5py
build (3.12)    Install dependencies    2025-08-01T23:39:18.8518610Z   Downloading h5py-3.14.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:18.9024615Z Collecting pytest
build (3.12)    Install dependencies    2025-08-01T23:39:18.9182368Z   Downloading pytest-8.4.1-py3-none-any.whl.metadata (7.7 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.0055958Z Collecting contourpy>=1.0.1 (from matplotlib)
build (3.12)    Install dependencies    2025-08-01T23:39:19.0215618Z   Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.0498272Z Collecting cycler>=0.10 (from matplotlib)
build (3.12)    Install dependencies    2025-08-01T23:39:19.0666717Z   Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.2152218Z Collecting fonttools>=4.22.0 (from matplotlib)
build (3.12)    Install dependencies    2025-08-01T23:39:19.2320816Z   Downloading fonttools-4.59.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (107 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.3278260Z Collecting kiwisolver>=1.3.1 (from matplotlib)
build (3.12)    Install dependencies    2025-08-01T23:39:19.3435224Z   Downloading kiwisolver-1.4.8-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.2 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.3871097Z Collecting packaging>=20.0 (from matplotlib)
build (3.12)    Install dependencies    2025-08-01T23:39:19.4030430Z   Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.5748062Z Collecting pillow>=8 (from matplotlib)
build (3.12)    Install dependencies    2025-08-01T23:39:19.5905427Z   Downloading pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.6354994Z Collecting pyparsing>=2.3.1 (from matplotlib)
build (3.12)    Install dependencies    2025-08-01T23:39:19.6506701Z   Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.6774820Z Collecting python-dateutil>=2.7 (from matplotlib)
build (3.12)    Install dependencies    2025-08-01T23:39:19.6930427Z   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.7542696Z Collecting pytz>=2020.1 (from pandas)
build (3.12)    Install dependencies    2025-08-01T23:39:19.7702365Z   Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.7993501Z Collecting tzdata>=2022.7 (from pandas)
build (3.12)    Install dependencies    2025-08-01T23:39:19.8146822Z   Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.8503398Z Collecting iniconfig>=1 (from pytest)
build (3.12)    Install dependencies    2025-08-01T23:39:19.8662464Z   Downloading iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.8932869Z Collecting pluggy<2,>=1.5 (from pytest)
build (3.12)    Install dependencies    2025-08-01T23:39:19.9086188Z   Downloading pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:19.9462672Z Collecting pygments>=2.7.2 (from pytest)
build (3.12)    Install dependencies    2025-08-01T23:39:19.9617076Z   Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.0069399Z Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib)
build (3.12)    Install dependencies    2025-08-01T23:39:20.0223176Z   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.0433590Z Downloading numpy-2.3.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.6 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.1663307Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.6/16.6 MB 153.1 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:20.1821761Z Downloading scipy-1.16.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.2 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.3444518Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.2/35.2 MB 219.0 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:20.3598051Z Downloading matplotlib-3.10.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.4036605Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 205.1 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:20.4193918Z Downloading pandas-2.3.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.0 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.4782528Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.0/12.0 MB 207.6 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:20.4992234Z Downloading h5py-3.14.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.9 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.5463500Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 104.3 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:20.5616550Z Downloading pytest-8.4.1-py3-none-any.whl (365 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.5801794Z Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.5979372Z Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (362 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.6165259Z Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.6342536Z Downloading fonttools-4.59.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (4.9 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.6760512Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 118.2 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:20.6915862Z Downloading iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.7093524Z Downloading kiwisolver-1.4.8-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.5 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.7186799Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 173.5 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:20.7340834Z Downloading packaging-25.0-py3-none-any.whl (66 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.7533347Z Downloading pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.7848233Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 221.6 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:20.8012032Z Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.8094885Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 168.0 MB/s  0:00:00
build (3.12)    Install dependencies    2025-08-01T23:39:20.8249793Z Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.8426680Z Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.8604831Z Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.8799995Z Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:20.8975109Z Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
build (3.12)    Install dependencies    2025-08-01T23:39:21.1036022Z Installing collected packages: pytz, tzdata, six, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, scipy, python-dateutil, pytest, h5py, contourpy, pandas, matplotlib
build (3.12)    Install dependencies    2025-08-01T23:39:35.9252521Z
build (3.12)    Install dependencies    2025-08-01T23:39:35.9279398Z Successfully installed contourpy-1.3.3 cycler-0.12.1 fonttools-4.59.0 h5py-3.14.0 iniconfig-2.1.0 kiwisolver-1.4.8 matplotlib-3.10.5 numpy-2.3.2 packaging-25.0 pandas-2.3.1 pillow-11.3.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.3 pytest-8.4.1 python-dateutil-2.9.0.post0 pytz-2025.2 scipy-1.16.1 six-1.17.0 tzdata-2025.2
build (3.12)    Install dependencies    2025-08-01T23:39:36.4450952Z Processing /home/runner/work/negative-energy-generator/negative-energy-generator
build (3.12)    Install dependencies    2025-08-01T23:39:36.4474885Z   Installing build dependencies: started
build (3.12)    Install dependencies    2025-08-01T23:39:37.4847705Z   Installing build dependencies: finished with status 'done'
build (3.12)    Install dependencies    2025-08-01T23:39:37.4852868Z   Getting requirements to build wheel: started
build (3.12)    Install dependencies    2025-08-01T23:39:38.0253674Z   Getting requirements to build wheel: finished with status 'done'
build (3.12)    Install dependencies    2025-08-01T23:39:38.0262929Z   Preparing metadata (pyproject.toml): started
build (3.12)    Install dependencies    2025-08-01T23:39:38.2156912Z   Preparing metadata (pyproject.toml): finished with status 'done'
build (3.12)    Install dependencies    2025-08-01T23:39:38.2188265Z Requirement already satisfied: numpy>=1.24 in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (from negative_energy_generator==0.1.0) (2.3.2)
build (3.12)    Install dependencies    2025-08-01T23:39:38.2192716Z Requirement already satisfied: scipy>=1.11 in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (from negative_energy_generator==0.1.0) (1.16.1)
build (3.12)    Install dependencies    2025-08-01T23:39:38.2197012Z Requirement already satisfied: matplotlib>=3.6 in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (from negative_energy_generator==0.1.0) (3.10.5)
build (3.12)    Install dependencies    2025-08-01T23:39:38.2201125Z Requirement already satisfied: pandas>=1.5 in /opt/hostedtoolcache/Python/3.12.11/x64/lib/python3.12/site-packages (from negative_energy_generator==0.1.0) (2.3.1)
build (3.12)    Install dependencies    2025-08-01T23:39:38.3358791Z INFO: pip is looking at multiple versions of negative-energy-generator to determine which version is compatible with other requirements. This could take a while.
build (3.12)    Install dependencies    2025-08-01T23:39:38.3361456Z ERROR: Could not find a version that satisfies the requirement lqg_first_principles_gravitational_constant>=0.1.0 (from negative-energy-generator) (from versions: none)
build (3.12)    Install dependencies    2025-08-01T23:39:38.3401492Z ERROR: No matching distribution found for lqg_first_principles_gravitational_constant>=0.1.0
build (3.12)    Install dependencies    2025-08-01T23:39:38.3962447Z ##[error]Process completed with exit code 1.
build (3.10)    Install dependencies    ﻿2025-08-01T23:39:14.8122512Z ##[group]Run python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-01T23:39:14.8122946Z python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-01T23:39:14.8123314Z # Install core dependencies and test tools
build (3.10)    Install dependencies    2025-08-01T23:39:14.8123686Z python -m pip install numpy scipy matplotlib pandas h5py pytest
build (3.10)    Install dependencies    2025-08-01T23:39:14.8124050Z # Install the project package
build (3.10)    Install dependencies    2025-08-01T23:39:14.8124306Z python -m pip install .
build (3.10)    Install dependencies    2025-08-01T23:39:14.8207683Z shell: /usr/bin/bash -e {0}
build (3.10)    Install dependencies    2025-08-01T23:39:14.8207929Z env:
build (3.10)    Install dependencies    2025-08-01T23:39:14.8208288Z   PYTHONPATH: /home/runner/work/negative-energy-generator/negative-energy-generator/src
build (3.10)    Install dependencies    2025-08-01T23:39:14.8208883Z   pythonLocation: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-01T23:39:14.8209330Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib/pkgconfig
build (3.10)    Install dependencies    2025-08-01T23:39:14.8209718Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-01T23:39:14.8210092Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-01T23:39:14.8210497Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-01T23:39:14.8210856Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib
build (3.10)    Install dependencies    2025-08-01T23:39:14.8211156Z ##[endgroup]
build (3.10)    Install dependencies    2025-08-01T23:39:17.5199286Z Requirement already satisfied: pip in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (25.1.1)
build (3.10)    Install dependencies    2025-08-01T23:39:17.6315495Z Collecting pip
build (3.10)    Install dependencies    2025-08-01T23:39:17.6664284Z   Downloading pip-25.2-py3-none-any.whl.metadata (4.7 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:17.6744827Z Downloading pip-25.2-py3-none-any.whl (1.8 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:17.7086236Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 85.0 MB/s eta 0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:17.7556176Z Installing collected packages: pip
build (3.10)    Install dependencies    2025-08-01T23:39:17.7558195Z   Attempting uninstall: pip
build (3.10)    Install dependencies    2025-08-01T23:39:17.7564101Z     Found existing installation: pip 25.1.1
build (3.10)    Install dependencies    2025-08-01T23:39:17.8079470Z     Uninstalling pip-25.1.1:
build (3.10)    Install dependencies    2025-08-01T23:39:17.8144558Z       Successfully uninstalled pip-25.1.1
build (3.10)    Install dependencies    2025-08-01T23:39:18.5925960Z Successfully installed pip-25.2
build (3.10)    Install dependencies    2025-08-01T23:39:19.7837447Z Collecting numpy
build (3.10)    Install dependencies    2025-08-01T23:39:19.8156747Z   Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:19.9648642Z Collecting scipy
build (3.10)    Install dependencies    2025-08-01T23:39:19.9686496Z   Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:20.1316584Z Collecting matplotlib
build (3.10)    Install dependencies    2025-08-01T23:39:20.1353273Z   Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:20.2845810Z Collecting pandas
build (3.10)    Install dependencies    2025-08-01T23:39:20.2891867Z   Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:20.3743451Z Collecting h5py
build (3.10)    Install dependencies    2025-08-01T23:39:20.3783384Z   Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:20.4242245Z Collecting pytest
build (3.10)    Install dependencies    2025-08-01T23:39:20.4285072Z   Downloading pytest-8.4.1-py3-none-any.whl.metadata (7.7 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:20.5235056Z Collecting contourpy>=1.0.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-01T23:39:20.5271953Z   Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:20.5468876Z Collecting cycler>=0.10 (from matplotlib)
build (3.10)    Install dependencies    2025-08-01T23:39:20.5536757Z   Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:20.7228297Z Collecting fonttools>=4.22.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-01T23:39:20.7264124Z   Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (107 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:20.8001236Z Collecting kiwisolver>=1.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-01T23:39:20.8038892Z   Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.2 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:20.8370296Z Collecting packaging>=20.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-01T23:39:20.8402944Z   Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.0648484Z Collecting pillow>=8 (from matplotlib)
build (3.10)    Install dependencies    2025-08-01T23:39:21.0686564Z   Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.1035345Z Collecting pyparsing>=2.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-01T23:39:21.1070646Z   Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.1250723Z Collecting python-dateutil>=2.7 (from matplotlib)
build (3.10)    Install dependencies    2025-08-01T23:39:21.1283993Z   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.1844720Z Collecting pytz>=2020.1 (from pandas)
build (3.10)    Install dependencies    2025-08-01T23:39:21.1879125Z   Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2044953Z Collecting tzdata>=2022.7 (from pandas)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2078613Z   Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2362897Z Collecting exceptiongroup>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2396081Z   Downloading exceptiongroup-1.3.0-py3-none-any.whl.metadata (6.7 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2510134Z Collecting iniconfig>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2543187Z   Downloading iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2675985Z Collecting pluggy<2,>=1.5 (from pytest)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2708312Z   Downloading pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2933988Z Collecting pygments>=2.7.2 (from pytest)
build (3.10)    Install dependencies    2025-08-01T23:39:21.2967767Z   Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.3137632Z Collecting tomli>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-01T23:39:21.3169384Z   Downloading tomli-2.2.1-py3-none-any.whl.metadata (10 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.3391587Z Collecting typing-extensions>=4.6.0 (from exceptiongroup>=1->pytest)
build (3.10)    Install dependencies    2025-08-01T23:39:21.3424261Z   Downloading typing_extensions-4.14.1-py3-none-any.whl.metadata (3.0 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.3630124Z Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib)
build (3.10)    Install dependencies    2025-08-01T23:39:21.3661686Z   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.3770248Z Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.4482266Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.8/16.8 MB 246.9 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:21.4517222Z Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.6067363Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 245.8 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:21.6102524Z Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.6480049Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 241.0 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:21.6518587Z Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.3 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.7053182Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.3/12.3 MB 238.6 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:21.7090471Z Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.7335787Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 197.8 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:21.7373863Z Downloading pytest-8.4.1-py3-none-any.whl (365 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.7447074Z Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.7503844Z Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.7576211Z Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.7635215Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.7693534Z Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.7936562Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 210.2 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:21.7970793Z Downloading iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.8050324Z Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.8154291Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 178.1 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:21.8190746Z Downloading packaging-25.0-py3-none-any.whl (66 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.8251874Z Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.8547759Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 239.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:21.8584597Z Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.8674281Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 154.2 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-01T23:39:21.8709545Z Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.8765358Z Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.8827779Z Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.8901031Z Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.8958304Z Downloading tomli-2.2.1-py3-none-any.whl (14 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.9012919Z Downloading typing_extensions-4.14.1-py3-none-any.whl (43 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:21.9068021Z Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
build (3.10)    Install dependencies    2025-08-01T23:39:22.1679921Z Installing collected packages: pytz, tzdata, typing-extensions, tomli, six, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, scipy, python-dateutil, h5py, exceptiongroup, contourpy, pytest, pandas, matplotlib
build (3.10)    Install dependencies    2025-08-01T23:39:35.1384221Z
build (3.10)    Install dependencies    2025-08-01T23:39:35.1428898Z Successfully installed contourpy-1.3.2 cycler-0.12.1 exceptiongroup-1.3.0 fonttools-4.59.0 h5py-3.14.0 iniconfig-2.1.0 kiwisolver-1.4.8 matplotlib-3.10.5 numpy-2.2.6 packaging-25.0 pandas-2.3.1 pillow-11.3.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.3 pytest-8.4.1 python-dateutil-2.9.0.post0 pytz-2025.2 scipy-1.15.3 six-1.17.0 tomli-2.2.1 typing-extensions-4.14.1 tzdata-2025.2
build (3.10)    Install dependencies    2025-08-01T23:39:36.1262502Z Processing /home/runner/work/negative-energy-generator/negative-energy-generator
build (3.10)    Install dependencies    2025-08-01T23:39:36.1288327Z   Installing build dependencies: started
build (3.10)    Install dependencies    2025-08-01T23:39:37.2782178Z   Installing build dependencies: finished with status 'done'
build (3.10)    Install dependencies    2025-08-01T23:39:37.2789230Z   Getting requirements to build wheel: started
build (3.10)    Install dependencies    2025-08-01T23:39:37.7541357Z   Getting requirements to build wheel: finished with status 'done'
build (3.10)    Install dependencies    2025-08-01T23:39:37.7551412Z   Preparing metadata (pyproject.toml): started
build (3.10)    Install dependencies    2025-08-01T23:39:37.9373423Z   Preparing metadata (pyproject.toml): finished with status 'done'
build (3.10)    Install dependencies    2025-08-01T23:39:37.9412886Z Requirement already satisfied: numpy>=1.24 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.2.6)
build (3.10)    Install dependencies    2025-08-01T23:39:37.9419514Z Requirement already satisfied: scipy>=1.11 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (1.15.3)
build (3.10)    Install dependencies    2025-08-01T23:39:37.9426253Z Requirement already satisfied: matplotlib>=3.6 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (3.10.5)
build (3.10)    Install dependencies    2025-08-01T23:39:37.9432458Z Requirement already satisfied: pandas>=1.5 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.3.1)
build (3.10)    Install dependencies    2025-08-01T23:39:38.0462561Z INFO: pip is looking at multiple versions of negative-energy-generator to determine which version is compatible with other requirements. This could take a while.
build (3.10)    Install dependencies    2025-08-01T23:39:38.0467004Z ERROR: Could not find a version that satisfies the requirement lqg_first_principles_gravitational_constant>=0.1.0 (from negative-energy-generator) (from versions: none)
build (3.10)    Install dependencies    2025-08-01T23:39:38.0477549Z ERROR: No matching distribution found for lqg_first_principles_gravitational_constant>=0.1.0
build (3.10)    Install dependencies    2025-08-01T23:39:38.1065331Z ##[error]Process completed with exit code 1.
````

## 2025-08-06

### In Progress
- Added `test_time_integration_basic.py` to validate `solve_klein_gordon` output shapes and finite values.

### Next Tasks
- Run the new test locally and confirm CI compatibility.
- Refine time integration accuracy by comparing against analytical solution for small dt.

### Progress Update
- Added `pytest.ini` to configure test discovery under `tests/` and set `pythonpath = src`, simplifying local and CI runs.

### Next Tasks
- Rerun CI and local tests to confirm configuration resolves any remaining import or discovery issues.
- Begin dynamic field evolution accuracy comparisons with analytical solutions.
