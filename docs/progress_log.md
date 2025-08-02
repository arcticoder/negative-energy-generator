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
I’ve added V&V tests for the `solve_semiclassical_metric` function and a UQ task for T00 sensitivity in backreaction. The roadmap is expanding through all three future-direction stages. 

Next, I’ll set up the CI workflow file adjustments if needed, or begin writing the HDF5 export integration test.
```
```progress
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
```oldest-progress
The CLI demo `lattice_sweep_demo.py` is in place and the progress log reflects that. The remaining tasks:

1. Add V&V and UQ tracker entries for lattice solver validation.
2. Ensure the `solve_klein_gordon` integration is fully implemented (already updated earlier).

Next, I’ll update the V&V and UQ ndjson files with validation and uncertainty tasks for the lattice solver. Let me know if that aligns!
```

```file-history
~/Code/asciimath$ find . -path "./.venv" -prune -o -type f -regex '.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' -print | while read file; do stat -c '%Y %n' "$file"; done | sort -nr | while read timestamp file; do echo "$(date -d @$timestamp '+%Y-%m-%d %H:%M:%S') $file"; done | head -n 40
2025-08-01 20:58:46 ./negative-energy-generator/docs/progress_log.md
2025-08-01 20:56:42 ./negative-energy-generator/tests/test_backreaction.py
2025-08-01 20:56:42 ./negative-energy-generator/src/simulation/backreaction.py
2025-08-01 20:56:42 ./negative-energy-generator/VnV-TODO.ndjson
2025-08-01 20:56:42 ./negative-energy-generator/UQ-TODO.ndjson
2025-08-01 20:49:41 ./casimir-nanopositioning-platform/validation_results/nanometer_validation_results.json
2025-08-01 20:49:41 ./casimir-nanopositioning-platform/nanometer_statistical_coverage_validator.py
2025-08-01 20:49:41 ./casimir-nanopositioning-platform/casimir_uq_resolution.py
2025-08-01 20:49:41 ./casimir-nanopositioning-platform/UQ_RESOLUTION_IMPLEMENTATION.py
2025-08-01 20:49:41 ./casimir-nanopositioning-platform/UQ-TODO.ndjson
2025-08-01 20:49:41 ./casimir-nanopositioning-platform/README.md
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/vessel_architecture_resolution.json
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/uq_resolution/test_critical_uq_resolution.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/uq_resolution/final_uq_resolution_analysis.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/uq_resolution/critical_uq_resolution_framework.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/unmanned_probe_design_20250712_161851.json
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tools/optimized_demo.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tools/minimal_demo.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tools/gpu_check.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tools/generate_uq_resolution_summary.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tests/test_unmanned_probe_design.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tests/test_imports.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tests/test_ftl_hull_design.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tests/test_framework.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tests/test_enhanced_framework.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tests/test_complete_integration.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/tests/test_advanced_hull_optimization.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/unmanned_probe_structural_validation.json
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/unmanned_probe_structural_framework.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/unmanned_probe_design_framework.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/tidal_force_analysis_framework.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/quantum_field_manipulator.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/probe_life_support_elimination_validation.json
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/probe_life_support_elimination.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/optimized_nanolattice_fabrication_framework.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/optimized_nanolattice_fabrication.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/optimized_carbon_nanolattice_algorithms.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/naval_architecture_framework.py
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/multi_physics_hull_coupling_analysis.json
2025-08-01 20:49:37 ./enhanced-simulation-hardware-abstraction-framework/src/multi_physics_hull_coupling.py
````

```test-history
~/Code/asciimath/negative-energy-generator$ /home/sherri3/Code/asciimath/negative-energy-generator/.venv/bin/python -m pytest --maxfail=1
bash: /home/sherri3/Code/asciimath/negative-energy-generator/.venv/bin/python: No such file or directory

~/Code/asciimath/negative-energy-generator$ gh run view 16689695793 --log-failed
build (3.10)    Install dependencies    ﻿2025-08-02T04:05:48.7611685Z ##[group]Run python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T04:05:48.7612270Z python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T04:05:48.7612786Z # Install core dependencies and test tools
build (3.10)    Install dependencies    2025-08-02T04:05:48.7613362Z python -m pip install numpy scipy matplotlib pandas h5py pytest
build (3.10)    Install dependencies    2025-08-02T04:05:48.7613901Z # Install the project package
build (3.10)    Install dependencies    2025-08-02T04:05:48.7614323Z python -m pip install .
build (3.10)    Install dependencies    2025-08-02T04:05:48.7707333Z shell: /usr/bin/bash -e {0}
build (3.10)    Install dependencies    2025-08-02T04:05:48.7707732Z env:
build (3.10)    Install dependencies    2025-08-02T04:05:48.7708259Z   PYTHONPATH: /home/runner/work/negative-energy-generator/negative-energy-generator/src
build (3.10)    Install dependencies    2025-08-02T04:05:48.7708948Z   pythonLocation: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:05:48.7709540Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib/pkgconfig
build (3.10)    Install dependencies    2025-08-02T04:05:48.7710142Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:05:48.7710817Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:05:48.7711376Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:05:48.7711909Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib
build (3.10)    Install dependencies    2025-08-02T04:05:48.7712360Z ##[endgroup]
build (3.10)    Install dependencies    2025-08-02T04:05:50.6572178Z Requirement already satisfied: pip in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (25.1.1)
build (3.10)    Install dependencies    2025-08-02T04:05:50.7755453Z Collecting pip
build (3.10)    Install dependencies    2025-08-02T04:05:50.8482225Z   Downloading pip-25.2-py3-none-any.whl.metadata (4.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:50.8628313Z Downloading pip-25.2-py3-none-any.whl (1.8 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:51.0104422Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 13.6 MB/s eta 0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:51.0484964Z Installing collected packages: pip
build (3.10)    Install dependencies    2025-08-02T04:05:51.0486953Z   Attempting uninstall: pip
build (3.10)    Install dependencies    2025-08-02T04:05:51.0492670Z     Found existing installation: pip 25.1.1
build (3.10)    Install dependencies    2025-08-02T04:05:51.1230663Z     Uninstalling pip-25.1.1:
build (3.10)    Install dependencies    2025-08-02T04:05:51.1293030Z       Successfully uninstalled pip-25.1.1
build (3.10)    Install dependencies    2025-08-02T04:05:51.8988096Z Successfully installed pip-25.2
build (3.10)    Install dependencies    2025-08-02T04:05:52.6872995Z Collecting numpy
build (3.10)    Install dependencies    2025-08-02T04:05:52.7393233Z   Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:52.9105502Z Collecting scipy
build (3.10)    Install dependencies    2025-08-02T04:05:52.9214236Z   Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:53.1061598Z Collecting matplotlib
build (3.10)    Install dependencies    2025-08-02T04:05:53.1170888Z   Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:53.2787495Z Collecting pandas
build (3.10)    Install dependencies    2025-08-02T04:05:53.2895555Z   Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:53.3894649Z Collecting h5py
build (3.10)    Install dependencies    2025-08-02T04:05:53.4002890Z   Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:53.4508058Z Collecting pytest
build (3.10)    Install dependencies    2025-08-02T04:05:53.4612538Z   Downloading pytest-8.4.1-py3-none-any.whl.metadata (7.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:53.5678837Z Collecting contourpy>=1.0.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:05:53.5783904Z   Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:53.6024385Z Collecting cycler>=0.10 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:05:53.6127119Z   Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:53.7961218Z Collecting fonttools>=4.22.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:05:53.8069289Z   Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (107 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:53.8957806Z Collecting kiwisolver>=1.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:05:53.9064410Z   Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.2 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:53.9424821Z Collecting packaging>=20.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:05:53.9527460Z   Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.1902465Z Collecting pillow>=8 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:05:54.2023148Z   Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.2427629Z Collecting pyparsing>=2.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:05:54.2546051Z   Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.2771059Z Collecting python-dateutil>=2.7 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:05:54.2873949Z   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.3520733Z Collecting pytz>=2020.1 (from pandas)
build (3.10)    Install dependencies    2025-08-02T04:05:54.3626802Z   Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.3864180Z Collecting tzdata>=2022.7 (from pandas)
build (3.10)    Install dependencies    2025-08-02T04:05:54.3972651Z   Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.4328361Z Collecting exceptiongroup>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:05:54.4428006Z   Downloading exceptiongroup-1.3.0-py3-none-any.whl.metadata (6.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.4607777Z Collecting iniconfig>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:05:54.4708690Z   Downloading iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.4925695Z Collecting pluggy<2,>=1.5 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:05:54.5030160Z   Downloading pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.5336506Z Collecting pygments>=2.7.2 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:05:54.5439944Z   Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.5742144Z Collecting tomli>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:05:54.5844854Z   Downloading tomli-2.2.1-py3-none-any.whl.metadata (10 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.6136047Z Collecting typing-extensions>=4.6.0 (from exceptiongroup>=1->pytest)
build (3.10)    Install dependencies    2025-08-02T04:05:54.6238393Z   Downloading typing_extensions-4.14.1-py3-none-any.whl.metadata (3.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.6521462Z Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:05:54.6636505Z   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:54.6812586Z Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.1161490Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.8/16.8 MB 40.3 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:55.1276242Z Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.4341566Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 123.6 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:55.4446876Z Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.4977517Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 168.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:55.5085900Z Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.3 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.5651563Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.3/12.3 MB 225.2 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:55.5783227Z Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.6061416Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 169.9 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:55.6165169Z Downloading pytest-8.4.1-py3-none-any.whl (365 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.6310285Z Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.6439898Z Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.6591221Z Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.6716954Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.6845456Z Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.7259722Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 119.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:55.7360830Z Downloading iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.7506104Z Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.7622101Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 150.1 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:55.7724589Z Downloading packaging-25.0-py3-none-any.whl (66 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.7861736Z Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.8175919Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 221.0 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:55.8281048Z Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.8378885Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 134.2 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:05:55.8483515Z Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.8609478Z Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.8751049Z Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.8896406Z Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.9030953Z Downloading tomli-2.2.1-py3-none-any.whl (14 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.9168781Z Downloading typing_extensions-4.14.1-py3-none-any.whl (43 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:55.9296836Z Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
build (3.10)    Install dependencies    2025-08-02T04:05:56.1872184Z Installing collected packages: pytz, tzdata, typing-extensions, tomli, six, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, scipy, python-dateutil, h5py, exceptiongroup, contourpy, pytest, pandas, matplotlib
build (3.10)    Install dependencies    2025-08-02T04:06:09.1041095Z
build (3.10)    Install dependencies    2025-08-02T04:06:09.1085494Z Successfully installed contourpy-1.3.2 cycler-0.12.1 exceptiongroup-1.3.0 fonttools-4.59.0 h5py-3.14.0 iniconfig-2.1.0 kiwisolver-1.4.8 matplotlib-3.10.5 numpy-2.2.6 packaging-25.0 pandas-2.3.1 pillow-11.3.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.3 pytest-8.4.1 python-dateutil-2.9.0.post0 pytz-2025.2 scipy-1.15.3 six-1.17.0 tomli-2.2.1 typing-extensions-4.14.1 tzdata-2025.2
build (3.10)    Install dependencies    2025-08-02T04:06:09.6200616Z Processing /home/runner/work/negative-energy-generator/negative-energy-generator
build (3.10)    Install dependencies    2025-08-02T04:06:09.6227265Z   Installing build dependencies: started
build (3.10)    Install dependencies    2025-08-02T04:06:10.6385692Z   Installing build dependencies: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T04:06:10.6392904Z   Getting requirements to build wheel: started
build (3.10)    Install dependencies    2025-08-02T04:06:11.1304828Z   Getting requirements to build wheel: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T04:06:11.1314272Z   Preparing metadata (pyproject.toml): started
build (3.10)    Install dependencies    2025-08-02T04:06:11.3160870Z   Preparing metadata (pyproject.toml): finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T04:06:11.3198304Z Requirement already satisfied: numpy>=1.24 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.2.6)
build (3.10)    Install dependencies    2025-08-02T04:06:11.3204890Z Requirement already satisfied: scipy>=1.11 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (1.15.3)
build (3.10)    Install dependencies    2025-08-02T04:06:11.3211654Z Requirement already satisfied: matplotlib>=3.6 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (3.10.5)
build (3.10)    Install dependencies    2025-08-02T04:06:11.3217779Z Requirement already satisfied: pandas>=1.5 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.3.1)
build (3.10)    Install dependencies    2025-08-02T04:06:11.4962372Z INFO: pip is looking at multiple versions of negative-energy-generator to determine which version is compatible with other requirements. This could take a while.
build (3.10)    Install dependencies    2025-08-02T04:06:11.4966088Z ERROR: Could not find a version that satisfies the requirement lqg_first_principles_gravitational_constant>=0.1.0 (from negative-energy-generator) (from versions: none)
build (3.10)    Install dependencies    2025-08-02T04:06:11.4974863Z ERROR: No matching distribution found for lqg_first_principles_gravitational_constant>=0.1.0
build (3.10)    Install dependencies    2025-08-02T04:06:11.5518271Z ##[error]Process completed with exit code 1.
````