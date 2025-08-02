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
- Created semiclassical backreaction module in `src/simulation/backreaction.py`.
- Developed backreaction CLI demo `scripts/backreaction_demo.py` with HDF5 export.
- Added integration tests in `tests/test_backreaction.py` and `tests/test_backreaction_export.py`.
- Created `scripts/survey_repos.py` to scan external repositories for functions and modules of interest.

### Next Tasks
1. Update V&V and UQ trackers with tasks for lattice solver validation and uncertainty quantification of lattice energy densities.
2. Implement finite-difference time-integration in `solve_klein_gordon` for dynamic field evolution.
3. Configure CI workflow (e.g., GitHub Actions) to run unit tests, integration tests, and CLI demo automatically.
4. Implement integration test for HDF5 export result validation (`test_parameter_sweep_export.py`).
5. Review and optimize time-integration accuracy in `solve_klein_gordon` based on discretization test outcomes.
6. Test and iterate on the CI workflow to ensure it runs unit tests and the CLI demo successfully.
7. Complete and validate the dynamic field evolution against discretization tests.
8. Add integration tests for HDF5 export result validation in `tests/test_parameter_sweep_export.py`.
9. Extend CI workflow to include backreaction demo execution and export validation.
10. Refine documentation for backreaction API and usage examples in `docs/future-directions.md`.
11. Run `scripts/survey_repos.py` and analyze `results/external_survey.json` for candidates.
12. Identify and import functions (e.g., ANEC checks, vacuum energy models) from external repos into our modules.
13. Update documentation with mappings of external modules to our negative-energy-generator framework.
14. Analyze numerical stability of metric evolution and adjust solver parameters.
15. Integrate backreaction UQ sensitivity tasks into automated analysis pipeline.
16. Review overall CI to ensure all demos (lattice_sweep and backreaction) run successfully.
17. Survey `lorentz-violation-pipeline` for stress-energy violation models to inform negative energy generation algorithms.
18. Review `lqg-anec-framework` for averaged null energy condition (ANEC) checks and integrate relevant inequality constraints.
19. Harvest vacuum energy prediction modules in `unified-lqg` and `unified-lqg-qft` to model macroscopic exotic state production.
20. Integrate exotic matter density routines from `warp-bubble-exotic-matter-density` into lattice QFT and backreaction demos for realistic density profiles.
21. Adapt advanced QFT toolchains from `warp-bubble-qft` to extend `evolve_QFT` and `detect_exotics` capabilities for larger-scale scenarios.

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
Survey script fixed and rerun successfully, and CI updated to execute the backreaction demo. I've updated the progress log with those changes. Now, I'll analyze external_survey.json to identify candidate modules to integrate. 
```
```progress
I’ve added a `survey_repos.py` script to catalog functions in the target repos, and updated the progress log with tasks to run the survey, identify useful modules, and integrate them into our project. Next step: run the survey, review the results, and begin importing relevant functions.
```
```progress
I’ve added tasks to survey and integrate relevant modules from the listed repos (Lorentz violation, ANEC, unified LQG, and warp-bubble QFT/density) into our macroscopic negative energy framework. Next, I can begin exploration in each repo and propose specific code imports or adaptations...
```
```progress
I’ve implemented the semiclassical backreaction module and demo, added export tests, and updated the progress log. Next I’ll integrate the backreaction demo into CI, enhance backreaction documentation, and refine solver stability. 
```
```progress
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
```oldest-progress
I’ve scaffolded the GitHub Actions CI workflow at ci.yml to install dependencies, run unit tests, and execute the CLI demo. The progress_log.md is updated accordingly. Next up, I’ll implement finite-difference time integration validation against discretization tests and write the HDF5 export integration test.
```

```file-history
~/Code/asciimath/negative-energy-generator$ find . -path "./.venv" -prune -o -type f -regex '.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' -print | while read file; do stat -c '%Y %n' "$file"; done | sort -nr | while read timestamp file; do echo "$(date -d @$timestamp '+%Y-%m-%d %H:%M:%S') $file"; done | head -n 40
2025-08-01 21:41:44 ./docs/progress_log.md
2025-08-01 21:40:13 ./scripts/survey_repos.py
2025-08-01 21:40:13 ./.github/workflows/ci.yml
2025-08-01 21:39:38 ./results/external_survey.json
2025-08-01 21:10:40 ./tests/test_backreaction_export.py
2025-08-01 21:10:40 ./scripts/backreaction_demo.py
2025-08-01 20:56:42 ./tests/test_backreaction.py
2025-08-01 20:56:42 ./src/simulation/backreaction.py
2025-08-01 20:56:42 ./VnV-TODO.ndjson
2025-08-01 20:56:42 ./UQ-TODO.ndjson
2025-08-01 20:30:15 ./tests/test_zero_initial_condition.py
2025-08-01 20:30:15 ./tests/test_time_integration_basic.py
2025-08-01 20:30:15 ./tests/test_qft_backend.py
2025-08-01 20:30:15 ./tests/test_parameter_sweep_export.py
2025-08-01 20:30:15 ./tests/test_lattice_energy.py
2025-08-01 20:30:15 ./tests/test_lattice_discretization.py
2025-08-01 20:30:15 ./tests/test_energy_conservation.py
2025-08-01 20:30:15 ./tests/test_analytical_solution.py
2025-08-01 20:30:15 ./src/simulation/qft_backend.py
2025-08-01 20:30:15 ./src/simulation/photonic_crystal_band.py
2025-08-01 20:30:15 ./src/simulation/parameter_sweep.py
2025-08-01 20:30:15 ./src/simulation/mechanical_fem.py
2025-08-01 20:30:15 ./src/simulation/lattice_qft.py
2025-08-01 20:30:15 ./src/simulation/electromagnetic_fdtd.py
2025-08-01 20:30:15 ./scripts/lattice_sweep_demo.py
2025-08-01 20:30:15 ./results/demo_sweep.h5
2025-08-01 20:30:15 ./pytest.ini
2025-08-01 20:30:15 ./pyproject.toml
2025-08-01 20:30:15 ./physics_driven_prototype_validation.py
2025-08-01 20:30:15 ./docs/literature_review.md
2025-08-01 20:30:15 ./docs/future-directions.md
2025-08-01 20:30:15 ./corrected_validation_results.json
2025-08-01 20:30:15 ./README.md
2025-08-01 20:30:15 ./.github/instructions/copilot-instructions.md
2025-07-31 19:25:44 ./working_validation_test.py
2025-07-31 19:25:44 ./working_negative_energy_generator.py
2025-07-31 19:25:44 ./verify_prototype_stack_fixed.py
2025-07-31 19:25:44 ./verify_prototype_stack.py
2025-07-31 19:25:44 ./validation_summary.json
2025-07-31 19:25:44 ./unified_exotic_matter_sourcing_results.json
````

```test-history
~/Code/asciimath/negative-energy-generator~/Code/asciimath/negative-energy-generator$ python -m pytest --maxfail=1
=============================================== test session starts ================================================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/echo_/Code/asciimath/negative-energy-generator
configfile: pytest.ini
testpaths: tests
collected 2 items / 1 error                                                                                        

====================================================== ERRORS ======================================================
________________________________ ERROR collecting tests/test_backreaction_export.py ________________________________
ImportError while importing test module '/home/echo_/Code/asciimath/negative-energy-generator/tests/test_backreaction_export.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../miniconda3/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_backreaction_export.py:5: in <module>
    import h5py
E   ModuleNotFoundError: No module named 'h5py'
================================================= warnings summary =================================================
src/simulation/quantum_circuit_sim.py:32
  /home/echo_/Code/asciimath/negative-energy-generator/src/simulation/quantum_circuit_sim.py:32: UserWarning: QuTiP not available. Install with: pip install qutip
    warnings.warn("QuTiP not available. Install with: pip install qutip")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================= short test summary info ==============================================
ERROR tests/test_backreaction_export.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
=========================================== 1 warning, 1 error in 0.12s ============================================

~/Code/asciimath/negative-energy-generator$ gh run view 16690052742 --log-failed
build (3.10)    Install dependencies    ﻿2025-08-02T04:52:33.5746952Z ##[group]Run python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T04:52:33.5747445Z python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T04:52:33.5747873Z # Install core dependencies and test tools
build (3.10)    Install dependencies    2025-08-02T04:52:33.5748350Z python -m pip install numpy scipy matplotlib pandas h5py pytest
build (3.10)    Install dependencies    2025-08-02T04:52:33.5748813Z # Install the project package
build (3.10)    Install dependencies    2025-08-02T04:52:33.5749171Z python -m pip install .
build (3.10)    Install dependencies    2025-08-02T04:52:33.5867857Z shell: /usr/bin/bash -e {0}
build (3.10)    Install dependencies    2025-08-02T04:52:33.5868199Z env:
build (3.10)    Install dependencies    2025-08-02T04:52:33.5868670Z   PYTHONPATH: /home/runner/work/negative-energy-generator/negative-energy-generator/src
build (3.10)    Install dependencies    2025-08-02T04:52:33.5869300Z   pythonLocation: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:52:33.5869834Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib/pkgconfig
build (3.10)    Install dependencies    2025-08-02T04:52:33.5870334Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:52:33.5870793Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:52:33.5871276Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:52:33.5871736Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib
build (3.10)    Install dependencies    2025-08-02T04:52:33.5872130Z ##[endgroup]
build (3.10)    Install dependencies    2025-08-02T04:52:35.6587177Z Requirement already satisfied: pip in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (25.1.1)
build (3.10)    Install dependencies    2025-08-02T04:52:35.7827536Z Collecting pip
build (3.10)    Install dependencies    2025-08-02T04:52:35.8533474Z   Downloading pip-25.2-py3-none-any.whl.metadata (4.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:35.8748180Z Downloading pip-25.2-py3-none-any.whl (1.8 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:35.9558111Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 37.9 MB/s eta 0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:35.9932472Z Installing collected packages: pip
build (3.10)    Install dependencies    2025-08-02T04:52:35.9935601Z   Attempting uninstall: pip
build (3.10)    Install dependencies    2025-08-02T04:52:35.9944481Z     Found existing installation: pip 25.1.1
build (3.10)    Install dependencies    2025-08-02T04:52:36.0490389Z     Uninstalling pip-25.1.1:
build (3.10)    Install dependencies    2025-08-02T04:52:36.0552715Z       Successfully uninstalled pip-25.1.1
build (3.10)    Install dependencies    2025-08-02T04:52:36.8381276Z Successfully installed pip-25.2
build (3.10)    Install dependencies    2025-08-02T04:52:37.6621808Z Collecting numpy
build (3.10)    Install dependencies    2025-08-02T04:52:37.7338697Z   Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:37.9003207Z Collecting scipy
build (3.10)    Install dependencies    2025-08-02T04:52:37.9183640Z   Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:38.0941033Z Collecting matplotlib
build (3.10)    Install dependencies    2025-08-02T04:52:38.1135645Z   Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:38.2717064Z Collecting pandas
build (3.10)    Install dependencies    2025-08-02T04:52:38.2889326Z   Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:38.3852059Z Collecting h5py
build (3.10)    Install dependencies    2025-08-02T04:52:38.4030119Z   Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:38.4594443Z Collecting pytest
build (3.10)    Install dependencies    2025-08-02T04:52:38.4759531Z   Downloading pytest-8.4.1-py3-none-any.whl.metadata (7.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:38.5842614Z Collecting contourpy>=1.0.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:52:38.6016717Z   Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:38.6291649Z Collecting cycler>=0.10 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:52:38.6450201Z   Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:38.8262446Z Collecting fonttools>=4.22.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:52:38.8441709Z   Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (107 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:38.9326551Z Collecting kiwisolver>=1.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:52:38.9524939Z   Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.2 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.0093260Z Collecting packaging>=20.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:52:39.0261823Z   Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.2554332Z Collecting pillow>=8 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:52:39.2723414Z   Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.3184142Z Collecting pyparsing>=2.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:52:39.3343354Z   Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.3627202Z Collecting python-dateutil>=2.7 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:52:39.3866889Z   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.4554416Z Collecting pytz>=2020.1 (from pandas)
build (3.10)    Install dependencies    2025-08-02T04:52:39.4715368Z   Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.5012532Z Collecting tzdata>=2022.7 (from pandas)
build (3.10)    Install dependencies    2025-08-02T04:52:39.5172112Z   Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.5575669Z Collecting exceptiongroup>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:52:39.5736109Z   Downloading exceptiongroup-1.3.0-py3-none-any.whl.metadata (6.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.5974334Z Collecting iniconfig>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:52:39.6141841Z   Downloading iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.6403352Z Collecting pluggy<2,>=1.5 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:52:39.6565291Z   Downloading pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.6924512Z Collecting pygments>=2.7.2 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:52:39.7089602Z   Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.7393640Z Collecting tomli>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:52:39.7554139Z   Downloading tomli-2.2.1-py3-none-any.whl.metadata (10 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.7898296Z Collecting typing-extensions>=4.6.0 (from exceptiongroup>=1->pytest)
build (3.10)    Install dependencies    2025-08-02T04:52:39.8066694Z   Downloading typing_extensions-4.14.1-py3-none-any.whl.metadata (3.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.8397322Z Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:52:39.8592094Z   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:39.8849472Z Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.0285860Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.8/16.8 MB 130.7 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:40.0484718Z Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.2656704Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 174.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:40.2831180Z Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.3302728Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 188.8 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:40.3483135Z Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.3 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.4216567Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.3/12.3 MB 170.8 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:40.4413324Z Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.4743367Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 142.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:40.4905920Z Downloading pytest-8.4.1-py3-none-any.whl (365 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.5099729Z Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.5286634Z Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.5490975Z Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.5671789Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.5860384Z Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.6151176Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 173.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:40.6311736Z Downloading iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.6511214Z Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.6656080Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 120.6 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:40.6816066Z Downloading packaging-25.0-py3-none-any.whl (66 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.7007042Z Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.7471217Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 146.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:40.7664538Z Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.7763784Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 143.5 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:52:40.7932196Z Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.8127189Z Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.8330492Z Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.8547044Z Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.8731965Z Downloading tomli-2.2.1-py3-none-any.whl (14 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.8915273Z Downloading typing_extensions-4.14.1-py3-none-any.whl (43 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:40.9101254Z Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
build (3.10)    Install dependencies    2025-08-02T04:52:41.1670842Z Installing collected packages: pytz, tzdata, typing-extensions, tomli, six, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, scipy, python-dateutil, h5py, exceptiongroup, contourpy, pytest, pandas, matplotlib
build (3.10)    Install dependencies    2025-08-02T04:52:54.2617125Z
build (3.10)    Install dependencies    2025-08-02T04:52:54.2661491Z Successfully installed contourpy-1.3.2 cycler-0.12.1 exceptiongroup-1.3.0 fonttools-4.59.0 h5py-3.14.0 iniconfig-2.1.0 kiwisolver-1.4.8 matplotlib-3.10.5 numpy-2.2.6 packaging-25.0 pandas-2.3.1 pillow-11.3.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.3 pytest-8.4.1 python-dateutil-2.9.0.post0 pytz-2025.2 scipy-1.15.3 six-1.17.0 tomli-2.2.1 typing-extensions-4.14.1 tzdata-2025.2
build (3.10)    Install dependencies    2025-08-02T04:52:54.7719222Z Processing /home/runner/work/negative-energy-generator/negative-energy-generator
build (3.10)    Install dependencies    2025-08-02T04:52:54.7744887Z   Installing build dependencies: started
build (3.10)    Install dependencies    2025-08-02T04:52:55.8946698Z   Installing build dependencies: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T04:52:55.8953171Z   Getting requirements to build wheel: started
build (3.10)    Install dependencies    2025-08-02T04:52:56.3820053Z   Getting requirements to build wheel: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T04:52:56.3829779Z   Preparing metadata (pyproject.toml): started
build (3.10)    Install dependencies    2025-08-02T04:52:56.5766764Z   Preparing metadata (pyproject.toml): finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T04:52:56.5804984Z Requirement already satisfied: numpy>=1.24 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.2.6)
build (3.10)    Install dependencies    2025-08-02T04:52:56.5811734Z Requirement already satisfied: scipy>=1.11 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (1.15.3)
build (3.10)    Install dependencies    2025-08-02T04:52:56.5818885Z Requirement already satisfied: matplotlib>=3.6 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (3.10.5)
build (3.10)    Install dependencies    2025-08-02T04:52:56.5825269Z Requirement already satisfied: pandas>=1.5 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.3.1)
build (3.10)    Install dependencies    2025-08-02T04:52:56.7570809Z INFO: pip is looking at multiple versions of negative-energy-generator to determine which version is compatible with other requirements. This could take a while.
build (3.10)    Install dependencies    2025-08-02T04:52:56.7574518Z ERROR: Could not find a version that satisfies the requirement lqg_first_principles_gravitational_constant>=0.1.0 (from negative-energy-generator) (from versions: none)
build (3.10)    Install dependencies    2025-08-02T04:52:56.7583615Z ERROR: No matching distribution found for lqg_first_principles_gravitational_constant>=0.1.0
build (3.10)    Install dependencies    2025-08-02T04:52:56.8126894Z ##[error]Process completed with exit code 1.
```