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
11. Analyze numerical stability of metric evolution and adjust solver parameters.
12. Integrate backreaction UQ sensitivity tasks into automated analysis pipeline.
13. Review overall CI to ensure all demos (lattice_sweep and backreaction) run successfully.
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
```progress
I’ve scaffolded the GitHub Actions CI workflow at ci.yml to install dependencies, run unit tests, and execute the CLI demo. The progress_log.md is updated accordingly. Next up, I’ll implement finite-difference time integration validation against discretization tests and write the HDF5 export integration test.
```
```progress
I’ll proceed to implement `solve_klein_gordon`’s time integration in lattice_qft.py. 
```
```progress
I’ve added a discretization unit test for the Laplacian and updated the progress log. Next, I recommend setting up a CI workflow to automatically run all tests and the CLI demo—ensuring stability—and writing the HDF5 export integration test. I can scaffold a GitHub Actions config if you’d like.
```
```oldest-progress
V&V and UQ trackers updated with high- and medium-priority tasks for the lattice QFT solver, HDF5 exports, and parameter sensitivities. The progress log and test/demo scripts are in place. Next, I recommend implementing the discretization unit test or refining the solver's accuracy. Let me know if you’d like to proceed with those, or I can set up a CI workflow next.
```

```file-history
~/Code/asciimath/negative-energy-generator$ find . -path "./.venv" -prune -o -type f -regex '.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' -print | while read file; do stat -c '%Y %n' "$file"; done | sort -nr | while read timestamp file; do echo "$(date -d @$timestamp '+%Y-%m-%d %H:%M:%S') $file"; done | head -n 40
2025-08-01 21:17:10 ./docs/progress_log.md
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
2025-08-01 20:30:15 ./.github/workflows/ci.yml
2025-08-01 20:30:15 ./.github/instructions/copilot-instructions.md
2025-07-31 19:25:44 ./working_validation_test.py
2025-07-31 19:25:44 ./working_negative_energy_generator.py
2025-07-31 19:25:44 ./verify_prototype_stack_fixed.py
2025-07-31 19:25:44 ./verify_prototype_stack.py
2025-07-31 19:25:44 ./validation_summary.json
2025-07-31 19:25:44 ./unified_exotic_matter_sourcing_results.json
2025-07-31 19:25:44 ./unified_exotic_matter_sourcing.py
2025-07-31 19:25:44 ./two_phase_summary.py
````

```test-history
~/Code/asciimath/negative-energy-generator$ /home/sherri3/Code/asciimath/negative-energy-generator/.venv/bin/python -m pytest --maxfail=1
bash: /home/sherri3/Code/asciimath/negative-energy-generator/.venv/bin/python: No such file or directory

gh run view 16689808072 --log-failed
build (3.10)    Install dependencies    ﻿2025-08-02T04:19:28.6299480Z ##[group]Run python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T04:19:28.6300842Z python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T04:19:28.6302126Z # Install core dependencies and test tools
build (3.10)    Install dependencies    2025-08-02T04:19:28.6303730Z python -m pip install numpy scipy matplotlib pandas h5py pytest
build (3.10)    Install dependencies    2025-08-02T04:19:28.6305268Z # Install the project package
build (3.10)    Install dependencies    2025-08-02T04:19:28.6306354Z python -m pip install .
build (3.10)    Install dependencies    2025-08-02T04:19:28.6401154Z shell: /usr/bin/bash -e {0}
build (3.10)    Install dependencies    2025-08-02T04:19:28.6402080Z env:
build (3.10)    Install dependencies    2025-08-02T04:19:28.6403490Z   PYTHONPATH: /home/runner/work/negative-energy-generator/negative-energy-generator/src
build (3.10)    Install dependencies    2025-08-02T04:19:28.6405492Z   pythonLocation: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:19:28.6407175Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib/pkgconfig
build (3.10)    Install dependencies    2025-08-02T04:19:28.6408935Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:19:28.6410452Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:19:28.6411953Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T04:19:28.6413432Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib
build (3.10)    Install dependencies    2025-08-02T04:19:28.6414660Z ##[endgroup]
build (3.10)    Install dependencies    2025-08-02T04:19:30.9314285Z Requirement already satisfied: pip in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (25.1.1)
build (3.10)    Install dependencies    2025-08-02T04:19:31.0126879Z Collecting pip
build (3.10)    Install dependencies    2025-08-02T04:19:31.0571500Z   Downloading pip-25.2-py3-none-any.whl.metadata (4.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:31.0655008Z Downloading pip-25.2-py3-none-any.whl (1.8 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:31.1050018Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 63.0 MB/s eta 0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:31.1413809Z Installing collected packages: pip
build (3.10)    Install dependencies    2025-08-02T04:19:31.1415697Z   Attempting uninstall: pip
build (3.10)    Install dependencies    2025-08-02T04:19:31.1421071Z     Found existing installation: pip 25.1.1
build (3.10)    Install dependencies    2025-08-02T04:19:31.1902560Z     Uninstalling pip-25.1.1:
build (3.10)    Install dependencies    2025-08-02T04:19:31.1964280Z       Successfully uninstalled pip-25.1.1
build (3.10)    Install dependencies    2025-08-02T04:19:31.9557658Z Successfully installed pip-25.2
build (3.10)    Install dependencies    2025-08-02T04:19:32.6777117Z Collecting numpy
build (3.10)    Install dependencies    2025-08-02T04:19:32.7101272Z   Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:32.8538417Z Collecting scipy
build (3.10)    Install dependencies    2025-08-02T04:19:32.8581087Z   Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.0170132Z Collecting matplotlib
build (3.10)    Install dependencies    2025-08-02T04:19:33.0210792Z   Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.1619350Z Collecting pandas
build (3.10)    Install dependencies    2025-08-02T04:19:33.1660592Z   Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.2452608Z Collecting h5py
build (3.10)    Install dependencies    2025-08-02T04:19:33.2491912Z   Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.2904288Z Collecting pytest
build (3.10)    Install dependencies    2025-08-02T04:19:33.2946097Z   Downloading pytest-8.4.1-py3-none-any.whl.metadata (7.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.3838637Z Collecting contourpy>=1.0.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:19:33.4044909Z   Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.4178103Z Collecting cycler>=0.10 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:19:33.4214092Z   Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.5860581Z Collecting fonttools>=4.22.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:19:33.5905293Z   Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (107 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.6637448Z Collecting kiwisolver>=1.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:19:33.6674661Z   Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.2 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.6950233Z Collecting packaging>=20.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:19:33.6987970Z   Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:33.9446616Z Collecting pillow>=8 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:19:33.9488569Z   Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.0378518Z Collecting pyparsing>=2.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:19:34.0415292Z   Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.3304268Z Collecting python-dateutil>=2.7 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:19:34.3344647Z   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.3889957Z Collecting pytz>=2020.1 (from pandas)
build (3.10)    Install dependencies    2025-08-02T04:19:34.3927398Z   Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.4098580Z Collecting tzdata>=2022.7 (from pandas)
build (3.10)    Install dependencies    2025-08-02T04:19:34.4135533Z   Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.4402535Z Collecting exceptiongroup>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:19:34.4439816Z   Downloading exceptiongroup-1.3.0-py3-none-any.whl.metadata (6.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.4547495Z Collecting iniconfig>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:19:34.4584777Z   Downloading iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.4723387Z Collecting pluggy<2,>=1.5 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:19:34.4761015Z   Downloading pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.4979771Z Collecting pygments>=2.7.2 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:19:34.5016896Z   Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.5182341Z Collecting tomli>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T04:19:34.5218611Z   Downloading tomli-2.2.1-py3-none-any.whl.metadata (10 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.5437440Z Collecting typing-extensions>=4.6.0 (from exceptiongroup>=1->pytest)
build (3.10)    Install dependencies    2025-08-02T04:19:34.5474518Z   Downloading typing_extensions-4.14.1-py3-none-any.whl.metadata (3.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.5674304Z Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib)
build (3.10)    Install dependencies    2025-08-02T04:19:34.5710163Z   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.5822281Z Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.7064305Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.8/16.8 MB 141.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:34.7103814Z Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.8658756Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 244.8 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:34.8698829Z Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.9136788Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 204.3 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:34.9174746Z Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.3 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:34.9768844Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.3/12.3 MB 212.2 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:34.9808792Z Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.0065600Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 188.6 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:35.0103886Z Downloading pytest-8.4.1-py3-none-any.whl (365 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.0177524Z Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.0237639Z Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.0309529Z Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.0366688Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.0424527Z Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.0697491Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 182.7 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:35.0734105Z Downloading iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.0795725Z Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.0912160Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 155.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:35.0949606Z Downloading packaging-25.0-py3-none-any.whl (66 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.1013082Z Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.1332449Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 218.4 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:35.1372646Z Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.1458461Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 163.7 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T04:19:35.1496052Z Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.1698859Z Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.1765785Z Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.1844925Z Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.1901674Z Downloading tomli-2.2.1-py3-none-any.whl (14 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.1958237Z Downloading typing_extensions-4.14.1-py3-none-any.whl (43 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.2021278Z Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
build (3.10)    Install dependencies    2025-08-02T04:19:35.4692301Z Installing collected packages: pytz, tzdata, typing-extensions, tomli, six, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, scipy, python-dateutil, h5py, exceptiongroup, contourpy, pytest, pandas, matplotlib
build (3.10)    Install dependencies    2025-08-02T04:19:48.3341378Z
build (3.10)    Install dependencies    2025-08-02T04:19:48.3394637Z Successfully installed contourpy-1.3.2 cycler-0.12.1 exceptiongroup-1.3.0 fonttools-4.59.0 h5py-3.14.0 iniconfig-2.1.0 kiwisolver-1.4.8 matplotlib-3.10.5 numpy-2.2.6 packaging-25.0 pandas-2.3.1 pillow-11.3.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.3 pytest-8.4.1 python-dateutil-2.9.0.post0 pytz-2025.2 scipy-1.15.3 six-1.17.0 tomli-2.2.1 typing-extensions-4.14.1 tzdata-2025.2
build (3.10)    Install dependencies    2025-08-02T04:19:48.8493979Z Processing /home/runner/work/negative-energy-generator/negative-energy-generator
build (3.10)    Install dependencies    2025-08-02T04:19:48.8520286Z   Installing build dependencies: started
build (3.10)    Install dependencies    2025-08-02T04:19:49.8023419Z   Installing build dependencies: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T04:19:49.8030351Z   Getting requirements to build wheel: started
build (3.10)    Install dependencies    2025-08-02T04:19:50.3874230Z   Getting requirements to build wheel: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T04:19:50.3884292Z   Preparing metadata (pyproject.toml): started
build (3.10)    Install dependencies    2025-08-02T04:19:50.5671190Z   Preparing metadata (pyproject.toml): finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T04:19:50.5708393Z Requirement already satisfied: numpy>=1.24 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.2.6)
build (3.10)    Install dependencies    2025-08-02T04:19:50.5715858Z Requirement already satisfied: scipy>=1.11 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (1.15.3)
build (3.10)    Install dependencies    2025-08-02T04:19:50.5722301Z Requirement already satisfied: matplotlib>=3.6 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (3.10.5)
build (3.10)    Install dependencies    2025-08-02T04:19:50.5727958Z Requirement already satisfied: pandas>=1.5 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.3.1)
build (3.10)    Install dependencies    2025-08-02T04:19:50.6730624Z INFO: pip is looking at multiple versions of negative-energy-generator to determine which version is compatible with other requirements. This could take a while.
build (3.10)    Install dependencies    2025-08-02T04:19:50.6734968Z ERROR: Could not find a version that satisfies the requirement lqg_first_principles_gravitational_constant>=0.1.0 (from negative-energy-generator) (from versions: none)
build (3.10)    Install dependencies    2025-08-02T04:19:50.6744612Z ERROR: No matching distribution found for lqg_first_principles_gravitational_constant>=0.1.0
build (3.10)    Install dependencies    2025-08-02T04:19:50.7282835Z ##[error]Process completed with exit code 1.
```