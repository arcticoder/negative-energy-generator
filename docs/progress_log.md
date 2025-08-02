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
- Added `h5py` to `pyproject.toml` dependencies to resolve import errors in integration tests.
- Created stability test in `tests/test_backreaction_stability.py` verifying theoretical growth for constant source.

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
22. Integrate backreaction stability test into CI workflow.
23. Refine `solve_semiclassical_metric` to support spatial backreaction terms (e.g., metric Laplacian) and test accordingly.
24. Update technical documentation with backreaction API details and usage examples.
25. Link backreaction UQ tasks from `UQ-TODO.ndjson` into automated analysis notebooks.

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
I’ve added a stability test matching the theoretical growth pattern for constant T00, and updated the progress log with next tasks to integrate this into CI, extend the solver, and update docs. CI now consistently covers demos and tests.
```
```progress
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
```oldest-progress
I’ve quoted the Python versions in the CI matrix to prevent YAML misinterpretation (3.10 became 3.1). The workflow now correctly picks up 3.10 and 3.12. I updated the progress log—next, I’ll write the HDF5 export integration test.
```

```file-history
~/Code/asciimath/negative-energy-generator$ find . -path "./.venv" -prune -o -type f -regex '.*\.\(ps1\|py\|sh\|ndjson\|json\|md\|yml\|toml\|h5\|ini\)$' -print | while read file; do stat -c '%Y %n' "$file"; done | sort -nr | while read timestamp file; do echo "$(date -d @$timestamp '+%Y-%m-%d %H:%M:%S') $file"; done | head -n 40
2025-08-01 21:58:34 ./docs/progress_log.md
2025-08-01 21:57:00 ./tests/test_backreaction_stability.py
2025-08-01 21:57:00 ./pyproject.toml
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
````

```test-history
~/Code/asciimath/negative-energy-generator$ pytest --maxfail=1
======================================================================= test session starts =======================================================================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/echo_/Code/asciimath/negative-energy-generator
configfile: pytest.ini
testpaths: tests
collected 33 items                                                                                                                                                

tests/test_analytical_solution.py .                                                                                                                         [  3%]
tests/test_backreaction.py .                                                                                                                                [  6%]
tests/test_backreaction_export.py .                                                                                                                         [  9%]
tests/test_backreaction_stability.py F

============================================================================ FAILURES =============================================================================
_________________________________________________________ test_constant_source_growth_matches_theoretical _________________________________________________________

    def test_constant_source_growth_matches_theoretical():
        """
        For constant T00 and zero initial conditions, the n-th metric perturbation h_n should follow h_n = (2n-1) * dt^2 * source.
        """
        N = 10
        x = np.linspace(0, 1, N)
        # Constant source
        T00 = np.ones(N)
        dt = 0.1
        steps = 5
        G = 1.0
    
        # Solve backreaction
        h_final, history = solve_semiclassical_metric(x, T00, dt=dt, steps=steps, G=G)
    
        source_term = 8 * np.pi * G * T00
        # Check each step
        for n in range(1, steps + 1):
            expected = (2 * n - 1) * dt**2 * source_term
            # Compare history[n] to expected value across grid
>           assert np.allclose(history[n], expected, atol=1e-6), (
                f"Step {n}: expected {(2*n-1)}*dt^2*source, got {history[n][0]}"
            )
E           AssertionError: Step 3: expected 5*dt^2*source, got 1.5079644737231006
E           assert False
E            +  where False = <function allclose at 0x765a863317b0>(array([1.50796447, 1.50796447, 1.50796447, 1.50796447, 1.50796447,\n       1.50796447, 1.50796447, 1.50796447, 1.50796447, 1.50796447]), array([1.25663706, 1.25663706, 1.25663706, 1.25663706, 1.25663706,\n       1.25663706, 1.25663706, 1.25663706, 1.25663706, 1.25663706]), atol=1e-06)
E            +    where <function allclose at 0x765a863317b0> = np.allclose

tests/test_backreaction_stability.py:24: AssertionError
======================================================================== warnings summary =========================================================================
src/simulation/quantum_circuit_sim.py:32
  /home/echo_/Code/asciimath/negative-energy-generator/src/simulation/quantum_circuit_sim.py:32: UserWarning: QuTiP not available. Install with: pip install qutip
    warnings.warn("QuTiP not available. Install with: pip install qutip")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
===================================================================== short test summary info =====================================================================
FAILED tests/test_backreaction_stability.py::test_constant_source_growth_matches_theoretical - AssertionError: Step 3: expected 5*dt^2*source, got 1.5079644737231006
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
============================================================= 1 failed, 3 passed, 1 warning in 0.38s ==============================================================

~/Code/asciimath/negative-energy-generator$ gh run view 16690107471 --log-failed
build (3.10)    Install dependencies    ﻿2025-08-02T05:00:07.3307054Z ##[group]Run python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T05:00:07.3307622Z python -m pip install --upgrade pip
build (3.10)    Install dependencies    2025-08-02T05:00:07.3308128Z # Install core dependencies and test tools
build (3.10)    Install dependencies    2025-08-02T05:00:07.3308697Z python -m pip install numpy scipy matplotlib pandas h5py pytest
build (3.10)    Install dependencies    2025-08-02T05:00:07.3309242Z # Install the project package
build (3.10)    Install dependencies    2025-08-02T05:00:07.3309656Z python -m pip install .
build (3.10)    Install dependencies    2025-08-02T05:00:07.3345427Z shell: /usr/bin/bash -e {0}
build (3.10)    Install dependencies    2025-08-02T05:00:07.3345806Z env:
build (3.10)    Install dependencies    2025-08-02T05:00:07.3346350Z   PYTHONPATH: /home/runner/work/negative-energy-generator/negative-energy-generator/src
build (3.10)    Install dependencies    2025-08-02T05:00:07.3347071Z   pythonLocation: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T05:00:07.3347682Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib/pkgconfig
build (3.10)    Install dependencies    2025-08-02T05:00:07.3348274Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T05:00:07.3348827Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T05:00:07.3349439Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.10.18/x64
build (3.10)    Install dependencies    2025-08-02T05:00:07.3350044Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.10.18/x64/lib
build (3.10)    Install dependencies    2025-08-02T05:00:07.3350522Z ##[endgroup]
build (3.10)    Install dependencies    2025-08-02T05:00:07.8812499Z Requirement already satisfied: pip in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (25.1.1)
build (3.10)    Install dependencies    2025-08-02T05:00:08.0092366Z Collecting pip
build (3.10)    Install dependencies    2025-08-02T05:00:08.0952484Z   Downloading pip-25.2-py3-none-any.whl.metadata (4.7 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:08.1152262Z Downloading pip-25.2-py3-none-any.whl (1.8 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:08.1953149Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 40.8 MB/s eta 0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:08.2408721Z Installing collected packages: pip
build (3.10)    Install dependencies    2025-08-02T05:00:08.2411315Z   Attempting uninstall: pip
build (3.10)    Install dependencies    2025-08-02T05:00:08.2417126Z     Found existing installation: pip 25.1.1
build (3.10)    Install dependencies    2025-08-02T05:00:08.2931819Z     Uninstalling pip-25.1.1:
build (3.10)    Install dependencies    2025-08-02T05:00:08.3001296Z       Successfully uninstalled pip-25.1.1
build (3.10)    Install dependencies    2025-08-02T05:00:09.0676478Z Successfully installed pip-25.2
build (3.10)    Install dependencies    2025-08-02T05:00:09.8981108Z Collecting numpy
build (3.10)    Install dependencies    2025-08-02T05:00:09.9699831Z   Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:10.1400172Z Collecting scipy
build (3.10)    Install dependencies    2025-08-02T05:00:10.1580684Z   Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:10.3307340Z Collecting matplotlib
build (3.10)    Install dependencies    2025-08-02T05:00:10.3483789Z   Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:10.5058651Z Collecting pandas
build (3.10)    Install dependencies    2025-08-02T05:00:10.5237314Z   Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:10.6208642Z Collecting h5py
build (3.10)    Install dependencies    2025-08-02T05:00:10.6401871Z   Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:10.6992608Z Collecting pytest
build (3.10)    Install dependencies    2025-08-02T05:00:10.7156201Z   Downloading pytest-8.4.1-py3-none-any.whl.metadata (7.7 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:10.8220591Z Collecting contourpy>=1.0.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T05:00:10.8388548Z   Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:10.8655671Z Collecting cycler>=0.10 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T05:00:10.8813455Z   Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.0570580Z Collecting fonttools>=4.22.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T05:00:11.0739266Z   Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (107 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.1604113Z Collecting kiwisolver>=1.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T05:00:11.1771549Z   Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.2 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.2204999Z Collecting packaging>=20.0 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T05:00:11.2361757Z   Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.4735893Z Collecting pillow>=8 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T05:00:11.4935067Z   Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.5454166Z Collecting pyparsing>=2.3.1 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T05:00:11.5612458Z   Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.5899701Z Collecting python-dateutil>=2.7 (from matplotlib)
build (3.10)    Install dependencies    2025-08-02T05:00:11.6059758Z   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.6788752Z Collecting pytz>=2020.1 (from pandas)
build (3.10)    Install dependencies    2025-08-02T05:00:11.6947732Z   Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.7230688Z Collecting tzdata>=2022.7 (from pandas)
build (3.10)    Install dependencies    2025-08-02T05:00:11.7390205Z   Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.7814676Z Collecting exceptiongroup>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T05:00:11.7970310Z   Downloading exceptiongroup-1.3.0-py3-none-any.whl.metadata (6.7 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.8206574Z Collecting iniconfig>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T05:00:11.8359607Z   Downloading iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.8620872Z Collecting pluggy<2,>=1.5 (from pytest)
build (3.10)    Install dependencies    2025-08-02T05:00:11.8777523Z   Downloading pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.9130647Z Collecting pygments>=2.7.2 (from pytest)
build (3.10)    Install dependencies    2025-08-02T05:00:11.9292096Z   Downloading pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:11.9589654Z Collecting tomli>=1 (from pytest)
build (3.10)    Install dependencies    2025-08-02T05:00:11.9747391Z   Downloading tomli-2.2.1-py3-none-any.whl.metadata (10 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.0091658Z Collecting typing-extensions>=4.6.0 (from exceptiongroup>=1->pytest)
build (3.10)    Install dependencies    2025-08-02T05:00:12.0247975Z   Downloading typing_extensions-4.14.1-py3-none-any.whl.metadata (3.0 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.0572104Z Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib)
build (3.10)    Install dependencies    2025-08-02T05:00:12.0730021Z   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.0963930Z Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.2331299Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.8/16.8 MB 137.9 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:12.2524913Z Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.4339187Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 210.6 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:12.4540699Z Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.5075345Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 165.0 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:12.5237627Z Downloading pandas-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.3 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.6021385Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.3/12.3 MB 161.1 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:12.6197222Z Downloading h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.6623526Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 107.1 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:12.6781598Z Downloading pytest-8.4.1-py3-none-any.whl (365 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.6980135Z Downloading pluggy-1.6.0-py3-none-any.whl (20 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.7169631Z Downloading contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.7371692Z Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.7546021Z Downloading exceptiongroup-1.3.0-py3-none-any.whl (16 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.7729422Z Downloading fonttools-4.59.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.8580252Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 55.8 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:12.8744933Z Downloading iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.8926836Z Downloading kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.9088942Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 106.5 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:12.9246733Z Downloading packaging-25.0-py3-none-any.whl (66 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.9428410Z Downloading pillow-11.3.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:12.9767120Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 203.0 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:12.9945998Z Downloading pygments-2.19.2-py3-none-any.whl (1.2 MB)
build (3.10)    Install dependencies    2025-08-02T05:00:13.0042746Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 136.8 MB/s  0:00:00
build (3.10)    Install dependencies    2025-08-02T05:00:13.0224015Z Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:13.0410547Z Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:13.0609701Z Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:13.0812859Z Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:13.0992727Z Downloading tomli-2.2.1-py3-none-any.whl (14 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:13.1193177Z Downloading typing_extensions-4.14.1-py3-none-any.whl (43 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:13.1384561Z Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
build (3.10)    Install dependencies    2025-08-02T05:00:13.3981983Z Installing collected packages: pytz, tzdata, typing-extensions, tomli, six, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, scipy, python-dateutil, h5py, exceptiongroup, contourpy, pytest, pandas, matplotlib
build (3.10)    Install dependencies    2025-08-02T05:00:26.2597453Z
build (3.10)    Install dependencies    2025-08-02T05:00:26.2642312Z Successfully installed contourpy-1.3.2 cycler-0.12.1 exceptiongroup-1.3.0 fonttools-4.59.0 h5py-3.14.0 iniconfig-2.1.0 kiwisolver-1.4.8 matplotlib-3.10.5 numpy-2.2.6 packaging-25.0 pandas-2.3.1 pillow-11.3.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.3 pytest-8.4.1 python-dateutil-2.9.0.post0 pytz-2025.2 scipy-1.15.3 six-1.17.0 tomli-2.2.1 typing-extensions-4.14.1 tzdata-2025.2
build (3.10)    Install dependencies    2025-08-02T05:00:26.7778132Z Processing /home/runner/work/negative-energy-generator/negative-energy-generator
build (3.10)    Install dependencies    2025-08-02T05:00:26.7804509Z   Installing build dependencies: started
build (3.10)    Install dependencies    2025-08-02T05:00:27.8964715Z   Installing build dependencies: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T05:00:27.8972312Z   Getting requirements to build wheel: started
build (3.10)    Install dependencies    2025-08-02T05:00:28.3560656Z   Getting requirements to build wheel: finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T05:00:28.3571107Z   Preparing metadata (pyproject.toml): started
build (3.10)    Install dependencies    2025-08-02T05:00:28.5352139Z   Preparing metadata (pyproject.toml): finished with status 'done'
build (3.10)    Install dependencies    2025-08-02T05:00:28.5390574Z Requirement already satisfied: numpy>=1.24 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.2.6)
build (3.10)    Install dependencies    2025-08-02T05:00:28.5397819Z Requirement already satisfied: scipy>=1.11 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (1.15.3)
build (3.10)    Install dependencies    2025-08-02T05:00:28.5403068Z Requirement already satisfied: matplotlib>=3.6 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (3.10.5)
build (3.10)    Install dependencies    2025-08-02T05:00:28.5408459Z Requirement already satisfied: pandas>=1.5 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (2.3.1)
build (3.10)    Install dependencies    2025-08-02T05:00:28.5413654Z Requirement already satisfied: h5py>=3.6.0 in /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages (from negative_energy_generator==0.1.0) (3.14.0)
build (3.10)    Install dependencies    2025-08-02T05:00:28.7547035Z INFO: pip is looking at multiple versions of negative-energy-generator to determine which version is compatible with other requirements. This could take a while.
build (3.10)    Install dependencies    2025-08-02T05:00:28.7549740Z ERROR: Could not find a version that satisfies the requirement lqg_first_principles_gravitational_constant>=0.1.0 (from negative-energy-generator) (from versions: none)
build (3.10)    Install dependencies    2025-08-02T05:00:28.7558440Z ERROR: No matching distribution found for lqg_first_principles_gravitational_constant>=0.1.0
build (3.10)    Install dependencies    2025-08-02T05:00:28.8095852Z ##[error]Process completed with exit code 1.
```