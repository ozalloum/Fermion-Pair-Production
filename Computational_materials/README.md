# PairPulse computational materials

PairPulse 0.2.2 calculates fermion occupations and one-dimensional densities in prescribed, homogeneous, fixed-axis electric fields. It implements two-component Dirac evolution, the equivalent quantum-kinetic equations, and an analytic Sauter benchmark. The software is distributed under the MIT license.

## Installation and examples

Create and activate a fresh virtual environment. From this directory, run:

```sh
python -m pip install -r requirements-validation.txt .
python -m unittest discover -s tests -v
python scripts/reproduce.py --output regenerated --points 41
python scripts/make_reproducibility_figures.py --output regenerated --points 41
python scripts/redraw_figures.py --data regenerated --output redrawn
python scripts/validate_study.py --output validation --workers 6
python scripts/summarize_validation.py --data validation
python scripts/check_analytic_precision.py --output analytic-check
python scripts/check_derivations.py --output derivation-check
python scripts/execute_notebook.py --output notebook-output
python scripts/verify_checksums.py
```

Use fresh output directories. Core dependencies in `pyproject.toml` are lower bounds and require Python 3.10 or later. The requirements files specify exact versions for reproducing the calculations and require Python 3.11 or later. The reported calculations used Python 3.13.9. Select an installed notebook kernel with `--kernel-name` if needed. Installation and execution qualifications are documented in `EXECUTION_NOTES.txt`.

## Numerical conventions and controls

Occupations are per channel. Density includes the measure dp/(2*pi), with no additional spin multiplicity. Finite-time occupations use the instantaneous energy basis. The potential and field must satisfy E=-dA/dt.

Both mode solvers accept an absolute `max_step`. Built-in pulses also limit steps to one eighth of the shortest active duration. A custom profile should provide `resolution_scale()` or an explicit step limit that resolves its narrowest relevant feature. Refine time endpoints, tolerances, step limits, momentum spacing and momentum range separately. Conservation of the state norm or kinetic invariant alone does not establish convergence.

`sauter_log_occupation` retains a logarithmic result when the ordinary occupation underflows. The analytic evaluation uses rationalized energy gaps and stable logarithmic factors. `integrated_density` rejects nonfinite data and duplicate momenta; `check_bounds=True` also checks the occupation bounds.

## Validation and supporting material

The 2,228-evaluation validation checks complete calculation coverage, finite values, physical bounds, and the numerical criteria stated in the manuscript. The study and summarizer report failure with a nonzero exit status. The supplied tests include malformed and incomplete input cases. The high-precision comparison covers 299 analytic values.

`REPRODUCIBILITY.txt` is used by the figure scripts and must remain with the source. See the accompanying `USER_GUIDE.txt` for instructions. The manuscript, numerical data and source are available at https://github.com/ozalloum/Fermion-Pair-Production. The source checksum manifest can be checked with `scripts/verify_checksums.py`.
