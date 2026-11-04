# PairPulse

PairPulse is a Python package for fermion pair-production spectra and one-dimensional densities in prescribed homogeneous electric fields. Version 0.2.2 includes Dirac and quantum-kinetic mode solvers, a stable analytic Sauter benchmark, and separate studies of integration, endpoint and momentum-integration convergence.

The [manuscript](manuscript.pdf) develops the theoretical conventions and derivations, presents the numerical results, and explains their scope. Its [LaTeX source](manuscript.tex) and figure PDFs are included.

## Code and data

- [Complete manuscript and computational package](PairPulse_CPC_package_v0.2.2.zip) — article PDF and LaTeX, figures, software, numerical data, user guide and equation-source verification.
- [Computational materials](Computational_materials/) — source, tests, calculation scripts and demonstration notebook.
- [User guide](USER_GUIDE.txt) — installation, numerical controls and reproduction commands.
- [Numerical data](data/) — the records underlying the manuscript figures and validation tables.
- [Source archive](PairPulse_source.zip) — source and user guide.
- [Software and data supplement](Supplementary_code.zip) — source, guide and supporting numerical data.

The source code is distributed under the [MIT license](LICENSE).

## Reproducing the results

Create and activate a Python virtual environment, then run:

```sh
cd Computational_materials
python -m pip install -r requirements-validation.txt .
python -m unittest discover -s tests -v
python scripts/redraw_figures.py --data ../data/reproduced --output redrawn
python scripts/summarize_validation.py --data ../data/validation
```

The source README and user guide give commands for repeating the integrations, high-precision comparisons and notebook examples. Use fresh output directories. The requirements files specify the reproduction environment; the reported calculations used Python 3.13.9. [Execution notes](Computational_materials/EXECUTION_NOTES.txt) state the installation and notebook qualifications for the machine used to validate this version.

The reported assessment includes 2,228 completed mode evaluations, 21 automated tests and 299 high-precision analytic comparisons. All satisfy the stated criteria. Accuracy beyond the tested cases requires independent convergence checks; norm conservation or agreement between formulations alone is insufficient.

The repository includes checksum manifests for integrity checks. The two supplementary workflow diagrams are retained as supporting software illustrations; the article contains Figures 1–4.
