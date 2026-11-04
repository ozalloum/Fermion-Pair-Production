# Numerical data

- `validation/`: all 2,228 mode results, calculation parameters, computing environment and summary tables. `protocol.json` specifies the expected calculations. `summary.json` records their acceptance checks.
- `reproduced/`: spectra, densities, endpoint refinements and the momentum–window error map underlying the figures, with calculation metadata.
- `analytic/`: 299 comparisons of the analytic Sauter expression with independent 100-digit evaluations.
- `derivations/`: numerical checks of the Dirac-to-kinetic transformation, Sauter connection coefficients, scalar mode equation and limiting cases.

The records are the results used in the manuscript. Run `scripts/redraw_figures.py` to reconstruct plots and `scripts/summarize_validation.py` to recompute validation tables, using the paths in the repository user guide. Repeating the integrations is optional for reconstructing these plots and tables.
