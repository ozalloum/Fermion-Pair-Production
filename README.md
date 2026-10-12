# Fermion pair production in pulsed electric fields: cross-validation of Dirac and quantum-kinetic calculations

**Othman H. Y. Zalloum**  
Department of Applied Mathematics and Physics, Palestine Polytechnic University, Hebron, Palestine

## Manuscript and code

- [Manuscript PDF](Physica_Scripta.pdf)
- [LaTeX source](Physica_Scripta.tex)
- [Computational source and reproduction materials](Supplementary_code.zip)
- [File checksums](CHECKSUMS.sha256)

The manuscript uses four scientific figures, supplied as vector PDFs. The bibliography is embedded in the LaTeX source. Two additional technical diagrams describe the reproducibility structure and computational workflow.

## Compile

Keep `Physica_Scripta.tex`, `iopjournal.cls` and the four scientific figure PDFs in the same directory. Run:

```sh
pdflatex Physica_Scripta.tex
pdflatex Physica_Scripta.tex
```

Alternatively, run `tectonic Physica_Scripta.tex`.

## Reproduce the calculations

Extract `Supplementary_code.zip` and follow `Computational_materials/README.md`. The archive contains the paired Dirac and quantum-kinetic solvers, analytic Sauter benchmark, tests, reproduction and validation scripts, notebook, execution notes and matching manuscript files. The scripts generate the numerical datasets and diagnostics.

`PairPulse_source.zip` is an identical copy retained for compatibility with the existing download link.

The model treats a prescribed, spatially homogeneous electric field in one spatial dimension. Numerical conventions, validation criteria and physical limitations are specified in the manuscript.

The computational source carries its MIT license inside the source archive. The IOP journal class retains its upstream license notice.
