# Fermion pair production in pulsed electric fields: cross-validation of Dirac and quantum-kinetic calculations

**Othman H. Y. Zalloum**  
Department of Applied Mathematics and Physics, Palestine Polytechnic University, Hebron, Palestine

Manuscript prepared for submission to *Physica Scripta*.

## Manuscript

- [Read the manuscript](Physica_Scripta.pdf)
- [LaTeX source](Physica_Scripta.tex)
- [Computational source and reproduction materials](PairPulse_source.zip)
- [File checksums](CHECKSUMS.sha256)

All repository files are stored in one flat directory. The manuscript uses four scientific figures and two appendix diagrams, supplied as vector PDFs. The bibliography is embedded in the LaTeX source.

## Compile

Keep the source, figures, `iopjournal.cls` and `orcid.pdf` in the same directory. Run:

```sh
pdflatex Physica_Scripta.tex
pdflatex Physica_Scripta.tex
```

Alternatively, run `tectonic Physica_Scripta.tex`.

## Reproduce the calculations

Extract `PairPulse_source.zip` and follow its README. It contains the paired Dirac and quantum-kinetic solvers, analytic Sauter benchmark, tests, reproduction scripts, validation scripts and notebook. Numerical datasets can be generated with the included scripts.

The model treats a prescribed, spatially homogeneous electric field in one spatial dimension. Numerical conventions, validation criteria and physical limitations are specified in the manuscript.

The computational source carries its MIT license inside the source archive. The IOP journal class retains its upstream license notice.
