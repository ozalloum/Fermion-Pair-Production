# Fermion pair production in pulsed electric fields

Cross-validation of Dirac and quantum-kinetic calculations.

**Othman H. Y. Zalloum**  
Department of Applied Mathematics and Physics, Palestine Polytechnic University, Hebron, Palestine

## Manuscript and code

- [Manuscript PDF](manuscript.pdf)
- [LaTeX source](manuscript.tex)
- [Source code and reproduction materials](Supplementary_code.zip)
- [File checksums](CHECKSUMS.sha256)

The manuscript uses the standard LaTeX article class and four vector figures. References are embedded in the source. Two additional technical diagrams describe the reproducibility structure and computational workflow.

## Compile

Keep `manuscript.tex` and the four scientific figure PDFs in the same directory, then run:

```sh
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Alternatively, run `tectonic manuscript.tex`.

## Reproduce the calculations

Extract `Supplementary_code.zip` and follow `Computational_materials/README.md`. It contains the Dirac and quantum-kinetic solvers, analytic Sauter benchmark, tests, reproduction and validation scripts, notebook, execution notes and manuscript files. The scripts generate the numerical datasets and diagnostics.

`PairPulse_source.zip` contains the same materials.

The model treats a prescribed, spatially homogeneous electric field in one spatial dimension. The manuscript specifies the numerical conventions, validation criteria and physical limitations. The source code is distributed under the MIT license included in the archive.
