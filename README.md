# Step size control for explicit relaxation Runge-Kutta methods preserving invariants

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10201246.svg)](https://doi.org/10.5281/zenodo.10201246)

This repository contains information and code to reproduce the results presented in the
article
```bibtex
@online{bleecke2023step,
  title={Step size control for explicit relaxation {R}unge-{K}utta
         methods preserving invariants},
  author={Bleecke, Sebastian and Ranocha, Hendrik},
  eprint={2311.14050},
  eprinttype={arxiv},
  eprintclass={math.NA},
  doi={10.48550/arXiv.2311.14050}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{bleecke2023stepRepro,
  title={Reproducibility repository for
         "{S}tep size control for explicit relaxation {R}unge-{K}utta
         methods preserving invariants"},
  author={Bleecke, Sebastian and Ranocha, Hendrik},
  year={2023},
  howpublished={\url{https://github.com/ranocha/2023_FSAL_relaxation}},
  doi={10.5281/zenodo.10201246}
}
```


## Abstract

Many time-dependent differential equations are equipped with invariants.
Preserving such invariants under discretization can be important, e.g.,
to improve the qualitative and quantitative properties of numerical
solutions. Recently, relaxation methods have been proposed as small
modifications of standard time integration schemes guaranteeing the
correct evolution of functionals of the solution. Here, we investigate
how to combine these relaxation techniques with efficient step size
control mechanisms based on local error estimates for explicit
Runge-Kutta methods. We demonstrate our results in several numerical
experiments including ordinary and partial differential equations.


## Numerical experiments

The numerical experiments presented in the paper use
[Julia](https://julialang.org/).

The subfolder `code` of this repository contains a `README.md` file with
instructions to reproduce the numerical experiments, including postprocessing.

The numerical experiments were carried out using Julia v1.9.3.


## Authors

- Sebastian Bleecke (Johannes Gutenberg University Mainz, Germany)
- [Hendrik Ranocha](https://ranocha.de) (Johannes Gutenberg University Mainz, Germany)


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
