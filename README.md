[![PyPI version](https://badge.fury.io/py/pyrff.svg)](https://badge.fury.io/py/pyrff)
[![pipeline](https://github.com/michaelosthege/pyrff/workflows/pipeline/badge.svg)](https://github.com/michaelosthege/pyrff/actions)
[![coverage](https://codecov.io/gh/michaelosthege/pyrff/branch/master/graph/badge.svg)](https://codecov.io/gh/michaelosthege/pyrff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3831380.svg)](https://doi.org/10.5281/zenodo.3831380)


# `pyrff`: Approximating Gaussian Process samples with Random Fourier Features
This project is a Python implementation of random fourier feature (RFF) approximations [1].

It is heavily inspired by the implementations from [2, 3] and generalizes the implementation to work with GP hyperparameters obtained from any GP library.

Examples are given as Jupyter notebooks for GPs fitted with [PyMC3](https://github.com/pymc-devs/pymc3) and [scikit-learn](https://scikit-learn.org):
+ [Example_RFF_PyMC3_1D](https://github.com/michaelosthege/pyrff/blob/master/notebooks/Example_RFF_1D_PyMC3.ipynb)
+ [Example_RFF_PyMC3_2D](https://github.com/michaelosthege/pyrff/blob/master/notebooks/Example_RFF_2D_PyMC3.ipynb)
+ [Thompson sampling, 1D with sklearn](https://github.com/michaelosthege/pyrff/blob/master/notebooks/TS_1D_sklearn.ipynb)

# Installation
`pyrff` is released on [PyPI](https://pypi.org/project/pyrff/):
```
pip install pyrff
```
# Usage and Citing
`pyrff` is licensed under the [GNU Affero General Public License v3.0](https://github.com/michaelosthege/pyrff/blob/master/LICENSE).

Head over to Zenodo to [generate a BibTeX citation](https://doi.org/10.5281/zenodo.3831380) for the latest release.

# References
1. Hernández-Lobato, 2014 [paper](https://arxiv.org/abs/1511.05467), [code](https://bitbucket.org/jmh233/codepesnips2014/src/ac843ba992ca1879190a472ac20c83a447e4e2c0/sourceFiles/sampleMinimum.m#lines-1)
2. PES implementation in Cornell-MOE [code](https://github.com/wujian16/Cornell-MOE/blob/df299d1be882d2af9796d7a68b3f9505cac7a53e/pes/PES/sample_minimum.py#L23)
3. Bradford, 2018 [paper](https://link.springer.com/article/10.1007/s10898-018-0609-2/), [code](https://github.com/Eric-Bradford/TS-EMO/blob/87151d94081db1d0f128a788ebdb789d2891ee9a/TSEMO_V4.m#L501)
