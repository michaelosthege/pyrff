[![PyPI version](https://badge.fury.io/py/pyrff.svg)](https://badge.fury.io/py/pyrff)
[![pipeline](https://github.com/michaelosthege/pyrff/workflows/pipeline/badge.svg)](https://github.com/michaelosthege/pyrff/actions)
[![coverage](https://codecov.io/gh/michaelosthege/pyrff/branch/master/graph/badge.svg)](https://codecov.io/gh/michaelosthege/pyrff)


# `pyrff`: Approximating Gaussian Process samples with Random Fourier Features
This project is a Python implementation of random fourier feature (RFF) approximations [1].

It is heavily inspired by the implementations from [2, 3] and generalizes the implementation to work with GP hyperparameters obtained from any GP library.

Examples are given as Jupyter notebooks for GPs fitted with [PyMC3](https://github.com/pymc-devs/pymc3):
+ [Example_RFF_PyMC3_1D](https://github.com/michaelosthege/pyrff/blob/master/notebooks/Example_RFF_1D_PyMC3.ipynb)
+ [Example_RFF_PyMC3_2D](https://github.com/michaelosthege/pyrff/blob/master/notebooks/Example_RFF_2D_PyMC3.ipynb)

# References
1. Hernández-Lobato, 2014 [paper](https://arxiv.org/abs/1511.05467), [code](https://bitbucket.org/jmh233/codepesnips2014/src/master/sourceFiles/sampleMinimum.m)
2. PES implementation in Cornell-MOE [code](https://github.com/wujian16/Cornell-MOE/blob/master/pes/PES/sample_minimum.py)
3. Bradford, 2018 [paper](https://link.springer.com/article/10.1007/s10898-018-0609-2/), [code](https://github.com/Eric-Bradford/TS-EMO/blob/master/TSEMO_V3.m#L495)
