# carmapy_light

This repository contains the Python code for the outlier detection and PIPs calualtion algorithm of CARMA, originally implemented in R ([CARMA on GitHub](https://github.com/ZikunY/CARMA)).

The full repository for the reimplementation of CARMA in Python can be found [here](https://github.com/hlnicholls/carmapy/tree/0.1.0).

This is a simplified version of CARMA with the following features:

1. It uses only Spike-slab effect size priors and Poisson model priors.
2. C++ is re-implemented in Python.
3. The way of storing the configuration list is changed. It uses a string with the list of indexes for causal SNPs instead of a sparse matrix.
4. Fixed bugs in PIP calculation.
5. No credible models.
6. No credible sets, only PIPs.
7. No functional annotations.
8. Removed unnecessary parameters.


**Use:**

No installation is required; it only has Python code (no C++ code).

See tests for examples.

**Notes:**

If you want to use it for outlier detection, you can specify the number of iterations to 1 (`all_iter=1`); it will be enough.

There are issues when using an LD matrix with a determinant equal to 0. This problem is inherited from the original version and requires investigation.
