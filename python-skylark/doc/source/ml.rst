Machine Learning and Statistical Data Analysis
================================================

.. currentmodule:: skylark.ml.nonlinear

Nonlinear Regression
-------------------------------------------------------

The examples below are currently not distributed, but they do illustrate sketching techniques for nonlinear modeling. 
A distributed version of Random Fourier Transform is available in Python (see C-API Accessing Lower Layers section). 

Regularized Least Squares with Gaussian Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: skylark.ml.nonlinear.rls
        :members:


Regularized Least Squares with Sketching Primitives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: skylark.ml.nonlinear.sketchrls
	:members:

Examples
^^^^^^^^^
.. literalinclude:: ../../skylark/examples/example_krr.py

Matrix Completion
------------------
To be done. Dependency on randomized low-rank approximations (SVD) implementation in the NLA layer.

Statistical Dimensionality Reduction
--------------------------------------

Principal Component Analysis and Canonical Component Analysis. Dependency on randomized low-rank approximations (SVD) implementation in the NLA layer.
