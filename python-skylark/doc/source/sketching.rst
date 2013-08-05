Sketching Transforms
=====================

In many settings, what makes sketching-based acceleration of NLA kernels such as :math:`l_p` regression possible is the existence of a suitable oblivious
subspace embedding (OSE). An OSE can be thought of as a data-independent random “sketching”
matrix :math:`S \in R^{s\times n}` whose approximate isometry properties (with respect to a norm :math:`\|\cdot\|_p`) over a subspace (e.g., over the column
space of a data input matrix Z, and regression target vector b) imply that,
 
.. math::
	\|S(Zx - b)\|\approx\|Zx-b\|
 
which in turn allows the regression coefficients, x,  to be optimized over a “sketched” dataset of much smaller size without losing
solution quality. Sketching matrices include Gaussian random matrices, structured random matrices which admit fast matrix multiplication via FFT-like operations, 
hashing-based transforms, among others.

Below, we give examples of how various sketching transforms on numpy matrices show the norm preservation property. 
We demonstrate a non-distributed usage implemented using numpy arrays. In C-api section on accessing lower layers we give an example of distributed JLT transform operating on Elemental matrices.


Johnson-Lindenstrauss Transform (JLT)
-------------------------------------------------------

.. autoclass:: skylark.sketch.JLT
	:members:

Faster Johnson-Lindenstrauss (FJLT) 
------------------------------------

.. autoclass:: skylark.sketch.FJLT
	:members:

Sparse Sketching Transforms
-----------------------------------

.. autoclass:: skylark.sketch.SparseJL
	:members:
	
.. currentmodule:: skylark.cskylark


