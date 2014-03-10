Introduction
==============

Goals
------

Matrix algorithms are the foundation for most methods in data analysis, scientific computing,
and engineering applications, and Numerical Linear Algebra (NLA) kernels are the key enabling technology for implementing
matrix algorithms on different hardware platforms. Without highly efficient and scalable NLA kernels, the challenge of big data cannot be met.

The goal of Skylark is to implement sketching-based NLA kernels for distributed computing platforms. Sketching reduces dimensionality
through randomization, and is one of only a few paradigms that have the
potential to deliver a true boost in performance in the presence of massive data, for an extensive
class of applications. Sketching methods have been developed, for over ten years now, mainly by the theoretical
computer science community. Sketching techniques are simple to implement, readily
parallelizable, and often robust to matrix sparsity patterns and other properties. Most of all,
they have fast asymptotic time and space complexity. Applications in population genetics, circuit testing, social network analysis, text classification, pattern matching and other domains, have demonstrated
the promise of sketching methods in practice.

Skylark is a software stack, being designed to be composed of distributed sketching-based NLA kernels, for fundamental
statistical and machine learning algorithms.

The NLA kernels consist of matrix multiplication, regression, and low-rank approximation
algorithms, as well as acceleration methods for matrix completion, missing data imputation,
nonlinear modeling (i.e., kernelized regularized least squares and support vector machines), and
statistical dimensionality reduction techniques. The implementations are guided by error
and sensitivity analyses, to give users a high confidence assurance for the accuracy of the
output of the software stack. Sketching techniques, when used in a straight-forward fashion and on their own, can deliver
relatively crude, i.e. low accuracy results, since the work to achieve an error of :math:`\epsilon` can be
proportional to :math:`1/\epsilon^2`. Such crude results can be relevant in themselves: in ranking applications
where only an ordinal rank is required rather than the exact value (PageRank, recommender
systems); in situations where data noise dominates algorithmic error; or where the tradeoff
between speed and accuracy favors “quick and dirty” results that can be readily refined. Also,
in statistical and Machine Learning (ML) settings, approximate solutions can be regarded as
a formal mechanism to avoid overfitting, that is, a form of regularization.

However, sketching techniques can also be used to precondition standard methods, so that
they accelerate computations but do not suffer from loss of accuracy. Skylark will also provide
sketching techniques as preconditioners, or accelerators, to a large variety of NLA computations.

Another central goal of Skylark is to serve as a workbench with which to determine the best sketching
techniques in practice from among the many that have been proposed in the theoretical
literature. This is challenging, because many sketching techniques come with error bounds that
are asymptotic only, and contain unknown constants. Furthermore, a combination of criteria
needs to considered: error due to randomization, numerical robustness and accuracy, matrix
access patterns (rows, columns, submatrices), arithmetic operation count and communication
requirements.

Sketching and Sampling Methods
-------------------------------

A wealth of techniques have been proposed in recent years for matrix problems, based
on random sampling of the input matrix via rows, columns, individual entries, or random linear
combinations of rows and columns. For brevity all these techniques will be collectively called
here sketching. Sketching yields “compressed” or “down-sized” versions of the input matrix,
that retain enough information to either yield approximate solutions, or to speed up methods
for finding high precision solutions, as discussed above.

In many situations we can represent the application of a sketching method to an n × d
matrix A as a product SA, where S has s x n rows. In general, accuracy or other sketch
properties vary with s. For some sketching approaches the matrix S is quite sparse, while in
others it is highly structured, so that computing SA involves an FFT or similar fast transform
per row of SA, and not much more. When each row of S has a single entry equal to one and all others zero, then SA represents
a subset of the rows of A. That is, S is a sampling matrix. Samples SA have very desirable
properties: under appropriate conditions, SA represents a collection of representative, “prototypical” rows of A, which is useful in certain data analytics applications; and if A is sparse,
then SA tends to be so as well, at least heuristically.

By reducing dimensionality while still preserving structure, sketching can be used for approximately solving a whole host of NLA problems,
with substantial speedups in computation time. Other benefits include reduced memory requirements, communication costs and IO overheads.
This, in turn, offers the potential to significantly improve the computational efficiency of solving higher-level statistical modeling and machine learning tasks,
by essentially replacing calls to traditional NLA kernels with their sketching counterparts.

More Technical Details
^^^^^^^^^^^^^^^^^^^^^^^

For more details, we refer the user to our whitepaper and references therein.

Implementation Plan and Current Status
---------------------------------------

Skylark builds on distributed Numerical Linear Algebra libraries for Dense and Sparse Matrices, with an MPI backend.
Organizationally, it comprises of three layers. The core is a library of sketching transforms whose main functionality
is to enable a variety of input matrix types to be sketched, using transforms specialized for various NLA kernels such as
least squares (:math:`l_2`) regression, robust regression (:math:`l_1`) and low-rank matrix approximations.
These kernels are being implemented in the Numerical Linear Algebra (NLA) layer.

Skylark uses external libraries to represent distributed input matrices. For
dense distributed matrices it uses `Elemental <http://libelemental.org/>`_.
Currently it uses the c-types interface of Elemental. For sparse distributed matrices it uses
`ComboBLAS <http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/>`_.

The accelerated NLA kernels are then used to accelerate higher level machine learning algorithms, e.g., kernel-based
nonlinear regression, matrix completion and statistical dimensionality reduction techniques such as Principal Component Analysis.
We also provide Python bindings to Skylark lower layers to enable rapid prototyping of data analysis algorithms, and exploration of the
sketching design space for optimal performance in a given domain.  Input matrices can also be local numpy or scipy arrays
for single node execution of Skylark.  We use Python bindings for Elemental and Combinatorial BLAS (through `KDT <http://kdt.sourceforge.net/wiki/index.php/Main_Page>`_).

We also plan to support a streaming interface for out-of-core sketching and matrix computations.

Currently, a number of distributed sketching transforms are supported in Skylark. The first seven transforms below are implemented, and form the basis of faster
:math:`l_2, l_1` regression and randomized SVD.

    * JLT - Johnson–Lindenstrauss Transform
    * FJLT - Fast Johnson–Lindenstrauss Transform
    * CT - Cauchy Transform
    * MMT - Meng-Mahoney Transform
    * CWT - Clarkson-Woodruff Transform
    * WZT - Woodruff-Zhang Transform
    * SRFUT - Subsampled Randomized Fast Unitary Transform

The following transforms are to be implemented soon:
    * FJLTp - "FJLT plus", that is SFD as usual but with S an hashing matrix (CWT, MMT, ...)/OSNAP instead of just sampling.
    * FCT - Fast Cauchy Transform
    * OSNAP - Oblivious Sparse Norm-Approximating Projection ( http://http://arxiv.org/abs/1211.1002)
    * PPT - Pahm-Pagh Transform (fast sketching of polynomial kernel/tensor sketch,  http://www.itu.dk/people/pagh/papers/tensorsketch.pdf)
    * RKP - Random Kronecker Product ( http://nuit-blanche.blogspot.fr/2008/05/cs-new-cs-measurement-matrices-some.html)
    * RBT - Random Butterfly Transform ( http://hal.inria.fr/docs/00/72/40/59/PDF/RR-8043.pdf)

.. image:: skylark_status.png

Notes:
	* Dense distributed matrices can have 1D or 2D decomposition. Currently we support 1D distribution, which fits tall-and-skinny matrices
(one of the primary use-cases). We plan to extend to 2D distribution as well.
	* Currently, Python interface uses ctypes. We may switch to SWIG to be consistent with Elemental and CVX.


Installation
--------------

We provide a separate install guide in pdf.


License and Copyright
----------------------

Copyright IBM Corporation, 2012-2014 ::

This program and the accompanying materials are made available under the terms of the Apache License, Version 2.0
which is available at `<http://www.apache.org/licenses/LICENSE-2.0>`_



