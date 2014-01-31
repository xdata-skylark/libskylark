Sketching Layer
===============

.. currentmodule:: skylark.sketch

Introduction
------------

'Sketching' is the core algorithmic foundation on which Skylark is built, and is
able to deliver faster NLA kernels and ML algorithmic.  

Dimensionality reduction in NLA and ML is often based on building an oblivious 
subspace embedding (OSE). An OSE can be thought of as a data-independent random 
“sketching” matrix :math:`S \in R^{s\times n}` whose approximate isometry 
properties (with respect to a norm :math:`\|\cdot\|_p`) over a subspace (e.g., 
over the column space of a data input matrix Z, and regression target vector b) 
imply that,
 
.. math::
	\|S(Zx - b)\|\approx\|Zx-b\|
 
which in turn allows the regression coefficients, x,  to be optimized over a 
“sketched”  dataset of much smaller size without losing solution quality. 
Sketching matrices include Gaussian random matrices, structured random matrices 
which admit fast matrix multiplication via FFT-like operations, hashing-based 
transforms, among others.


Skylark's Sketching Layer
-------------------------

The purpose of the sketching layer is to provide optimized implementations 
of various sketching transforms, for various matrix arrangement in memory
(e.g. local matrices, distributed matrices, sparse matrices ...).
The majority of the sketching library is implemented in C++, but it is 
accessible in Python through :mod:`skylark.sketch`. Skylark also provides
`pure Python` implementations of the various transforms, which it will 
default in case the C++ layers of Skylark are not compiled. Some transforms
are currently implemented only in Python, but there are plans to 
implement them in C++ as well.

Here we describe access to the sketching layer only through the Python
interface (:mod:`skylark.sketch`). For description of access through 
C++ we refer the reader to the C++ documentation.

Dependencies
------------

Skylark uses external libraries to represent distributed matrices. For
dense distributed matrices it uses `Elemental <http://libelemental.org/>`_.
Currently it uses the c-types interface of Elemental, so be sure install
that as well. For sparse distributed matrices it uses
`ComboBLAS <http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/>`_ interfaced
through `KDT <http://kdt.sourceforge.net/wiki/index.php/Main_Page>`_.

The lower layers use MPI so it is advisable an MPI interface to Python be
installed. One option is to use `mpi4py <http://mpi4py.scipy.org/>`_.

Initialization and Finalization
-------------------------------

Skylark is automatically initialized with a random seed when you import
sketch. However, you can reinitialize it to a specific seed by calling
initialize. While not required, you can finalize the library using
finalize. However, note that that will not cause allocated objects
(e.g. sketch transforms) to be freed. They are freed by the garbage collector
when detected as garbage (no references).

.. autofunction:: initialize

.. autofunction:: finalize

SketchTransform
---------------

.. autoclass:: _SketchTransform
                :members:

                .. automethod:: _SketchTransform.__mul__

                .. automethod:: _SketchTransform.__div__

Supported Sketches
------------------

.. autoclass:: JLT
.. autoclass:: FJLT
.. autoclass:: SJLT
.. autoclass:: CT
.. autoclass:: CWT
.. autoclass:: MMT
.. autoclass:: WZT
.. autoclass:: GaussianRFT
.. autoclass:: LaplacianRFT
.. autoclass:: URST

Example
-------
.. literalinclude:: ../../skylark/examples/example_sketch.py



