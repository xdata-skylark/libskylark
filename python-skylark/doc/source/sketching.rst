Sketching Layer
*****************

.. currentmodule:: skylark.sketch

Introduction
=============

'Sketching' is the core algorithmic foundation on which Skylark is built, and is
able to deliver faster NLA kernels and ML algorithmic.

Dimensionality reduction in NLA and ML is often based on building an oblivious
subspace embedding (OSE). An OSE can be thought of as a data-independent random
“sketching” matrix :math:`S \in R^{s\times n}` whose approximate isometry
properties (with respect to a norm :math:`\|\cdot\|_p`) over a subspace (e.g.,
over the column space of a data input matrix A, and regression target vector b)
imply that,

.. math::
	\|S(A x - b)\|\approx\|A x-b\|

which in turn allows the regression coefficients, :math:`x`, to be optimized
over a “sketched” dataset - namely :math:`S*A` and :math:`S*b` - of much smaller
size without losing solution quality significantly.
Sketching matrices include Gaussian random matrices, structured random matrices
which admit fast matrix multiplication via FFT-like operations, hashing-based
transforms, among others.


Overview of High-performance Distributed Sketching Implementation
==================================================================

Sketching a matrix A typically amounts to multiplying it by a random matrix
:math:`S`, i.e., :math:`A * S` for compressing the size of its rows
(row-wise sketching) or :math:`S * A` for compressing the size of its
columns (column-wise sketching).

In a distributed setting, this matrix-matrix multiplication primitive
(sketching GEMM operation) is special in the sense that any part of matrix
:math:`S` can be constructed without communication.
In addition, depending on the relative sizes of :math:`A`, :math:`S` and the
sketched matrix (output), we can organize the distributed GEMM so that no part
of the largest-size matrix is communicated (SUMMA approach), thus resulting in
communication savings.
Further optimizing, we can perform local computations over blocks of :math:`A`,
:math:`S`, and also assume transposed views of the operands for memory and
cache use efficiency.

In particular when both :math:`A` and :math:`S` are distributed dense matrices
we represent them as Elemental matrices and support sketching over a rich set
of combinations of vector and matrix-oriented data distributions: in vector
distributions different processes own complete rows of columns of the matrix
that are p apart (p is the number of processes) while in matrix distributions
each process owns a strided view of the matrix with strides along rows and
columns being equal to the dimensions of the process grid.

In our sketching GEMM, local entries of the random matrix S are computed
independently by indexing into a global stream of random values provided by a
counter-based Parallel random number generator (supplied by
`Random123 library <http://www.deshawresearch.com/resources_random123.html>`_).
No entries of :math:`S` are communicated since they can be locally generated
instead.
:math:`A` can be squarish (aka "matrix") or tall-and-thin or short-and-fat
(aka "panel").
In multiplying with :math:`S`, matrix-panel, panel-matrix, inner-panel-panel
or outer-panel-panel products may arise.
We provide separate implementations for these cases organized around the
principle of communication-avoidance for the largest of the matrix terms
involved in the GEMM, for each of the input/output matrix-data distribution
combinations.
The user can optionally set the relative sizes that differentiate between these
cases.
Local :math:`S` entries can be incrementally realized in a distribution format that
best matches the matrix indices of the local GEMM operation that follows
it.
Resulting :math:`S` blocks typically traverse the smallest of matrix sizes in
increments that can optionally be specified by the user.
This has the extra benefit of minimizing the communication volume of a
collective operation that generally follows this local GEMM - essentially to
compensate for the stride-indexed matrix entries in the factors.

As an example we provide a *pseudocode* snippet (in Python syntax) that describes
the rowwise sketching of a squarish input matrix :math:`A`, initially distributed
across the process grid in `[MC, MR]` format (please refer
`here <http://libelemental.org/documentation/0.83/core/dist_matrix/DM.html>`_ for a
comprehensive documentation of distribution formats, here appearing in brackets):
:math:`S` is first realized (its random entries are actually computed in the desired
distribution format - in embarrassingly parallel mode) and then the local parts of
:math:`A` and :math:`S` are multiplied together. Finally collective communications within
subsets of the process grid take place to produce the resulting sketched matrix
(`C[MC, MR]`). The corresponding C++ code (allowing also for incremental realization of
:math:`S`) can be found in
:file:`libskylark/sketch/dense_transform_Elemental_mc_mr.hpp`.

.. code-block:: python

    def matrix_panel(A[MC, MR], S):
        S[MR, STAR]       = realize(S)
        C_hat[MC, STAR]   = local_gemm(A[MC, MR], S[MR, STAR])
        C[MC, MR]         = reduce_scatter_within_process_rows(C_hat[MC, STAR))
        return C[MC, MR]

Quite interestingly and depending on the distribution format of the input and sketched
matrices, sketching can be *communication free*. The following snippet illustrates this
remark when both input and sketched matrices are in `[VC, STAR]` or `[VR, STAR]`
distribution formats - same scenario as before, rowwise sketching of a squarish input matrix:

.. code-block:: python

    def matrix_panel(A[VC/VR, STAR], S):
        S[STAR, STAR]     = realize(S)
        C[VC/VR, STAR]    = local_gemm(A[VC/VR, STAR], S[STAR, STAR])
        return C[VC/VR, STAR]


Sparse matrices :math:`A` are currently represented as
:abbr:`CombBLAS (Combinatorial BLAS)` matrices.
As for dense sketch matrices, any part of the sparse sketch matrix can be
realized without communication. Since the sketch matrix is sparse, we only
require a "sparse" realization of the sketch matrix :math:`S` and the sketching
GEMM can be computed on the random stream directly.
Similar to the SUMMA approach for dense matrices we select what will be
communicated depending on input and output dimensions.

It is possible to sketch from a sparse matrix to a dense (and vice versa).
The only restriction when using CombBLAS is that total number of processors
has to be a square number.



Libskylark's Sketching Layer
==============================

The purpose of the sketching layer is to provide optimized implementations
of various sketching transforms and for various matrix arrangement in memory
(e.g. local matrices, distributed matrices, sparse matrices ...).
The majority of the sketching library is implemented in C++, but it is
accessible in Python through :mod:`skylark.sketch`.

Sketching Transforms
--------------------

The following table lists the sketching transforms currently provided by LibSkylark.
These transforms are appropriate for specific downstream tasks, e.g.
:math:`l2`-regression, :math:`l1`-regression, or kernel methods.

The implementations are provided under *libskylark/sketch*.

============== ============================================= =================
Abbreviation   Name                                          Reference
============== ============================================= =================
JLT            Johnson-Lindenstrauss Transform               Johnson and Lindenstrauss, 1984
FJLT           Fast Johnson-Lindenstrauss Transform          `Ailon and Chazelle, 2009 <http://www.cs.princeton.edu/~chazelle/pubs/FJLT-sicomp09.pdf>`_
CT             Cauchy Transform                              `Sohler and Woodruff, 2011 <http://researcher.ibm.com/files/us-dpwoodru/sw.pdf>`_
MMT            Meng-Mahoney Transform                        `Meng and Mahoney, 2013 <http://arxiv.org/abs/1210.3135>`_
CWT            Clarkson-Woodruff Transform                   `Clarkson and Woodruff, 2013 <http://arxiv.org/abs/1207.6365>`_
WZT            Woodruff-Zhang Transform                      `Woodruff and Zhang, 2013 <http://homes.soic.indiana.edu/qzhangcs/papers/subspace-full.pdf>`_
PPT            Pahm-Pagh Transform                           `Pahm and Pagh, 2013 <http://www.itu.dk/people/ndap/TensorSketch.pdf>`_
ESRLT          Random Laplace Transform (Exp-semigroup)      `Yang et al, 2014 <http://vikas.sindhwani.org/RandomLaplace.pdf>`_
LRFT           Laplacian Random Fourier Transform            `Rahimi and Recht, 2007 <http://www.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf>`_
GRFT           Gaussian Random Fourier Transform             `Rahimi and Recht, 2007 <http://www.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf>`_
FGRFT          Fast Gaussian Random Fourier Transform        `Le, Sarlos and Smola, 2013 <http://jmlr.org/proceedings/papers/v28/le13.html>`_
============== ============================================= =================

Sketching Layer in C++
------------------------

The above sketch transforms can be instantiated for various combinations of
distributed and local, sparse and dense input matrices and output sketches.
The following table lists the input-output combinations currently implemented
in the C++ sketching layer.

In the table below, *LocalDense* refers to Elemental `sequential matrix
<http://libelemental.org/documentation/0.83/core/matrix.html>`_ type,
while STAR/STAR, VR/STAR, VC/STAR, STAR/VR, STAR/VC, MC/MR refer to
specializations of `Elemental distributed matrices
<http://libelemental.org/documentation/0.83/core/dist_matrix/DM.html>`_.
Each specialization involves choosing a sensical pairing of distributions for
the rows and columns of the matrix:
* CIRC : Only give the data to a single process
* STAR : Give the data to every process
* MC : Distribute round-robin within each column of the 2D process grid (M atrix C olumn)
* MR : Distribute round-robin within each row of the 2D process grid (M atrix R ow)
* VC : Distribute round-robin within a column-major ordering of the entire 2D process grid (V ector C olumn)
* VR : Distribute round-robin within a row-major ordering of the entire 2D process grid (V ector R ow)

*LocalSparse* refers to a libskylark-provided class for representing local
sparse matrices, while *DistSparse* refers to CombBLAS sparse matrices.

.. image:: files/sketch_transf_in_out_cpp.png
    :width: 750 px
    :align: center

.. raw:: html

    <i>Schematic views of input and output types for various sketch transforms.
    The <font color="#154685">blue</font> color marks sparse matrix or
    transforms, <font color="#ee9428">orange</font> is used for dense matrix or
    transforms.</i>

.. note:: In the near future the local dense matrix will be replaced by
    CIRC/CIRC and STAR/STAR matrices.


.. _sketching-transforms-label:

Sketching Transforms
^^^^^^^^^^^^^^^^^^^^^

.. cpp:type:: class skylark::sketch::sketch_transform_t<InputMatrixType, OutputMatrixType>

    **Query dimensions**

    .. cpp:function:: int get_N() const

        Get input dimension.

    .. cpp:function:: int get_S() const

        Get output dimension.

    **Sketch application**

    .. cpp:function:: void apply (const InputMatrixType& A, OutputMatrixType& sketch_of_A, columnwise_tag dimension) const

        Apply the sketch transform in column dimension.

    .. cpp:function:: void apply (const InputMatrixType& A, OutputMatrixType& sketch_of_A, rowwise_tag dimension) const

        Apply the sketch transform in row dimension.

    **Serialization**

    .. cpp:function:: boost::property_tree::ptree to_ptree() const

        Serialize the sketch transform to a ptree structure.

    .. cpp:function:: static sketch_transform_t* from_ptree(const boost::property_tree::ptree& pt)

        Load a sketch transform from a ptree structure.

    **Accessors**

    .. cpp:function:: const sketch_transform_data_t* get_data()

        Get the underlaying transform data.


The sketch transformation class is coupled to a data class that is responsible
to initialize and provide a lazy view on the random data required when
applying the sketch transform.

The sketching direction is specified using the following types:

    .. cpp:type:: skylark::sketch::columnwise_tag

    .. cpp:type:: skylark::sketch::rowwise_tag


Using the C++ Sketching layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get a flavour of using the sketching layer, we provide a C++ code snippet
here where an Elemental 1D-distributed matrix is sketched to reduce the column
dimensionality (number of rows).
The sketched matrix -- the output of the sketching operation -- is a local
matrix.
The sketching is done using *Johnson-Lindenstrauss (JLT)* transform.


.. code-block:: cpp

     #include <elemental.hpp>
     #include <skylark.hpp>
     ...

     /* Local Matrix Type */
     typedef elem::Matrix<double> MatrixType;

     /* Row distributed Matrix Type */
     typedef elem::DistMatrix<double, elem::VC, elem::STAR> DistMatrixType;

      /* Initialize skylark context with a seed */
     skylark::base::context_t context (12345);

     /* Row distributed Elemental Matrix A of size N x M */
     elem::DistMatrix<double, elem::VR, elem::STAR> A(grid);
     elem::Uniform (A, N, M);

     /* Create the Johnson-Lindenstrauss Sketch object to map R^N to R^S*/
     skys::JLT_t<DistMatrixType, MatrixType> JLT (N,S, context);

     /* Create space for the sketched matrix with number of rows compressed to S */
     MatrixType sketch_A(S, M);

     /* Apply the sketch. We call this columnwise sketching since the column dimensionality is reduced. */
     JLT.apply (A, sketch_A, skys::columnwise_tag());



Python Sketching Interface
------------------------------

Skylark also provides `pure Python` implementations of the various transforms, which it will default in case the C++ layers of
Skylark are not compiled. Some transforms are currently implemented only in Python, but there are plans to implement them in
C++ as well. Likewise, some transforms currently implemented in the C++ layer will be extended to Python in near-term
releases.

Skylark uses external libraries to represent distributed matrices. For dense distributed matrices it uses `Elemental
<http://libelemental.org/>`_. Currently it uses the c-types interface of Elemental, so be sure install that as well. For
sparse distributed matrices it uses `CombBLAS <http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/>`_ interfaced through `KDT
<http://kdt.sourceforge.net/wiki/index.php/Main_Page>`_.

The lower layers use MPI so it is advisable an MPI interface to Python be
installed. One option is to use `mpi4py <http://mpi4py.scipy.org/>`_.

The following table lists currently supported sketching transforms available through Python.

.. image:: files/sketch_transf_in_out_py.png
    :width: 750 px
    :align: center

.. raw:: html

    <i>Schematic views of input and output types for various sketch transforms.
    The <font color="#154685">blue</font> color marks sparse matrix or
    transforms, <font color="#ee9428">orange</font> is used for dense matrix or
    transforms.</i>

.. note:: In the near future the local dense matrix will be replaced by
    CIRC/CIRC and STAR/STAR matrices.


Using the Python interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Skylark is automatically initialized with a random seed when you import
sketch. However, you can reinitialize it to a specific seed by calling
initialize. While not required, you can finalize the library using
finalize. However, note that that will not cause allocated objects
(e.g. sketch transforms) to be freed. They are freed by the garbage collector
when detected as garbage (no references).

.. autofunction:: initialize

.. autofunction:: finalize

Python sketch classes inherit from the *SketchTransform* class.

.. autoclass:: _SketchTransform
                :members:

                .. automethod:: _SketchTransform.__mul__

                .. automethod:: _SketchTransform.__div__

Specific python sketch classes are documented below.

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

Python Examples
-----------------

**Sketching Dense Distributed Matrices**

.. literalinclude:: ../../skylark/examples/example_sketch.py

**Sketching Sparse Distributed Matrices**

.. literalinclude:: ../../skylark/examples/example_sparse_sketch.py


