.. highlight:: rst

Base Layer
**********

This layer provides the foundation of the entire libSkylark library.

Random Values
-------------

Random values are generated in a lazy fashion using
`Random123 library <http://www.deshawresearch.com/resources_random123.html>`_.
The access to the random streams is wrapped in two classes, which abstracts random
number arrays.

.. cpp:type:: struct skylark::utility::random_samples_array_t<Distribution>

    .. cpp:function:: random_samples_array_t(size_t base, size_t size, int seed, Distribution& distribution)

        Random-access array of samples drawn from a distribution starting from
        `base` up to a total of `size` samples. The `distribution` determines
        how samples are drawn.

    .. cpp:function:: random_samples_array_t(const random_samples_array_t& other)

    .. cpp:function:: random_samples_array_t& operator=(const random_samples_array_t& other)

    **Accessors**

    .. cpp:function:: value_type operator[](size_t index) const

        Returns the random value (following the specified distribution) at
        position `index` in the random stream.

libSkylark Context
------------------

The library provides and tracks the state of the random number streams in the `context`
class. Any use of functionality that generates random numbers will need to provide a context
object.

.. cpp:type:: struct skylark::base::context_t

    **Constructor**

    .. cpp:function:: context_t (int seed, int counter=0)

        Initialize a context with a seed that is used for all computations.

    **Serialization**

    .. cpp:function:: context_t (const boost::property_tree::ptree& json)

        Load a context from a serialized `ptree` structure.

    .. cpp:function:: boost::property_tree::ptree to_ptree() const

        Serialize the context to a Boost `ptree`.

    **Query**

    .. cpp:function:: size_t get_counter()

        Returns the current position in the random stream.

    **Accessors**

    .. cpp:function:: skylark::utility::random_samples_array_t<Distribution> allocate_random_samples_array(size_t size, Distribution& distribution)

        Returns a container of samples drawn from a distribution to be accessed
        as an array. The container contains `size` samples drawn from the
        specified `distribution`.

    .. cpp:function:: std::vector<typename Distribution::result_type> generate_random_samples_array(size_t size, Distribution& distribution)

        Returns a vector of samples drawn from a distribution. The vector
        contains `size` samples drawn from the specified `distribution`.

    .. cpp:function:: skylark::utility::random_array_t allocate_random_array(size_t size)

        Returns a container of random numbers to be accessed as an array. The
        container lazily provides `size` samples.

    .. cpp:function:: int random_int()

        Returns an integer random number.

Local sparse matrices
---------------------

This implements a very crude CSC sparse matrix container only intended to
hold local sparse matrices.

* Row indices are not sorted.
* Structure is always constants, and can only be attached by Attached.
* Values of non-zeros can be modified.


.. cpp:type:: struct skylark::base::sparse_matrix_t<T>

    **Constructor**

    .. cpp:function:: sparse_matrix_t ()

        Creates a new sparse matrix.

    .. cpp:function:: sparse_matrix_t sparse_matrix_t(sparse_matrix_t<ValueType>&& A)

        Move constructor.


    **Query**

    .. cpp:function:: int height() const

        Height of the matrix (number of rows).

    .. cpp:function:: int width() const

        Width of the matrix (number of cols).

    .. cpp:function:: int nonzeros() const

        Number of non-zeros.

    .. cpp:function:: bool struct_updated() const

        Flag that encodes if CSC structure has been modified.

    .. cpp:function:: void reset_update_flag()

        Reset the modified flag (to false).


    **Accessors**

    .. cpp:function:: const int* indptr() const

        Access to indices pointer array.

    .. cpp:function:: const int* indices() const

        Access to non-zero column indices array.

    .. cpp:function:: T* values()

        Access to non-zero value array.

    .. cpp:function:: const T* locked_values() const

        Read only view on non-zero values.


    **Modifiers**

    .. cpp:function:: void detach<IdxType, ValType>(IdxType* indptr, IdxType* indices, ValType* values) const

        Copy CSC data to external buffer.

    .. cpp:function:: void attach(const int* indptr, const int* indices, double* values, int nnz, int n_rows, int n_cols, bool _own = false)

        Attach CSC matrix data.

    .. cpp:function:: void attach(const int* indptr, const int* indices, double* values, int nnz, int n_rows, int n_cols, bool ownindptr, bool ownindices, bool ownvalues)

        Attach CSC matrix data where the caller can decide who retains the
        ownership.

Random Dense Matrices
---------------------

The library provides functions for filling Elemental dense matrices (local or distributed) using the library's random number generator.

    .. cpp:function:: void RandomMatrix(El::Matrix<T> &A, El::Int m, El::Int n, DistributionType<T> &dist, context_t &context)

    .. cpp:function:: void RandomMatrix(El::DistMatrix<T, CD, RD> &A, El::Int m, El::Int n, DistributionType<T> &dist, context_t &context)

       Resize matrix to m-by-n and fill with entries which are i.i.d samples of the distribution.

    .. cpp:function:: void GaussianMatrix(MatrixType &A, El::Int m, El::Int n, context_t &context)

       Resize matrix to m-by-n and fill with entries which are i.i.d samples from standard normal distribution.

    .. cpp:function:: void UniformMatrix(MatrixType &A, El::Int m, El::Int n, context_t &context)

       Resize matrix to m-by-n and fill with entries which are i.i.d samples from a uniform distrbution on [0,1).

Cross matrix-type GEMM and other linear algebra routines
--------------------------------------------------------

To be added...

