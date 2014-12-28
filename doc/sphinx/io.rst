IO in libSkylark
========================

IO from C++
------------
Currently, IO from the C++ layer is implemented in libskylark/ml/io.hpp.

Reading
^^^^^^^

.. cpp:function:: void read(const boost::mpi::communicator& comm, int fileformat, std::string filename, InputType& X, LabelType& Y, int d=0)

    Reads data from `filename` as a pair of matrices `X` and `Y`. Data are in the form of instances each carrying a label; an instance in turn consists of a series of feature/value pairs. Feature/value pairs are the column indices/entries for matrix `X` with its row indices coming from a running index of the instances (the minimum number of features is `d`). Matrix `Y` carries the vector of successive labels corresponding to this running index. `fileformat` can have any of the following values of `FileFormatType` enum (as defined in libskylark/ml/options.hpp):

    * HDF5_DENSE

    * HDF5_SPARSE

    * LIBSVM_DENSE

    * LIBSVM_SPARSE

`*_DENSE` correspond to the case of elem::Matrix<T> (dense) matrix `X`; `*_SPARSE` is for the skylark::base::sparse_matrix_t<T> (sparse) matrix `X`.
`HDF5_*` are for `Hierarchical Data Format, version 5 (HDF5) <http://www.hdfgroup.org/HDF5/>`_ files  while `LIBSVM_*` are for `Library for Support Vector Machines (LIBSVM) <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ ones.


**HDF5-specific routines**

.. cpp:function:: void read_hdf5(const boost::mpi::communicator& comm, std::string filename, elem::Matrix<double>& X, elem::Matrix<double>& Y, int blocksize = 10000)

.. cpp:function:: void read_hdf5(const boost::mpi::communicator& comm, std::string filename, skylark::base::sparse_matrix_t<double>& X, elem::Matrix<double>& Y, int min_d = 0)

    Reads datasets `X` and `Y` from HDF5 `filename` as a pair of matrices `X`, `Y`. Instances are read in bunches of `blocksize` each; `min_d` is the minimum number of features.


**LIBSVM-specific routines**

.. cpp:function:: void read_libsvm(const boost::mpi::communicator& comm, std::string filename, elem::Matrix<T>& X, elem::Matrix<T>& Y, int min_d = 0, int blocksize = 10000)

.. cpp:function:: void read_libsvm(const boost::mpi::communicator& comm, std::string filename, skylark::base::sparse_matrix_t<T>& X, elem::Matrix<T>& Y, int min_d = 0)

    Reads data from LIBSVM `filename` as a pair of matrices `X`, `Y`. Instances are read in bunches of `blocksize` each; `min_d` is the minimum number of features.


**Utility routines**

.. cpp:function:: void read_hdf5_dataset(H5::H5File& file, std::string name, int* buffer, int offset, int count)

.. cpp:function:: void read_hdf5_dataset(H5::H5File& file, std::string name, double* buffer, int offset, int count)

    Reads `count` entries from the `name` dataset in HDF5 `file` starting at `offset` and collects them into `buffer`.


.. cpp:function:: void read_model_file(std::string filename, elem::Matrix<double>& W)

    Reads the values from model file `filename` into matrix `W`; model file is expected to start with a line of the form:
    `# Dimensions m n` for an `m x n` matrix `W` followed by lines containing the values of successive rows of the matrix.


.. cpp:function:: std::string read_header(const boost::mpi::communicator& comm, std::string filename)

    Returns the first line of `filename`.



Writing
^^^^^^^

.. cpp:function:: int write_hdf5(const boost::mpi::communicator& comm, std::string filename, elem::Matrix<double>& X, elem::Matrix<double>& Y)

.. cpp:function:: int write_hdf5(std::string filename, skylark::base::sparse_matrix_t<double>& X, elem::Matrix<double>& Y)

    Writes feature/value pairs (in matrix `X`) and labels (in matrix `Y`) to the HDF5 `filename` and returns 0 to signal successfully completed operation.

     * For the case of dense matrix input `X` (elem::Matrix<double> type) 2 datasets named `X` and `Y` are created in the file for all data.

     * For the case of sparse matrix `X` (skylark::base::sparse_matrix_t<double> type) 5 datasets are created in the file for all data: `dimensions` (for `X` matrix size), `indptr`, `indices`, `values` (for `X` matrix) and `Y` (for `Y` matrix).


.. note::

    Currently IO from C++ and Python layers are not in sync; we expect a convergence of offerings from these layers as depending libraries (most notably Elemental) freeze their IO facilities and typical/common application IO scenaria are clearly identified.


IO from Python
---------------
.. automodule:: skylark.io
	:members:
 	:private-members:

Example
^^^^^^^^

.. literalinclude:: ../../python-skylark/skylark/examples/example_IO.py

Sketch Serialization
---------------------

As mentioned in Section :ref:`sketching-transforms-label` sketches can be
serialized using the following API:

.. cpp:function:: boost::property_tree::ptree to_ptree() const

    Serialize the sketch transform to a ptree structure.

.. cpp:function:: static sketch_transform_t* from_ptree(const boost::property_tree::ptree& pt)

    Load a sketch transform from a ptree structure.


The following small snippet shows how a sketch can be serialized and loaded in
C++.

.. code-block:: cpp

    //[> 1. Create the sketching matrix and dump JSON <]
    skylark::sketch::CWT_t<DistMatrixType, DistMatrixType>
        Sparse(n, n_s, context);

    // dump to property tree
    boost::property_tree::ptree pt = Sparse.get_data()->to_ptree();

    //[> 2. Dump the JSON string to file <]
    std::ofstream out("sketch.json");
    write_json(out, pt);
    out.close();

    //[> 3. Create a sketch from the JSON file. <]
    std::ifstream file;
    std::stringstream json;
    file.open("sketch.json", std::ios::in);

    boost::property_tree::ptree json_tree;
    boost::property_tree::read_json(file, json_tree);

    skylark::sketch::CWT_t<DistMatrixType, DistMatrixType> tmp(json_tree);


This functionality is also exposed to the Python layer in two ways.
The first, is straightforward serialize method of sketch objects, and a 
corresponding deserialize_sketch on the module level. The following is a 
short example.

.. code-block:: python

    S = sketch.CWT(10, 6)
    sketch_dict = S.serialize()

    S_clone = sketch.deserialize_sketch(sketch_dict)

Alternatively, sketch objects also support the Pickle and cPickle modules.
This allows them to be easily used by external libraries or software like Spark.
