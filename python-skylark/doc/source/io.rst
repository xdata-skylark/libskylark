IO in Skylark
========================

IO from C++
------------
Currently, IO from the C++ layer is implemented in libskylark/ml/io.hpp.

Reading
^^^^^^^

.. cpp:function:: void read(const boost::mpi::communicator &comm, int fileformat, std::string filename, InputType& X, LabelType& Y, int d=0)

    Reads data from `filename` as a pair of matrices `X` and `Y`. Data are in the form of instances each carrying a label; an instance in turn consists of a series of feature/value pairs. Feature/value pairs are the column indices/entries for matrix `X` with its row indices coming from a running index of the instances. Matrix `Y` carries the vector of successive labels corresponding to this running index. `fileformat` can be any of the following:

    * HDF5_DENSE

    * HDF5_SPARSE

    * LIBSVM_DENSE

    * LIBSVM_SPARSE

`*_DENSE` correspond to the case of elem::Matrix<T> (dense) matrix `X` (`*_SPARSE` is for skylark::base::sparse_matrix_t<T> (sparse) matrix `X`).
`HDF5_*` are for HDF5-formatted files  while `LIBSVM_*` are for LIBSVM ones.


**HDF5-specific routines**

.. cpp:function:: void read_hdf5(const boost::mpi::communicator &comm, std::string fName, LocalMatrixType&  Xlocal, LocalMatrixType& Ylocal, int blocksize = 10000)

.. cpp:function:: void read_hdf5(const boost::mpi::communicator &comm, std::string fName, sparse_matrix_t& X, elem::Matrix<double>& Y, int min_d = 0)

    Reads data from datasets `X` and `Y` in Hierarchical Data Format (HDF5) file `fName` as a pair of matrices `Xlocal`, `Ylocal` (`X`, `Y`). Instances are read in bunches of `blocksize` each.


**LIBSVM-specific routines**

.. cpp:function:: void read_libsvm(const boost::mpi::communicator &comm, std::string fName, elem::Matrix<T>& Xlocal, elem::Matrix<T>& Ylocal, int min_d = 0, int blocksize = 10000)

.. cpp:function:: void read_libsvm(const boost::mpi::communicator &comm, std::string fName,  skylark::base::sparse_matrix_t<T>& X, elem::Matrix<T>& Y, int min_d = 0)

    Reads data from Library for Support Vector Machines (LIBSVM) formatted file `fName` as a pair of matrices `Xlocal`, `Ylocal` (`X`, `Y`). Instances are read in bunches of `blocksize` each.


**Utility routines**

.. cpp:function:: void read_hdf5_dataset(H5::H5File& file, std::string name, int* buf, int offset, int count)

.. cpp:function:: void read_hdf5_dataset(H5::H5File& file, std::string name, double* buf, int offset, int count)

    Reads `count` entries from the `name` dataset in HDF5 `file` starting at `offset` and collects them into buffer `buf`.


.. cpp:function:: void read_model_file(std::string fName, elem::Matrix<double>& W)

    Reads the values from model file `fName` into matrix `W`; model file is expected to start with a line of the form:
    `# Dimensions m n` for an `m x n` matrix `W` followed by lines containing the values of successive rows of the matrix.


.. cpp:function:: std::string read_header(const boost::mpi::communicator &comm, std::string fName)

    Returns the first line of file `fName`.



Writing
^^^^^^^

.. cpp:function:: int write_hdf5(const boost::mpi::communicator &comm, std::string fName, elem::Matrix<double>& X, elem::Matrix<double>& Y)

.. cpp:function:: int write_hdf5(std::string fName, sparse_matrix_t& X, elem::Matrix<double>& Y)

    Writes feature/value pairs (in matrix `X`) and labels (in matrix `Y`) to the HDF5 file `fName` and returns 0 to signal successfully completed operation.For the case of dense matrix input `X` (of elem::Matrix<double> type) two datasets named `X` and `Y` are created in the file for all data. For the case of sparse matrix `X` (of sparse_matrix_t type, aka  skylark::base::sparse_matrix_t<T>) five datasets are created: `dimensions` (matrix size), `indptr`, `indices`, `values` (`X` matrix) and `Y` (`Y` matrix).


.. note::

    Currently IO from C++ and Python layers are not in sync; we expect a convergence of offerings from these layers as depending libraries (most notably Elemental) freeze their IO facilities and typical/common application IO scenaria are clearly identified.


IO from Python
---------------
.. automodule:: skylark.io
	:members:
 	:private-members:

Example
^^^^^^^^

.. literalinclude:: ../../skylark/examples/example_IO.py

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


This functionality is also exposed to the Python layer.

.. code-block:: python

    S = sketch.CWT(10, 6)
    sketch_dict = S.serialize()

    S_clone = deserialize_sketch(sketch_dict)


