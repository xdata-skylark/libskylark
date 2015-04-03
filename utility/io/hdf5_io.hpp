#ifndef SKYLARK_HDF5_IO_HPP
#define SKYLARK_HDF5_IO_HPP

#include <H5Cpp.h>

namespace skylark { namespace utility { namespace io {

namespace internal {

template<typename T>
class hdf5_type_mapper_t {

};

template<>
class hdf5_type_mapper_t<float> {
public:
    static const H5::DataType& get_type() {
        return H5::PredType::NATIVE_FLOAT;
    }
};

template<>
class hdf5_type_mapper_t<double> {
public:
    static const H5::DataType& get_type() {
        return H5::PredType::NATIVE_DOUBLE;
    }
};

template<>
class hdf5_type_mapper_t<int> {
public:
    static const H5::DataType& get_type() {
        return H5::PredType::NATIVE_INT;
    }
};

} // namspace internal

/**
 * Reads a matrix from an HDF5 file. 
 * Output is an Elemental dense matrices.
 *
 * IMPORTANT: HDF5 keeps matrices in row-major format, while Elemental works
 *            with column-major. The cosquence is that matrices are read in
 *            transpose form. That is, the output matrix is a transpose of the
 *            one you will see in, say, h5dump.
 *
 * @param in HDF5 file to operate on.
 * @param name name of the dataset/group holding the matrix.
 * @param X output matrix.
 */
template<typename T>
void ReadHDF5(H5::H5File& in, const std::string& name, El::Matrix<T>& X) {

    H5::DataSet dataset = in.openDataSet(name);
    H5::DataSpace fs = dataset.getSpace();
    hsize_t dims[2];
    fs.getSimpleExtentDims(dims);
    hsize_t m = dims[0];
    hsize_t n = fs.getSimpleExtentNdims() > 1 ? dims[1] : 1;

    X.Resize(n, m);

    dataset.read(X.Buffer(),  internal::hdf5_type_mapper_t<T>::get_type());
    dataset.close();
}

/**
 * Reads a matrix from an HDF5 file.
 * Output is a local sparse matrix.
 *
 * The matrix should be in a group identified by name. The format is the same
 * as the one used by MATLAB to store sparse matrices (jc, ir, values).
 *
 * @param in HDF5 file to operate on.
 * @param name name of the dataset/group holding the matrix.
 * @param X output matrix.
 * @param min_m minimum value of m. 
 */
template<typename T>
void ReadHDF5(H5::H5File& in, const std::string& name,
    base::sparse_matrix_t<T>& X, int min_m = -1) {
    hsize_t sz;

    H5::Group group = in.openGroup(name);

    // Read colptr
    H5::DataSet dsjc = group.openDataSet("jc");
    H5::DataSpace fs = dsjc.getSpace();
    fs.getSimpleExtentDims(&sz);
    int n = sz - 1;
    int *colptr = new int[n + 1];
    dsjc.read(colptr, internal::hdf5_type_mapper_t<int>::get_type());
    dsjc.close();

    // Read indices
    H5::DataSet dsir = group.openDataSet("ir");
    fs = dsir.getSpace();
    fs.getSimpleExtentDims(&sz);
    int nnz = sz;
    int *indices = new int[nnz];
    dsir.read(indices, internal::hdf5_type_mapper_t<int>::get_type());
    dsir.close();

    // Read data
    H5::DataSet dsdata = group.openDataSet("data");
    T *values = new T[nnz];
    dsdata.read(values, internal::hdf5_type_mapper_t<T>::get_type());
    dsdata.close();

    // Done with HDF5
    group.close();
    // Figure out number of rows
    int m = min_m;
    for(int i = 0; i < nnz; i++)
        if (indices[i] + 1 > m)
            m = indices[i] + 1;

    X.attach(colptr, indices, values, nnz, m, n, true);
}

} } } // namespace skylark::utility::io
#endif
