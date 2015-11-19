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

template<>
class hdf5_type_mapper_t<long long> {
public:
    static const H5::DataType& get_type() {
        return H5::PredType::NATIVE_LLONG;
    }
};

} // namspace internal

/**
 * Reads a matrix from an HDF5 file.
 * Output is an Elemental dense local matrix.
 *
 * IMPORTANT: HDF5 keeps matrices in row-major format, while Elemental works
 *            with column-major. The consequence is that matrices are read in
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

/**
 * Reads a matrix from an HDF5 file.
 * Output is an Elemental dense matrix.
 *
 * IMPORTANT: HDF5 keeps matrices in row-major format, while Elemental works
 *            with column-major. The consequence is that matrices are read in
 *            transpose form. That is, the output matrix is a transpose of the
 *            one you will see in, say, h5dump.
 *
 * @param in HDF5 file to operate on.
 * @param name name of the dataset/group holding the matrix.
 * @param X output matrix.
 * @param max_n, max_m optionally set max height and width
 * @param block_size
 */
template<typename T, El::Distribution U, El::Distribution V>
void ReadHDF5(H5::H5File& in, const std::string& name,
    El::DistMatrix<T, U, V>& X, hsize_t max_n = -1, hsize_t max_m = -1,
              int block_size = 10000) {
    // NOTE: -1 will wrap up to max value because of hsize_t definition

    boost::mpi::communicator comm = skylark::utility::get_communicator(X);
    int rank = X.Grid().Rank();

    // Read matrix size
    H5::DataSet dataset;
    H5::DataSpace fs;
    hsize_t m, n;

    if (rank == 0) {
        dataset = in.openDataSet(name);
        fs = dataset.getSpace();
        hsize_t dims[2];
        fs.getSimpleExtentDims(dims);
        m = std::min(dims[0], max_m);
        n = std::min(fs.getSimpleExtentNdims() > 1 ? dims[1] : 1, max_n);
    }

    boost::mpi::broadcast(comm, m, 0);
    boost::mpi::broadcast(comm, n, 0);

    X.Resize(n, m);

    hsize_t remainm = m;
    hsize_t startm = 0;

    El::DistMatrix<T, U, V> Xv;
    El::DistMatrix<T, El::CIRC, El::CIRC> XB;

    while (remainm > 0) {
        hsize_t mym = remainm < 2 * block_size ? remainm : block_size;
        XB.Resize(n, mym);
        if (rank == 0) {
            hsize_t fo[2], fst[2], fc[2];
            fo[0] = startm;
            fo[1] = 0;
            fc[0] = mym;
            fc[1] = n;
            fs.selectHyperslab(H5S_SELECT_SET, fc, fo);

            hsize_t md[2];
            md[0] = mym;
            md[1] = n;
            H5::DataSpace ms(2, md);

            dataset.read(XB.Buffer(),  internal::hdf5_type_mapper_t<T>::get_type(),
                ms, fs);
        }

        base::ColumnView(Xv, X, startm, mym);
        Xv = XB;

        remainm -= mym;
        startm += mym;
    }

    if (rank == 0)
        dataset.close();

    /*
    H5::DataSet dataset = in.openDataSet(name);
    H5::DataSpace fs = dataset.getSpace();
    hsize_t dims[2];
    fs.getSimpleExtentDims(dims);
    hsize_t m = std::min(dims[0], max_m);
    hsize_t n = std::min(fs.getSimpleExtentNdims() > 1 ? dims[1] : 1, max_n);

    X.Resize(n, m);

    if (fs.getSimpleExtentNdims() == 2) {
        hsize_t fo[2], fst[2], fc[2];
        fo[0] = X.RowShift();
        fo[1] = X.ColShift();
        fst[0] = X.RowStride();
        fst[1] = X.ColStride();
        fc[0] = X.LocalWidth();
        fc[1] = X.LocalHeight();
        fs.selectHyperslab(H5S_SELECT_SET, fc, fo, fst);

        hsize_t md[2];
        md[0] = X.LocalWidth();
        md[1] = X.LocalHeight();
        H5::DataSpace ms(2, md);

        dataset.read(X.Buffer(),  internal::hdf5_type_mapper_t<T>::get_type(),
            ms, fs);
    }

    if (fs.getSimpleExtentNdims() == 1) {
        El::DistMatrix<T, El::CIRC, El::CIRC> X1(n, m);
        if (X1.LocalHeight() != 0) {
            hsize_t cnt[1], fo[1];
            fo[0] = 0;
            cnt[0] = m;
            fs.selectHyperslab(H5S_SELECT_SET, cnt, fo);
            H5::DataSpace ms(1, cnt);

            dataset.read(X1.Buffer(),
                internal::hdf5_type_mapper_t<T>::get_type(), ms, fs);
        }
        El::Copy(X1, X);
    }

    dataset.close();
    */

}

} } } // namespace skylark::utility::io

#endif
