#ifndef SKYLARK_LIBSVM_IO_HPP
#define SKYLARK_LIBSVM_IO_HPP

namespace skylark { namespace utility { namespace io {

/**
 * Reads X and Y from a file in libsvm format.
 * X and Y are Elemental dense matrices.
 *
 * IMPORTANT: output is in column-major format (the rows are features).
 *
 * @param fname input file name.
 * @param X output X
 * @param Y output Y
 * @param min_d minimum number of rows in the matrix.
 */
template<typename T>
void ReadLIBSVM(const std::string& fname,
    El::Matrix<T>& X, El::Matrix<T>& Y,
    int min_d = 0) {

    std::string line;
    std::string token, val, ind;
    float label;
    unsigned int start = 0;
    unsigned int delim, t;
    int n = 0;
    int d = 0;
    int i, j, last;
    char c;

    std::ifstream in(fname);

    // make one pass over the data to figure out dimensions - 
    // will pay in terms of preallocated storage.
    while(!in.eof()) {
        getline(in, line);
        if(line.length()==0)
            break;
        delim = line.find_last_of(":");
        if(delim > line.length())
            continue;
        n++;
        t = delim;
        while(line[t]!=' ') {
            t--;
        }
        val = line.substr(t+1, delim - t);
        last = atoi(val.c_str());
        if (last>d)
            d = last;
    }
    if (min_d > 0)
        d = std::max(d, min_d);

    // prepare for second pass
    in.clear();
    in.seekg(0, std::ios::beg);

    X.Resize(d, n);
    Y.Resize(1, n);

    T *Xdata = X.Buffer();
    T *Ydata = Y.Buffer();
    int ldX = X.LDim();

    for (t = 0; t < n; t++) {
        getline(in, line);
        if( line.length()==0)
            break;

        std::istringstream tokenstream (line);
        tokenstream >> label;
        Ydata[t] = label;

        while (tokenstream >> token) {
            delim  = token.find(':');
            ind = token.substr(0, delim);
            val = token.substr(delim+1); //.substr(delim+1);
            j = atoi(ind.c_str()) - 1;
            Xdata[t * ldX + j] = atof(val.c_str());
        }
    }
}

/**
 * Reads X and Y from a file in libsvm format.
 * X and Y are Elemental distributed matrices.
 *
 * IMPORTANT: output is in column-major format (the rows are features).
 *
 * @param fname input file name.
 * @param X output X
 * @param Y output Y
 * @param min_d minimum number of rows in the matrix.
 * @param blocksize blocksize for blocking of read.
 */
template<typename T, El::Distribution UX, El::Distribution VX,
         El::Distribution UY, El::Distribution VY>
void ReadLIBSVM(const std::string& fname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<T, UY, VY>& Y,
    int min_d = 0, int blocksize = 10000) {


    std::string line;
    std::string token, val, ind;
    float label;
    unsigned int start = 0;
    unsigned int delim, t;
    int n = 0;
    int d = 0;
    int i, j, last;
    char c;

    std::ifstream in(fname);

    // TODO check that X and Y have the same grid.
    boost::mpi::communicator comm = skylark::utility::get_communicator(X);
    int rank = X.Grid().Rank();

    // make one pass over the data to figure out dimensions - 
    // will pay in terms of preallocated storage.
    if (rank==0) {
        while(!in.eof()) {
            getline(in, line);
            if(line.length()==0)
                break;
            delim = line.find_last_of(":");
            if(delim > line.length())
                continue;
            n++;
            t = delim;
            while(line[t]!=' ') {
                t--;
            }
            val = line.substr(t+1, delim - t);
            last = atoi(val.c_str());
            if (last>d)
                d = last;
        }
        if (min_d > 0)
            d = std::max(d, min_d);

        // prepare for second pass
        in.clear();
        in.seekg(0, std::ios::beg);
    }

    boost::mpi::broadcast(comm, n, 0);
    boost::mpi::broadcast(comm, d, 0);

    int numblocks = ((int) n/ (int) blocksize); // of size blocksize
    int leftover = n % blocksize;
    int block = blocksize;

    X.Resize(d, n);
    Y.Resize(1, n);

    El::DistMatrix<T, El::CIRC, El::CIRC> XB(X.Grid()), YB(Y.Grid());
    El::DistMatrix<T, UX, VX> Xv(X.Grid());
    El::DistMatrix<T, UY, VY> Yv(Y.Grid());
    for(int i=0; i<numblocks+1; i++) {
        if (i==numblocks)
            block = leftover;
        if (block==0)
            break;

        El::Zeros(XB, d, block);
        El::Zeros(YB, 1, block);

        if(rank==0) {
            T *Xdata = XB.Matrix().Buffer();
            T *Ydata = YB.Matrix().Buffer();
            int ldX = XB.Matrix().LDim();

            t = 0;
            while(!in.eof() && t<block) {
                getline(in, line);
                if( line.length()==0)
                    break;

                std::istringstream tokenstream (line);
                tokenstream >> label;
                Ydata[t] = label;

                while (tokenstream >> token) {
                    delim  = token.find(':');
                    ind = token.substr(0, delim);
                    val = token.substr(delim+1); //.substr(delim+1);
                    j = atoi(ind.c_str()) - 1;
                    Xdata[t * ldX + j] = atof(val.c_str());
                }

                t++;
            }
        }

        // The calls below should distribute the data to all the nodes.
        El::View(Xv, X, 0, i*blocksize, d, block);
        El::View(Yv, Y, 0, i*blocksize, 1, block);

        Xv = XB;
        Yv = YB;
    }
}

/**
 * Reads X and Y from a file in libsvm format.
 * X is a Skylark local sparse matrix, and Y is Elemental dense matrices.
 *
 * IMPORTANT: output is in column-major format (the rows are features).
 *
 * @param fname input file name
 * @param X output X
 * @param Y output Y
 * @param min_d minimum number of rows in the matrix.
 */
template<typename T>
void ReadLIBSVM(const std::string& fname,
    base::sparse_matrix_t<T>& X, El::Matrix<T>& Y, int min_d = 0) {

    std::string line;
    std::string token, val, ind;
    float label;
    unsigned int start = 0;
    unsigned int delim, t;
    int n = 0;
    int d = 0;
    int i, j, last;
    char c;
    int nnz=0;
    int nz;

    std::ifstream in(fname);

    // make one pass over the data to figure out dimensions and nnz
    // will pay in terms of preallocated storage.
    while(!in.eof()) {
        getline(in, line);
        if(line.length()==0)
            break;

        delim = line.find_last_of(":");
        if(delim > line.length())
            continue;
        n++;
        t = delim;
        while(line[t]!=' ')
            t--;

        val = line.substr(t+1, delim - t);
        last = atoi(val.c_str());
        if (last>d)
            d = last;

        std::istringstream tokenstream (line);
        tokenstream >> label;
        while (tokenstream >> token)
            nnz++;
    }

    T *values = new T[nnz];
    int *rowind = new int[nnz];
    int *col_ptr = new int[n + 1];

    Y.Resize(1, n);
    T *Ydata = Y.Buffer();

    // prepare for second pass
    in.clear();
    in.seekg(0, std::ios::beg);
    nnz = 0;

    for (t = 0; t < n; t++) {
        getline(in, line);
        if( line.length()==0)
            break;

        std::istringstream tokenstream (line);
        tokenstream >> Ydata[t];

        col_ptr[t] = nnz;
        while (tokenstream >> token) {
            delim  = token.find(':');
            ind = token.substr(0, delim);
            val = token.substr(delim+1); //.substr(delim+1);
            j = atoi(ind.c_str()) - 1;
            rowind[nnz] = j;
            values[nnz] = atof(val.c_str());
            nnz++;
        }
    }
    col_ptr[n] = nnz; // last entry (total number of nnz)

    X.attach(col_ptr, rowind, values, nnz, d, n, true);
}

} } } // namespace skylark::utility::io
#endif
