#ifndef SKYLARK_LIBSVM_IO_HPP
#define SKYLARK_LIBSVM_IO_HPP

#include <unordered_map>

namespace skylark { namespace utility { namespace io {

/**
 * Reads X and Y from a file in libsvm format.
 * X and Y are Elemental dense matrices.
 *
 * @param fname input file name.
 * @param X output X
 * @param Y output Y
 * @param direction whether the examples are to be put in rows or columns
 * @param min_d minimum number of rows in the matrix.
 */
template<typename T, typename R>
void ReadLIBSVM(const std::string& fname,
    El::Matrix<T>& X, El::Matrix<R>& Y,
    base::direction_t direction, int min_d = 0) {

    std::string line;
    std::string token, val, ind;
    R label;
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

    if (direction == base::COLUMNS) {
        X.Resize(d, n);
        Y.Resize(1, n);
    } else {
        X.Resize(n, d);
        Y.Resize(n, 1);
    }

    T *Xdata = X.Buffer();
    R *Ydata = Y.Buffer();
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
            if (direction == base::COLUMNS)
                Xdata[t * ldX + j] = atof(val.c_str());
            else
                Xdata[j * ldX + t] = atof(val.c_str());
        }
    }
}

/**
 * Reads X and Y from a file in libsvm format.
 * X and Y are Elemental distributed matrices.
 *
 * @param fname input file name.
 * @param X output X
 * @param Y output Y
 * @param direction whether the examples are to be put in rows or columns
 * @param min_d minimum number of rows in the matrix.
 * @param blocksize blocksize for blocking of read.
 */
template<typename T, El::Distribution UX, El::Distribution VX,
         typename R, El::Distribution UY, El::Distribution VY>
void ReadLIBSVM(const std::string& fname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<R, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {


    std::string line;
    std::string token, val, ind;
    R label;
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

    if (direction == base::COLUMNS) {
        X.Resize(d, n);
        Y.Resize(1, n);
    } else {
        X.Resize(n, d);
        Y.Resize(n, 1);
    }

    El::DistMatrix<T, El::CIRC, El::CIRC> XB(X.Grid());
    El::DistMatrix<R, El::CIRC, El::CIRC> YB(Y.Grid());
    El::DistMatrix<T, UX, VX> Xv(X.Grid());
    El::DistMatrix<R, UY, VY> Yv(Y.Grid());
    for(int i=0; i<numblocks+1; i++) {
        if (i==numblocks)
            block = leftover;
        if (block==0)
            break;

        if (direction == base::COLUMNS) {
            El::Zeros(XB, d, block);
            El::Zeros(YB, 1, block);
        } else {
            El::Zeros(XB, block, d);
            El::Zeros(YB, block, 1);
        }

        if(rank==0) {
            T *Xdata = XB.Matrix().Buffer();
            R *Ydata = YB.Matrix().Buffer();
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
                    if (direction == base::COLUMNS)
                        Xdata[t * ldX + j] = atof(val.c_str());
                    else
                        Xdata[j * ldX + t] = atof(val.c_str());
                }

                t++;
            }
        }

        // The calls below should distribute the data to all the nodes.
        if (direction == base::COLUMNS) {
            El::View(Xv, X, 0, i*blocksize, d, block);
            El::View(Yv, Y, 0, i*blocksize, 1, block);
        } else {
            El::View(Xv, X, i*blocksize, 0, block, d);
            El::View(Yv, Y, i*blocksize, 0, block, 1);
        }

        Xv = XB;
        Yv = YB;
    }
}

/**
 * Reads X and Y from a file in libsvm format.
 * X is a Skylark local sparse matrix, and Y is Elemental dense matrices.
 *
 * @param fname input file name
 * @param X output X
 * @param Y output Y
 * @param direction whether the examples are to be put in rows or columns
 * @param min_d minimum number of rows in the matrix.
 */
template<typename T, typename R>
void ReadLIBSVM(const std::string& fname,
    base::sparse_matrix_t<T>& X, El::Matrix<R>& Y, 
    base::direction_t direction, int min_d = 0) {

    std::string line;
    std::string token;
    R label;
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
    // Also find number of non-zeros per column. 
    std::unordered_map<int, int> colsize;

    while(!in.eof()) {
        getline(in, line);
        if(line.length()==0)
            break;
        n++;

        if (direction == base::COLUMNS) {
            delim = line.find_last_of(":");
            if(delim > line.length())
                continue;
            t = delim;
            while(line[t]!=' ')
                t--;
            std::string val = line.substr(t+1, delim - t);
            last = atoi(val.c_str());
            if (last>d)
                d = last;


            std::istringstream tokenstream (line);
            tokenstream >> label;
            while (tokenstream >> token)
                nnz++;
        } else {
            std::istringstream tokenstream (line);
            tokenstream >> label;

            while (tokenstream >> token) {
                nnz++;
                delim  = token.find(':');
                int ind = atoi(token.substr(0, delim).c_str());

                colsize[ind-1]++;

                if (ind > d)
                    d = ind;
            }
        }
    }

    T *values = new T[nnz];
    int *rowind = new int[nnz];
    int *col_ptr = new int[direction == base::COLUMNS ? n + 1 : d + 1];

    if (direction == base::ROWS) {
        col_ptr[0] = 0;
        for(int i = 1; i <= d; i++)
            col_ptr[i] = col_ptr[i-1] + colsize[i-1];
        Y.Resize(n, 1);
    } else
        Y.Resize(1, n);
    R *Ydata = Y.Buffer();

    // prepare for second pass
    in.clear();
    in.seekg(0, std::ios::beg);
    if (direction == base::COLUMNS)
        nnz = 0;

    colsize.clear();
    for (t = 0; t < n; t++) {
        getline(in, line);
        if( line.length()==0)
            break;

        std::istringstream tokenstream (line);
        tokenstream >> Ydata[t];

        if (direction == base::COLUMNS)
            col_ptr[t] = nnz;

        while (tokenstream >> token) {
            delim  = token.find(':');
            std::string ind = token.substr(0, delim);
            std::string val = token.substr(delim+1); //.substr(delim+1);
            j = atoi(ind.c_str()) - 1;

            if (direction == base::COLUMNS) {
                rowind[nnz] = j;
                values[nnz] = atof(val.c_str());
                nnz++;
            } else {
                rowind[col_ptr[j] + colsize[j]] = t;
                values[col_ptr[j] + colsize[j]] = atof(val.c_str());
                colsize[j]++;
            }
        }
    }

    if (min_d > 0)
        d = std::max(d, min_d);

    if (direction == base::COLUMNS) {
        col_ptr[n] = nnz; // last entry (total number of nnz)
        X.attach(col_ptr, rowind, values, nnz, d, n, true);
    } else
        X.attach(col_ptr, rowind, values, nnz, n, d, true);
}

/**
 * Write X and Y from a file in libsvm format.
 * X and Y are Elemental dense matrices.
 *
 * @param fname output file name.
 * @param X input X
 * @param Y output Y
 * @param direction whether the examples are in the rows or columns of X and Y
 */
template<typename T, typename R>
void WriteLIBSVM(const std::string& fname,
    El::Matrix<T>& X, El::Matrix<R>& Y,
    base::direction_t direction) {

    std::ofstream out(fname);
    El::Int n, d;

    if (direction == base::COLUMNS) {
        n = X.Width();
        d = X.Height();
    } else {
        n = X.Height();
        d = X.Width();
    }

    for(El::Int j = 0; j < n; j++) {
        if (direction == base::COLUMNS) {
            out << Y.Get(0, j) << " ";
            for(El::Int r = 0; r < d; r++) {
                T val = X.Get(r, j);
                if (val != 0.0)
                    out << (r+1) << ":" << val << " ";
            }
            out << std::endl;
        } else {
            out << Y.Get(j, 0) << " ";
            for(El::Int r = 0; r < d; r++) {
                T val = X.Get(j, r);
                if (val != 0.0)
                    out << (r+1) << ":" << val << " ";
            }
            out << std::endl;
        }
    }

    out.close();
}

/**
 * Write X and Y from a file in libsvm format.
 * X and Y are Elemental distributed matrices.
 *
 * @param fname output file name.
 * @param X input X
 * @param Y output Y
 * @param direction whether the examples are in the rows or columns of X and Y
 * @param blocksize blocksize for blocking of read.
 */
template<typename T, El::Distribution UX, El::Distribution VX,
         typename R, El::Distribution UY, El::Distribution VY>
void WriteLIBSVM(const std::string& fname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<R, UY, VY>& Y,
    base::direction_t direction, int blocksize = 10000) {

    int rank = X.Grid().Rank();

    std::ofstream out(fname);
    El::Int n, d;

    if (direction == base::COLUMNS) {
        n = X.Width();
        d = X.Height();
    } else {
        n = X.Height();
        d = X.Width();
    }

    El::Int numblocks = ((int) n/ (int) blocksize); 
    El::Int leftover = n % blocksize;
    El::Int block = blocksize;

    El::DistMatrix<T, El::CIRC, El::CIRC> XB(X.Grid());
    El::DistMatrix<R, El::CIRC, El::CIRC> YB(Y.Grid());
    El::DistMatrix<T, UX, VX> Xv(X.Grid());
    El::DistMatrix<R, UY, VY> Yv(Y.Grid());
    for(El::Int i = 0; i < numblocks + 1; i++) {
        if (i == numblocks)
            block = leftover;
        if (block == 0)
            break;

        // The calls below should distribute the data to all the nodes.
        if (direction == base::COLUMNS) {
            El::View(Xv, X, 0, i*blocksize, d, block);
            El::View(Yv, Y, 0, i*blocksize, 1, block);
        } else {
            El::View(Xv, X, i*blocksize, 0, block, d);
            El::View(Yv, Y, i*blocksize, 0, block, 1);
        }

        XB = Xv;
        YB = Yv;

        if(rank==0)
            for(El::Int j = 0; j < block; j++) {
                if (direction == base::COLUMNS) {
                    out << YB.Get(0, j) << " ";
                    for(El::Int r = 0; r < d; r++) {
                        T val = XB.Get(r, j);
                        if (val != 0.0)
                            out << (r+1) << ":" << val << " ";
                    }
                    out << std::endl;
                } else {
                        out << YB.Get(j, 0) << " ";
                        for(El::Int r = 0; r < d; r++) {
                            T val = XB.Get(j, r);
                            if (val != 0.0)
                                out << (r+1) << ":" << val << " ";
                        }
                        out << std::endl;
                }
            }
    }
    out.close();
}

} } } // namespace skylark::utility::io
#endif
