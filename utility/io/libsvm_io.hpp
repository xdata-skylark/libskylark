#ifndef SKYLARK_LIBSVM_IO_HPP
#define SKYLARK_LIBSVM_IO_HPP

#include <memory>

#if SKYLARK_HAVE_BOOST_FILESYSTEM
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
namespace boostfs = boost::filesystem;
#endif

#include <unordered_map>
#include <boost/serialization/list.hpp>

#include "../types.hpp"
#include "../get_communicator.hpp"

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
 * @param max_n maximum number of columns in the matrix.
 * @param blocksize blocksize for reading for distributed outputs.
 */
template<typename T, typename R>
void ReadLIBSVM(const std::string& fname,
    El::Matrix<T>& X, El::Matrix<R>& Y,
    base::direction_t direction, int min_d = 0, int max_n = -1,
    int blocksize=10000) {

    std::string line;
    std::string token, val, ind;
    R label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;

    std::ifstream in(fname);

    if ((in.rdstate() & std::ifstream::failbit) != 0)
        SKYLARK_THROW_EXCEPTION (
           base::io_exception()
               << base::error_msg(
                "Failed to open file " + fname));

    // make one pass over the data to figure out dimensions -
    // will pay in terms of preallocated storage.
    while(!in.eof() && n != max_n) {
        getline(in, line);

        // Ignore empty lines and comment lines (begin with #)
        if(line.length() == 0 || line[0] == '#')
            break;

        n++;

        // Figure out number of targets (only first line)
        if (n == 1) {
            std::string tstr;
            std::istringstream tokenstream (line);
            tokenstream >> tstr;
            while (tstr.find(":") == std::string::npos) {
                nt++;
                if (tokenstream.eof())
                    break;
                tokenstream >> tstr;
            }
        }

        size_t delim = line.find_last_of(":");
        if(delim == std::string::npos)
            continue;

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
        Y.Resize(nt, n);
    } else {
        X.Resize(n, d);
        Y.Resize(n, nt);
    }

    T *Xdata = X.Buffer();
    R *Ydata = Y.Buffer();
    int ldX = X.LDim();
    int ldY = Y.LDim();

    for (t = 0; t < n; t++) {
        getline(in, line);

        // Ignore empty lines and comment lines (begin with #)
        if(line.length() == 0 || line[0] == '#')
            break;

        std::istringstream tokenstream(line);

        for(int r = 0; r < nt; r++) {
            tokenstream >> label;
            if (direction == base::COLUMNS)
                Ydata[t * ldY + r] = label;
            else
                Ydata[r * ldY + t] = label;
        }

        while (tokenstream >> token) {
            size_t delim  = token.find(':');
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
 * @param max_n stop reading after n rows. If -1 then will read all rows.
 * @param min_d minimum number of rows in the matrix.
 * @param blocksize blocksize for blocking of read.
 */
template<typename T, El::Distribution UX, El::Distribution VX,
         typename R, El::Distribution UY, El::Distribution VY>
void ReadLIBSVM(const std::string& fname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<R, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int max_n = -1,
    int blocksize = 10000) {

    std::string line;
    std::string token, val, ind;
    R label;
    unsigned int start = 0;
    size_t delim;
    unsigned int t;
    int n = 0, nt = 0;
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
        while(!in.eof() && n != max_n) {
            getline(in, line);

            // Ignore empty lines and comment lines (begin with #)
            if(line.length() == 0 || line[0] == '#')
                break;

            n++;

            // Figure out number of targets (only first line)
            if (n == 1) {
                std::string tstr;
                std::istringstream tokenstream (line);
                tokenstream >> tstr;
                while (tstr.find(":") == std::string::npos) {
                    nt++;
                    if (tokenstream.eof())
                        break;
                    tokenstream >> tstr;
                }
            }

            size_t delim = line.find_last_of(":");
            if(delim == std::string::npos)
                continue;

            t = delim;
            while(line[t]!=' ')
                t--;

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
    boost::mpi::broadcast(comm, nt, 0);

    int numblocks = ((int) n/ (int) blocksize); // of size blocksize
    int leftover = n % blocksize;
    int block = blocksize;

    if (direction == base::COLUMNS) {
        X.Resize(d, n);
        Y.Resize(nt, n);
    } else {
        X.Resize(n, d);
        Y.Resize(n, nt);
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
            El::Zeros(YB, nt, block);
        } else {
            El::Zeros(XB, block, d);
            El::Zeros(YB, block, nt);
        }

        if(rank==0) {
            T *Xdata = XB.Matrix().Buffer();
            R *Ydata = YB.Matrix().Buffer();
            int ldX = XB.Matrix().LDim();
            int ldY = YB.Matrix().LDim();

            t = 0;
            while(!in.eof() && t<block) {
                getline(in, line);

                // Ignore empty lines and comment lines (begin with #)
                if(line.length() == 0 || line[0] == '#')
                    break;

                std::istringstream tokenstream(line);

                for(int r = 0; r < nt; r++) {
                    tokenstream >> label;
                    if (direction == base::COLUMNS)
                        Ydata[t * ldY + r] = label;
                    else
                        Ydata[r * ldY + t] = label;
                }

                while (tokenstream >> token) {
                    size_t delim  = token.find(':');
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
            int ldX = XB.Matrix().LDim();
            El::View(Xv, X, 0, i*blocksize, d, block);
            El::View(Yv, Y, 0, i*blocksize, nt, block);
        } else {
            El::View(Xv, X, i*blocksize, 0, block, d);
            El::View(Yv, Y, i*blocksize, 0, block, nt);
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
 * @param max_n maximum number of cols in the matrix.
 * @param blocksize blocksize for reading for distributed outputs.
 */
template<typename T, typename R>
void ReadLIBSVM(const std::string& fname,
    base::sparse_matrix_t<T>& X, El::Matrix<R>& Y,
    base::direction_t direction, int min_d = 0, int max_n = -1,
    int blocksize = 10000) {

    std::string line;
    std::string token;
    R label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
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

    while(!in.eof() && n != max_n) {
        getline(in, line);

        // Ignore empty lines and comment lines (begin with #)
        if(line.length() == 0 || line[0] == '#')
            break;

        n++;

        // Figure out number of targets (only first line)
        if (n == 1) {
            std::string tstr;
            std::istringstream tokenstream (line);
            tokenstream >> tstr;
            while (tstr.find(":") == std::string::npos) {
                nt++;
                if (tokenstream.eof())
                    break;
                tokenstream >> tstr;
            }
        }

        if (direction == base::COLUMNS) {
            size_t delim = line.find_last_of(":");
            if(delim == std::string::npos)
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
                size_t delim  = token.find(':');
                int ind = atoi(token.substr(0, delim).c_str());

                colsize[ind-1]++;

                if (ind > d)
                    d = ind;
            }
        }
    }

    if (min_d > 0)
        d = std::max(d, min_d);

    T *values = new T[nnz];
    int *rowind = new int[nnz];
    int *col_ptr = new int[direction == base::COLUMNS ? n + 1 : d + 1];

    if (direction == base::ROWS) {
        col_ptr[0] = 0;
        for(int i = 1; i <= d; i++)
            col_ptr[i] = col_ptr[i-1] + colsize[i-1];
        Y.Resize(n, nt);
    } else
        Y.Resize(nt, n);
    R *Ydata = Y.Buffer();
    int ldY = Y.LDim();

    // prepare for second pass
    in.clear();
    in.seekg(0, std::ios::beg);
    if (direction == base::COLUMNS)
        nnz = 0;

    colsize.clear();
    for (t = 0; t < n; t++) {
        getline(in, line);

        // Ignore empty lines and comment lines (begin with #)
        if(line.length() == 0 || line[0] == '#')
            break;

        std::istringstream tokenstream (line);

        for(int r = 0; r < nt; r++) {
            tokenstream >> label;
            if (direction == base::COLUMNS)
                Ydata[t * ldY + r] = label;
            else
                Ydata[r * ldY + t] = label;
        }

        if (direction == base::COLUMNS)
            col_ptr[t] = nnz;

        while (tokenstream >> token) {
            size_t delim  = token.find(':');
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
 * Reads X and Y from a file in libsvm format.
 * X is a sparse distributed VC/STAR matrix and Y is a dense distributed
 * VC/STAR matrix.
 *
 * IMPORTANT: output is in column-major format (the rows are features).
 *
 * @param fname input file name.
 * @param X output X
 * @param Y output Y
 * @param direction whether the examples are to be put in rows or columns
 * @param min_d minimum number of rows in the matrix.
 * @param blocksize blocksize for blocking of read.
 */
template<typename T,
         typename R, El::Distribution UY, El::Distribution VY>
void ReadLIBSVM(const std::string& fname,
    base::sparse_vc_star_matrix_t<T>& X, El::DistMatrix<R, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int max_n = -1,
    int blocksize = 10000) {


    std::string line;
    std::string token, val, ind;
    T label;
    unsigned int start = 0;
    unsigned int delim, t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;

    std::ifstream in(fname);

    boost::mpi::communicator comm = skylark::utility::get_communicator(Y);
    int rank = comm.rank();
    int size = comm.size();

    std::vector< std::list<int> > non_local_updates_j(size);
    std::vector< std::list<int> > non_local_updates_row(size);
    std::vector< std::list<T> > non_local_updates_v(size);

    // make one pass over the data to figure out dimensions -
    // will pay in terms of preallocated storage.
    if (rank==0) {
        while(!in.eof() && n != max_n) {
            getline(in, line);
            if(line.length()==0)
                break;
            delim = line.find_last_of(":");
            if(delim > line.length())
                continue;
            n++;

            // Figure out number of targets
            if (n == 1) {
                std::string tstr;
                std::istringstream tokenstream (line);
                tokenstream >> tstr;
                while (tstr.find(":") == std::string::npos) {
                    nt++;
                    tokenstream >> tstr;
                }
            }

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
    boost::mpi::broadcast(comm, nt, 0);

    int numblocks = ((int) n/ (int) blocksize); // of size blocksize
    int leftover = n % blocksize;
    int block = blocksize;

    if (direction == base::COLUMNS) {
        X.resize(d, n);
        Y.Resize(nt, n);
    } else {
        X.resize(n, d);
        Y.Resize(n, nt);
    }

    El::DistMatrix<T, El::CIRC, El::CIRC> YB(Y.Grid());
    El::DistMatrix<T, El::VC, El::STAR> Yv(Y.Grid());

    int row = 0;
    for(int i=0; i<numblocks+1; i++) {
        if (i==numblocks)
            block = leftover;
        if (block==0)
            break;

        if (direction == base::COLUMNS) {
            El::Zeros(YB, nt, block);
        } else {
            El::Zeros(YB, block, nt);
        }

        if(rank==0) {
            T *Ydata = YB.Matrix().Buffer();
            int ldY = YB.Matrix().LDim();

            t = 0;
            while(!in.eof() && t<block) {
                getline(in, line);
                if( line.length()==0)
                    break;

                std::istringstream tokenstream (line);
                for(int r = 0; r < nt; r++) {
                    tokenstream >> label;
                    if (direction == base::COLUMNS)
                        Ydata[t * ldY + r] = label;
                    else
                        Ydata[r * ldY + t] = label;
                }

                while (tokenstream >> token) {
                    delim  = token.find(':');
                    ind = token.substr(0, delim);
                    val = token.substr(delim+1); //.substr(delim+1);
                    j = atoi(ind.c_str()) - 1;
                    int owner = (direction == base::COLUMNS) ?
                        X.owner(j, row) : X.owner(row, j);
                    if (owner == 0) {
                        if (direction == base::COLUMNS)
                            X.queue_update(j, row, atof(val.c_str()));
                        else
                            X.queue_update(row, j, atof(val.c_str()));
                    } else {
                        non_local_updates_j[owner].push_back(j);
                        non_local_updates_row[owner].push_back(row);
                        non_local_updates_v[owner].push_back(atof(val.c_str()));
                    }
                }

                row++;
                t++;
            }

            for (int rk = 1; rk < size; rk++) {
                comm.send(rk, 0, non_local_updates_j[rk]);
                comm.send(rk, 0, non_local_updates_row[rk]);
                comm.send(rk, 0, non_local_updates_v[rk]);

                non_local_updates_j[rk].clear();
                non_local_updates_row[rk].clear();
                non_local_updates_v[rk].clear();
            }
        } else {
                comm.recv(0, 0, non_local_updates_j[rank]);
                comm.recv(0, 0, non_local_updates_row[rank]);
                comm.recv(0, 0, non_local_updates_v[rank]);

                auto it_j = non_local_updates_j[rank].begin();
                auto it_row = non_local_updates_row[rank].begin();
                auto it_v = non_local_updates_v[rank].begin();

                for(; it_j != non_local_updates_j[rank].end();) {

                    int j = *it_j, row = *it_row;
                    T val = *it_v;

                    if (direction == base::COLUMNS)
                        X.queue_update(j, row, val);
                    else
                        X.queue_update(row, j, val);

                    it_j++; it_row++; it_v++;
                }

                non_local_updates_j[rank].clear();
                non_local_updates_row[rank].clear();
                non_local_updates_v[rank].clear();
        }

        // The calls below should distribute the data to all the nodes.
        if (direction == base::COLUMNS) {
            El::View(Yv, Y, 0, i*blocksize, nt, block);
        } else {
            El::View(Yv, Y, i*blocksize, 0, block, nt);
        }

        Yv = YB;
    }

    X.finalize();
}

void ReadLIBSVM(const std::string& fname,
    boost::any X, boost::any Y,
    base::direction_t direction, int min_d = 0, int max_n = -1,
    int blocksize = 10000) {

#define SKYLARK_READLIBSVM_APPLY_DISPATCH(XT, YT)                   \
    if (X.type() == typeid(XT*) && Y.type() == typeid(YT*))  {      \
        ReadLIBSVM(fname, *boost::any_cast<XT*>(X),                 \
            *boost::any_cast<YT*>(Y), direction,                    \
            min_d, max_n, blocksize);                               \
            return;                                                 \
    }                                       \

#if !(defined SKYLARK_NO_ANY)
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::matrix_t, mdtypes::matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::matrix_t, mftypes::matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::matrix_t, mdtypes::matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::matrix_t, mftypes::matrix_t);

    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::shared_matrix_t,
        mdtypes::shared_matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::shared_matrix_t,
        mftypes::shared_matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::shared_matrix_t,
        mdtypes::shared_matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::shared_matrix_t,
        mftypes::shared_matrix_t);

    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::root_matrix_t,
        mdtypes::root_matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::root_matrix_t,
        mftypes::root_matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::root_matrix_t,
        mdtypes::root_matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::root_matrix_t,
        mftypes::root_matrix_t);

    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
        mdtypes::matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
        mftypes::matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::sparse_matrix_t,
        mdtypes::matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::sparse_matrix_t,
        mftypes::matrix_t);

    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_t,
        mdtypes::dist_matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_t,
        mftypes::dist_matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_t,
        mdtypes::dist_matrix_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_t,
        mftypes::dist_matrix_t);

    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
        mdtypes::dist_matrix_vc_star_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
        mftypes::dist_matrix_vc_star_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
        mdtypes::dist_matrix_vc_star_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
        mftypes::dist_matrix_vc_star_t);

    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
        mdtypes::dist_matrix_vr_star_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
        mftypes::dist_matrix_vr_star_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
        mdtypes::dist_matrix_vr_star_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
        mftypes::dist_matrix_vr_star_t);

    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
        mdtypes::dist_matrix_star_vc_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
        mftypes::dist_matrix_star_vc_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
        mdtypes::dist_matrix_star_vc_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
        mftypes::dist_matrix_star_vc_t);

    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
        mdtypes::dist_matrix_star_vr_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
        mftypes::dist_matrix_star_vr_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
        mdtypes::dist_matrix_star_vr_t);
    SKYLARK_READLIBSVM_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
        mftypes::dist_matrix_star_vr_t);

#endif

    SKYLARK_THROW_EXCEPTION (
        base::io_exception()
          << base::error_msg(
           "This combination has not yet been implemented for ReadLIBSVM"));

#undef SKYLARK_READLIBSVM_APPLY_DISPATCH
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

#if SKYLARK_HAVE_BOOST_FILESYSTEM

/**
 * Reads X and Y from a directory of files in libsvm format.
 * X and Y are Elemental dense matrices.
 *
 * @param fname input file name
 * @param X output X
 * @param Y output Y
 * @param direction whether the examples are to be put in rows or columns
 * @param min_d minimum number of rows in the matrix.
 */
template<typename T, typename R>
void ReadDirLIBSVM(const std::string& dname,
    El::Matrix<T>& X, El::Matrix<R>& Y,
    base::direction_t direction, int min_d = 0) {

    std::string line;
    std::string token;
    R label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;
    int nnz=0;
    int nz;

    boostfs::path full_path(boostfs::system_complete(boostfs::path(dname)));
    boostfs::directory_iterator end_iter;

    for(boostfs::directory_iterator dirit(full_path); dirit != end_iter;
        dirit++) {

        std::string fname = dirit->path().filename().string();
        if (fname == "." || fname == ".." || fname[0] == '.')
            continue;

        std::ifstream in(dirit->path().string());

        while(!in.eof()) {
            getline(in, line);

            // Ignore empty lines and comment lines (begin with #)
            if(line.length() == 0 || line[0] == '#')
                break;

            n++;

            // Figure out number of targets (only first line)
            if (n == 1) {
                std::string tstr;
                std::istringstream tokenstream (line);
                tokenstream >> tstr;
                while (tstr.find(":") == std::string::npos) {
                    nt++;
                    if (tokenstream.eof())
                        break;
                    tokenstream >> tstr;
                }
            }

            if (direction == base::COLUMNS) {
                size_t delim = line.find_last_of(":");
                if(delim == std::string::npos)
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
                    size_t delim  = token.find(':');
                    int ind = atoi(token.substr(0, delim).c_str());

                    if (ind > d)
                        d = ind;
                }
            }
        }

        in.close();
    }

    if (min_d > 0)
        d = std::max(d, min_d);


    if (direction == base::ROWS) {
        X.Resize(n, d);
        Y.Resize(n, nt);
    } else {
        X.Resize(d, n);
        Y.Resize(nt, n);
    }

    T *Xdata = X.Buffer();
    El::Int ldX = X.LDim();
    R *Ydata = Y.Buffer();
    El::Int ldY = Y.LDim();

    // prepare for second pass
    t = 0;

    for(boostfs::directory_iterator dirit(full_path); dirit != end_iter;
        dirit++) {

        std::string fname = dirit->path().filename().string();
        if (fname == "." || fname == ".." || fname[0] == '.')
            continue;

        std::ifstream in(dirit->path().string());
        while(!in.eof()) {
            getline(in, line);

            // Ignore empty lines and comment lines (begin with #)
            if(line.length() == 0 || line[0] == '#')
                break;
            t++;

            std::istringstream tokenstream (line);

            for(int r = 0; r < nt; r++) {
                tokenstream >> label;
                if (direction == base::COLUMNS)
                    Ydata[t * ldY + r] = label;
                else
                    Ydata[r * ldY + t] = label;
            }

            while (tokenstream >> token) {
                size_t delim  = token.find(':');
                std::string ind = token.substr(0, delim);
                std::string val = token.substr(delim+1); //.substr(delim+1);
                j = atoi(ind.c_str()) - 1;

                if (direction == base::COLUMNS)
                    Xdata[t * ldX + j] = atof(val.c_str());
                else
                    Xdata[j * ldX + t] = atof(val.c_str());

            }
        }
    }
}

/**
 * Reads X and Y from a directory of files in libsvm format.
 * X is a Skylark local sparse matrix, and Y is Elemental dense matrices.
 *
 * @param fname input file name
 * @param X output X
 * @param Y output Y
 * @param direction whether the examples are to be put in rows or columns
 * @param min_d minimum number of rows in the matrix.
 */
template<typename T, typename R>
void ReadDirLIBSVM(const std::string& dname,
    base::sparse_matrix_t<T>& X, El::Matrix<R>& Y,
    base::direction_t direction, int min_d = 0) {

    std::string line;
    std::string token;
    R label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;
    int nnz=0;
    int nz;

    boostfs::path full_path(boostfs::system_complete(boostfs::path(dname)));
    boostfs::directory_iterator end_iter;

    // make one pass over the data to figure out dimensions and nnz
    // will pay in terms of preallocated storage.
    // Also find number of non-zeros per column.
    std::unordered_map<int, int> colsize;

    for(boostfs::directory_iterator dirit(full_path); dirit != end_iter;
        dirit++) {

        std::string fname = dirit->path().filename().string();
        if (fname == "." || fname == ".." || fname[0] == '.')
            continue;

        std::ifstream in(dirit->path().string());

        while(!in.eof()) {
            getline(in, line);

            // Ignore empty lines and comment lines (begin with #)
            if(line.length() == 0 || line[0] == '#')
                break;

            n++;

            // Figure out number of targets (only first line)
            if (n == 1) {
                std::string tstr;
                std::istringstream tokenstream (line);
                tokenstream >> tstr;
                while (tstr.find(":") == std::string::npos) {
                    nt++;
                    if (tokenstream.eof())
                        break;
                    tokenstream >> tstr;
                }
            }

            if (direction == base::COLUMNS) {
                size_t delim = line.find_last_of(":");
                if(delim == std::string::npos)
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
                    size_t delim  = token.find(':');
                    int ind = atoi(token.substr(0, delim).c_str());

                    colsize[ind-1]++;
                    if (ind > d)
                        d = ind;
                }
            }
        }

        in.close();
    }

    if (min_d > 0)
        d = std::max(d, min_d);

    T *values = new T[nnz];
    int *rowind = new int[nnz];
    int *col_ptr = new int[direction == base::COLUMNS ? n + 1 : d + 1];

    if (direction == base::ROWS) {
        col_ptr[0] = 0;
        for(int i = 1; i <= d; i++)
            col_ptr[i] = col_ptr[i-1] + colsize[i-1];
        Y.Resize(n, nt);
    } else
        Y.Resize(nt, n);

    R *Ydata = Y.Buffer();
    int ldY = Y.LDim();

    // prepare for second pass
    colsize.clear();
    t = 0;

    for(boostfs::directory_iterator dirit(full_path); dirit != end_iter;
        dirit++) {

        std::string fname = dirit->path().filename().string();
        if (fname == "." || fname == ".." || fname[0] == '.')
            continue;

        std::ifstream in(dirit->path().string());
        while(!in.eof()) {
            getline(in, line);

            // Ignore empty lines and comment lines (begin with #)
            if(line.length() == 0 || line[0] == '#')
                break;
            t++;

            std::istringstream tokenstream (line);

            for(int r = 0; r < nt; r++) {
                tokenstream >> label;
                if (direction == base::COLUMNS)
                    Ydata[t * ldY + r] = label;
                else
                    Ydata[r * ldY + t] = label;
            }
            if (direction == base::COLUMNS)
                col_ptr[t] = nnz;

            while (tokenstream >> token) {
                size_t delim  = token.find(':');
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
    }

    if (direction == base::COLUMNS) {
        col_ptr[n] = nnz; // last entry (total number of nnz)
        X.attach(col_ptr, rowind, values, nnz, d, n, true);
    } else
        X.attach(col_ptr, rowind, values, nnz, n, d, true);
}

/**
 * reads x and y from a directory of files in libsvm format.
 * x and y are elemental distributed matrices.
 *
 * @param fname input file name.
 * @param x output x
 * @param y output y
 * @param direction whether the examples are to be put in rows or columns
 * @param min_d minimum number of rows in the matrix.
 * @param blocksize blocksize for blocking of read.
 */
template<typename T, El::Distribution UX, El::Distribution VX,
         typename R, El::Distribution UY, El::Distribution VY>
void ReadDirLIBSVM(const std::string& dname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<R, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {


    std::string line;
    std::string token, val, ind;
    R label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;

    // TODO check that X and Y have the same grid.
    boost::mpi::communicator comm = skylark::utility::get_communicator(X);
    int rank = X.Grid().Rank();

    boostfs::path full_path(boostfs::system_complete(boostfs::path(dname)));
    boostfs::directory_iterator end_iter;

    // make one pass over the data to figure out dimensions -
    // will pay in terms of preallocated storage.
    if (rank==0) {
        for(boostfs::directory_iterator dirit(full_path); dirit != end_iter;
            dirit++) {

            std::string fname = dirit->path().filename().string();
            if (fname == "." || fname == ".." || fname[0] == '.')
                continue;

            std::ifstream in(dirit->path().string());

            while(!in.eof()) {
                getline(in, line);

                // Ignore empty lines and comment lines (begin with #)
                if(line.length() == 0 || line[0] == '#')
                    break;

                n++;

                // Figure out number of targets (only first line)
                if (n == 1) {
                    std::string tstr;
                    std::istringstream tokenstream (line);
                    tokenstream >> tstr;
                    while (tstr.find(":") == std::string::npos) {
                        nt++;
                        if (tokenstream.eof())
                            break;
                        tokenstream >> tstr;
                    }
                }

                size_t delim = line.find_last_of(":");
                if(delim == std::string::npos)
                    continue;

                t = delim;
                while(line[t]!=' ') {
                    t--;
                }
                val = line.substr(t+1, delim - t);
                last = atoi(val.c_str());
                if (last>d)
                    d = last;
            }

            in.close();
        }

        if (min_d > 0)
            d = std::max(d, min_d);
    }

    boost::mpi::broadcast(comm, n, 0);
    boost::mpi::broadcast(comm, d, 0);
    boost::mpi::broadcast(comm, nt, 0);

    int numblocks = ((int) n/ (int) blocksize); // of size blocksize
    int leftover = n % blocksize;
    int block = blocksize;

    if (direction == base::COLUMNS) {
        X.Resize(d, n);
        Y.Resize(nt, n);
    } else {
        X.Resize(n, d);
        Y.Resize(n, nt);
    }

    El::DistMatrix<T, El::CIRC, El::CIRC> XB(X.Grid()), YB(Y.Grid());
    El::DistMatrix<T, UX, VX> Xv(X.Grid());
    El::DistMatrix<R, UY, VY> Yv(Y.Grid());
    boostfs::directory_iterator dirit(full_path);

    std::ifstream in;
    if (rank == 0) {
        std::string fname = dirit->path().filename().string();
        while (fname == "." || fname == ".." || fname[0] == '.') {
            dirit++;
            if (dirit == end_iter)
                break;
            fname = dirit->path().filename().string();
        }

        in.open(dirit->path().string());

        dirit++;
    }

    for(int i=0; i<numblocks+1; i++) {
        if (i==numblocks)
            block = leftover;
        if (block==0)
            break;

        if (direction == base::COLUMNS) {
            El::Zeros(XB, d, block);
            El::Zeros(YB, nt, block);
        } else {
            El::Zeros(XB, block, d);
            El::Zeros(YB, block, nt);
        }

        if(rank==0) {
            T *Xdata = XB.Matrix().Buffer();
            R *Ydata = YB.Matrix().Buffer();
            int ldX = XB.Matrix().LDim();
            int ldY = YB.Matrix().LDim();

            t = 0;
            while(t<block) {
                if (in.eof()) {
                    if (dirit == end_iter)
                        break;

                    in.close();

                    std::string fname = dirit->path().filename().string();
                    while (fname == "." || fname == ".." || fname[0] == '.') {
                        dirit++;
                        if (dirit == end_iter)
                            break;

                        fname = dirit->path().filename().string();
                    }

                    if (dirit == end_iter)
                        break;

                    in.open(dirit->path().string());
                    dirit++;
                }

                getline(in, line);

                // Ignore empty lines and comment lines (begin with #)
                if(line.length() == 0 || line[0] == '#')
                    break;

                std::istringstream tokenstream (line);
                for(int r = 0; r < nt; r++) {
                    tokenstream >> label;
                    if (direction == base::COLUMNS)
                        Ydata[t * ldY + r] = label;
                    else
                        Ydata[r * ldY + t] = label;
                }

                while (tokenstream >> token) {
                    size_t delim  = token.find(':');
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
            int ldX = XB.Matrix().LDim();
            El::View(Xv, X, 0, i*blocksize, d, block);
            El::View(Yv, Y, 0, i*blocksize, nt, block);
        } else {
            El::View(Xv, X, i*blocksize, 0, block, d);
            El::View(Yv, Y, i*blocksize, 0, block, nt);
        }

        Xv = XB;
        Yv = YB;
    }

    if (rank == 0)
        in.close();
}

template<typename T,
         typename R, El::Distribution UY, El::Distribution VY>
void ReadDirLIBSVM(const std::string& dname,
    base::sparse_vc_star_matrix_t<T>& X, El::DistMatrix<R, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {

    SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
        skylark::base::error_msg(
            "readdirlibsvm not implemented for sparse_vc_star_matrix_t!"));
}

#else

template<typename XType, typename YType>
void ReadDirLIBSVM(const std::string& dname, XType& X, YType& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {

    SKYLARK_THROW_EXCEPTION(base::io_exception() <<
        base::error_msg("Install Boost Filesystem for ReadDir support!"));

}

#endif

#if SKYLARK_HAVE_LIBHDFS

/**
 * Reads X and Y from a file in libsvm format (from HDFS filesystem).
 * X and Y are Elemental dense matrices.
 *
 * @param fs hdfs filesystem
 * @param fname input file name.
 * @param X output X
 * @param Y output Y
 * @param direction whether the examples are to be put in rows or columns
 * @param min_d minimum number of rows in the matrix.
 */
template<typename T, typename R>
void ReadLIBSVM(const hdfsFS &fs, const std::string& fname,
    El::Matrix<T>& X, El::Matrix<R>& Y,
    base::direction_t direction, int min_d = 0) {

    std::string line;
    std::string token, val, ind;
    R label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;

    hdfs_line_streamer_iterator_t itr(fs, fname, 1000);
    auto in = itr.next();

    // make one pass over the data to figure out dimensions -
    // will pay in terms of preallocated storage.
    while(in != nullptr) {
        while(!in->eof()) {
            in->getline(line);

            // Ignore empty lines and comment lines (begin with #)
            if(line.length() == 0 || line[0] == '#')
                break;

            n++;

            // Figure out number of targets (only first line)
            if (n == 1) {
                std::string tstr;
                std::istringstream tokenstream (line);
                tokenstream >> tstr;
                while (tstr.find(":") == std::string::npos) {
                    nt++;
                    if (tokenstream.eof())
                        break;
                    tokenstream >> tstr;
                }
            }

            size_t delim = line.find_last_of(":");
            if(delim == std::string::npos)
                continue;

            t = delim;
            while(line[t]!=' ') {
                t--;
            }
            val = line.substr(t+1, delim - t);
            last = atoi(val.c_str());
            if (last>d)
                d = last;
        }

        in = itr.next();
    }

    if (min_d > 0)
        d = std::max(d, min_d);

    // prepare for second pass
    itr.reset();
    in = itr.next();

    if (direction == base::COLUMNS) {
        X.Resize(d, n);
        Y.Resize(nt, n);
    } else {
        X.Resize(n, d);
        Y.Resize(n, nt);
    }

    T *Xdata = X.Buffer();
    R *Ydata = Y.Buffer();
    int ldX = X.LDim();
    int ldY = Y.LDim();

    t = 0;
    while(in != nullptr) {

        in->rewind();
        while(!in->eof()) {
            in->getline(line);

            // Ignore empty lines and comment lines (begin with #)
            if(line.length() == 0 || line[0] == '#')
                break;

            std::istringstream tokenstream (line);

            for(int r = 0; r < nt; r++) {
                tokenstream >> label;
                if (direction == base::COLUMNS)
                    Ydata[t * ldY + r] = label;
                else
                    Ydata[r * ldY + t] = label;
            }

            while (tokenstream >> token) {
                size_t delim  = token.find(':');
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

        in = itr.next();
    }
}

/**
 * Reads X and Y from a file in libsvm format (from HDFS filesystem).
 * X is a Skylark local sparse matrix, and Y is Elemental dense matrices.
 *
 * @param fname input file name
 * @param X output X
 * @param Y output Y
 * @param direction whether the examples are to be put in rows or columns
 * @param min_d minimum number of rows in the matrix.
 */
template<typename T, typename R>
void ReadLIBSVM(hdfsFS &fs, const std::string& fname,
    base::sparse_matrix_t<T>& X, El::Matrix<R>& Y,
    base::direction_t direction, int min_d = 0) {

    std::string line;
    std::string token;
    R label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;
    int nnz=0;
    int nz;

    hdfs_line_streamer_iterator_t itr(fs, fname, 1000);
    auto in = itr.next();

    // make one pass over the data to figure out dimensions and nnz
    // will pay in terms of preallocated storage.
    // Also find number of non-zeros per column.
    std::unordered_map<int, int> colsize;

    while(in != nullptr) {

        while(!in->eof()) {
            in->getline(line);

            // Ignore empty lines and comment lines (begin with #)
            if(line.length() == 0 || line[0] == '#')
                break;

            n++;

            // Figure out number of targets (only first line)
            if (n == 1) {
                std::string tstr;
                std::istringstream tokenstream (line);
                tokenstream >> tstr;
                while (tstr.find(":") == std::string::npos) {
                    nt++;
                    if (tokenstream.eof())
                        break;
                    tokenstream >> tstr;
                }
            }

            if (direction == base::COLUMNS) {
                size_t delim = line.find_last_of(":");
                if(delim == line.length())
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
                    size_t delim  = token.find(':');
                    int ind = atoi(token.substr(0, delim).c_str());

                    colsize[ind-1]++;

                    if (ind > d)
                        d = ind;
                }
            }
        }

        in = itr.next();
    }

    if (min_d > 0)
        d = std::max(d, min_d);

    T *values = new T[nnz];
    int *rowind = new int[nnz];
    int *col_ptr = new int[direction == base::COLUMNS ? n + 1 : d + 1];

    if (direction == base::ROWS) {
        col_ptr[0] = 0;
        for(int i = 1; i <= d; i++)
            col_ptr[i] = col_ptr[i-1] + colsize[i-1];
        Y.Resize(n, nt);
    } else
        Y.Resize(1, nt);

    R *Ydata = Y.Buffer();
    int ldY = Y.LDim();

    // prepare for second pass
    itr.reset();
    in = itr.next();

    if (direction == base::COLUMNS)
        nnz = 0;

    colsize.clear();

    t = 0;
    while(in != nullptr) {
        in->rewind();

        while(!in->eof()) {
            in->getline(line);

            // Ignore empty lines and comment lines (begin with #)
            if(line.length() == 0 || line[0] == '#')
                break;

            std::istringstream tokenstream (line);

            for(int r = 0; r < nt; r++) {
                tokenstream >> label;
                if (direction == base::COLUMNS)
                    Ydata[t * ldY + r] = label;
                else
                    Ydata[r * ldY + t] = label;
            }

            if (direction == base::COLUMNS)
                col_ptr[t] = nnz;

            while (tokenstream >> token) {
                size_t delim  = token.find(':');
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

            t++;
        }

        in->close();
        in = itr.next();
    }

    if (direction == base::COLUMNS) {
        col_ptr[n] = nnz; // last entry (total number of nnz)
        X.attach(col_ptr, rowind, values, nnz, d, n, true);
    } else
        X.attach(col_ptr, rowind, values, nnz, n, d, true);
}

/**
 * Reads X and Y from a file in libsvm format (from HDFS filesystem).
 * X and Y are Elemental distributed matrices.
 * Note that all the data is read on rank 0 (WHY??).
 *
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
void ReadLIBSVM(hdfsFS &fs, const std::string& fname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<R, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {

    std::string line;
    std::string token, val, ind;
    R label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;

    // TODO check that X and Y have the same grid.
    boost::mpi::communicator comm = skylark::utility::get_communicator(X);
    int rank = X.Grid().Rank();

    //FIXME: only required on rank 0
    //detail::hdfs_line_streamer_iterator_t itr(fs, fname, 1000);
    std::unique_ptr<hdfs_line_streamer_iterator_t> itr(nullptr);

    // make one pass over the data to figure out dimensions -
    // will pay in terms of preallocated storage.
    if (rank==0) {

        itr.reset(new hdfs_line_streamer_iterator_t (fs, fname, 1000));
        auto in = itr->next();

        while(in != nullptr) {
            while(!in->eof()) {
                in->getline(line);

                // Ignore empty lines and comment lines (begin with #)
                if(line.length() == 0 || line[0] == '#')
                    break;

                n++;

                // Figure out number of targets
                if (n == 1) {
                    std::string tstr;
                    std::istringstream tokenstream (line);
                    tokenstream >> tstr;
                    while (tstr.find(":") == std::string::npos) {
                        nt++;
                        if (tokenstream.eof())
                            break;
                        tokenstream >> tstr;
                    }
                }

                size_t delim = line.find_last_of(":");
                if(delim == std::string::npos)
                    continue;

                t = delim;
                while(line[t]!=' ') {
                    t--;
                }
                val = line.substr(t+1, delim - t);
                last = atoi(val.c_str());
                if (last>d)
                    d = last;
            }

            in = itr->next();
        }

        if (min_d > 0)
            d = std::max(d, min_d);
    }

    boost::mpi::broadcast(comm, n, 0);
    boost::mpi::broadcast(comm, d, 0);
    boost::mpi::broadcast(comm, nt, 0);

    int numblocks = ((int) n/ (int) blocksize); // of size blocksize
    int leftover = n % blocksize;
    int block = blocksize;

    if (direction == base::COLUMNS) {
        X.Resize(d, n);
        Y.Resize(nt, n);
    } else {
        X.Resize(n, d);
        Y.Resize(n, nt);
    }

    El::DistMatrix<T, El::CIRC, El::CIRC> XB(X.Grid()), YB(Y.Grid());
    El::DistMatrix<T, UX, VX> Xv(X.Grid());
    El::DistMatrix<R, UY, VY> Yv(Y.Grid());
    for(int i=0; i<numblocks+1; i++) {
        if (i==numblocks)
            block = leftover;
        if (block==0)
            break;

        if (direction == base::COLUMNS) {
            El::Zeros(XB, d, block);
            El::Zeros(YB, nt, block);
        } else {
            El::Zeros(XB, block, d);
            El::Zeros(YB, block, nt);
        }

        if(rank==0) {
            T *Xdata = XB.Matrix().Buffer();
            R *Ydata = YB.Matrix().Buffer();
            int ldX = XB.Matrix().LDim();
            int ldY = YB.Matrix().LDim();

            itr->reset();
            auto in = itr->next();

            t = 0;
            while(in != nullptr) {

                in->rewind();
                while(!in->eof() && t<block) {
                    in->getline(line);

                    // Ignore empty lines and comment lines (begin with #)
                    if(line.length() == 0 || line[0] == '#')
                        break;
                    std::istringstream tokenstream (line);
                    for(int r = 0; r < nt; r++) {
                        tokenstream >> label;
                        if (direction == base::COLUMNS)
                            Ydata[t * ldY + r] = label;
                        else
                            Ydata[r * ldY + t] = label;
                    }

                    while (tokenstream >> token) {
                        size_t delim  = token.find(':');
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

                in = itr->next();
            }
        }

        // The calls below should distribute the data to all the nodes.
        if (direction == base::COLUMNS) {
            int ldX = XB.Matrix().LDim();
            El::View(Xv, X, 0, i*blocksize, d, block);
            El::View(Yv, Y, 0, i*blocksize, nt, block);
        } else {
            El::View(Xv, X, i*blocksize, 0, block, d);
            El::View(Yv, Y, i*blocksize, 0, block, nt);
        }

        Xv = XB;
        Yv = YB;
    }
}

template<typename T,
         typename R, El::Distribution UY, El::Distribution VY>
void ReadLIBSVM(const hdfsFS &fs, const std::string& fname,
    base::sparse_vc_star_matrix_t<T>& X, El::DistMatrix<R, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {

    //TODO: implement
    SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
        skylark::base::error_msg(
            "ReadLIBSVM from HDFS not implemented for sparse_vc_star_matrix_t!"));
}



#endif

} } } // namespace skylark::utility::io

#endif
