#ifndef SKYLARK_LIBSVM_IO_HPP
#define SKYLARK_LIBSVM_IO_HPP

#if SKYLARK_HAVE_BOOST_FILESYSTEM
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
namespace boostfs = boost::filesystem;
#endif

#if SKYLARK_HAVE_LIBHDFS
#include "hdfs.h"
#endif

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
template<typename T>
void ReadLIBSVM(const std::string& fname,
    El::Matrix<T>& X, El::Matrix<T>& Y,
    base::direction_t direction, int min_d = 0) {

    std::string line;
    std::string token, val, ind;
    float label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;

    std::ifstream in(fname);

    // make one pass over the data to figure out dimensions -
    // will pay in terms of preallocated storage.
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
    T *Ydata = Y.Buffer();
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
 * @param min_d minimum number of rows in the matrix.
 * @param blocksize blocksize for blocking of read.
 */
template<typename T, El::Distribution UX, El::Distribution VX,
         El::Distribution UY, El::Distribution VY>
void ReadLIBSVM(const std::string& fname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<T, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {


    std::string line;
    std::string token, val, ind;
    T label;
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
    if (rank == 0) {
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

    El::DistMatrix<T, El::CIRC, El::CIRC> XB(X.Grid()), YB(Y.Grid());
    El::DistMatrix<T, UX, VX> Xv(X.Grid());
    El::DistMatrix<T, UY, VY> Yv(Y.Grid());
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
            T *Ydata = YB.Matrix().Buffer();
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
 */
template<typename T>
void ReadLIBSVM(const std::string& fname,
    base::sparse_matrix_t<T>& X, El::Matrix<T>& Y,
    base::direction_t direction, int min_d = 0) {

    std::string line;
    std::string token;
    float label;
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

        if(line.length()==0)
            break;
        n++;

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

    T *Ydata = Y.Buffer();
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
    if (direction == base::COLUMNS) {
        col_ptr[n] = nnz; // last entry (total number of nnz)
        X.attach(col_ptr, rowind, values, nnz, d, n, true);
    } else
        X.attach(col_ptr, rowind, values, nnz, n, d, true);
}


#if SKYLARK_HAVE_BOOST_FILESYSTEM

/**
 * Reads X and Y from a directory of files in libsvm format.
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
         El::Distribution UY, El::Distribution VY>
void ReadDirLIBSVM(const std::string& dname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<T, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {


    std::string line;
    std::string token, val, ind;
    T label;
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
            if (min_d > 0)
                d = std::max(d, min_d);

            in.close();
        }
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
    El::DistMatrix<T, UY, VY> Yv(Y.Grid());
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
            T *Ydata = YB.Matrix().Buffer();
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

#else

template<typename T, El::Distribution UX, El::Distribution VX,
         El::Distribution UY, El::Distribution VY>
void ReadDirLIBSVM(const std::string& dname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<T, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {

    SKYLARK_THROW_EXCEPTION(base::io_exception() <<
        base::error_msg("Install Boost Filesystem for ReadDir support!"));

}

#endif

#if SKYLARK_HAVE_LIBHDFS

namespace detail {

struct hdfs_line_streamer_t {

    hdfs_line_streamer_t(hdfsFS &fs, hdfsFile &fid, int bufsize) :
        _bufsize(bufsize), _fs(fs), _fid(fid),
        _readbuf(new char[bufsize]), _eof(false), _closed(false),
        _emptybuf(true), _readsize(0), _loc(0) {

    }

    ~hdfs_line_streamer_t() {
        close();
        delete _readbuf;
    }

    void getline(std::string &line) {
        line = "";
        while (!_eof) {
            if (_emptybuf) {
                _readsize = hdfsRead(_fs, _fid, _readbuf, _bufsize - 1);
                if (_readsize == 0) {
                    _eof = true;
                    break;
                }
                _readbuf[_readsize] = 0;
                _emptybuf = false;
                _loc = 0;
            }

            char *el = std::strchr(_readbuf + _loc, '\n');
            if (el != NULL) {
                *el = 0;
                line += std::string(_readbuf + _loc);
                _loc += el - (_readbuf + _loc) + 1;
                if (_loc == _readsize)
                    _emptybuf = true;
                return;
            } else {
                line += std::string(_readbuf + _loc);
                _emptybuf = true;
            }
        }
    }

    void rewind() {
        _eof = false;
        _emptybuf = true;
        _readsize = 0;
        hdfsSeek(_fs, _fid, 0);
    }

    void close() {
        if (!_closed)
            hdfsCloseFile(_fs, _fid);
        _closed = true;
    }

    bool eof() {
        return _eof;
    }

private:
    const int _bufsize;
    hdfsFS &_fs;
    hdfsFile &_fid;
    char *_readbuf;
    bool _eof, _closed, _emptybuf;
    int _readsize;
    int _loc;

};

}

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
template<typename T>
void ReadLIBSVM(const hdfsFS &fs, const std::string& fname,
    El::Matrix<T>& X, El::Matrix<T>& Y,
    base::direction_t direction, int min_d = 0) {

    std::string line;
    std::string token, val, ind;
    float label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;

    hdfsFile fid = hdfsOpenFile(fs, fname.c_str(), O_RDONLY, 0, 0, 0);
    if(!fid) {
        std::stringstream ss;
        ss << "Failed to open HDFS file " << fname;
        SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
            skylark::base::error_msg(ss.str()))
    }
    detail::hdfs_line_streamer_t in(fs, fid, 1000);

    // make one pass over the data to figure out dimensions -
    // will pay in terms of preallocated storage.
    while(!in.eof()) {
        in.getline(line);

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
    in.rewind();

    if (direction == base::COLUMNS) {
        X.Resize(d, n);
        Y.Resize(nt, n);
    } else {
        X.Resize(n, d);
        Y.Resize(n, nt);
    }

    T *Xdata = X.Buffer();
    T *Ydata = Y.Buffer();
    int ldX = X.LDim();
    int ldY = Y.LDim();

    for (t = 0; t < n; t++) {
        in.getline(line);

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
    }
}

/**
 * Reads X and Y from a file in libsvm format.
 * X and Y are Elemental distributed matrices.
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
         El::Distribution UY, El::Distribution VY>
void ReadLIBSVM(hdfsFS &fs, const std::string& fname,
    El::DistMatrix<T, UX, VX>& X, El::DistMatrix<T, UY, VY>& Y,
    base::direction_t direction, int min_d = 0, int blocksize = 10000) {


    std::string line;
    std::string token, val, ind;
    T label;
    unsigned int start = 0;
    unsigned int t;
    int n = 0, nt = 0;
    int d = 0;
    int i, j, last;
    char c;

    // TODO check that X and Y have the same grid.
    boost::mpi::communicator comm = skylark::utility::get_communicator(X);
    int rank = X.Grid().Rank();

    detail::hdfs_line_streamer_t *in = nullptr;
    hdfsFile fid;
    if (rank == 0) {
        fid = hdfsOpenFile(fs, fname.c_str(), O_RDONLY, 0, 0, 0);
        if(!fid) {
            std::stringstream ss;
            ss << "Failed to open HDFS file " << fname;
            SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
                skylark::base::error_msg(ss.str()))
        }

        in = new detail::hdfs_line_streamer_t(fs, fid, 1000);
    }

    // make one pass over the data to figure out dimensions -
    // will pay in terms of preallocated storage.
    if (rank==0) {

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
        if (min_d > 0)
            d = std::max(d, min_d);

        in->rewind();
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
    El::DistMatrix<T, UY, VY> Yv(Y.Grid());
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
            T *Ydata = YB.Matrix().Buffer();
            int ldX = XB.Matrix().LDim();
            int ldY = YB.Matrix().LDim();

            t = 0;
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

        if (rank == 0)
            delete in;
    }
}

#endif

} } } // namespace skylark::utility::io
#endif
