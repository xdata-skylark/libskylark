#ifndef SKYLARK_ARC_LIST_HPP_
#define SKYLARK_ARC_LIST_HPP_

#include <string>
#include <sstream>
#include <iostream>
#include <climits>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

//XXX: add a Boost serializer for our edge tuples: (index, index, value)
namespace boost { namespace serialization {

    template<class Archive, typename index_t, typename value_t>
    void serialize(Archive & ar, std::tuple<index_t, index_t, value_t> &t,
                   const unsigned int version) {
        ar & std::get<0>(t);
        ar & std::get<1>(t);
        ar & std::get<2>(t);
    }

} }


namespace skylark { namespace utility { namespace io {

//FIXME: move to io util header
namespace detail {

void parallelChunkedRead(
        const std::string& fname, boost::mpi::communicator &comm,
        int num_partitions, std::stringstream& data) {

    int rank = comm.rank();

    MPI_File file;
    int rc = MPI_File_open(comm, const_cast<char *>(fname.c_str()),
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
     if(rc)
        SKYLARK_THROW_EXCEPTION (
            base::io_exception() << base::error_msg("Unable to open file"));

    // First we just divide the file across all the available processors
    // in the communicator.
    MPI_Offset size;
    MPI_File_get_size(file, &size);

    size_t myStart = static_cast<size_t>(static_cast<double>(rank) *
                     (static_cast<double>(size) / num_partitions));

    size_t myEnd = static_cast<size_t>(static_cast<double>(rank + 1) *
                   (static_cast<double>(size) / num_partitions));

    if(rank == num_partitions)
        myEnd = size;

    size_t mySize  = myEnd - myStart;

    if(mySize > static_cast<size_t>(INT_MAX)) {
        std::ostringstream os;
        os << "ERROR: file " << fname << " cannot be opened."
           << std::endl
           << "Current implementation does not support one process "
           << "to read more than " << INT_MAX << " bytes (limited "
           << "by INT_MAX)."
           << std::endl;
        os << "To avoid this problem, use more processes to read the data."
           << std::endl;
        SKYLARK_THROW_EXCEPTION (
            base::io_exception() << base::error_msg(os.str()));
    }

    std::vector<char> buff(mySize);

    // Reading a portion of the file that is an initial guess of what
    // should belong to this process. Might not consist of entire lines,
    // balance later.
    MPI_Status readStatus;
    MPI_Offset offset = myStart;
    int err =
        MPI_File_read_at(file, offset, &buff[0], mySize, MPI_BYTE, &readStatus);
    if (err != MPI_SUCCESS)
        SKYLARK_THROW_EXCEPTION (
            base::io_exception()
                << base::error_msg("Error while MPI_File_read_ordered!"));


    data << std::string(buff.begin(), buff.end());

    //FIXME: why does this not work on the BGQ?
    //std::stringstream data("", std::ios::app | std::ios::out | std::ios::in);
    //data.rdbuf()->pubsetbuf(&buff[0], mySize);


    if(num_partitions > 1) {

        // now we go about "redistributing" (reading the correct offsets)
        // corresponding to the underlaying CombBLAS distribution.
        // 1) we need to figure out where our line ends and read the rest of the
        //    line.
        // 2) we need to find out what we own in the CombBLAS matrix and comm (?)
        //    the appropriate values

        MPI_Request recReq, sendReq;
        MPI_Status  recStatus, sendStatus;

        int myNumExtraBytes   = 0;
        int prevNumExtraBytes = 0;

        // pre-post receives
        if (rank != num_partitions - 1) {
            // expecting to receive the number of bytes until the first
            // endline character of the next portion
            MPI_Irecv(&myNumExtraBytes, 1, MPI_INT, rank + 1,
                      0, comm, &recReq);
        }

        // make sure that we have "full" lines on each processor
        if (rank == 0) {
            // and not sending anything just waiting for
            // the number of extra bytes to arrive
            MPI_Wait(&recReq, &recStatus);
        } else {
            std::string firstLine;
            // find the position of the first endline
            std::getline(data, firstLine);
            prevNumExtraBytes = data.tellg();
            if (data.eof() && rank != num_partitions - 1) {
                // this ranks data does not contain any line ending, so send
                // all my data preceding rank
                prevNumExtraBytes = mySize;

                MPI_Wait(&recReq, &recStatus);
                prevNumExtraBytes += myNumExtraBytes;

                myNumExtraBytes = 0;
                MPI_Isend(&prevNumExtraBytes, 1, MPI_INT, rank - 1,
                          0, comm, &sendReq);
            } else {
                // did find a line ending, send the number of bytes to
                // preceding rank
                MPI_Isend(&prevNumExtraBytes, 1, MPI_INT, rank - 1,
                          0, comm, &sendReq);
                if (rank == num_partitions - 1) myNumExtraBytes = 0;
                else MPI_Wait(&recReq, &recStatus);

                // strip the first prevNumExtraBytes from the stream
                //data.rdbuf()->pubsetbuf(&buff[prevNumExtraBytes],
                                        //mySize - prevNumExtraBytes);
            }
        }

        // Reading the extra bytes at the end.
        MPI_Status extraReadStatus;
        if (myNumExtraBytes > 0) {
            std::vector<char> extbuff(myNumExtraBytes);
            err = MPI_File_read_at(file, myEnd, &extbuff[0], myNumExtraBytes,
                                   MPI_BYTE, &extraReadStatus);
            if (err != MPI_SUCCESS)
                SKYLARK_THROW_EXCEPTION (
                    base::io_exception()
                        << base::error_msg("Error while MPI_file_read_at!"));

            // append the extra data
            data << std::string(extbuff.begin(), extbuff.end());
            //buff.insert(buff.end(), extbuff.begin(), extbuff.end());
            //data.rdbuf()->pubsetbuf(&buff[prevNumExtraBytes], mySize +
                                    //myNumExtraBytes - prevNumExtraBytes);
        }

        // waiting for the completion of all send requests
        if (rank != 0) MPI_Wait(&sendReq, &sendStatus);
    }

    MPI_File_close(&file);
}

} // namespace detail


/**
 *  Read arc list, assuming the file contains triplets (from, to, weigh)
 *  separated by spaces or tabs.
 *
 *  Current implementation reads a "fair" part of the file and determines from
 *  the read indices which portions it should get. This corresponds to
 *  computing a distributed index map and then do a final read to get the
 *  right values.
 *
 *  Determines an initial guess of where the current process should
 *  start reading the file. Can also be used to find where to stop
 *  using the rank following the rank of this process.
 *  Current implementation aims to divide the file into equal portions
 *  in bytes.
 *
 *  FIXME:
 *      - read for a subcommunicator...
 *      - 0-based vnodes s. 1-based nodes
 *      - relabeling?
 *      - on BG/Q there seems to be an issue with setting the rdbuf (why?)
 *
 *  @param fname input file name
 *  @param X output distributed sparse matrix
 *  @param comm communicator
 *  @param symmetrize make the matrix symmetric by returning (A + A')/2
 */
template <typename value_t>
void ReadArcList(const std::string& fname,
    base::sparse_vc_star_matrix_t<value_t>& X,
    boost::mpi::communicator &comm, bool symmetrize = false) {

    assert(X.is_finalized() == false);

    typedef std::tuple<El::Int, El::Int, value_t> tuple_type;
    typedef std::vector<tuple_type> edge_list_t;

    int rank = comm.rank();
    int num_partitions = comm.size();

    std::stringstream data;
    detail::parallelChunkedRead(fname, comm, num_partitions, data);

    // parse local buffer and insert edges into temporary list
    size_t max_row = 0, max_col = 0;
    edge_list_t edge_list;

    std::string line;
    std::getline(data, line);

    while (!data.eof()) {
        if (!data.good())
            SKYLARK_THROW_EXCEPTION (
                base::io_exception() << base::error_msg("Stream went bad!"));

        if (line[0] == '#')
            continue;

        std::vector<std::string> values;
        boost::split(values, line, boost::is_any_of("\t "));

        //XXX: maybe we need to trim left/right?
        //boost::trim_left_if(values[0],  boost::is_any_of("\t "));

        size_t from = 0, to = 0;

        try {
            from = boost::lexical_cast<size_t>(values[0]);
        } catch(boost::bad_lexical_cast &) {
            std::cerr << rank << ": cannot convert: \"" << values[0]
                      << "\" of line \""
                      << line << "\"" << std::flush << std::endl;
        }

        try {
            to = boost::lexical_cast<size_t>(values[1]);
        } catch(boost::bad_lexical_cast &) {
            std::cerr << rank << ": cannot convert: \"" << values[1]
                      << "\" of line \""
                      << line << "\"" << std::flush << std::endl;
        }

        value_t value = 1.0;
        if(values.size() > 2) {
            try {
                value = boost::lexical_cast<value_t>(values[2]);
            } catch(boost::bad_lexical_cast &) {
                std::cerr << rank << ": cannot convert: \"" << values[2]
                          << "\" of line \""
                          << line << "\"" << std::flush << std::endl;
            }
        }

        max_col = std::max(to, max_col);
        max_row = std::max(from, max_row);

        if (symmetrize) {
            edge_list.push_back(std::make_tuple(from, to, value / 2));
            edge_list.push_back(std::make_tuple(to, from, value / 2));
        } else
            edge_list.push_back(std::make_tuple(from, to, value));

        std::getline(data, line);
    }

    //XXX: boost::mpi::inplace_t was added in version 1.55
    size_t ncol = 0, nrow = 0;
    boost::mpi::all_reduce(comm, max_col, ncol, boost::mpi::maximum<size_t>());
    boost::mpi::all_reduce(comm, max_row, nrow, boost::mpi::maximum<size_t>());

    if (symmetrize) {
        size_t n = std::max(ncol, nrow);
        ncol = n;
        nrow = n;
    }

    //if(rank == 0)
    //    std::cout << "Read matrix of size " << nrow << " x " << ncol << std::endl;

    // FIXME: 0-based??
    X.resize(nrow + 1, ncol + 1);

    // finally we can redistribute the data, create a plan
    std::vector<edge_list_t> proc_set(comm.size());
    std::vector<size_t> proc_count(comm.size(), 0);
    typename edge_list_t::const_iterator itr;
    for(itr = edge_list.begin(); itr != edge_list.end(); itr++) {
        const size_t target_rank = X.owner(
                static_cast<El::Int>(get<0>(*itr)),
                static_cast<El::Int>(get<1>(*itr)));
        assert(target_rank < comm.size());
        proc_set[target_rank].push_back(*itr);
        proc_count[target_rank]++;
    }

    //XXX: what comm strategy: p2p, collective, one-sided?
    //XXX: what to send: send data or file offsets?

    // first communicate sizes that we will receive from other procs
    std::vector< std::vector<size_t> > vector_proc_counts;
    boost::mpi::all_gather(comm, proc_count, vector_proc_counts);

    size_t total_count = 0;
    for(size_t i = 0; i < vector_proc_counts.size(); ++i)
        total_count += vector_proc_counts[i][rank];

    // creating a local structure to hold sparse data triplets
    std::vector<tuple_type> matrix_data;
    try {
       matrix_data.resize(total_count);
    } catch (std::bad_alloc &e) {
        std::cout << "BAD ALLOC" << std::endl;
    }

    //FIXME: for now use synchronous comm, because Boost has an issue with
    //       comms:
    //  https://www.mail-archive.com/boost-mpi@lists.boost.org/msg00080.html
    //
    // pre-post receives
    //size_t idx = 0;
    std::vector<boost::mpi::request> requests(vector_proc_counts.size());
#if 0
    for(size_t i = 0; i < vector_proc_counts.size(); ++i) {
        // actual number of tuples we will receive from rank i
        size_t size = vector_proc_counts[i][rank];

        // pre-post receive for message of size of rank i
        requests[i] = comm.irecv(i, i, &(matrix_data[idx]), size);

        idx += size;
    }
#endif

    // send data
    for(size_t i = 0; i < proc_set.size(); ++i) {
        const edge_list_t &edges = proc_set[i];
        if(edges.size() == 0) continue;
        requests[i] = comm.isend(
                i, (i * comm.size()) + rank, &(edges[0]), edges.size());
    }

    size_t idx = 0;
    for(size_t i = 0; i < vector_proc_counts.size(); ++i) {
        // actual number of tuples we will receive from rank i
        size_t size = vector_proc_counts[i][rank];
        if(size == 0) continue;
        comm.recv(i, (rank * comm.size()) + i, &matrix_data[idx], size);
        idx += size;
    }

    // and wait for all requests to finish
    boost::mpi::wait_all(&requests[0], &requests[vector_proc_counts.size() - 1]);

    // insert all values the processor owns
    typename std::vector<tuple_type>::iterator matrix_itr;
    for(matrix_itr = matrix_data.begin(); matrix_itr != matrix_data.end();
        matrix_itr++) {

        // converts global indices to local
        X.queue_update(
            get<0>(*matrix_itr), get<1>(*matrix_itr), get<2>(*matrix_itr));
    }

    X.finalize();
}

template <typename value_t>
void ReadArcList(const std::string& fname,
    base::sparse_matrix_t<value_t>& X,
    boost::mpi::communicator &comm) {

    // TODO: temp!
    SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

}

template <typename T>
void ReadArcList(const std::string& fname,
    El::Matrix<T>& X, boost::mpi::communicator &comm) {

    // TODO: should we add this?
    SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

}

template <typename value_t, El::Distribution U, El::Distribution V>
void ReadArcList(const std::string& fname,
    El::DistMatrix<value_t, U, V>& X,
    boost::mpi::communicator &comm) {

    // TODO: should we add this?
    SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

}


} // namespace io
} // namespace util
} // namespace skylark


#endif // SKYLARK_ARC_LIST_HPP_

