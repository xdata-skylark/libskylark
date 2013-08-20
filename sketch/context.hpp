#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <boost/random.hpp>
#include <boost/mpi.hpp>

namespace skylark {
namespace sketch {

/**
 * A structure that holds basic information about the MPI world and what the
 * user wants to implement.
 */
struct context_t {
    /// Communicator to use for MPI
    boost::mpi::communicator comm;
    /// Rank of the current process
    int rank;
    /// Number of processes in the group
    int size;
    /** PRNG that generates seeds to define the transforms.
      * Should be seeded the same across ranks, and always have
      * the calls coordinated */
    boost::random::mt19937 prng;

    /**
     * Initilize context with a seed and the communicator.
     * @param[in] seed Random seed to be used for all computations.
     * @param[in] orig Communicator that is duplicated and used with SKYLARK i
     *
     * @caveat This is a global operation since all MPI ranks need to
     * participate in the duplication of the communicator.
     */
    context_t (int seed,
               const boost::mpi::communicator& orig) :
        comm(orig, boost::mpi::comm_duplicate),
        rank(comm.rank()),
        size(comm.size()),
        prng(seed) {}

    /**
     * Return a new seed, that can be used to generate an independent stream.
     * Or at least we hope the stream is independent (need to read on PRNG).
     */
    int newseed() {
        return prng();
    }
};

} // namespace sketch
} // namespace skylark

#endif // CONTEXT_HPP
