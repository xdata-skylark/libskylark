#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "../utility/randgen.hpp"
#include <boost/mpi.hpp>

namespace skylark { namespace sketch {

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
private:
    /// Internal counter identifying the start of next stream of random numbers
    int _counter;
    /// The seed used for initializing the context
    int _seed;

public:
    /**
     * Initialize context with a seed and the communicator.
     * @param[in] seed Random seed to be used for all computations.
     * @param[in] orig Communicator that is duplicated and used with SKYLARK.
     *
     * @caveat This is a global operation since all MPI ranks need to
     * participate in the duplication of the communicator.
     */
    context_t (int seed,
               const boost::mpi::communicator& orig) :
        comm(orig, boost::mpi::comm_duplicate),
        rank(comm.rank()),
        size(comm.size()),
        _counter(0),
        _seed(seed) {}

    /**
     * Returns pointer to a meta random number generator.
     * This can later be used as a random access array of generators.
     * @param[in] size The size of the array of generators provided.
     *
     * @caveat This should be used as a global operation to keep the
     * the internal state of the context synchronized.
     */
    skylark::utility::rng_array_t* allocate_rng_array(int size) {
        skylark::utility::rng_array_t* rng_array_ptr =
            new skylark::utility::rng_array_t(_counter, size, _seed);
        _counter = _counter + size;
        return rng_array_ptr;
    }

    /**
     * Returns an integer random number.
     *
     * It uses the meta random generator to make sure the context is informed.
     *
     * @caveat This should be used as a global operation to keep the
     * the internal state of the context synchronized.
     */
     int random_int() {
        skylark::utility::rng_array_t* rng_array_ptr = allocate_rng_array(1);
        int sample = static_cast<int>(((*rng_array_ptr)[0])());
        delete rng_array_ptr;
        return sample;
    }
};

} } /** skylark::sketch */

#endif // CONTEXT_HPP
