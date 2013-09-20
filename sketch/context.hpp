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
     * Returns a container of samples drawn from a distribution
     * to be accessed as an array.
     * @param[in] size The size of the container.
     * @param[in] distribution The distribution to draw samples from.
     * @return Random samples' container.
     *
     * @caveat This should be used as a global operation to keep the
     * the internal state of the context synchronized.
     */
    template <typename ValueType,
              typename Distribution>
    skylark::utility::random_samples_array_t<ValueType, Distribution>
    allocate_random_samples_array(int size,
        Distribution& distribution) {
        skylark::utility::rng_array_t rng_array(_counter, size, _seed);
        _counter = _counter + size;
        return skylark::utility::random_samples_array_t<ValueType, Distribution>
            (rng_array, distribution);
    }


    /**
     * Returns a container of random numbers to be accessed as an array.
     * @param[in] size The size of the container.
     * @return Random numbers' container.
     *
     * @caveat This should be used as a global operation to keep the
     * the internal state of the context synchronized.
     */
    skylark::utility::random_array_t allocate_random_array(int size) {
        skylark::utility::rng_array_t rng_array(_counter, size, _seed);
        _counter = _counter + size;
        return skylark::utility::random_array_t(rng_array);
    }


    /**
     * Returns an integer random number.
     * @return Random integer number.
     *
     * @todo Temporarily used as a replacement for newseed().
     *
     * @caveat This should be used as a global operation to keep the
     * the internal state of the context synchronized.
     */
     int random_int() {
         skylark::utility::random_array_t random_array =
             allocate_random_array(1);
         int sample = random_array[0];
         return sample;
    }
};

} } /** skylark::sketch */

#endif // CONTEXT_HPP
