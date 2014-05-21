#ifndef SKYLARK_CONTEXT_HPP
#define SKYLARK_CONTEXT_HPP

#include "../config.h"

#include "exception.hpp"
#include "../utility/randgen.hpp"

#include "boost/smart_ptr.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

namespace skylark { namespace base {

/**
 * A structure that holds basic information about the state of the
 * random number stream.
 */
struct context_t {
    /**
     * Initialize context with a seed.
     * @param[in] seed Random seed to be used for all computations.
     */
    context_t (int seed, int counter=0) :
        _counter(counter),
        _seed(seed) {}

#if 0
    context_t (context_t&& ctxt) :
        _counter(std::move(ctxt._counter)), _seed(std::move(ctxt._seed))
    {}

    context_t(const context_t& other) {
        _seed    = other._seed;
        _counter = other._counter;
    }

    context_t& operator=(const context_t& other) {
        _seed    = other._seed;
        _counter = other._counter;
        return *this;
    }
#endif

    /**
     * Load context from a serialized JSON structure.
     * @param[in] filename of JSON structure encoding serialized state.
     */
    context_t (const boost::property_tree::ptree& json) {
        _counter = json.get<size_t>("seed");
        _seed = json.get<int>("counter");
    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "context");
        pt.put("skylark_version", VERSION);
        pt.put("seed", _seed);
        pt.put("counter", _counter);
        return pt;
    }


    /**
     * Returns a container of samples drawn from a distribution
     * to be accessed as an array.
     * @param[in] size The size of the container.
     * @param[in] distribution The distribution to draw samples from.
     * @return Random samples' container.
     *
     * @details This is the main facility for creating a "stream" of
     * samples of given size and distribution. size is needed for
     * reserving up-front a portion of the linear space of the 2^64 samples
     * that can be provided by a context with a fixed seed.
     *
     * @internal We currently use Random123 library, most specifically
     * Threefry4x64 counter-based generator, as wrapped by the uniform
     * random generator, MicroURNG. For each sample we instantiate
     * a MicroURNG instance. Each such instance needs 2 arrays of 4 uint64
     * numbers each, namely a counter and a key: we successively increment only
     * the first uint64 component in counter (counter[0]) and fix key to be
     * the seed. This instance is then passed to the distribution. This
     * in turn calls operator () on the instance and increments counter[3]
     * accordingly (for multiple calls), thus ensuring the independence
     * of successive samples. operator () can either trigger a run of
     * the Threefry4x64 algorithm for creating a fresh result array
     * (also consisting of 4 uint64's) and use one or more of its components or
     * use those components from a previous run that are not processed yet.
     *
     * @caveat This should be used as a global operation to keep the
     * the internal state of the context synchronized.
     */
     template <typename Distribution>
     skylark::utility::random_samples_array_t<Distribution>
     allocate_random_samples_array(size_t size, Distribution& distribution) {
         skylark::utility::random_samples_array_t<Distribution>
             random_samples_array(_counter, size, _seed, distribution);
         _counter += size;
         return random_samples_array;
     }

    /**
     * Returns a vector of samples drawn from a distribution.
     * @param[in] size The size of the vector.
     * @param[in] distribution The distribution to draw samples from.
     * @return Random samples' vector.
     */
    template <typename Distribution >
    std::vector<typename Distribution::result_type>
      generate_random_samples_array(size_t size,
        Distribution& distribution) {
        skylark::utility::random_samples_array_t<Distribution>
            allocated_random_samples_array(_counter, size, _seed, distribution);
        _counter += size;
        std::vector<typename Distribution::result_type> random_samples_array;
        try {
            random_samples_array.resize(size);
        } catch (std::bad_alloc ba) {
            SKYLARK_THROW_EXCEPTION (
                base::allocation_exception()
                    << base::error_msg(ba.what()) );
        }
        for(size_t i = 0; i < size; i++) {
            random_samples_array[i] = allocated_random_samples_array[i];
        }
        return random_samples_array;
    }


    /**
     * Returns a container of random numbers to be accessed as an array.
     * @param[in] size The size of the container.
     * @return Random numbers' container.
     *
     * @caveat This should be used as a global operation to keep the
     * the internal state of the context synchronized.
     */
    skylark::utility::random_array_t allocate_random_array(size_t size) {
        skylark::utility::random_array_t random_array(_counter, size, _seed);
        _counter += size;
        return random_array;
    }


    /**
     * Returns an integer random number.
     * @return Random integer number.
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


    size_t get_counter() { return _counter; }

    /**
     * Serializes the context to a JSON structure.
     * @param[out] JSON encoded state of the context.
     */
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk, const context_t &data);
private:

    /// Internal counter identifying the start of next stream of random numbers
    size_t _counter;
    /// The seed used for initializing the context
    int _seed;
};

boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const context_t &data) {
        sk.put("sketch.context.seed", data._seed);
        sk.put("sketch.context.counter", data._counter);
        return sk;
}

} } /** namespace skylark::base */

#endif // SKYLARK_CONTEXT_HPP
