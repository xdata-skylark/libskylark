#ifndef RANDGEN_HPP
#define RANDGEN_HPP

#include <Random123/threefry.h>
#include <Random123/MicroURNG.hpp>

namespace skylark { namespace utility {

/**
 * Convenience nicknames for Random123 types
 */
typedef r123::Threefry4x64 RNG_t;
typedef RNG_t::ctr_type ctr_t;
typedef RNG_t::key_type key_t;
typedef r123::MicroURNG<RNG_t> URNG_t;


/*
 * Helper routine for handling array initializer in pre C++11 compilers
 */
static key_t _seed_to_key(int seed) {
    key_t key;
    key.v[0] = seed;
    return key;
}


/**
 * Random-access array of random generators
 */
struct rng_array_t {

public:

    rng_array_t()
        : _base(0), _size(0), _key(_seed_to_key(0)) {}

    rng_array_t(int base, int size, int seed)
        : _base(base), _size(size), _key(_seed_to_key(seed)) {}

    URNG_t operator[] (int index) const {
        // TODO: Assert the ranges of the bounds
        ctr_t ctr;
        // The high 32 bits of the highest word in ctr must be zero.
        // MicroURNG uses these bits to "count".
        for(int i = 0; i < 4; i++)
            ctr.v[i] = 0;
        ctr.v[0] = _base + index;
        URNG_t urng(ctr, _key);
        return urng;
    }

private:
    // TODO: Useful for bounds' assertion
    // TODO: Enforce constness
    int _base;
    int _size;
    key_t _key;
};


/**
 * Random-access array of samples drawn from a distribution.
 * It is templated over the types of each sample value and the distribution.
 */
template <typename ValueType,
          typename Distribution>
struct random_samples_array_t {

public:

    random_samples_array_t() {}

    random_samples_array_t(rng_array_t& rng_array,
        Distribution& distribution)
        : _rng_array(rng_array), _distribution(distribution) {}

    ValueType operator[](int index) {
        URNG_t urng = _rng_array[index];
        return _distribution(urng);
    }

private:
    rng_array_t _rng_array;
    Distribution _distribution;
};


/**
 * Random-access array of random numbers.
 */
struct random_array_t {

public:

    random_array_t() {}

    random_array_t(rng_array_t& rng_array)
        : _rng_array(rng_array) {}

    int operator[](int index) {
        URNG_t urng = _rng_array[index];
        return urng();
    }

private:
    rng_array_t _rng_array;
};

} } /** skylark::utility */

#endif // RANDGEN_HPP
