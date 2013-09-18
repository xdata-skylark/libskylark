#ifndef RANDGEN_HPP
#define RANDGEN_HPP

#include <Random123/threefry.h>
#include <Random123/MicroURNG.hpp>

namespace skylark { namespace utility {

typedef r123::Threefry4x64 RNG_t;
typedef RNG_t::ctr_type ctr_t;
typedef RNG_t::key_type key_t;
typedef r123::MicroURNG<RNG_t> URNG_t;

/**
 * Random-access array of random generators
 **/
struct rng_array_t {
    int _base;
    int _size;
    ctr_t _ctr;
    key_t _key;
    int _seed;

    rng_array_t(int base, int size, int seed) {
        // TODO
        // If base + size > max then seed++
        _base = base;
        _size = size;
        _seed = seed;
        for(int i = 0; i < 4; i++) {
            _ctr.v[i] = 0;
        }
        _key.v[0] = _seed;
    }

    URNG_t operator[](int index) {
        // TODO
        // Assert the ranges of the bounds
        _ctr.v[0] = _base + index;
        URNG_t urng(_ctr, _key);
        return urng;
    }
};

} } /** skylark::utility */

#endif // RANDGEN_HPP
