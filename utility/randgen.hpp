#ifndef RANDGEN_HPP
#define RANDGEN_HPP

#include <sstream>

#include <Random123/threefry.h>
#include <Random123/MicroURNG.hpp>

namespace skylark { namespace utility {


/**
 * Random-access array of samples drawn from a distribution.
 * It is templated over the types of each sample value and the distribution.
 */
template <typename ValueType,
          typename Distribution>
struct random_samples_array_t {

public:
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
        for(int i = 1; i < 4; i++)
            key.v[i] =  static_cast<key_t::value_type>(0);
        key.v[0] = static_cast<key_t::value_type>(seed);
        return key;
    }

    /**
     * Random-access array of samples drawn from a distribution.
     * @param[in] base Start location within a global stream.
     * @param[in] size The number of samples provided.
     * @param[in] seed The seed for the array.
     * @param[in] distribution Distribution from which samples are drawn.
     *
     * @internal The seed serves as the identifier for the global stream.
     * The i-th element in the array is essentially the (base + i)-th
     * element in this global stream. i should be in [0, size).
     *
     */
    random_samples_array_t(size_t base, size_t size, int seed,
        Distribution& distribution)
        : _base(base), _size(size),
          _key(_seed_to_key(seed)),
          _distribution(distribution) {
        if(std::numeric_limits<size_t>::max() - base < size - 1) {
            std::ostringstream msg;
            msg << "Usupported operation:\n";
            msg << "The random number generator stream cannot supply the ";
            msg << "requested number of samples without being exhausted\n";
            throw std::runtime_error( msg.str() );
        }
        _distribution.reset();
    }

    /**
     * @internal The samples could be generated during the sketch apply().
     * apply() are const methods so this [] operator should be const too.
     * A distribution object however as provided e.g. by boost may modify its
     * state between successive invocations of the passed in generator object.
     * (e.g. normal distribution). So the reason for copying is the
     * const-correctness.
     */
    ValueType operator[](size_t index) const {
        // Could be more specific exception std::out_of_range
        if( index >= _size ) {
            std::ostringstream msg;
            msg << "Index is out of bounds:\n";
            msg << "index = " << index << " not in expected ";
            msg << "[0, " << _size << ") range\n";
            throw std::logic_error( msg.str() );
        }
        ctr_t ctr;
        for(int i = 1; i < 4; i++)
            ctr.v[i] = static_cast<ctr_t::value_type>(0);
        ctr.v[0] = static_cast<ctr_t::value_type>(_base + index);
        URNG_t urng(ctr, _key);
        Distribution cloned_distribution = _distribution;
        return cloned_distribution(urng);
    }

private:
    const size_t _base;
    const size_t _size;
    const key_t _key;
    Distribution _distribution;
};


/**
 * Random-access array of random numbers.
 */
struct random_array_t {

public:
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
        for(int i = 1; i < 4; i++)
            key.v[i] =  static_cast<key_t::value_type>(0);
        key.v[0] = static_cast<key_t::value_type>(seed);
        return key;
    }

    /**
     * Random-access array of random numbers.
     * @param[in] base Start location within a global stream.
     * @param[in] size The number of random numbers provided.
     * @param[in] seed The seed for the array.
     *
     * @todo Bounds checking.
     */

    random_array_t(size_t base, size_t size, int seed)
        : _base(base), _size(size), _key(_seed_to_key(seed)) {
        if(std::numeric_limits<size_t>::max() - base + 1 < size) {
            std::ostringstream msg;
            msg << "Usupported operation:\n";
            msg << "The random number generator stream cannot supply the ";
            msg << "requested number of samples without being exhausted\n";
            throw std::runtime_error( msg.str() );
        }
    }

    int operator[](size_t index) const {
        // Could be more specific exception std::out_of_range
        if( index >= _size ) {
            std::ostringstream msg;
            msg << "Index is out of bounds:\n";
            msg << "index = " << index << " not in expected ";
            msg << "[0, " << _size << ") range\n";
            throw std::logic_error( msg.str() );
        }
        ctr_t ctr;
        for(int i = 1; i < 4; i++)
            ctr.v[i] = static_cast<ctr_t::value_type>(0);
        ctr.v[0] = static_cast<ctr_t::value_type>(_base + index);
        URNG_t urng(ctr, _key);
        return urng();
    }

private:
    const size_t _base;
    const size_t _size;
    const key_t _key;
};

} } /** skylark::utility */

#endif // RANDGEN_HPP
