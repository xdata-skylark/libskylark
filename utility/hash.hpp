#ifndef SKYLARK_HASH_HPP
#define SKYLARK_HASH_HPP

#include <boost/functional/hash.hpp>

namespace skylark {
namespace utility {

/**
 * A class that defines an hash function for pair of objects.
 */
struct pair_hasher_t {
    template<typename S, typename T>
    inline size_t operator()(const std::pair<S, T> & v) const {
        size_t seed = 0;
        boost::hash_combine(seed, v.first);
        boost::hash_combine(seed, v.second);
        return seed;
    }
};


} }  // namespace skylark::utility

#endif // SKYLARK_HASH_HPP
