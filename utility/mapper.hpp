#ifndef MAPPER_HPP
#define MAPPER_HPP

#include <utility>
#include <iterator>
#include <vector>
#include <algorithm>

namespace skylark {
namespace utility {

/* A mapper that can take in a number and return which range it is in */
template <typename ValueType>
struct interval_mapper_t {
    typedef ValueType value_type;

    /** A structure to compare two values and return which occurs before */
    struct less_than_or_equal {
        bool operator()(const value_type& a,
                        const value_type& b) const {
            return !(a > b);
        }
    };

private:
    /** variables that are required */
    std::vector<value_type> interval_map;
    less_than_or_equal compare;

public:
    /**
     * Copy construct from another mapper.
     */
    interval_mapper_t (const interval_mapper_t& other) :
        interval_map (other.interval_map) {}

    /**
     * Construct from a given range of numbers.
     */
    template <typename InputIterator>
    interval_mapper_t (InputIterator map_first,
                       InputIterator map_last) :
        interval_map (map_first, map_last) {}

    /**
     * Construct from an iterator and size.
     */
    template <typename InputIterator>
    interval_mapper_t (InputIterator map_first,
                       ValueType num_elements) : interval_map(num_elements) {
        InputIterator map_last (map_first);
        std::advance (map_last, num_elements);
        std::copy (map_first, map_last, interval_map.begin());
    }

    /**
     * Initialize from a given range.
     */
    template <typename InputIterator>
    void set (InputIterator map_first, InputIterator map_last) {
        interval_map.resize (map_last - map_first);
        std::copy (map_first, map_last, interval_map.begin());
    }

    /**
     * Return the owner of the requested element.
     */
    value_type operator()(const value_type& number) const {
        return (std::lower_bound(interval_map.begin(),
                                 interval_map.end(),
                                 number,
                                 compare)) - interval_map.begin() - 1;
    }

    /**
     * Give the range of the given element.
     */
    value_type range (const value_type& number) const {
        return (interval_map[number + 1] - interval_map[number]);
    }

    /**
     * Give the start of the partition.
     */
    value_type begin (const value_type& number) const {
        return (interval_map[number]);
    }

    /**
     * Give the end of the partition.
     */
    value_type end (const value_type& number) const {
        return (interval_map[number + 1]);
    }

    /**
     * Print the ownership information.
     */
    void pretty_print () const {
        std::cout << "Ownership map = ";
        typename std::vector<value_type>::const_iterator iter = interval_map.begin();
        while (iter != interval_map.end()) std::cout << *iter++ << " ";
        std::cout << std::endl;
    }
};

} // namespace utility
} // namespace skylark

#endif // MAPPER_HPP
