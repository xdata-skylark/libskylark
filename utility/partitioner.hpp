#ifndef PARTITIONER_HPP
#define PARTITIONER_HPP

#include <utility>

#include "config.h"

namespace skylark {
namespace utility {

template <typename IntegralType>
struct partitioner_t {
    /* Define a type for the iteration space */
    typedef std::pair<IntegralType, IntegralType> space_type;

    /**
     * Begin and end are given as [begin,end), so end is NOT included. Rank and
     * size help in determining the subrange of this particular range_partitioner
     */
    static space_type create (const IntegralType& begin,
                              const IntegralType& end,
                              const int rank = 0,
                              const int size = 1) {
        const IntegralType range = (end - begin);
        const IntegralType chunk = (range / size);
        const IntegralType leftover = (range % size);
        const IntegralType my_portion = chunk + ((rank < leftover) ? 1 : 0);
        const IntegralType my_begin =
            begin + chunk * rank + ((rank < leftover) ? rank : leftover);
        const IntegralType my_end = my_begin + my_portion;

        return space_type (my_begin, my_end);
    }

    /**
     * @param[in] begin  The starting point of the range.
     * @param[in] end    The ending of the range (not included).
     * @param[in] size   The number of partitions to create.
     * @param[in] result The array to write the intervals out.
     *
     * This procedure creates a running sum of intervals so that a binary search
     * can later be applied to find out the owner of each chunk of range points.
     */
    template <typename OutputIterator>
    static void intervals (const IntegralType& begin,
                           const IntegralType& end,
                           const int size,
                           OutputIterator result) {

        for (int i = 0; i < size; ++i) {
            space_type current_space = create (begin, end, i, size);
            *result++ = current_space.first;
        }
        *result = end;
    }

    /**
     * @param[in] begin  The starting point of the range.
     * @param[in] end    The ending of the range (not included).
     * @param[in] size   The number of partitions to create.
     * @param[in] result The array to write the counts out.
     *
     * This procedure creates an array, where each entry contains the range of
     * elements computed by each rank.
     */
    template <typename OutputIterator>
    static void counts (const IntegralType& begin,
                        const IntegralType& end,
                        const int size,
                        OutputIterator result) {

        for (int i = 0; i < size; ++i) {
            space_type current_space = create (begin, end, i, size);
            *result++ = current_space.second - current_space.first;
        }
    }
};

} // namespace utility
} // namespace skylark

#endif // PARTITIONER_HPP
