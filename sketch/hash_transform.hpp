#ifndef SKYLARK_HASH_TRANSFORM_HPP
#define SKYLARK_HASH_TRANSFORM_HPP

#include "../config.h"
#include "../utility/distributions.hpp"

#include "transforms.hpp"
#include "hash_transform_data.hpp"

namespace skylark { namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename> class IdxDistributionType,
           template <typename> class ValueDistribution
        >
struct hash_transform_t {
    // To be specialized and derived. Just some guards here.
    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;
    typedef hash_transform_data_t<IdxDistributionType, ValueDistribution>
    data_type;

    hash_transform_t(int N, int S, base::context_t& context) 
        : data_type(N, S, context) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for HashTransform"));
    }

    hash_transform_t(const data_type& other_data)
        : data_type(other_data) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for HashTransform"));
    }

    hash_transform_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for HashTransform"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for HashTransform"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for HashTransform"));
    }
};

} } /** namespace skylark::sketch */

#if SKYLARK_HAVE_ELEMENTAL
#include "hash_transform_Elemental.hpp"
#endif

#if SKYLARK_HAVE_COMBBLAS
#include "hash_transform_CombBLAS.hpp"
#endif

#if SKYLARK_HAVE_ELEMENTAL and SKYLARK_HAVE_COMBBLAS
#include "hash_transform_Mixed.hpp"
#endif

#include "hash_transform_local_sparse.hpp"

#endif // SKYLARK_HASH_TRANSFORM_HPP
