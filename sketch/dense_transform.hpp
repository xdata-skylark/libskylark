#ifndef SKYLARK_DENSE_TRANSFORM_HPP
#define SKYLARK_DENSE_TRANSFORM_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

namespace skylark { namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename> class ValueDistribution>
class dense_transform_t {

    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;
    typedef dense_transform_data_t<ValueDistribution> data_type;

    dense_transform_t(int N, int S, base::context_t& context) 
        : data_type(N, S, context) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for DenseTransform"));
    }

    dense_transform_t(const data_type& other_data)
        : data_type(other_data) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for DenseTransform"));
    }

    dense_transform_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for DenseTransform"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for DenseTransform"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for DenseTransform"));
    }
};


} } /** namespace skylark::sketch */

#if SKYLARK_HAVE_ELEMENTAL
#include "dense_transform_Elemental.hpp"
#endif

#endif // SKYLARK_DENSE_TRANSFORM_HPP
