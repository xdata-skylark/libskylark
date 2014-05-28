#ifndef SKYLARK_DENSE_TRANSFORM_HPP
#define SKYLARK_DENSE_TRANSFORM_HPP

#include "../config.h"
#include "dense_transform_data.hpp"

namespace skylark { namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename> class ValueDistribution>
class dense_transform_t {
    // Concrete transforms, like JLT, can derive this class and set the scale.
    // This enables also adding parameters to constuctor, adding methods,
    // renaming the class.

    // Without deriving, the scale should be 1.0, so this is just a
    // random matrix with enteries from the specified distribution.
    // sets a scale variable, but can add methods.

    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;
    typedef dense_transform_data_t<ValueDistribution> data_type;

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

private:
    dense_transform_t(int N, int S, base::context_t& context);
};


} } /** namespace skylark::sketch */

#if SKYLARK_HAVE_ELEMENTAL
#include "dense_transform_Elemental.hpp"
#endif

#endif // SKYLARK_DENSE_TRANSFORM_HPP
