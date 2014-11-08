#ifndef SKYLARK_DIST_DENSE_TRANSFORM_HPP
#define SKYLARK_DIST_DENSE_TRANSFORM_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

namespace skylark { namespace sketch {

/**
 * Just an adaptor for the template type
 */
template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename> class DistributionType>
class dist_dense_transform_t :
        public dense_transform_t<InputMatrixType, OutputMatrixType,
         utility::random_samples_array_t<DistributionType<double> > >  {

public:

    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;

    // We use composition to defer calls to dense_transform_t
    typedef dense_transform_t<InputMatrixType, OutputMatrixType,
         utility::random_samples_array_t<DistributionType<double> >  > base_t;
    typedef dist_dense_transform_data_t<DistributionType> data_type;

    dist_dense_transform_t(int N, int S, double scale, base::context_t& context)
        : base_t(N, S, scale, context) {

    }

    dist_dense_transform_t (const dist_dense_transform_t<matrix_type,
        output_matrix_type,
        DistributionType>& other)
        : data_type(other) {

    }

    dist_dense_transform_t(const data_type& other_data)
        : data_type(other_data) {

    }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DIST_DENSE_TRANFORM
