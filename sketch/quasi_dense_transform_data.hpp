#ifndef SKYLARK_QUASI_DENSE_TRANSFORM_DATA_HPP
#define SKYLARK_QUASI_DENSE_TRANSFORM_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>
#include <boost/math/distributions.hpp>

#include "../utility/quasirand.hpp"
#include "boost/smart_ptr.hpp"

namespace skylark { namespace sketch {

namespace internal {

template <template <typename, typename> class Distribution,
          template <typename> class QMCSequenceType,
          typename ValueType>
struct quasi_random_samples_array_t {

public:

    typedef ValueType value_type;
    typedef Distribution<value_type, boost::math::policies::policy<> >
    distribution_type;
    typedef QMCSequenceType<value_type> sequence_type;

    quasi_random_samples_array_t() :
        _d(0), _N(0), _distribution(), _sequence(), _skip(0) {

    }

    quasi_random_samples_array_t(int d, int N,
        const distribution_type& distribution,
        const sequence_type& sequence, int skip = 0) :
        _d(d), _N(N), _distribution(distribution),
        _sequence(sequence), _skip(skip) {

    }



    quasi_random_samples_array_t(const quasi_random_samples_array_t& other) :
        _d(other._d), _N(other._N), _distribution(other._distribution),
        _sequence(other._sequence), _skip(other._skip) {

    }

    quasi_random_samples_array_t& operator=(const quasi_random_samples_array_t& other) {
        _d = other._d;
        _N = other._N;
        _distribution = other._distribution;
        _skip = other._skip;
        _sequence = other._sequence;
        return *this;
    }

    value_type operator[](size_t index) const {
        value_type baseval =
            _sequence.coordinate(_skip + (index / _d), index % _d);
        return boost::math::quantile(_distribution, baseval);
    }

private:
    size_t _d;
    size_t _N;
    distribution_type _distribution;
    sequence_type _sequence;
    size_t _skip;
};

}  // namespace internal

template <template <typename, typename> class ValueDistribution,
          template <typename> class QMCSequenceType>
struct quasi_dense_transform_data_t :
        public dense_transform_data_t<
    internal::quasi_random_samples_array_t<ValueDistribution,
                                           QMCSequenceType, double > > {

    // Note: we always generate doubles for array values,
    // but when applying to floats the size can be reduced.
    typedef double value_type;
    typedef QMCSequenceType<value_type> sequence_type;
    typedef ValueDistribution<value_type, boost::math::policies::policy<> >
    distribution_type;
    typedef internal::quasi_random_samples_array_t<ValueDistribution,
                                                   QMCSequenceType,
                                                   value_type> value_accessor_type;

    typedef dense_transform_data_t<value_accessor_type> base_t;

    /**
     * Regular constructor
     */
    quasi_dense_transform_data_t (int N, int S, double scale,
        const sequence_type& sequence, int skip,  base::context_t& context)
        : base_t(N, S, scale, context, "DistributionDenseTransform"),
          _skip(skip), _distribution(), _sequence(sequence) {

        // No scaling in "raw" form
        context = build();
    }

    virtual
    boost::property_tree::ptree to_ptree() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Do not yet support serialization of generic dense transform"));

        return boost::property_tree::ptree();
    }

    quasi_dense_transform_data_t(const quasi_dense_transform_data_t& other)
        : base_t(other), _distribution(other._distribution),
          _sequence(other._sequence) {

    }


protected:

    quasi_dense_transform_data_t (int N, int S, double scale,
        const sequence_type& sequence, int skip,
        const base::context_t& context, std::string type)
        : base_t(N, S, scale, context, type),
          _skip(skip), _distribution(), _sequence(sequence) {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();

        base_t::entries =
            value_accessor_type(base_t::_N, base_t::_S, _distribution,
                _sequence, _skip);

        return ctx;
    }

    int _skip;
    distribution_type _distribution; /**< Distribution for samples */
    sequence_type _sequence;

};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_QUASI_DENSE_TRANSFORM_DATA_HPP */
