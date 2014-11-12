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
          typename ValueType>
struct quasi_random_samples_array_t {

public:

    typedef ValueType value_type;
    typedef Distribution<value_type, boost::math::policies::policy<> >
    distribution_type;

    quasi_random_samples_array_t() :
        _d(0), _N(0), _distribution(), _skip(0), _qmc_sequence() {

    }

    quasi_random_samples_array_t(int d, int N,
        const distribution_type& distribution,
        int skip = 0, int  leap = -1) :
        _d(d), _N(N), _distribution(distribution),
        _skip(skip), _qmc_sequence(d + 1, leap) {

    }



    quasi_random_samples_array_t(const quasi_random_samples_array_t& other) :
        _d(other._d), _N(other._N), _distribution(other._distribution),
        _qmc_sequence(other._qmc_sequence) {

    }

    quasi_random_samples_array_t& operator=(const quasi_random_samples_array_t& other) {
        _d = other._d;
        _N = other._N;
        _distribution = other._distribution;
        _skip = other._skip;
        _qmc_sequence = other._qmc_sequence;
        return *this;
    }

    value_type operator[](size_t index) const {
        size_t skpidx = index + _skip;
        value_type baseval = _qmc_sequence.coordinate(skpidx / _d, skpidx % _d);
        return boost::math::quantile(_distribution, baseval);
    }

private:
    size_t _d;
    size_t _N;
    distribution_type _distribution;
    size_t _skip;
    utility::leaped_halton_sequence_t<value_type> _qmc_sequence;
};

}  // namespace internal

template <template <typename, typename> class ValueDistribution>
struct quasi_dense_transform_data_t :
        public dense_transform_data_t<
    internal::quasi_random_samples_array_t< ValueDistribution, double > > {

    // Note: we always generate doubles for array values,
    // but when applying to floats the size can be reduced.
    typedef double value_type;
    typedef ValueDistribution<value_type, boost::math::policies::policy<> >
    distribution_type;
    typedef internal::quasi_random_samples_array_t<ValueDistribution, value_type>
        value_accessor_type;

    typedef dense_transform_data_t<value_accessor_type> base_t;

    /**
     * Regular constructor
     */
    quasi_dense_transform_data_t (int N, int S, double scale,
        int skip, int leap, base::context_t& context)
        : base_t(N, S, scale, context, "DistributionDenseTransform"),
          _skip(skip), _leap(leap), _distribution() {

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
        : base_t(other), _distribution(other._distribution) {

    }


protected:

    quasi_dense_transform_data_t (int N, int S, double scale,
        int skip, int leap, const base::context_t& context, std::string type)
        : base_t(N, S, scale, context, type),
          _skip(skip), _leap(leap), _distribution() {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();

        base_t::random_samples =
            value_accessor_type(base_t::_N, base_t::_S, _distribution,
                _skip, _leap);

        return ctx;
    }

    int _skip, _leap;
    distribution_type _distribution; /**< Distribution for samples */


};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_QUASI_DENSE_TRANSFORM_DATA_HPP */
