#ifndef SKYLARK_QUASI_DENSE_TRANSFORM_DATA_HPP
#define SKYLARK_QUASI_DENSE_TRANSFORM_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

#include "../utility/randgen.hpp"
#include "boost/smart_ptr.hpp"

namespace skylark { namespace sketch {

namespace internal {

template <typename Distribution>
struct quasi_random_samples_array_t {

public:

    typedef typename Distribution::result_type value_type;


    quasi_random_samples_array_t() :
        _d(0), _N(0), _skip(0), _leap(0), _distribution() {

    }

    quasi_random_samples_array_t(int d, int N, int skip, int  leap, 
        const Distribution& distribution) :
        _d(d), _N(N), _skip(skip), _leap(leap), _distribution(distribution) {
        _distribution.reset();
    }



    quasi_random_samples_array_t(const quasi_random_samples_array_t& other) :
        _d(other._d), _N(other._N), _skip(other._skip), _leap(other._leap),
        _distribution(other._distribution) {

    }

    quasi_random_samples_array_t& operator=(const quasi_random_samples_array_t& other) {
        _d = other._d;
        _N = other._N;
        _skip = other._skip;
        _leap = other._leap;
        _distribution = other._distribution;
        return *this;
    }

    /**
     * @internal The samples could be generated during the sketch apply().
     * apply() are const methods so this [] operator should be const too.
     * A distribution object however as provided e.g. by boost may modify its
     * state between successive invocations of the passed in generator object.
     * (e.g. normal distribution). So the reason for copying is the
     * const-correctness.
     */
    value_type operator[](size_t index) const {
        return 0.0;
    }

private:
    size_t _d;
    size_t _N;
    size_t _skip;
    size_t _leap;
    Distribution _distribution;
};

}  // namespace internal

template <template <typename> class ValueDistribution>
struct quasi_dense_transform_data_t :
        public dense_transform_data_t<
    internal::quasi_random_samples_array_t< ValueDistribution<double> > > {

    // Note: we always generate doubles for array values,
    // but when applying to floats the size can be reduced.
    typedef double value_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef internal::quasi_random_samples_array_t<value_distribution_type>
        value_accessor_type;

    typedef dense_transform_data_t<value_accessor_type> base_t;

    /**
     * Regular constructor
     */
    quasi_dense_transform_data_t (int N, int S, double scale,
        int skip, int leap, base::context_t& context)
        : base_t(N, S, scale, context, "DistributionDenseTransform"),
          _skip(skip), _leap(leap), distribution() {

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
        : base_t(other), distribution(other.distribution) {

    }


protected:

    quasi_dense_transform_data_t (int N, int S, double scale,
        int skip, int leap, const base::context_t& context, std::string type)
        : base_t(N, S, scale, context, type),
          _skip(skip), _leap(leap), distribution() {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();

        base_t::random_samples =
            value_accessor_type(base_t::_N, base_t::_S, _skip, _leap,
                distribution);

        return ctx;
    }

    int _skip, _leap;
    value_distribution_type distribution; /**< Distribution for samples */


};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_QUASI_DENSE_TRANSFORM_DATA_HPP */
