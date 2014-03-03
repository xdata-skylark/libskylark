#ifndef SKYLARK_PPT_DATA_HPP
#define SKYLARK_PPT_DATA_HPP

#include <vector>

#include "../utility/distributions.hpp"
#include "context.hpp"
#include "CWT_data.hpp"

namespace skylark { namespace sketch {

/**
 * Pham-Pagh Transform aka TensorSketch (data).
 *
 * Sketches the monomial expansion of a vector.
 *
 * See:
 * Ninh Pham and Rasmus Pagh
 * Fast and Scalable Polynomial Kernels via Explicit Feature Maps
 * KDD 2013
 */
template <typename ValueType>
struct PPT_data_t {


    /**
     * Regular constructor
     */
    PPT_data_t (int N, int S,
        int q, double c, double gamma,
        skylark::sketch::context_t& context)
        : _N(N), _S(S), _q(q), _c(c), _gamma(gamma), _context(context) {
        for(int i = 0; i < q; i++)
            _cwts_data.push_back(_CWT_data_t(N, S, context));
        boost::random::uniform_int_distribution<int> distidx(0, S-1);
        _hash_idx = context.generate_random_samples_array(q, distidx);
        utility::rademacher_distribution_t<double> distval;
        _hash_val = context.generate_random_samples_array(q, distval);
    }

protected:

    typedef CWT_data_t<size_t, ValueType> _CWT_data_t;

    const int _N;         /**< Input dimension  */
    const int _S;         /**< Output dimension  */
    const int _q;         /**< Polynomial degree */
    const double _c;
    const double _gamma;

    // Hashing info for the homogenity parameter c
    std::vector<int> _hash_idx;
    std::vector<double> _hash_val;


    skylark::sketch::context_t& _context; /**< Context for this sketch */
    std::list< _CWT_data_t > _cwts_data;
  };

} } /** namespace skylark::sketch */

#endif /** SKYLARK_PPT_DATA_HPP */
