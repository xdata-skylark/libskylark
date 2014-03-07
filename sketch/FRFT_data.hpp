#ifndef SKYLARK_FRFT_DATA_HPP
#define SKYLARK_FRFT_DATA_HPP

#include <vector>

#include "context.hpp"
#include "dense_transform_data.hpp"
#include "../utility/randgen.hpp"

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Fast Random Features Transform (data)
 *
 * Sketch transform into Eucledian space of fuctions in an RKHS
 * implicitly defined by a vector and a shift invaraint kernel. Fast variant
 * (also known as Fastfood).
 *
 * See:
 * Q. Le, T. Sarlos, A. Smola
 * Fastfood - Approximating Kernel Expansions in Loglinear Time
 * ICML 2013.
 */
template< typename ValueType >
struct FastRFT_data_t {

    typedef ValueType value_type;

    /**
     * Regular constructor
     */
    FastRFT_data_t (int N, int S, skylark::sketch::context_t& context)
        : N(N), S(S), NB(std::pow(2, std::ceil(std::log(N)/std::log(2)))),
          context(context),
          numblks(1 + ((S - 1) / NB)), scale(std::sqrt(2.0 / S)),
          Sm(numblks * NB)  {

        const double pi = boost::math::constants::pi<value_type>();
        bstrand::uniform_real_distribution<value_type> dist_shifts(0, 2 * pi);
        shifts = context.generate_random_samples_array(S, dist_shifts);
        utility::rademacher_distribution_t<value_type> dist_B;
        B = context.generate_random_samples_array(numblks * NB, dist_B);
        bstrand::normal_distribution<value_type> dist_G;
        G = context.generate_random_samples_array(numblks * NB, dist_G);

        // For the permutation we use Fisher-Yates (Knuth)
        // The following will generate the indexes for the swaps. However
        // the scheme here might have a small bias if NB is small
        // (has to be really small).
        bstrand::uniform_int_distribution<int> dist_P(0);
        P = context.generate_random_samples_array(numblks * (NB - 1), dist_P);
        for(int i = 0; i < numblks; i++)
            for(int j = NB - 1; j >= 1; j--)
                P[i * (NB - 1) + NB - 1 - j] =
                    P[i * (NB - 1) + NB - 1 - j] % (j + 1);

        // Fill scaling matrix with 1. Subclasses (which are adapted to concrete
        // kernels) should modify this.
        std::fill(Sm.begin(), Sm.end(), 1.0);
    }


protected:
    const int N; /**< Input dimension  */
    const int S; /**< Output dimension  */
    skylark::sketch::context_t& context; /**< Context for this sketch */

    const int NB; /**< Block size -- closet power of two of N */
    const int numblks;
    const value_type scale; /** Scaling for trigonometric factor */
    std::vector<value_type> Sm; /** Scaling based on kernel (filled by subclass) */
    std::vector<value_type> B;
    std::vector<value_type> G;
    std::vector<int> P;
    std::vector<value_type> shifts; /** Shifts for scaled trigonometrfic factor */
};

template<typename ValueType>
struct FastGaussianRFT_data_t :
        public FastRFT_data_t<ValueType> {

    typedef FastRFT_data_t<ValueType> base_t;
    typedef typename base_t::value_type value_type;

    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    FastGaussianRFT_data_t(int N, int S, value_type sigma,
        skylark::sketch::context_t& context)
        : base_t(N, S, context), sigma(sigma) {

        std::fill(base_t::Sm.begin(), base_t::Sm.end(), 1.0 / (sigma * std::sqrt(N)));
    }

protected:
    const value_type sigma; /**< Bandwidth (sigma)  */

};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_FRFT_DATA_HPP */
