#ifndef SKYLARK_FRFT_DATA_HPP
#define SKYLARK_FRFT_DATA_HPP

#include <vector>

#include "../base/context.hpp"
#include "sketch_transform_data.hpp"
#include "dense_transform_data.hpp"
#include "../utility/randgen.hpp"

namespace skylark { namespace sketch {

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT

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
struct FastRFT_data_t : public sketch_transform_data_t {

    typedef ValueType value_type;
    typedef sketch_transform_data_t base_t;

    /**
     * Regular constructor
     */
    FastRFT_data_t (int N, int S, skylark::base::context_t& context,
                    std::string type = "FastRFT")
        : base_t(N, S, context, type, true), _NB(N),
          numblks(1 + ((base_t::_S - 1) / _NB)),
          scale(std::sqrt(2.0 / base_t::_S)),
          Sm(numblks * _NB)  {

        context = build();
    }

    FastRFT_data_t (boost::property_tree::ptree &json)
        : base_t(json, true), _NB(base_t::_N),
          numblks(1 + ((base_t::_S - 1) / _NB)),
          scale(std::sqrt(2.0 / base_t::_S)),
          Sm(numblks * _NB)  {

         build();
    }

protected:
    FastRFT_data_t (int N, int S, skylark::base::context_t& context,
        std::string type = "FastRFT", bool nobuild = true)
        : base_t(N, S, context, type), _NB(N),
          numblks(1 + ((base_t::_S - 1) / _NB)),
          scale(std::sqrt(2.0 / base_t::_S)),
          Sm(numblks * _NB)  {

    }

    FastRFT_data_t (boost::property_tree::ptree &json, bool nobuild)
        : base_t(json), _NB(base_t::_N),
          numblks(1 + ((base_t::_S - 1) / _NB)),
          scale(std::sqrt(2.0 / base_t::_S)),
          Sm(numblks * _NB)  {

    }

    const int _NB; /**< Block size -- closet power of two of N */

    const int numblks;
    const value_type scale; /** Scaling for trigonometric factor */
    std::vector<value_type> Sm; /** Scaling based on kernel (filled by subclass) */
    std::vector<value_type> B;
    std::vector<value_type> G;
    std::vector<int> P;
    std::vector<value_type> shifts; /** Shifts for scaled trigonometric factor */

    base::context_t build() {
        base::context_t ctx = base_t::build();
        const double pi = boost::math::constants::pi<value_type>();
        bstrand::uniform_real_distribution<value_type> dist_shifts(0, 2 * pi);
        shifts = ctx.generate_random_samples_array(base_t::_S, dist_shifts);
        utility::rademacher_distribution_t<value_type> dist_B;

        B = ctx.generate_random_samples_array(numblks * _NB, dist_B);
        bstrand::normal_distribution<value_type> dist_G;
        G = ctx.generate_random_samples_array(numblks * _NB, dist_G);

        // For the permutation we use Fisher-Yates (Knuth)
        // The following will generate the indexes for the swaps. However
        // the scheme here might have a small bias if NB is small
        // (has to be really small).
        bstrand::uniform_int_distribution<int> dist_P(0);
        P = ctx.generate_random_samples_array(numblks * (_NB - 1), dist_P);
        for(int i = 0; i < numblks; i++)
            for(int j = _NB - 1; j >= 1; j--)
                P[i * (_NB - 1) + _NB - 1 - j] =
                    P[i * (_NB - 1) + _NB - 1 - j] % (j + 1);

        // Fill scaling matrix with 1. Subclasses (which are adapted to concrete
        // kernels) should modify this.
        std::fill(Sm.begin(), Sm.end(), 1.0);

        return ctx;
    }


    // TODO there is also the issue of type of FUT, which now depends on what
    //      is installed. For seralization we need to add an indicator on type
    //      of the underlying FUT.
    static int block_size(int N) {
#if SKYLARK_HAVE_FFTW
        return N;
#elif SKYLARK_HAVE_SPIRALWHT
        return (int)std::pow(2, std::ceil(std::log(N)/std::log(2)));
#endif
    }

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
        skylark::base::context_t& context)
        : base_t(N, S, context, "FastGaussianRFT", true), _sigma(sigma) {

        std::fill(base_t::Sm.begin(), base_t::Sm.end(),
                1.0 / (_sigma * std::sqrt(base_t::_N)));
        context = base_t::build();
    }

    FastGaussianRFT_data_t(boost::property_tree::ptree &json)
        : base_t(json, true),
        _sigma(json.get<value_type>("sketch.sigma")) {

        std::fill(base_t::Sm.begin(), base_t::Sm.end(),
                1.0 / (_sigma * std::sqrt(base_t::_N)));
        base_t::build();
    }

    template <typename ValueT>
    friend boost::property_tree::ptree& operator<<(
        boost::property_tree::ptree &sk,
        const FastGaussianRFT_data_t<ValueT> &data);

protected:
    FastGaussianRFT_data_t(int N, int S, value_type sigma,
        skylark::base::context_t& context, bool nobuild)
        : base_t(N, S, context, "FastGaussianRFT", true), _sigma(sigma) {

        std::fill(base_t::Sm.begin(), base_t::Sm.end(),
                1.0 / (_sigma * std::sqrt(base_t::_N)));
    }

    FastGaussianRFT_data_t(boost::property_tree::ptree &json, bool nobuild)
        : base_t(json, true),
        _sigma(json.get<value_type>("sketch.sigma")) {

        std::fill(base_t::Sm.begin(), base_t::Sm.end(),
                1.0 / (_sigma * std::sqrt(base_t::_N)));
    }


    const value_type _sigma; /**< Bandwidth (sigma)  */

};

template <typename ValueType>
boost::property_tree::ptree& operator<<(
        boost::property_tree::ptree &sk,
        const FastGaussianRFT_data_t<ValueType> &data) {

    sk << static_cast<const sketch_transform_data_t&>(data);
    sk.put("sketch.sigma", data._sigma);
    return sk;
}

} } /** namespace skylark::sketch */

#endif  // SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT
#endif
