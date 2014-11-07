#ifndef SKYLARK_FRFT_DATA_HPP
#define SKYLARK_FRFT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

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
struct FastRFT_data_t : public sketch_transform_data_t {

    typedef sketch_transform_data_t base_t;

    FastRFT_data_t (int N, int S, skylark::base::context_t& context)
        : base_t(N, S, context, "FastRFT"), _NB(N),
          numblks(1 + ((base_t::_S - 1) / _NB)),
          scale(std::sqrt(2.0 / base_t::_S)),
          Sm(numblks * _NB)  {

        context = build();
    }

    FastRFT_data_t (const boost::property_tree::ptree &pt)
        : base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "FastRFT"),
          _NB(base_t::_N),
          numblks(1 + ((base_t::_S - 1) / _NB)),
          scale(std::sqrt(2.0 / base_t::_S)),
          Sm(numblks * _NB)  {

         build();
    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
     */
    virtual
    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        sketch_transform_data_t::add_common(pt);
        return pt;
    }


protected:
    FastRFT_data_t (int N, int S, const skylark::base::context_t& context,
        std::string type)
        : base_t(N, S, context, type), _NB(N),
          numblks(1 + ((base_t::_S - 1) / _NB)),
          scale(std::sqrt(2.0 / base_t::_S)),
          Sm(numblks * _NB)  {

    }

    const int _NB; /**< Block size -- closet power of two of N */

    const int numblks;
    const double scale; /** Scaling for trigonometric factor */
    std::vector<double> Sm; /** Scaling based on kernel (filled by subclass) */
    std::vector<double> B;
    std::vector<double> G;
    std::vector<size_t> P;
    std::vector<double> shifts; /** Shifts for scaled trigonometric factor */

    base::context_t build() {
        base::context_t ctx = base_t::build();
        const double pi = boost::math::constants::pi<double>();
        bstrand::uniform_real_distribution<double> dist_shifts(0, 2 * pi);
        shifts = ctx.generate_random_samples_array(base_t::_S, dist_shifts);
        utility::rademacher_distribution_t<double> dist_B;

        B = ctx.generate_random_samples_array(numblks * _NB, dist_B);
        bstrand::normal_distribution<double> dist_G;
        G = ctx.generate_random_samples_array(numblks * _NB, dist_G);

        // For the permutation we use Fisher-Yates (Knuth)
        // The following will generate the indexes for the swaps. However
        // the scheme here might have a small bias if NB is small
        // (has to be really small).
        bstrand::uniform_int_distribution<size_t> dist_P(0);
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

struct FastGaussianRFT_data_t :
        public FastRFT_data_t {

    typedef FastRFT_data_t base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double sigma) : sigma(sigma) {

        }

        const double sigma;
    };

    FastGaussianRFT_data_t(int N, int S, double sigma,
        skylark::base::context_t& context)
        : base_t(N, S, context, "FastGaussianRFT"), _sigma(sigma) {

        context = build();
    }

    FastGaussianRFT_data_t(int N, int S, const params_t& params,
        skylark::base::context_t& context)
        : base_t(N, S, context, "FastGaussianRFT"), _sigma(params.sigma) {

        context = build();
    }

    FastGaussianRFT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "FastGaussianRFT"),
        _sigma(pt.get<double>("sigma")) {

        build();
    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
     */
    virtual
    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        sketch_transform_data_t::add_common(pt);
        pt.put("sigma", _sigma);
        return pt;
    }

protected:
    FastGaussianRFT_data_t(int N, int S, double sigma,
        const skylark::base::context_t& context, std::string type)
        : base_t(N, S, context, type), _sigma(sigma) {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();

        std::fill(base_t::Sm.begin(), base_t::Sm.end(),
                1.0 / (_sigma * std::sqrt(base_t::_N)));

        return ctx;
    }
    const double _sigma; /**< Bandwidth (sigma)  */

};

struct FastMaternRFT_data_t :
        public FastRFT_data_t {

    typedef FastRFT_data_t base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(int order) : order(order) {

        }

        const int order;
    };

    FastMaternRFT_data_t(int N, int S, int order,
        skylark::base::context_t& context)
        : base_t(N, S, context, "FastMaternRFT"), _order(order) {

        context = build();
    }

    FastMaternRFT_data_t(int N, int S, const params_t& params,
        skylark::base::context_t& context)
        : base_t(N, S, context, "FastMaternRFT"), _order(params.order) {

        context = build();
    }

    FastMaternRFT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "FastMaternRFT"),
        _order(pt.get<int>("order")) {

        build();
    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
     */
    virtual
    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        sketch_transform_data_t::add_common(pt);
        pt.put("order", _order);
        return pt;
    }

protected:
    FastMaternRFT_data_t(int N, int S, int order,
        const skylark::base::context_t& context, std::string type)
        : base_t(N, S, context, type), _order(order) {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();

        // TODO (?) If we pregenerate, we can speed this up with OMP
        boost::random::normal_distribution<double> distnrml;
        std::vector<double> xi(_N), xii(_N);
        for(auto it = Sm.begin(); it != Sm.end(); it++) {
            std::fill(xi.begin(), xi.end(), 0.0);
            for(int i = 0; i < _order; i++) {
                xii = ctx.generate_random_samples_array(_N, distnrml);
                double dot = 0.0;
                for(int j = 0; j < _N; j++)
                    dot += xii[j] * xii[j];
                double nrm = std::sqrt(dot);
                for(int j = 0; j < _N; j++)
                    xi[j] += xii[j] / nrm;
            }
            double dot = 0.0;
            for(int j = 0; j < _N; j++)
                dot += xi[j] * xi[j];
            *it = std::sqrt(dot);
        }

        return ctx;
    }

private:
    const int _order; /**< The order of the kernel */

};

} } /** namespace skylark::sketch */

#endif  // SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT
#endif
