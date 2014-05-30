#ifndef SKYLARK_WZT_DATA_HPP
#define SKYLARK_WZT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include "../utility/distributions.hpp"

namespace skylark { namespace sketch {

/**
 * Woodruff-Zhang Transform (data)
 *
 * Woodruff-Zhang Transform is very similar to the Clarkson-Woodruff Transform:
 * it replaces the +1/-1 diagonal with reciprocal exponential random entries.
 * It is suitable for lp regression with 1 <= p <= 2.
 *
 * Reference:
 * D. Woodruff and Q. Zhang
 * Subspace Embeddings and L_p Regression Using Exponential Random
 * COLT 2013
 *
 * TODO current implementation is only one sketch index, when for 1 <= p <= 2
 *      you want more than one.
 */
struct WZT_data_t : public hash_transform_data_t<
    boost::random::uniform_int_distribution,
    boost::random::exponential_distribution > {

    typedef hash_transform_data_t<
        boost::random::uniform_int_distribution,
        boost::random::exponential_distribution >  base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double p) : p(p) {

        }

        const double p;
    };

    WZT_data_t(int N, int S, double p, base::context_t& context)
        : base_t(N, S, context, "WZT"), _P(p) {

        // TODO verify that p is in the correct range.
        if(p < 1 || 2 < p)
            SKYLARK_THROW_EXCEPTION (
                base::sketch_exception()
                    << base::error_msg("WZT parameter p has to be in (1, 2)") );

        context = build();
    }

    WZT_data_t(int N, int S, const params_t& params, base::context_t& context)
        : base_t(N, S, context, "WZT"), _P(params.p) {

        // TODO verify that p is in the correct range.
        if(_P < 1 || 2 < _P)
            SKYLARK_THROW_EXCEPTION (
                base::sketch_exception()
                    << base::error_msg("WZT parameter p has to be in (1, 2)") );

        context = build();
    }


    WZT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "WZT"),
        _P(pt.get<double>("P")) {

        base_t::build();
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
        pt.put("P", _P);
        return pt;
    }

protected:
    WZT_data_t(int N, int S, double p, const base::context_t& context,
        std::string type)
        : base_t(N, S, context, type), _P(p) {

        // TODO verify that p is in the correct range.
        if(p < 1 || 2 < p)
            SKYLARK_THROW_EXCEPTION (
                base::sketch_exception()
                    << base::error_msg("WZT parameter p has to be in (1, 2)") );
    }

private:
    double _P;

    base::context_t build() {

        // Since the distribution depends on the target p we have to pass p as
        // a parameter. We also cannot just use the distribution as template.
        // The only solution I found is to let the base class generate the
        // numbers and then modify them to the correct distribution.
        // We also need it to +/- with equal probability. This solves this as
        // well.
        base::context_t ctx = base_t::build();
        utility::rademacher_distribution_t<double> pmdist;
        std::vector<double> pmvals =
            ctx.generate_random_samples_array(base_t::_N, pmdist);
        for(int i = 0; i < base_t::_N; i++)
             base_t::row_value[i] =
                 pmvals[i] * pow(1.0 / base_t::row_value[i], 1.0 / _P);
        return ctx;
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_MMT_DATA_HPP
