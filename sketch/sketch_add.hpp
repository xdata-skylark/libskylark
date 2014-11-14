#ifndef SKYLARK_SKETCH_ADD_HPP
#define SKYLARK_SKETCH_ADD_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

/**
 * Methods that can only be defined here because they require previous classes
 * to be defined.
 */

namespace skylark { namespace sketch {

sketch_transform_data_t*
sketch_transform_data_t::from_ptree(const boost::property_tree::ptree& pt) {
    std::string type = pt.get<std::string>("sketch_type");

# define AUTO_LOAD_DISPATCH(T, C)                    \
    if (type == #T)                                  \
        return new C(pt);

    AUTO_LOAD_DISPATCH(JLT, JLT_data_t);
    AUTO_LOAD_DISPATCH(CT,  CT_data_t);
    AUTO_LOAD_DISPATCH(CWT, CWT_data_t);
    AUTO_LOAD_DISPATCH(MMT, MMT_data_t);
    AUTO_LOAD_DISPATCH(WZT, WZT_data_t);
    AUTO_LOAD_DISPATCH(PPT, PPT_data_t);

    AUTO_LOAD_DISPATCH(GaussianRFT,  GaussianRFT_data_t);
    AUTO_LOAD_DISPATCH(LaplacianRFT, LaplacianRFT_data_t);

    AUTO_LOAD_DISPATCH(GaussianQRFT,
        GaussianQRFT_data_t<utility::qmc_sequence_container_t>);
    AUTO_LOAD_DISPATCH(LaplacianQRFT,
        LaplacianQRFT_data_t<utility::qmc_sequence_container_t>);

    AUTO_LOAD_DISPATCH(ExpSemigroupRLT, ExpSemigroupRLT_data_t);
    AUTO_LOAD_DISPATCH(FastGaussianRFT, FastGaussianRFT_data_t);

#if SKYLARK_HAVE_FFTW
    AUTO_LOAD_DISPATCH(FJLT, FJLT_data_t);
#endif

#undef AUTO_LOAD_DISPATCH

    SKYLARK_THROW_EXCEPTION(base::sketch_exception());

    return nullptr;
}

template<typename IT, typename OT>
sketch_transform_t<IT, OT>*
sketch_transform_t<IT, OT>::from_ptree(const boost::property_tree::ptree& pt) {
    std::string type = pt.get<std::string>("sketch_type");

# define AUTO_LOAD_DISPATCH(T, C)                    \
    if (type == #T)                                  \
        return new C<IT,OT>(pt);

    AUTO_LOAD_DISPATCH(JLT, JLT_t);
    AUTO_LOAD_DISPATCH(CT,  CT_t);
    AUTO_LOAD_DISPATCH(CWT, CWT_t);
    AUTO_LOAD_DISPATCH(MMT, MMT_t);
    AUTO_LOAD_DISPATCH(WZT, WZT_t);
    AUTO_LOAD_DISPATCH(PPT, PPT_t);

    AUTO_LOAD_DISPATCH(GaussianRFT,  GaussianRFT_t);
    AUTO_LOAD_DISPATCH(LaplacianRFT, LaplacianRFT_t);

    if (type == "GaussianQRFT")
        return new GaussianQRFT_t<IT, OT, utility::qmc_sequence_container_t>(pt);
    if (type == "LaplacianQRFT")
        return new LaplacianQRFT_t<IT, OT, utility::qmc_sequence_container_t>(pt);

    AUTO_LOAD_DISPATCH(GaussianQRFT,  GaussianRFT_t);
    AUTO_LOAD_DISPATCH(LaplacianQRFT, LaplacianRFT_t);


    AUTO_LOAD_DISPATCH(ExpSemigroupRLT, ExpSemigroupRLT_t);
    AUTO_LOAD_DISPATCH(FastGaussianRFT, FastGaussianRFT_t);

#if SKYLARK_HAVE_FFTW
    AUTO_LOAD_DISPATCH(FJLT, FJLT_t);
#endif

#undef AUTO_LOAD_DISPATCH

    SKYLARK_THROW_EXCEPTION(base::sketch_exception() <<
        base::error_msg("Unknown sktech type"));

    return nullptr;
}

} }    // namespace skylark::sketch

#endif // SKYLARK_SKETCH_ADD_HPP
