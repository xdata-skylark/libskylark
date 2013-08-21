#ifndef RFT_ELEMENTAL_HPP
#define RFT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"

namespace skylark {
namespace sketch {


/**
 * Specialization distributed input and output in [*, SOMETHING]
 */
template <typename ValueType,
          elem::Distribution ColDist,
          template <typename> class DistributionType>
struct RFT_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    DistributionType> {

public:
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> output_matrix_type;

private:

    typedef skylark::sketch::dense_transform_t<matrix_type,
                                               output_matrix_type,
                                               DistributionType> dense_type;

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                     output_matrix_type& sketch_of_A,
                     skylark::sketch::columnwise_tag tag) const {
        _dense.apply(A, sketch_of_A, tag);
        elem::Matrix<value_type> &Al = sketch_of_A.Matrix();
        for(int j = 0; j < Al.Width(); j++)
            for(int i = 0; i < _S; i++) {
                value_type val = Al.Get(i, j);
                value_type trans =
                    _scale * std::cos((val / _sigma) + _shifts[i]);
                Al.Set(i, j, trans);
            }
    }

    /**
      * Apply the sketching transform that is described in by the sketch_of_A.
      * Implementation for [VR/VC, *] and rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {
        // TODO verify sizes etc.

        _dense.apply(A, sketch_of_A, tag);
        elem::Matrix<value_type> &Al = sketch_of_A.Matrix();
        for(int j = 0; j < _S; j++)
            for(int i = 0; i < Al.Height(); i++) {
                value_type val = Al.Get(i, j);
                value_type trans =
                    _scale * std::cos((val / _sigma) + _shifts[j]);
                Al.Set(i, j, trans);
            }
    }

    /// Input dimension
    const int _N;
    /// Output dimension
    const int _S;
    /// Bandwidth (sigma)
    const double _sigma;
    /// Context for this sketch
    skylark::sketch::context_t& _context;
    /// Underlying dense_transform
    dense_type _dense;
    /// Shifts
    std::vector<double> _shifts;
    /// Scale
    const double _scale;


public:
    /**
     * Constructor
     */
    RFT_t (int N, int S, double sigma, skylark::sketch::context_t& context)
        : _N(N), _S(S), _sigma(sigma), _context(context),
          _dense(N, S, context), _shifts(S),
          _scale(std::sqrt(2.0 / _S)) {
        boost::random::mt19937 prng(context.newseed());
        const double pi = boost::math::constants::pi<double>();
        boost::random::uniform_real_distribution<> distribution(0, 2 * pi);
        for (int i = 0; i < _S; i++)
            _shifts[i] = distribution(prng);
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {

        switch(ColDist) {
        case elem::VR:
        case elem::VC:
            apply_impl_vdist (A, sketch_of_A, dimension);
            break;

        default:
            std::cerr << "Unsupported for now..." << std::endl;
            break;
        }
    }
};

} // namespace sketch
} // namespace skylark

#endif // RFT_ELEMENTAL_HPP
