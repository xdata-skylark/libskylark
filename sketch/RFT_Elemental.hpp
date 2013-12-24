#ifndef SKYLARK_RFT_ELEMENTAL_HPP
#define SKYLARK_RFT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"
#include "RFT_data.hpp"
#include "../utility/exception.hpp"


namespace skylark {
namespace sketch {

/**
 * Specialization distributed input and output in [*, SOMETHING]
 */
template <typename ValueType,
          elem::Distribution ColDist,
          template <typename> class KernelDistribution>
struct RFT_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    KernelDistribution> :
        public RFT_data_t<ValueType,
                          KernelDistribution> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<value_type,
                             ColDist, elem::STAR> output_matrix_type;
    typedef RFT_data_t<ValueType,
                       KernelDistribution> base_data_t;
private:
    typedef skylark::sketch::dense_transform_t <matrix_type,
                                                output_matrix_type,
                                                KernelDistribution>
    underlying_t;


public:
    /**
     * Regular constructor
     */
    RFT_t (int N, int S, double sigma, skylark::sketch::context_t& context)
        : base_data_t (N, S, sigma, context) {

    }

    /**
     * Copy constructor
     */
    RFT_t(const RFT_t<matrix_type,
                      output_matrix_type,
                      KernelDistribution>& other)
        : base_data_t(other.get_data()) {

    }

    /**
     * Constructor from data
     */
    RFT_t(const base_data_t& other_data)
        : base_data_t(other_data.get_data()) {

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
            try {
            apply_impl_vdist (A, sketch_of_A, dimension);
            } catch (std::logic_error e) {
                SKYLARK_THROW_EXCEPTION (
                    utility::elemental_exception()
                        << utility::error_msg(e.what()) );
            } catch(boost::mpi::exception e) {
                SKYLARK_THROW_EXCEPTION (
                    utility::mpi_exception()
                        << utility::error_msg(e.what()) );
            }

            break;

        default:
            SKYLARK_THROW_EXCEPTION (
                utility::unsupported_matrix_distribution() );
        }
    }

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                     output_matrix_type& sketch_of_A,
                     skylark::sketch::columnwise_tag tag) const {
        underlying_t underlying(base_data_t::underlying_data);
        underlying.apply(A, sketch_of_A, tag);
        elem::Matrix<value_type> &Al = sketch_of_A.Matrix();
        for(int j = 0; j < Al.Width(); j++)
            for(int i = 0; i < base_data_t::S; i++) {
                value_type val = Al.Get(i, j);
                value_type trans =
                    base_data_t::scale * std::cos((val / base_data_t::sigma) +
                        base_data_t::shifts[i]);
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
        underlying_t underlying(base_data_t::underlying_data);
        underlying.apply(A, sketch_of_A, tag);
        elem::Matrix<value_type> &Al = sketch_of_A.Matrix();
        for(int j = 0; j < base_data_t::S; j++)
            for(int i = 0; i < Al.Height(); i++) {
                value_type val = Al.Get(i, j);
                value_type trans =
                    base_data_t::scale * std::cos((val / base_data_t::sigma) +
                        base_data_t::shifts[j]);
                Al.Set(i, j, trans);
            }
    }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_RFT_ELEMENTAL_HPP
