#ifndef SKYLARK_RFT_ELEMENTAL_HPP
#define SKYLARK_RFT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "../base/context.hpp"
#include "transforms.hpp"
#include "RFT_data.hpp"
#include "../utility/exception.hpp"


namespace skylark {
namespace sketch {

/**
 * Specialization for local input (dense or sparse), local output (dense)
 */
template <typename ValueType,
          template <typename> class InputType,
          template <typename> class KernelDistribution>
struct RFT_t <
    InputType<ValueType>,
    elem::Matrix<ValueType>,
    KernelDistribution> :
        public RFT_data_t<ValueType,
                          KernelDistribution> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef InputType<value_type> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef RFT_data_t<ValueType,
                       KernelDistribution> data_type;
private:
    typedef skylark::sketch::dense_transform_t <matrix_type,
                                                output_matrix_type,
                                                KernelDistribution>
    underlying_t;


protected:
    /**
     * Regular constructor - allow creation only by subclasses
     */
    RFT_t (int N, int S, skylark::base::context_t& context)
        : data_type (N, S, context) {

    }

public:
    /**
     * Copy constructor
     */
    RFT_t(const RFT_t<matrix_type,
                      output_matrix_type,
                      KernelDistribution>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    RFT_t(const data_type& other_data)
        : data_type(other_data) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        try {
            apply_impl(A, sketch_of_A, dimension);
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                utility::elemental_exception()
                    << utility::error_msg(e.what()) );
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                utility::mpi_exception()
                    << utility::error_msg(e.what()) );
        }
    }

private:
    /**
     * Apply the sketching transform on A and write to sketch_of_A.
     * Implementation for columnwise.
     */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag tag) const {

        // TODO verify sizes etc.
        underlying_t underlying(data_type::_underlying_data);
        underlying.apply(A, sketch_of_A, tag);

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for(int j = 0; j < A.Width(); j++)
            for(int i = 0; i < data_type::_S; i++) {
                value_type x = sketch_of_A.Get(i, j);
                x *= data_type::_val_scale;
                x += data_type::_shifts[i];

#               ifdef SKYLARK_EXACT_COSINE
                x = std::cos(x);
#               else
                // x = std::cos(x) is slow
                // Instead use low-accuracy approximation
                if (x < -3.14159265) x += 6.28318531;
                else if (x >  3.14159265) x -= 6.28318531;
                x += 1.57079632;
                if (x >  3.14159265)
                    x -= 6.28318531;
                x = (x < 0) ?
                    1.27323954 * x + 0.405284735 * x * x :
                    1.27323954 * x - 0.405284735 * x * x;
#               endif

                x = data_type::_scale * x;
                sketch_of_A.Set(i, j, x);
            }
    }

    /**
      * Apply the sketching transform on A and write to  sketch_of_A.
      * Implementation rowwise.
      */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        // TODO verify sizes etc.
        underlying_t underlying(data_type::_underlying_data);
        underlying.apply(A, sketch_of_A, tag);

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for(int j = 0; j < data_type::_S; j++)
            for(int i = 0; i < A.Height(); i++) {
                value_type x = sketch_of_A.Get(i, j);
                x *= data_type::_val_scale;
                x += data_type::_shifts[j];

#               ifdef SKYLARK_EXACT_COSINE
                x = std::cos(x);
#               else
                // x = std::cos(x) is slow
                // Instead use low-accuracy approximation
                if (x < -3.14159265) x += 6.28318531;
                else if (x >  3.14159265) x -= 6.28318531;
                x += 1.57079632;
                if (x >  3.14159265)
                    x -= 6.28318531;
                x = (x < 0) ?
                    1.27323954 * x + 0.405284735 * x * x :
                    1.27323954 * x - 0.405284735 * x * x;
#               endif

                x = data_type::_scale * x;
                sketch_of_A.Set(i, j, x);
            }
    }
};

/**
 * Specialization distributed input and output in [SOMETHING, *]
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
                       KernelDistribution> data_type;
private:
    typedef skylark::sketch::dense_transform_t <matrix_type,
                                                output_matrix_type,
                                                KernelDistribution>
    underlying_t;

protected:

    // Allow only creation by subclasses.

    /**
     * Regular constructor -- Allow only creation by subclasses
     */
    RFT_t (int N, int S, skylark::base::context_t& context)
        : data_type (N, S, context) {

    }

public:

    /**
     * Copy constructor
     */
    RFT_t(const RFT_t<matrix_type,
                      output_matrix_type,
                      KernelDistribution>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    RFT_t(const data_type& other_data)
        : data_type(other_data) {

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
        underlying_t underlying(data_type::_underlying_data);
        underlying.apply(A, sketch_of_A, tag);
        elem::Matrix<value_type> &Al = sketch_of_A.Matrix();
        for(int j = 0; j < Al.Width(); j++)
            for(int i = 0; i < data_type::_S; i++) {
                value_type val = Al.Get(i, j);
                value_type trans =
                    data_type::_scale * std::cos((val * data_type::_val_scale) +
                        data_type::_shifts[i]);
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
        underlying_t underlying(data_type::_underlying_data);
        underlying.apply(A, sketch_of_A, tag);
        elem::Matrix<value_type> &Al = sketch_of_A.Matrix();
        for(int j = 0; j < data_type::_S; j++)
            for(int i = 0; i < Al.Height(); i++) {
                value_type val = Al.Get(i, j);
                value_type trans =
                    data_type::_scale * std::cos((val * data_type::_val_scale) +
                        data_type::_shifts[j]);
                Al.Set(i, j, trans);
            }
    }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_RFT_ELEMENTAL_HPP
