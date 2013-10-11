#ifndef RFUT_ELEMENTAL_HPP
#define RFUT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "RFUT_data.hpp"
#include "transforms.hpp"
#include "../utility/randgen.hpp"

namespace skylark { namespace sketch {

/**
 * Specialization for [*, SOMETHING]
 */
template < typename ValueType,
           typename FUT,
           elem::Distribution RowDist,
           typename ValueDistributionType>
struct RFUT_t<
    elem::DistMatrix<ValueType, elem::STAR, RowDist>,
    FUT,
    ValueDistributionType> :
        public RFUT_data_t<ValueType,
                           ValueDistributionType> {
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::Matrix<ValueType> local_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, RowDist> matrix_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, RowDist>
        output_matrix_type;
    // Typedef distribution
    typedef ValueDistributionType value_distribution_type;
    typedef RFUT_data_t<ValueType,
                        ValueDistributionType> base_data_t;

    /**
     * Apply the transform to compute mixed_A.
     * Implementation for the application on the columns.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& mixed_A,
                           skylark::sketch::columnwise_tag) const {
        // TODO verify that A has the correct size

        FUT T;

        // Scale
        const local_type& local_A = A.LockedMatrix();
        local_type& local_TA = mixed_A.Matrix();
        value_type scale = T.scale(local_A);
        for (int j = 0; j < local_A.Width(); j++)
            for (int i = 0; i < base_data_t::N; i++)
                local_TA.Set(i, j,
                    scale * base_data_t::D[i] * local_A.Get(i, j));

        // Apply underlying transform
        T.apply(local_TA, skylark::sketch::columnwise_tag());
    }

public:
    RFUT_t(int N, skylark::sketch::context_t& context)
        : base_data_t (N, context) {}

    template <typename MatrixType>
    RFUT_t (RFUT_t<MatrixType,
                   FUT,
                   ValueDistributionType>& other) :
        base_data_t(other.get_data()) {}

    RFUT_t(RFUT_data_t<ValueType,
        ValueDistributionType>& other_data) :
        base_data_t(other_data.get_data()) {}

    /**
     * Apply the transform that is described in by the mixed_A.
     * mixed_A can be the same as A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& mixed_A,
                Dimension dimension) const {

        switch (RowDist) {
            case elem::VC:
            case elem::VR:
                apply_impl_vdist(A, mixed_A, dimension);
                break;

            default:
                std::cerr << "Unsupported for now..." << std::endl;
                break;
        }
    }
};

/**
 * Specialization for [SOMETHING, *]
 */
template < typename ValueType,
           typename FUT,
           elem::Distribution RowDist,
           typename ValueDistributionType>
struct RFUT_t<
    elem::DistMatrix<ValueType, RowDist, elem::STAR>,
    FUT,
    ValueDistributionType> :
        public RFUT_data_t<ValueType,
                           ValueDistributionType> {
    /** Typedef matrix type so that we can use it regularly */
    typedef ValueType value_type;
    typedef elem::Matrix<ValueType> local_type;
    typedef elem::DistMatrix<ValueType, RowDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<ValueType, RowDist, elem::STAR> output_matrix_type;

    typedef elem::DistMatrix<ValueType, elem::STAR, RowDist> intermediate_type;
    /**< Intermediate type for columnwise applications */

    // Typedef distribution
    typedef ValueDistributionType value_distribution_type;
    typedef RFUT_data_t<ValueType,
                        ValueDistributionType> base_data_t;

    /**
     * Apply the transform to compute mixed_A.
     * Implementation for the application on the rows.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& mixed_A,
                           skylark::sketch::rowwise_tag) const {
        // TODO verify that A has the correct size

        FUT T;

        // Scale
        const local_type& local_A = A.LockedMatrix();
        local_type& local_TA = mixed_A.Matrix();
        value_type scale = T.scale(local_A);
        for (int j = 0; j < base_data_t::N; j++)
            for (int i = 0; i < local_A.Height(); i++)
                local_TA.Set(i, j,
                    scale * base_data_t::D[j] * local_A.Get(i, j));

        // Apply underlying transform
        T.apply(local_TA, skylark::sketch::rowwise_tag());
    }

    /**
     * Apply the transform to compute mixed_A.
     * Implementation for the application on the columns.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& mixed_A,
                           skylark::sketch::columnwise_tag) const {
        // TODO verify that A has the correct size
        // TODO A and mixed_A have to match

        FUT T;

        // Rearrange matrix
        intermediate_type inter_A(A.Grid());
        inter_A = A;

        // Scale
        local_type& local_A = inter_A.Matrix();
        value_type scale = T.scale(local_A);
        for (int j = 0; j < local_A.Width(); j++)
            for (int i = 0; i < base_data_t::N; i++)
                local_A.Set(i, j,
                    scale * base_data_t::D[i] * local_A.Get(i, j));

        // Apply underlying transform
        T.apply(local_A, skylark::sketch::columnwise_tag());

        // Rearrange back
        mixed_A = inter_A;
    }

    /**
     * Apply the transform to compute mixed_A.
     * Implementation for the application on the columns.
     */
    void apply_inverse_impl_vdist  (const matrix_type& A,
                                    output_matrix_type& mixed_A,
                                    skylark::sketch::columnwise_tag) const {

        FUT T;

        // TODO verify that A has the correct size
        // TODO A and mixed_A have to match

        // Rearrange matrix
        intermediate_type inter_A(A.Grid());
        inter_A = A;

        // Apply underlying transform
        local_type& local_A = inter_A.Matrix();
        T.apply_inverse(local_A, skylark::sketch::columnwise_tag());

        // Scale
        value_type scale = T.scale(local_A);
        for (int j = 0; j < local_A.Width(); j++)
            for (int i = 0; i < base_data_t::N; i++)
                local_A.Set(i, j,
                    scale * base_data_t::D[i] * local_A.Get(i, j));

        // Rearrange back
        mixed_A = inter_A;
    }

public:
    RFUT_t(int N, skylark::sketch::context_t& context)
        : base_data_t (N, context) {}

    template <typename MatrixType>
    RFUT_t (RFUT_t<MatrixType,
                   FUT,
                   ValueDistributionType>& other) :
        base_data_t(other.get_data()) {}

    RFUT_t(RFUT_data_t<ValueType,
        ValueDistributionType>& other_data) :
        base_data_t(other_data.get_data()) {}

    /**
     * Apply the transform that is described in by the mixed_A.
     */
    template <typename Dimension>
    void apply(const matrix_type& A,
               output_matrix_type& mixed_A,
               Dimension dimension) const {

        switch (RowDist) {
            case elem::VC:
            case elem::VR:
                apply_impl_vdist(A, mixed_A, dimension);
                break;

            default:
                std::cerr << "Unsupported for now..." << std::endl;
                break;
        }
    }

    template <typename Dimension>
    void apply_inverse(const matrix_type& A,
                       output_matrix_type& mixed_A,
                       Dimension dimension) const {

        switch (RowDist) {
            case elem::VC:
            case elem::VR:
                apply_inverse_impl_vdist(A, mixed_A, dimension);
                break;

            default:
                std::cerr << "Unsupported for now..." << std::endl;
                break;
        }
    }
};

} // namespace sketch
} // namespace skylark

#endif // RFUT_HPP
