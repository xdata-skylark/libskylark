#ifndef RFUT_ELEMENTAL_HPP
#define RFUT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "config.h"

#include "context.hpp"
#include "transforms.hpp"

namespace skylark {
namespace sketch {

/**
 * Specialization for [*, SOMETHING]
 */
template < typename ValueType,
           typename FUT,
           elem::Distribution RowDist,
           typename Distribution>
struct RFUT_t<
    elem::DistMatrix<ValueType, elem::STAR, RowDist>, FUT, Distribution> {

    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::Matrix<ValueType> local_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, RowDist> matrix_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, RowDist>
        output_matrix_type;

private:

    // List of variables associated with this sketch
    /// The transform matrix is N-by-N
    const int N;
    /// The random diagonal part
    std::vector<ValueType> D;
    /// context for this sketch
    skylark::sketch::context_t& context;

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
            for (int i = 0; i < N; i++)
                local_TA.Set(i, j, scale * D[i] * local_A.Get(i, j));

        // Apply underlying transform
        T.apply(local_TA, skylark::sketch::columnwise_tag());
    }

public:
    RFUT_t(int N, skylark::sketch::context_t& context) :
        N(N), D(N), context(context) {
        boost::random::mt19937 prng(context.newseed());
        Distribution distribution;
        for (int i = 0; i < N; i++)
            D[i] = distribution(prng) ? +1 : -1;
    }

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
           typename Distribution>
struct RFUT_t<
    elem::DistMatrix<ValueType, RowDist, elem::STAR>, FUT, Distribution> {

    /** Typedef matrix type so that we can use it regularly */
    typedef ValueType value_type;
    typedef elem::Matrix<ValueType> local_type;
    typedef elem::DistMatrix<ValueType, RowDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<ValueType, RowDist, elem::STAR> output_matrix_type;

    typedef elem::DistMatrix<ValueType, elem::STAR, RowDist> intermediate_type;
    /**< Intermediate type for columnwise applications */

private:

    /** List of variables associated with this sketch */
    const int N; /**< The transform matrix is N-by-N */
    const std::vector<ValueType> D; /**< The random diagonal part */
    skylark::sketch::context_t& context; /**< context for this sketch */

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
        for (int j = 0; j < N; j++)
            for (int i = 0; i < local_A.Height(); i++)
                local_TA.Set(i, j, scale * D[j] * local_A.Get(i, j));

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
            for (int i = 0; i < N; i++)
                local_A.Set(i, j, scale * D[i] * local_A.Get(i, j));

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
            for (int i = 0; i < N; i++)
                local_A.Set(i, j, scale * D[i] * local_A.Get(i, j));

        // Rearrange back
        mixed_A = inter_A;
    }

public:
    RFUT_t(int N, skylark::sketch::context_t& context) :
        N(N), D(N), context(context) {
        boost::random::mt19937 prng(context.newseed());
        Distribution distribution;
        for (int i = 0; i < N; i++)
            D[i] = distribution(prng) ? +1 : -1;
    }

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
