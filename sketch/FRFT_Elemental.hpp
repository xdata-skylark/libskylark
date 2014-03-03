#ifndef SKYLARK_FRFT_ELEMENTAL_HPP
#define SKYLARK_FRFT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"
#include "FUT.hpp"
#include "FRFT_data.hpp"
#include "../utility/exception.hpp"


namespace skylark {
namespace sketch {

/**
 * Specialization for local input, local output
 */
template <typename ValueType>
struct FastRFT_t <
    elem::Matrix<ValueType>,
    elem::Matrix<ValueType> > :
        public FastRFT_data_t<ValueType> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::Matrix<value_type> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef FastRFT_data_t<ValueType> base_data_t;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    FastRFT_t(const FastRFT_t<matrix_type,
                      output_matrix_type>& other)
        : base_data_t(other), _dct(base_data_t::N) {

    }

    /**
     * Constructor from data
     */
    FastRFT_t(const base_data_t& other_data)
        : base_data_t(other_data), _dct(base_data_t::N) {

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

        // Create a work array W
        matrix_type W(A.Height(), A.Width());

        output_matrix_type B(base_data_t::N, 1), G(base_data_t::N, 1);
        output_matrix_type Sm(base_data_t::N, 1);
        for(int i = 0; i < base_data_t::numblks; i++) {
            int s = i * base_data_t::N;
            int e = std::min(s + base_data_t::N, base_data_t::S);

            W = A;

            // Set the local values of B, G and S
            value_type scal =
                std::sqrt(base_data_t::N) * _dct.scale(W, tag);
            for(int j = 0; j < base_data_t::N; j++) {
                B.Set(j, 0, base_data_t::B[i * base_data_t::N + j]);
                G.Set(j, 0, scal * base_data_t::G[i * base_data_t::N + j]);
                Sm.Set(j, 0, scal * base_data_t::Sm[i * base_data_t::N + j]);
            }

            int c, l, idx1, idx2;
            double *w;
#ifdef SKYLARK_HAVE_OPENMP
#pragma omp parallel for default(shared) private(c, l, w, idx1, idx2)
#endif
            for(c = 0; c < A.Width(); c++) {
                matrix_type Wc;
                elem::View(Wc, W, 0, c, W.Height(), 1);

                elem::DiagonalScale(elem::LEFT, elem::NORMAL, B, Wc);

                _dct.apply(Wc, tag);

                w = Wc.Buffer();
                for(l = 0; l < base_data_t::N - 1; l++) {
                    idx1 = base_data_t::N - 1 - l;
                    idx2 = base_data_t::P[i * (base_data_t::N - 1) + l];
                    std::swap(w[idx1], w[idx2]);
                }

                elem::DiagonalScale(elem::LEFT, elem::NORMAL, G, Wc);

                _dct.apply(Wc, tag);

                elem::DiagonalScale(elem::LEFT, elem::NORMAL, Sm, Wc);
            }

            // Copy that part to the output
            output_matrix_type view_sketch_of_A;
            elem::View(view_sketch_of_A, sketch_of_A, s, 0, e - s, A.Width());
            matrix_type view_W;
            elem::View(view_W, W, 0, 0, e - s, A.Width());
            view_sketch_of_A = view_W;
        }

        for(int j = 0; j < A.Width(); j++)
            for(int i = 0; i < base_data_t::S; i++) {
                value_type val = sketch_of_A.Get(i, j);
                value_type trans =
                    base_data_t::scale * std::cos(val + base_data_t::shifts[i]);
                sketch_of_A.Set(i, j, trans);
            }
    }

    /**
      * Apply the sketching transform on A and write to  sketch_of_A.
      * Implementation rowwise.
      */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        // Create a work array W
        matrix_type W(A.Height(), A.Width());

        output_matrix_type B(base_data_t::N, 1), G(base_data_t::N, 1);
        output_matrix_type Sm(base_data_t::N, 1);
        for(int i = 0; i < base_data_t::numblks; i++) {
            int s = i * base_data_t::N;
            int e = std::min(s + base_data_t::N, base_data_t::S);

            W = A;

            // Set the local values of B, G and S
            value_type scal =
                std::sqrt(base_data_t::N) * _dct.scale(W, tag);
            for(int j = 0; j < base_data_t::N; j++) {
                B.Set(j, 0, base_data_t::B[i * base_data_t::N + j]);
                G.Set(j, 0, scal * base_data_t::G[i * base_data_t::N + j]);
                Sm.Set(j, 0, scal * base_data_t::Sm[i * base_data_t::N + j]);
            }

            elem::DiagonalScale(elem::RIGHT, elem::NORMAL, B, W);

            _dct.apply(W, tag);

            double *w = W.Buffer();
            for(int c = 0; c < W.Height(); c++)
                for(int l = 0; l < base_data_t::N - 1; l++) {
                    int idx1 = c + (base_data_t::N - 1 - l) * W.LDim();
                    int idx2 = c  +
                        (base_data_t::P[i * (base_data_t::N - 1) + l]) * W.LDim();
                    std::swap(w[idx1], w[idx2]);
                }

            elem::DiagonalScale(elem::RIGHT, elem::NORMAL, G, W);

            _dct.apply(W, tag);

            elem::DiagonalScale(elem::RIGHT, elem::NORMAL, Sm, W);

            // Copy that part to the output
            output_matrix_type view_sketch_of_A;
            elem::View(view_sketch_of_A, sketch_of_A, 0, s, A.Height(), e - s);
            matrix_type view_W;
            elem::View(view_W, W, 0, 0, A.Height(), e - s);
            view_sketch_of_A = view_W;
        }

        for(int j = 0; j < base_data_t::S; j++)
            for(int i = 0; i < A.Height(); i++) {
                value_type val = sketch_of_A.Get(i, j);
                value_type trans =
                    base_data_t::scale * std::cos(val + base_data_t::shifts[j]);
                sketch_of_A.Set(i, j, trans);
            }
    }

private:
    typename fft_futs<ValueType>::DCT_t _dct;
};



} } /** namespace skylark::sketch */

#endif // SKYLARK_FRFT_ELEMENTAL_HPP
