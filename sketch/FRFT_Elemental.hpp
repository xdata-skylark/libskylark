#ifndef SKYLARK_FRFT_ELEMENTAL_HPP
#define SKYLARK_FRFT_ELEMENTAL_HPP

#include "../base/base.hpp"

#include "../base/context.hpp"
#include "transforms.hpp"
#include "FUT.hpp"
#include "FRFT_data.hpp"
#include "../utility/exception.hpp"


namespace skylark {
namespace sketch {

#if SKYLARK_HAVE_ELEMENTAL && (SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT)

/**
 * Specialization local input (sparse of dense), local output.
 * InputType should either be elem::Matrix, or base:spare_matrix_t.
 */
template <typename ValueType,
          template <typename> class InputType>
struct FastRFT_t <
    InputType<ValueType>,
    elem::Matrix<ValueType> > :
        public FastRFT_data_t<ValueType> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef InputType<value_type> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef FastRFT_data_t<ValueType> base_data_t;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    FastRFT_t(const FastRFT_t<matrix_type,
                      output_matrix_type>& other)
        : base_data_t(other), _fut(base_data_t::_N) {

    }

    /**
     * Constructor from data
     */
    FastRFT_t(const base_data_t& other_data)
        : base_data_t(other_data), _fut(base_data_t::_N) {

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

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel
#       endif
        {
        output_matrix_type W(base_data_t::_NB, 1);
        double *w = W.Buffer();

        output_matrix_type Ac(base_data_t::_NB, 1);
        double *ac = Ac.Buffer();

        output_matrix_type Acv;
        elem::View(Acv, Ac, 0, 0, base_data_t::_N, 1);

        double *sa = sketch_of_A.Buffer();
        int ldsa = sketch_of_A.LDim();

        value_type scal =
            std::sqrt(base_data_t::_NB) * _fut.scale();

        output_matrix_type B(base_data_t::_NB, 1), G(base_data_t::_NB, 1);
        output_matrix_type Sm(base_data_t::_NB, 1);

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp for
#       endif
        for(int c = 0; c < A.Width(); c++) {
            const matrix_type Acs = base::ColumnView(A, c, 1);
            base::DenseCopy(Acs, Acv);
            std::fill(ac + base_data_t::_N, ac + base_data_t::_NB, 0);

            for(int i = 0; i < base_data_t::numblks; i++) {

                int s = i * base_data_t::_NB;
                int e = std::min(s + base_data_t::_NB,  base_data_t::_S);

                // Set the local values of B, G and S
                for(int j = 0; j < base_data_t::_NB; j++) {
                    B.Set(j, 0, base_data_t::B[i * base_data_t::_NB + j]);
                    G.Set(j, 0, scal * base_data_t::G[i * base_data_t::_NB + j]);
                    Sm.Set(j, 0, scal * base_data_t::Sm[i * base_data_t::_NB + j]);
                }

                W = Ac;

                elem::DiagonalScale(elem::LEFT, elem::NORMAL, B, W);
                _fut.apply(W, tag);
                for(int l = 0; l < base_data_t::_NB - 1; l++) {
                    int idx1 = base_data_t::_NB - 1 - l;
                    int idx2 = base_data_t::P[i * (base_data_t::_NB - 1) + l];
                    std::swap(w[idx1], w[idx2]);
                }
                elem::DiagonalScale(elem::LEFT, elem::NORMAL, G, W);
                _fut.apply(W, tag);
                elem::DiagonalScale(elem::LEFT, elem::NORMAL, Sm, W);

                double *sac = sa + ldsa * c;
                for(int l = s; l < e; l++) {
                    value_type x = w[l - s];
                    x += base_data_t::shifts[l];

#                   ifdef SKYLARK_EXACT_COSINE
                    x = std::cos(x);
#                   else
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
#                   endif

                    x = base_data_t::scale * x;
                    sac[l - s] = x;
                }
            }
        }

        }
    }

    /**
      * Apply the sketching transform on A and write to  sketch_of_A.
      * Implementation rowwise.
      */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        // TODO this version is really bad: it completely densifies the matrix
        //      on the begining.
        // TODO this version does not work with _NB and N
        // TODO this version is not as optimized as the columnwise version.

        // Create a work array W
        output_matrix_type W(A.Height(), A.Width());

        output_matrix_type B(base_data_t::_N, 1), G(base_data_t::_N, 1);
        output_matrix_type Sm(base_data_t::_N, 1);
        for(int i = 0; i < base_data_t::numblks; i++) {
            int s = i * base_data_t::_N;
            int e = std::min(s + base_data_t::_N, base_data_t::_S);

            base::DenseCopy(A, W);

            // Set the local values of B, G and S
            value_type scal =
                std::sqrt(base_data_t::_N) * _fut.scale();
            for(int j = 0; j < base_data_t::_N; j++) {
                B.Set(j, 0, base_data_t::B[i * base_data_t::_N + j]);
                G.Set(j, 0, scal * base_data_t::G[i * base_data_t::_N + j]);
                Sm.Set(j, 0, scal * base_data_t::Sm[i * base_data_t::_N + j]);
            }

            elem::DiagonalScale(elem::RIGHT, elem::NORMAL, B, W);

            _fut.apply(W, tag);

            double *w = W.Buffer();
            for(int c = 0; c < W.Height(); c++)
                for(int l = 0; l < base_data_t::_N - 1; l++) {
                    int idx1 = c + (base_data_t::_N - 1 - l) * W.LDim();
                    int idx2 = c  +
                        (base_data_t::P[i * (base_data_t::_N - 1) + l]) * W.LDim();
                    std::swap(w[idx1], w[idx2]);
                }

            elem::DiagonalScale(elem::RIGHT, elem::NORMAL, G, W);

            _fut.apply(W, tag);

            elem::DiagonalScale(elem::RIGHT, elem::NORMAL, Sm, W);

            // Copy that part to the output
            output_matrix_type view_sketch_of_A;
            elem::View(view_sketch_of_A, sketch_of_A, 0, s, A.Height(), e - s);
            output_matrix_type view_W;
            elem::View(view_W, W, 0, 0, A.Height(), e - s);
            view_sketch_of_A = view_W;
        }

        for(int j = 0; j < base_data_t::_S; j++)
            for(int i = 0; i < A.Height(); i++) {
                value_type x = sketch_of_A.Get(i, j);
                x += base_data_t::shifts[j];

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

                x = base_data_t::scale * x;
                sketch_of_A.Set(i, j, x);
            }
    }

private:

#ifdef SKYLARK_HAVE_FFTW
    typename fft_futs<ValueType>::DCT_t _fut;
#elif SKYLARK_HAVE_SPIRALWHT
    WHT_t<double> _fut;
#endif

};

#endif // SKYLARK_HAVE_ELEMENTAL && (SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT)

} } /** namespace skylark::sketch */

#endif // SKYLARK_FRFT_ELEMENTAL_HPP
