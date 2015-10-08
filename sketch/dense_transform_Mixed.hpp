#ifndef SKYLARK_DENSE_TRANSFORM_MIXED_HPP
#define SKYLARK_DENSE_TRANSFORM_MIXED_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/get_communicator.hpp"
#include "../base/sparse_vc_star_matrix.hpp"

#include "sketch_params.hpp"

namespace skylark { namespace sketch {
/**
 * Specialization distributed input sparse_vc_star_matrix_t and output in [SOMETHING, *]
 */
template <typename ValueType, El::Distribution ColDist,
          typename ValuesAccessor>
struct dense_transform_t <
    base::sparse_vc_star_matrix_t<ValueType>,
    El::DistMatrix<ValueType, ColDist, El::STAR>,
    ValuesAccessor> :
        public dense_transform_data_t<ValuesAccessor> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef base::sparse_vc_star_matrix_t<value_type> matrix_type;
    typedef El::DistMatrix<value_type, ColDist, El::STAR>
    output_matrix_type;
    typedef dense_transform_data_t<ValuesAccessor> data_type;

    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, double scale, base::context_t& context)
        : data_type (N, S, scale, context) {

    }

    /**
     * Copy constructor
     */
    dense_transform_t (dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValuesAccessor>& other)
        : data_type(other) {}

    /**
     * Constructor from data
     */
    dense_transform_t(const data_type& other_data)
        : data_type(other_data) {}

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {

        switch(ColDist) {
        case El::VR:
        case El::VC:
            try {
                // FIXME: for now only one implementation
                outer_panel_gemm(A, sketch_of_A, dimension);
            } catch (std::logic_error e) {
                SKYLARK_THROW_EXCEPTION (
                    base::elemental_exception()
                        << base::error_msg(e.what()) );
            } catch(boost::mpi::exception e) {
                SKYLARK_THROW_EXCEPTION (
                    base::mpi_exception()
                        << base::error_msg(e.what()) );
            }

            break;

        default:
            SKYLARK_THROW_EXCEPTION (
                base::unsupported_matrix_distribution() );
        }
    }

    int get_N() const { return this->_N; } /**< Get input dimension. */
    int get_S() const { return this->_S; } /**< Get output dimension. */

    const sketch_transform_data_t* get_data() const { return this; }

private:

    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        El::Zero(sketch_of_A);
        const El::Grid& grid = sketch_of_A.Grid();

        // create temporary matrix for sketch
        El::DistMatrix<value_type, El::STAR, El::STAR> R1(grid);
        R1.AlignWith(sketch_of_A);

        // if no blocksize is specified use full width
        El::Int blocksize = get_blocksize();
        if (blocksize == 0) {
            blocksize = A.width();
        }

        const int* indptr  = A.indptr();
        const int* indices = A.indices();
        const value_type *values = A.locked_values();

        El::Int base = 0;
        El::Int remaining_width = A.width();
        while (base < A.width()) {

            El::Int b = std::min(A.width() - base, blocksize);
            data_type::realize_matrix_view(R1, 0, base, sketch_of_A.Width(), b);

            // Perform local Gemm starting from column base with with b
            // FIXME: we should provide a proper View on sparse matrices, even
            //        though that may not be optimal for all distributions.
#if SKYLARK_HAVE_OPENMP
            #pragma omp parallel for
#endif
            for(int i = 0; i < sketch_of_A.Width(); i++)
                for(int col = base; col < base + b; col++) {
                    int g_col = A.global_col(col);
                    for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                        int row = A.global_row(indices[j]);
                        value_type val = values[j];
                        sketch_of_A.Update(row, i, val * R1.Get(i, g_col));
                    }
                }

            base += b;
        }
    }

    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

    }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_MIXED_HPP
