#ifndef SKYLARK_DENSE_TRANSFORM_DATA_HPP
#define SKYLARK_DENSE_TRANSFORM_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

#include "../utility/randgen.hpp"
#include "boost/smart_ptr.hpp"

namespace skylark { namespace sketch {

//FIXME: WHY DO WE NEED TO ALLOW COPY CONSTRUCTOR HERE (or more precisely in
//       dense_transform_Elemental)?
/**
 * This is the base data class for dense transforms. Essentially, it
 * holds the input and sketched matrix sizes and the array of samples
 * to be lazily computed.
 */
template <template <typename> class ValueDistribution>
struct dense_transform_data_t : public sketch_transform_data_t {
    typedef sketch_transform_data_t base_t;

    // Note: we always generate doubles for array values,
    // but when applying to floats the size can be reduced.
    typedef ValueDistribution<double> value_distribution_type;

    typedef double value_type;

    /**
     * Regular constructor
     */
    dense_transform_data_t (int N, int S, double scale, 
        base::context_t& context)
        : base_t(N, S, context, "DenseTransform"),
          scale(scale), distribution() {

        // No scaling in "raw" form
        context = build();
    }

    virtual
    boost::property_tree::ptree to_ptree() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Do not yet support serialization of generic dense transform"));

        return boost::property_tree::ptree();
    }

    dense_transform_data_t(const dense_transform_data_t& other)
        : base_t(other), scale(other.scale), distribution(other.distribution),
          random_samples(other.random_samples)  {

    }


    void realize_matrix_view(elem::Matrix<value_type>& A) const {
        realize_matrix_view(A, 0, 0, _S, _N);
    }


    void realize_matrix_view(elem::Matrix<value_type>& A,
        int i, int j, int height, int width) const {
        realize_matrix_view(A, i, j, height, width, 1, 1);
    }


    void realize_matrix_view(elem::Matrix<value_type>& A,
        int i, int j, int height, int width,
        int col_stride, int row_stride) const {

        A.Resize(height, width);
        value_type *data = A.Buffer();

#ifdef SKYLARK_HAVE_OPENMP
#pragma omp parallel for
#endif
        for(int j_loc = 0; j_loc < width; j_loc++) {
            int j_glob = j + j_loc * row_stride;
            for (int i_loc = 0; i_loc < height; i_loc++) {
                int i_glob = i + i_loc * col_stride;
                value_type sample =
                    random_samples[j_glob * _S + i_glob];
                data[j_loc * height + i_loc] = scale * sample;
            }
        }
    }


    template<elem::Distribution ColDist,
             elem::Distribution RowDist>
    void realize_matrix_view(elem::DistMatrix<value_type,
                                              ColDist,
                                              RowDist>& A) const {
        realize_matrix_view<ColDist, RowDist>(A, 0, 0, _S, _N);
    }


    template<elem::Distribution ColDist,
             elem::Distribution RowDist>
    void realize_matrix_view(elem::DistMatrix<value_type, ColDist, RowDist>& A,
        int i, int j, int height, int width) const {

        elem::DistMatrix<value_type, ColDist, RowDist> parent;
        const elem::Grid& grid = parent.Grid();

        // for view (A) and parent matrices: stride, rank are the same
        const int col_stride    = parent.ColStride();
        const int row_stride    = parent.RowStride();
        const int col_rank      = parent.ColRank();
        const int row_rank      = parent.RowRank();

        // for view (A) and parent matrices: alignment (and shift) are different
        const int parent_col_alignment = parent.ColAlign();
        const int parent_row_alignment = parent.RowAlign();

        const int col_alignment = (parent_col_alignment + i) % col_stride;
        const int row_alignment = (parent_row_alignment + j) % row_stride;
        const int col_shift     =
            elem::Shift(col_rank, col_alignment, col_stride);
        const int row_shift     =
            elem::Shift(row_rank, row_alignment, row_stride);
        const int local_height        =
            elem::Length(height, col_shift, col_stride);
        const int local_width         =
            elem::Length(width,  row_shift, row_stride);

        A.Empty();

        A = elem::DistMatrix<value_type, ColDist, RowDist>(height, width,
            col_alignment, row_alignment, grid);

        elem::Matrix<value_type>& local_matrix = A.Matrix();
        realize_matrix_view(local_matrix,
            i + col_shift, j + row_shift,
            local_height, local_width,
            col_stride, row_stride);
    }


protected:

    dense_transform_data_t (int N, int S, double scale, 
        const base::context_t& context, std::string type)
        : base_t(N, S, context, type),
          scale(scale),
          distribution() {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();
        random_samples = ctx.allocate_random_samples_array(_N * _S, distribution);
        return ctx;
    }

    double scale; /**< Scaling factor for the samples */
    value_distribution_type distribution; /**< Distribution for samples */
    skylark::utility::random_samples_array_t <value_distribution_type>
    random_samples;
    /**< Array of samples, to be lazily computed */


};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_DENSE_TRANSFORM_DATA_HPP */
