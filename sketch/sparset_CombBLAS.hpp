#ifndef SPARSET_COMBBLAS_HPP
#define SPARSET_COMBBLAS_HPP

#include "config.h"

#include "context.hpp"
#include "transforms.hpp"

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpParMat.h"


namespace skylark {
namespace sketch {


template <typename ValueType,
          typename IdxDistributionType,
          template <typename> class ValueDistributionType>
struct hash_transform_t <SpParMat<size_t, ValueType, SpDCCols<size_t, double> >,
                         SpParMat<size_t, ValueType, SpDCCols<size_t, double> >,
                         IdxDistributionType,
                         ValueDistributionType > {

public:
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef SpDCCols< size_t, value_type > col_t;
    typedef SpParMat< size_t, value_type, col_t > matrix_t;
    typedef SpParMat< size_t, value_type, col_t > output_matrix_t;
    typedef IdxDistributionType idx_distribution_type;
    typedef ValueDistributionType<value_type> value_distribution_type;

    hash_transform_t (int N, int S, skylark::sketch::context_t& context)
        : _N(N), _S(S), _context(context) {

        _row_idx.resize(N);
        _row_value.resize(N);

        boost::random::mt19937 prng(context.newseed());
        idx_distribution_type   row(0, _S - 1);
        value_distribution_type row_value;

        for (int i = 0; i < N; ++i) {
            _row_idx[i]   = row(prng);
            _row_value[i] = row_value(prng);
        }
    }

    template <typename Dimension>
    void apply (matrix_t &A, output_matrix_t &sketch_of_A,
                Dimension dimension) {

        apply_impl (A, sketch_of_A, dimension);
    }


private:

    /// Input dimension
    const int _N;
    /// Output dimension
    const int _S;
    /// context for this sketch
    skylark::sketch::context_t& _context;

    // precomputed row index and value per column of Pi
    std::vector<int> _row_idx;
    std::vector<value_type> _row_value;

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the column-wise direction of sketching.
     */
    void apply_impl (matrix_t &A,
                     output_matrix_t &sketch_of_A,
                     skylark::sketch::columnwise_tag) {

        // extract columns of matrix
        col_t &data = A.seq();
        size_t matrix_size = sketch_of_A.getncol() * sketch_of_A.getnrow();

        for(typename col_t::SpColIter col = data.begcol();
            col != data.endcol(); col++) {
            for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
                nz != data.endnz(col); nz++) {

                FullyDistVec<size_t, double> cols(matrix_size, 0.0);
                FullyDistVec<size_t, double> rows(matrix_size, 0.0);
                FullyDistVec<size_t, double> vals(matrix_size, 0.0);

                size_t row_begin = col.colid();
                size_t pos = row_begin + _row_idx[nz.rowid()] * data.getncol();

                cols.SetElement(pos, col.colid());
                rows.SetElement(pos, _row_idx[nz.rowid()]);
                vals.SetElement(pos, _row_value[nz.rowid()] * nz.value());

                output_matrix_t tmp(sketch_of_A.getnrow(),
                        sketch_of_A.getncol(), rows, cols, vals);

                sketch_of_A += tmp;
            }
        }

        //TODO: pull everything to rank 0?
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the row-wise direction of sketching.
     */
    void apply_impl (matrix_t &A,
                     output_matrix_t &sketch_of_A,
                     skylark::sketch::rowwise_tag) {

        // extract columns of matrix
        col_t &data = A.seq();
        size_t matrix_size = sketch_of_A.getncol() * sketch_of_A.getnrow();

        for(typename col_t::SpColIter col = data.begcol();
            col != data.endcol(); col++) {
            for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
                nz != data.endnz(col); nz++) {

                FullyDistVec<size_t, double> cols(matrix_size, 0.0);
                FullyDistVec<size_t, double> rows(matrix_size, 0.0);
                FullyDistVec<size_t, double> vals(matrix_size, 0.0);

                // new value at (nz.rowid(), col.colid())
                size_t pos = nz.rowid() + _row_idx[col.colid()] * data.getncol();

                rows.SetElement(pos, nz.rowid());
                cols.SetElement(pos, _row_idx[col.colid()]);
                vals.SetElement(pos, _row_value[col.colid()] * nz.value());

                output_matrix_t tmp(sketch_of_A.getnrow(),
                        sketch_of_A.getncol(), rows, cols, vals);

                sketch_of_A += tmp;
            }
        }

        //TODO: pull everything to rank 0?
    }
};

} // namespace sketch
} // namespace skylark

#endif // SPARSET_COMBBLAS_HPP
