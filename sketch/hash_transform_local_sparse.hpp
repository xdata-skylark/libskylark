#ifndef SKYLARK_HASH_TRANSFORM_LOCAL_SPARSE_HPP
#define SKYLARK_HASH_TRANSFORM_LOCAL_SPARSE_HPP

#include "../base/sparse_matrix.hpp"
#include "../utility/exception.hpp"

#include "context.hpp"
#include "transforms.hpp"
#include "hash_transform_data.hpp"

namespace skylark { namespace sketch {

/* Specialization: local SpMat for input, output */
template <typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    base::sparse_matrix_t<ValueType>,
    base::sparse_matrix_t<ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<size_t,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution>,
        virtual public sketch_transform_t<base::sparse_matrix_t<ValueType>,
                                          base::sparse_matrix_t<ValueType> >  {
    typedef size_t index_type;
    typedef ValueType value_type;
    typedef base::sparse_matrix_t<ValueType> matrix_type;
    typedef base::sparse_matrix_t<ValueType> output_matrix_type;
    typedef IdxDistributionType<index_type> idx_distribution_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef hash_transform_data_t<index_type,
                                  value_type,
                                  IdxDistributionType,
                                  ValueDistribution> base_data_t;


    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, skylark::sketch::context_t& context) :
        base_data_t(N, S, context) {}

    /**
     * Copy constructor
     */
    template <typename InputMatrixType,
              typename OutputMatrixType>
    hash_transform_t (hash_transform_t<InputMatrixType,
                                       OutputMatrixType,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        base_data_t(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (hash_transform_data_t<index_type,
                                            value_type,
                                            IdxDistributionType,
                                            ValueDistribution>& other_data) :
        base_data_t(other_data) {}

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type &A, output_matrix_type &sketch_of_A,
                columnwise_tag dimension) const {
        try {
            apply_impl (A, sketch_of_A, dimension);
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                utility::mpi_exception()
                    << utility::error_msg(e.what()) );
        } catch (std::string e) {
            SKYLARK_THROW_EXCEPTION (
                utility::combblas_exception()
                    << utility::error_msg(e) );
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                utility::combblas_exception()
                    << utility::error_msg(e.what()) );
        }
    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type &A, output_matrix_type &sketch_of_A,
                rowwise_tag dimension) const {
        try {
            apply_impl (A, sketch_of_A, dimension);
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                utility::mpi_exception()
                    << utility::error_msg(e.what()) );
        } catch (std::string e) {
            SKYLARK_THROW_EXCEPTION (
                utility::combblas_exception()
                    << utility::error_msg(e) );
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                utility::combblas_exception()
                    << utility::error_msg(e.what()) );
        }
    }

    int get_N() const { return this->N; } /**< Get input dimension. */
    int get_S() const { return this->S; } /**< Get output dimension. */

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply_impl (const matrix_type &A,
                     output_matrix_type &sketch_of_A,
                     Dimension dist) const {

        typename output_matrix_type::coords_t coords;

        index_type col_idx = 0;

        const int* indptr = A.indptr();
        const int* indices = A.indices();
        const value_type* values = A.locked_values();

        for(index_type col = 0; col < A.width(); col++) {
            for(index_type idx = indptr[col]; idx < indptr[col + 1]; idx++) {

                index_type coltmp = col;
                index_type row = indices[idx];
                value_type value = values[idx] * get_value(row, coltmp, dist);

                final_pos(row, coltmp, dist);
                typename output_matrix_type::coord_tuple_t
                    new_entry(row,  coltmp, value);

                coords.push_back(new_entry);
            }
        }

        index_type n_rows = sketch_rows(A, dist);
        index_type n_cols = sketch_cols(A, dist);

        sketch_of_A.set(coords, n_rows, n_cols);
    }

    inline void final_pos(index_type &rowid, index_type &colid,
        columnwise_tag) const {
        rowid = base_data_t::row_idx[rowid];
    }

    inline void final_pos(index_type &rowid, index_type &colid,
        rowwise_tag) const {
        colid = base_data_t::row_idx[colid];
    }

    inline value_type get_value(index_type rowid, index_type colid,
        columnwise_tag) const {
        return base_data_t::row_value[rowid];
    }

    inline value_type get_value(index_type rowid, index_type colid,
        rowwise_tag) const {
        return base_data_t::row_value[colid];
    }

    inline index_type sketch_rows(const matrix_type &A, columnwise_tag) const {
        return base_data_t::S;
    }

    inline index_type sketch_rows(const matrix_type &A, rowwise_tag) const {
        return A.height();
    }

    inline index_type sketch_cols(const matrix_type &A, columnwise_tag) const {
        return A.width();
    }

    inline index_type sketch_cols(const matrix_type &A, rowwise_tag) const {
        return base_data_t::S;
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_LOCAL_SPARSE_HPP
