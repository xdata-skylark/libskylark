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
                                     ValueDistribution> {
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
        base_data_t(other.get_data()) {}

    /**
     * Constructor from data
     */
    hash_transform_t (hash_transform_data_t<index_type,
                                            value_type,
                                            IdxDistributionType,
                                            ValueDistribution>& other_data) :
        base_data_t(other_data.get_data()) {}

    template <typename Dimension>
    void apply (const matrix_type &A,
                output_matrix_type &sketch_of_A,
                Dimension dimension) const {
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
        typename matrix_type::const_ind_itr_range_t citr = A.indptr_itr();
        typename matrix_type::const_ind_itr_range_t ritr = A.indices_itr();
        typename matrix_type::const_val_itr_range_t vitr = A.values_itr();

        for(; citr.first + 1 != citr.second; citr.first++, ++col_idx) {
            for(index_type idx = 0; idx < (*(citr.first + 1) - *citr.first);
                ritr.first++, vitr.first++, ++idx) {

                index_type col     =  col_idx;
                index_type row_idx = *ritr.first;
                value_type value   = *vitr.first * get_value(row_idx, col_idx, dist);

                final_pos(row_idx, col, dist);
                typename output_matrix_type::coord_tuple_t
                    new_entry(row_idx,  col, value);

                coords.push_back(new_entry);
            }
        }

        index_type n_rows = sketch_rows(A, dist);
        index_type n_cols = sketch_cols(A, dist);

        sketch_of_A.attach(coords, n_rows, n_cols);
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
