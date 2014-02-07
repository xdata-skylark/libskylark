#ifndef SKYLARK_HASH_TRANSFORM_LOCAL_SPARSE_HPP
#define SKYLARK_HASH_TRANSFORM_LOCAL_SPARSE_HPP

#include "../utility/sparse_matrix.hpp"
#include "../utility/exception.hpp"

#include "context.hpp"
#include "transforms.hpp"
#include "hash_transform_data.hpp"

namespace skylark { namespace sketch {

/* Specialization: local SpMat for input, output */
template <typename IndexType,
          typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    utility::sparse_matrix_t<IndexType, ValueType>,
    utility::sparse_matrix_t<IndexType, ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IndexType,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution> {
    typedef IndexType index_type;
    typedef ValueType value_type;
    typedef utility::sparse_matrix_t<IndexType, ValueType> matrix_type;
    typedef utility::sparse_matrix_t<IndexType, ValueType> output_matrix_type;
    typedef hash_transform_data_t<IndexType,
                                  ValueType,
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

        index_type row_idx = 0;
        typename matrix_type::const_ind_itr_range_t ritr = A.indptr_itr();
        typename matrix_type::const_ind_itr_range_t citr = A.indices_itr();
        typename matrix_type::const_val_itr_range_t vitr = A.values_itr();

        for(; ritr.first + 1 != ritr.second; ritr.first++, ++row_idx) {
            for(size_t idx = 0; idx < (*(ritr.first + 1) - *ritr.first);
                citr.first++, vitr.first++, ++idx) {

                index_type col_idx = *citr.first;
                value_type value   = *vitr.first *
                                     getRowValue(row_idx, col_idx, dist);

                newPos(row_idx, col_idx, dist);
                typename output_matrix_type::coord_tuple_t new_entry(
                        row_idx, col_idx, value);

                coords.push_back(new_entry);
            }
        }

        sketch_of_A.Attach(coords);
    }

    inline void newPos(index_type &rowid, index_type &colid, columnwise_tag) const {
        rowid = base_data_t::row_idx[rowid];
    }

    inline void newPos(index_type &rowid, index_type &colid, rowwise_tag) const {
        colid = base_data_t::row_idx[colid];
    }

    inline value_type getRowValue(index_type rowid, index_type colid,
        columnwise_tag) const {
        return base_data_t::row_value[rowid];
    }

    inline value_type getRowValue(index_type rowid, index_type colid,
        rowwise_tag) const {
        return base_data_t::row_value[colid];
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_LOCAL_SPARSE_HPP
