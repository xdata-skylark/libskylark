#ifndef SKYLARK_FRFT_ELEMENTAL_HPP
#define SKYLARK_FRFT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"
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
private:
    // DEFINE DCT

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    FastRFT_t(const FastRFT_t<matrix_type,
                      output_matrix_type>& other)
        : base_data_t(other) {

    }

    /**
     * Constructor from data
     */
    FastRFT_t(const base_data_t& other_data)
        : base_data_t(other_data) {

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
        matrix_type W = A;

        for(int i = 0; i < base_data_t::numblks; i++) {
            int s = i * base_data_t::N;
            int e = std::min(s + base_data_t::N, base_data_t::S);

            // DO SOME MAGIC HERE!

            // Copy that part to the output
            output_matrix_type view_sketch_of_A;
            elem::View(view_sketch_of_A, sketch_of_A, s, 0, e - s, A.Width());
            matrix_type view_W;
            elem::View(view_W, W, 0, 0, e - s, A.Width());
            view_sketch_of_A = W;
        }
    }

    /**
      * Apply the sketching transform on A and write to  sketch_of_A.
      * Implementation rowwise.
      */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {


    }
};



} } /** namespace skylark::sketch */

#endif // SKYLARK_FRFT_ELEMENTAL_HPP
