#ifndef SKYLARK_PPT_HPP
#define SKYLARK_PPT_HPP

#include "PPT_data.hpp"

namespace skylark { namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct PPT_t :
        public PPT_data_t,
        virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {
    // To be specilized and derived. Just some guards here.
    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;
    typedef PPT_data_t data_type;


    PPT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }

private:
    PPT_t(int N, int S, base::context_t& context);

};

} } /** namespace skylark::sketch */

#if SKYLARK_HAVE_ELEMENTAL
# include "PPT_Elemental.hpp"
#endif

#endif // SKYLARK_PPT_HPP
