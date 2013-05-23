/**
 * @file JLT.cpp
 * @author pkambadu
 *
 * Python bindings for the JLT object.
 */

#include <boost/python.hpp>
#include <boost/mpi.hpp>

#include <sketch/JLT.hpp>
#include <sketch/context.hpp>
#include <sketch/JLT.hpp>

#include <utility/mapper.hpp>
#include <utility/matrix.hpp>
#include <utility/py_iterator.hpp>

namespace skylark {
namespace python {

/**
 * A function to export all the definitions for JLT object.
 */
void export_JLT () {

    using namespace boost::python;
    using namespace skylark;
    using namespace skylark::sketch;
    using namespace skylark::utility;

    /// We define all our dense matrix version for doubles and int mapper
    typedef py_iterator_t<double> dbl_iterator_t;
    typedef py_iterator_t<int> int_iterator_t;
    typedef interval_mapper_t<int> int_mapper_t;

    /// Helper variable
    typedef dbl_iterator_t::difference_type size_type;

    /// Now, typedef some of the frequently used matrix types
    typedef dense_1D_t < dbl_iterator_t,
                         int_mapper_t,
                         row_major_tag > dense_1D_row_major_t;

    typedef dense_1D_t < dbl_iterator_t,
                         int_mapper_t,
                         col_major_tag > dense_1D_col_major_t;

    typedef JLT_t<dense_1D_row_major_t> JLT_1D_row_major_t;
    typedef JLT_t<dense_1D_col_major_t> JLT_1D_col_major_t;

    class_<JLT_1D_row_major_t>("JLT", init<int, int, context_t&>())
        .def ("preview",
            &JLT_1D_row_major_t::preview<columnwise_tag>)
        .def ("apply",
            &JLT_1D_row_major_t::apply<columnwise_tag>)
    ;

}

} // namespace python
} // namespace skylark
