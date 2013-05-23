/**
 * @file matrix.cpp
 * @author pkambadu
 *
 * Python bindings for matrix number generators.
 */

#include <boost/python.hpp>
#include <utility/matrix.hpp>

namespace skylark {
namespace python {

/**
 * A function to export all the definitions for matrix object.
 */
void export_matrix () {

    using namespace boost::python;
    using namespace skylark::utility;

    class_<row_major_tag>("row_major");
    class_<col_major_tag>("col_major");
    class_<one_D_tag>("one_D");
    class_<two_D_tag>("two_D");
    class_<dense_tag>("dense");
    class_<sparse_tag>("sparse");

}

} // namespace python
} // namespace skylark
