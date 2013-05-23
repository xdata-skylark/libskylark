/**
 * @file py_iterator.cpp
 * @author pkambadu
 *
 * Python bindings for random access iterators.
 */

#include <boost/python.hpp>
#include <iterator>

#include <utility/py_iterator.hpp>

namespace skylark {
namespace python {

/**
 * A function to export all the definitions for py_iterator object.
 */
void export_iterator () {

    using namespace boost::python;
    using namespace skylark::utility;

    class_<py_iterator_t<int> >("int_iterator")
        .def(init<object>())
        .def(init<object, int>());
    class_<py_iterator_t<double> >("dbl_iterator")
        .def(init<object>())
        .def(init<object, int>());

}

} // namespace python
} // namespace skylark
