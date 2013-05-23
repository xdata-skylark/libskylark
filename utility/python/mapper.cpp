/**
 * @file mapper.cpp
 * @author pkambadu
 *
 * Python bindings for the mapper object.
 */

#include <boost/python.hpp>
#include <boost/mpi.hpp>

#include <utility/mapper.hpp>
#include <utility/py_iterator.hpp>

namespace skylark {
namespace python {

/**
 * A function to export all the definitions for mapper object.
 */
void export_mapper () {

    using namespace boost::python;
    using namespace skylark::utility;

    typedef py_iterator_t<int> int_iterator_t;
    typedef interval_mapper_t<int> int_mapper_t;

    class_<int_mapper_t>("mapper", init<int_iterator_t, int_iterator_t>())
    .def (init<int_iterator_t, int>())
    .def (init<const int_mapper_t&>())
    .def ("set", &int_mapper_t::set<int_iterator_t>)
    .def ("range", &int_mapper_t::range)
    .def ("begin", &int_mapper_t::begin)
    .def ("end", &int_mapper_t::end)
    .def ("pretty_print", &int_mapper_t::pretty_print)
    ;

}

} // namespace python
} // namespace skylark
