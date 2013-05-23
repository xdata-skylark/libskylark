/**
 * @file context.cpp
 * @author pkambadu
 *
 * Python bindings for the context object.
 */

#include <boost/python.hpp>
#include <boost/mpi.hpp>
#include <sketch/context.hpp>

namespace skylark {
namespace python {

/**
 * A function to export all the definitions for context object.
 */
void export_context () {

    using namespace boost::mpi;
    using namespace boost::python;
    using namespace skylark::sketch;

    class_<context_t>("context", init<int, const communicator&>())
        .add_property("rank", &context_t::rank)
        .add_property("size", &context_t::size)
        .def("newseed", &context_t::newseed)
    ;

}

} // namespace python
} // namespace skylark
