/**
 * @file partitioner.cpp
 * @author pkambadu
 *
 * Python bindings for the partitioner object.
 */

#include <boost/python.hpp>
#include <boost/mpi.hpp>

#include <utility/partitioner.hpp>
#include <utility/py_iterator.hpp>

namespace skylark {
namespace python {

extern const char* partitioner_divide_docstring;

#define DEFINE_PARTITIONER(TYPE) \
  typedef partitioner_t<TYPE ## _t> TYPE ## _t_partitioner_t; \
  typedef py_iterator_t<TYPE ## _t> TYPE ## _t_iterator_t; \
  class_<TYPE ## _t_partitioner_t>(#TYPE "_partitioner") \
    .def("divide", \
          &TYPE## _t_partitioner_t::intervals<TYPE## _t_iterator_t>, \
          partitioner_divide_docstring) \
    .staticmethod("divide")

/**
 * A function to export all the definitions for partitioner object.
 */
void export_partitioner () {

    using namespace boost::mpi;
    using namespace boost::python;
    using namespace skylark::utility;

    DEFINE_PARTITIONER(int32);
    DEFINE_PARTITIONER(int64);

}

} // namespace python
} // namespace skylark
