/**
 * @file module.cpp
 * @author pkambadu
 *
 * This file contains all the sources that are used for the python bindings.
 */

#include <boost/python.hpp>

namespace skylark {
namespace python {


extern void export_partitioner();
extern void export_iterator();
extern void export_mapper();
extern void export_matrix();
extern void export_dense_1D();

extern const char* module_docstring;

BOOST_PYTHON_MODULE(_utility) {

    using namespace boost::python;

    scope().attr("__doc__") = module_docstring;
    scope().attr("__author__") = "XDATA";
    scope().attr("__date__") = "xx.xx.xxxx";
    scope().attr("__version__") = "1";
    scope().attr("__copyright__") = "Copyright (C) 2013 IBM";
    scope().attr("__license__") = "http://www.eclipse.org/legal/epl-v10.html";

    skylark::python::export_partitioner();
    skylark::python::export_iterator();
    skylark::python::export_mapper();
    skylark::python::export_matrix();
    skylark::python::export_dense_1D();

}

} // namespace python
} // namespace skylark
