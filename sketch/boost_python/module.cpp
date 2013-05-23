/**
 * @file module.cpp
 * @author pkambadu
 *
 * This file contains all the sources that are used for the python bindings.
 */

#include <boost/python.hpp>

namespace skylark {
namespace python {


extern void export_context();
extern void export_transforms();
extern void export_JLT();

extern const char* module_docstring;

BOOST_PYTHON_MODULE(_sketch) {

    using namespace boost::python;

    scope().attr("__doc__") = module_docstring;
    scope().attr("__author__") = "XDATA";
    scope().attr("__date__") = "xx.xx.xxxx";
    scope().attr("__version__") = "1";
    scope().attr("__copyright__") = "Copyright (C) 2013 IBM";
    scope().attr("__license__") = "http://www.eclipse.org/legal/epl-v10.html";

    skylark::python::export_context();
    skylark::python::export_transforms();
    skylark::python::export_JLT();

}

} // namespace python
} // namespace skylark
