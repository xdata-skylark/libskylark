/**
 * @file sketch.cpp
 * @author pkambadu
 *
 * Python bindings for sketch number generators.
 */

#include <boost/python.hpp>
#include <sketch/transforms.hpp>

namespace skylark {
namespace python {

/**
 * A function to export all the definitions for sketch object.
 */
void export_transforms () {

    using namespace boost::python;
    using namespace skylark::sketch;

    class_<columnwise_tag>("left");
    class_<rowwise_tag>("right");

}

} // namespace python
} // namespace skylark
