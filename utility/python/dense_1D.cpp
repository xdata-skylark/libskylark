/**
 * @file dense_1D.cpp
 * @author pkambadu
 *
 * Python bindings for the dense_1D object.
 */

#include <boost/python.hpp>
#include <boost/mpi.hpp>

#include <utility/dense_1D.hpp>
#include <utility/mapper.hpp>
#include <utility/matrix.hpp>
#include <utility/py_iterator.hpp>
#include <sketch/context.hpp>

namespace skylark {
namespace python {

/**
 * A function to export all the definitions for dense_1D object.
 */
void export_dense_1D () {

    using namespace boost::python;
    using namespace skylark;
    using namespace skylark::sketch;
    using namespace skylark::python;
    using namespace skylark::utility;

    /** We define all our dense matrix version for doubles and int mapper */
    typedef py_iterator_t<double> dbl_iterator_t;
    typedef py_iterator_t<int> int_iterator_t;
    typedef interval_mapper_t<int> int_mapper_t;

    /** Helper variable */
    typedef dbl_iterator_t::difference_type size_type;

    /** Now, typedef some of the frequently used matrix types */
    typedef dense_1D_t < dbl_iterator_t,
                         int_mapper_t,
                         row_major_tag > dense_1D_row_major_t;

    typedef dense_1D_t < dbl_iterator_t,
                         int_mapper_t,
                         col_major_tag > dense_1D_col_major_t;

    /**
     * We have to mention the default constructor to use in the opening line ---
     * since dense_1D_t does not have a "()" constructor, it is important to
     * put one of the two constructors within the opening line.
     */
    class_<dense_1D_row_major_t>("dense_1D_row", init < std::string,
                                                        size_type,
                                                        size_type,
                                                        int_mapper_t,
                                                        context_t& > ())
        .def (init < std::string,
                     size_type,
                     size_type,
                     int_mapper_t,
                     context_t&,
                     dbl_iterator_t > ())
        .add_property ("M", &dense_1D_row_major_t::M)
        .add_property ("N", &dense_1D_row_major_t::N)
        .def ("owner", &dense_1D_row_major_t::owner)
        .def ("size", &dense_1D_row_major_t::size)
        .def ("range", &dense_1D_row_major_t::size)
        .def ("set_buffer", &dense_1D_row_major_t::set_buffer)
        .def ("randomize", &dense_1D_row_major_t::randomize)
        .def ("read", &dense_1D_row_major_t::read)
        .def ("pretty_print",
            (void (dense_1D_row_major_t::*)(bool) const)
                &dense_1D_row_major_t::pretty_print,
            (arg("print_values") = true))
    ;

    class_<dense_1D_col_major_t>("dense_1D_col", init < std::string,
                                                        size_type,
                                                        size_type,
                                                        int_mapper_t,
                                                        context_t& > ())
        .def (init < std::string,
                     size_type,
                     size_type,
                     int_mapper_t,
                     context_t&,
                     dbl_iterator_t > ())
        .add_property ("M", &dense_1D_col_major_t::M)
        .add_property ("N", &dense_1D_col_major_t::N)
        .def ("owner", &dense_1D_col_major_t::owner)
        .def ("size", &dense_1D_col_major_t::size)
        .def ("range", &dense_1D_col_major_t::size)
        .def ("set_buffer", &dense_1D_col_major_t::set_buffer)
        .def ("randomize", &dense_1D_col_major_t::randomize)
        .def ("read", &dense_1D_col_major_t::read)
        .def ("pretty_print",
            (void (dense_1D_col_major_t::*)(bool) const)
                &dense_1D_row_major_t::pretty_print,
            (arg("print_values") = true))
    ;
}

} // namsepace python
} // namespace skylark
