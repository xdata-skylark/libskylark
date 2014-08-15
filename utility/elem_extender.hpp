#ifndef SKYLARK_ELEM_EXTENDER_HPP
#define SKYLARK_ELEM_EXTENDER_HPP

#include <elemental.hpp>

#include "typer.hpp"
namespace skylark { namespace utility {

/**
 * Sometimes elemental class lack a few operations that make life a bit
 * easier in some codes (i.e. operator []). This class adds those.
 */
template<typename ET>
struct elem_extender_t : public ET {

    // Once we have c'tor inheritance (e.g. gcc-4.8) we can simply
    // use the following line:
    // using ET::ET;
    // For now, I am just implementing the most basic c'tors. More
    // will be added as neccessary.
    elem_extender_t(const ET& other) : ET(other) {  }
    elem_extender_t(const ET&& other) : ET(other) {  }
    elem_extender_t(int m, int n, const elem::Grid& grid=elem::DefaultGrid(),
        int root=0) : ET(m, n, grid, root) { }

private:
    typedef typename utility::typer_t<ET>::value_type value_type;

public:
    /// Returns a reference to local i in Buffer() (ignores LDim!!!)
    value_type &operator[](int i) {
        return *(ET::Buffer() + i);
    }

    /// Returns a reference to local i in Buffer() (ignores LDim!!!)
    const value_type &operator[](int i) const {
        return *(ET::Buffer() + i);
    }
};


} }
#endif
