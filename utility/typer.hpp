#ifndef SKYLARK_TYPER_HPP
#define SKYLARK_TYPER_HPP

namespace skylark {
namespace utility {

template<typename T>
struct typer_t {
    typedef typename T::value_type value_type;
    typedef typename T::index_type index_type;
};

#ifdef SKYLARK_HAVE_ELEMENTAL

#include <elemental.hpp>

template<typename F>
struct typer_t<elem::Matrix<F>> {
    typedef F value_type;
    typedef int index_type;
};

template<typename F, elem::Distribution CD, elem::Distribution RD>
struct typer_t< elem::DistMatrix<F, CD, RD> > {
    typedef F value_type;
    typedef int index_type;
};

#endif

#ifdef SKYLARK_HAVE_COMBBLAS

#include <CombBLAS.h>

template<typename IT, typename VT, typename DT>
struct typer_t< SpParMat<IT, VT, DT> > {
    typedef VT value_type;
    typedef IT index_type;
};

template<typename IT, typename VT>
struct typer_t< FullyDistVec<IT, VT> > {
    typedef VT value_type;
    typedef IT index_type;
};

#endif

} }  // namespace skylark::utility

#endif // SKYLARK_TYPER_HPP
