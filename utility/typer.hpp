#ifndef SKYLARK_TYPER_HPP
#define SKYLARK_TYPER_HPP

#include <El.hpp>
#ifdef SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#endif

namespace skylark {
namespace utility {

template<typename T>
struct typer_t {
    typedef typename T::value_type value_type;
    typedef typename T::index_type index_type;
};


template<typename F>
struct typer_t<El::Matrix<F> > {
    typedef F value_type;
    typedef El::Int index_type;
};

template<typename F>
struct typer_t< El::AbstractDistMatrix<F> > {
    typedef F value_type;
    typedef El::Int index_type;
};

template<typename F>
struct typer_t< El::ElementalMatrix<F> > {
    typedef F value_type;
    typedef El::Int index_type;
};

template<typename F, El::Distribution CD, El::Distribution RD>
struct typer_t< El::DistMatrix<F, CD, RD> > {
    typedef F value_type;
    typedef El::Int index_type;
};

#ifdef SKYLARK_HAVE_COMBBLAS
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
